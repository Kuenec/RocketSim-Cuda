// 2026-06-10: Monolithic GPU step.
//
// Architecture (per project requirements): ONE __global__ kernel executes the ENTIRE
// physics tick for all arenas in the batch, looping ticks inside the kernel so a whole
// gate run is a single launch with zero per-tick CPU work. Each physics phase is a
// __device__ function called from the kernel in CPU reference order:
//
//   per tick:
//     1. ball zero-vel sleep check        (Arena::Step head)            [not ported]
//     2. car pre-tick (vehicle+game)      (Car::_PreTickUpdate)         [not ported]
//     3. pad pre-tick                     (BoostPad::_PreTickUpdate)    [not ported]
//     4. ball pre-tick                    (Ball::_PreTickUpdate)        [not ported]
//     5. bullet: predict (damping)        (predictUnconstraintMotion)   [PORTED]
//     6. bullet: collision detection      (broadphase+narrowphase+cb)   [not ported]
//     7. bullet: islands + solver         (btSequentialImpulse...)      [not ported]
//     8. bullet: integrate transforms     (integrateTransforms)         [PORTED]
//     9. bullet: update activation        (updateActivationState)       [PORTED]
//    10. car post-tick + finish + pads    (Car::_PostTickUpdate)        [not ported]
//    11. pad post-tick, ball finish       (Arena::Step tail)            [not ported]
//
// The kernel does NOT yet produce correct simulation output (unported phases above).
// CUDA% stays 0 until the full step matches the CPU reference. --gpu-compare in
// rocketsim_bench validates each PORTED phase bit-exactly against CPU capture points.

#include "CudaStep.h"
#include "CudaMath.cuh"
#include "CudaRaycast.cuh"
#include "CudaVehicle.cuh"
#include "CudaCollision.cuh"
#include "CudaCollideWorld.cuh"
#include "CudaSolver.cuh"
#include "CudaGameTail.cuh"
#include "RsqrtHost.h"

#include <cuda_runtime.h>
#include <cuda/atomic>
#include "GpuAllocationError.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace rscuda {

static void CudaCheck(cudaError_t err, const char* what) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
		exit(2);
	}
}

void ThrowAllocationError(RocketSim::GpuAllocationKind kind, const char* buffer, size_t bytes,
	int cudaError) {
	cudaGetLastError();
	throw RocketSim::GpuAllocationError(kind, buffer, bytes,
		cudaGetErrorString((cudaError_t)cudaError));
}

template <class T>
static void DeviceAlloc(T*& ptr, size_t bytes, const char* buffer) {
	void* allocated = nullptr;
	const cudaError_t err = cudaMalloc(&allocated, bytes);
	if (err != cudaSuccess) {
		ptr = nullptr;
		ThrowAllocationError(RocketSim::GpuAllocationKind::DEVICE, buffer, bytes, err);
	}
	ptr = (T*)allocated;
}

template <class T>
static void PinnedHostAlloc(T*& ptr, size_t bytes, const char* buffer) {
	void* allocated = nullptr;
	const cudaError_t err = cudaMallocHost(&allocated, bytes);
	if (err != cudaSuccess) {
		ptr = nullptr;
		ThrowAllocationError(RocketSim::GpuAllocationKind::PINNED_HOST, buffer, bytes, err);
	}
	ptr = (T*)allocated;
}

struct GpuBatch;
static cudaStream_t BatchStream(GpuBatch* b);

constexpr int CU_MAX_PADS = 64;
constexpr int CU_STEP_BLOCK_THREADS = 32;
constexpr unsigned CU_SLOW_BLOCK_DIVISOR = 4;

struct CuStepConsts {
	GpuCarConsts car;
	GpuTailConsts tail;
	GpuPadConfig pads[CU_MAX_PADS];
};

// Stream-ordered equivalent of a blocking cudaMemcpy: identical bytes,
// identical returns-after-completion semantics, but ordered on the batch's
// private stream so it synchronizes with the batch's kernels (and never with
// unrelated work on the legacy default stream).
static void MemcpyOnBatchStream(GpuBatch* b, void* dst, const void* src,
	size_t bytes, cudaMemcpyKind kind, const char* what) {
	CudaCheck(cudaMemcpyAsync(dst, src, bytes, kind, BatchStream(b)), what);
	CudaCheck(cudaStreamSynchronize(BatchStream(b)), what);
}

bool Available() {
	int n = 0;
	if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0)
		return false;
	// 2026-06-10: make the host thread SLEEP (interrupt-driven) while the step
	// kernel runs instead of the driver's default spin-poll, so "the CPU does
	// nothing during simulation" holds at the OS level too. Must run before the
	// CUDA context exists; harmless no-op error if it already does.
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	return true;
}

struct GpuBatch {
	int numArenas;
	int numCars;        // per arena
	int numPads;        // per arena
	int bodiesPerArena; // 1 ball + numCars
	int lanesPerArena;
	int arenasPerBlock;
	int numBlocks;

	cudaStream_t stream = nullptr;
	cudaEvent_t downloadEvent = nullptr;
	cudaStream_t resetStream = nullptr;
	cudaEvent_t resetEvent = nullptr;
	unsigned* dArenaDone = nullptr;
	unsigned launchGen = 0;
	unsigned* dBlockCycles = nullptr;
	GpuArenaStaging staging = {};
	unsigned long long* dProfCycles = nullptr; // [GPU_PHASE_COUNT], diagnostic
	int* dArenaList = nullptr; // [numArenas], subset-step index list (lazy alloc)
	cudaEvent_t copySourceEvent = nullptr;
	cudaEvent_t copyDoneEvent = nullptr;
	int2* copyPairs = nullptr;
	int2* dCopyPairs = nullptr;

	GpuBodyState* dBodies;      // [numArenas * bodiesPerArena]
	GpuBodyConsts* dConsts;     // [bodiesPerArena]
	GpuCarGameState* dCars;     // [numArenas * numCars]
	GpuPadState* dPads;         // [numArenas * numPads]
	GpuCarControls* dSchedule;  // [numArenas * numCars * scheduleTicks]
	int scheduleTicks;
	uint64_t tickCount;

	// static world geometry (device buffers + descriptor passed to kernels by value)
	GpuPlane* dPlanes;
	GpuMeshInfo* dMeshes;
	GpuVec3* dTris;
	GpuWorldGeo geo; // holds device pointers

	CuStepConsts stepConsts;

	// mesh BVHs + internal-edge data (uploaded once)
	GpuMeshBvh* dMeshBvhs;
	GpuBvhNode* dBvhNodes;
	GpuVec3* dHullPoints;  // snowday puck (null otherwise)
	GpuCarControls* dConstControls; // per-car constant controls (chunked RL stepping)
	GpuTileShape* dTiles;          // dropshot tiles (null otherwise)
	GpuTileTables* dTileTables;
	uint8_t* dTileDamage;          // [numArenas * numTiles]
	uint8_t* dTileNoResponse;      // [numArenas * numTiles]
	int numTiles;
	uint4* dBvhSubtreeBoxes;
	GpuTriInfo* dTriInfos;
	CuTriCull* dTriCull;
	float* dMeshMargins;

	// collision orchestration (uploaded once / persistent per arena)
	GpuCollideConsts* dCollideConsts;
	GpuStaticObj* dStatics;
	int numStatics;
	int* dCellStaticCount;
	CuCellStatic* dCellStatics;
	int cellStaticStride;
	GpuGridParams grid;
	GpuDynProxy* dDynProxies;     // [numArenas * bodiesPerArena]
	int* dRegCounters;            // [numArenas]
	GpuBallGameState* dBallStates;// [numArenas]
	CuManifold* dManifolds;       // [numArenas * GPU_MAX_MANIFOLDS]
	int* dNumManifolds;           // [numArenas]
	int2* dPairs;                 // [numArenas * 96] (collide -> solver islands)
	int* dNumPairs;               // [numArenas]
	CuSolveCtx* dSolveScratch;    // [numArenas]
	bool hasCollideData;

	// game-tail data (full-step mode)
	GpuDynBody* dDynTemplates;    // [bodiesPerArena] shape/filter templates
	int* dCarOrder;               // [numArenas * numCars] CPU _cars iteration order
	uint32_t* dRngStates;         // [numArenas] minstd_rand0 state (demo respawns)
	bool hasTailData;
};

// Capture this CPU's RSQRTSS into the device emulation table; verify the emulation
// hypothesis on host first so a different CPU fails loudly, never silently.
static void InitRsqrtEmulation() {
	static bool done = false;
	if (done) return;
	static uint32_t tab[2][4096];
	RsBuildRsqrtTable(tab);
	uint64_t bad = RsVerifyRsqrtEmulation(tab, 4000000);
	if (bad) {
		fprintf(stderr, "FATAL: RSQRTSS emulation hypothesis fails on this CPU "
			"(%llu/4000000 mismatches) - GPU exactness impossible without a new scheme\n",
			(unsigned long long)bad);
		exit(2);
	}
	if (!CuUploadRsqrtTable(tab)) {
		fprintf(stderr, "FATAL: rsqrt table upload failed\n");
		exit(2);
	}
	done = true;
}

GpuBatch* CreateBatch(int numArenas, int numCarsPerArena, int numPads) {
	InitRsqrtEmulation();
	// EPA penetration (collision phase GJK fallback) recurses deeply; the default
	// 1KB device stack overflows (proven in cuda_math_test). 2026-06-11: the
	// reserve is allocated for every resident thread (~74k on a 5070 = GBs of
	// VRAM), capping trainer arena counts. 16KB fails ALL gates; 32KB passes
	// the full suite and halves the reserve vs the old 64KB.
	CudaCheck(cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024), "set stack size");
	GpuBatch* b = new GpuBatch{};
	b->numArenas = numArenas;
	b->numCars = numCarsPerArena;
	if (numPads > CU_MAX_PADS) {
		fprintf(stderr, "CreateBatch: numPads %d exceeds the kernel's %d-pad limit\n", numPads,
			CU_MAX_PADS);
		std::abort();
	}
	if (numCarsPerArena + 2 > CU_MAX_SOLVER_BODIES) {
		fprintf(stderr, "CreateBatch: numCarsPerArena %d exceeds the solver's %d-car limit\n",
			numCarsPerArena, CU_MAX_SOLVER_BODIES - 2);
		std::abort();
	}
	b->numPads = numPads;
	b->bodiesPerArena = 1 + numCarsPerArena;
	b->lanesPerArena = 8;
	while (b->lanesPerArena < b->bodiesPerArena)
		b->lanesPerArena *= 2;
	b->arenasPerBlock = CU_STEP_BLOCK_THREADS / b->lanesPerArena;
	b->numBlocks = (numArenas + b->arenasPerBlock - 1) / b->arenasPerBlock;
	b->scheduleTicks = 0;
	b->tickCount = 0;
	try {
		DeviceAlloc(b->dBodies, sizeof(GpuBodyState) * numArenas * b->bodiesPerArena,
			"alloc bodies");
		DeviceAlloc(b->dConsts, sizeof(GpuBodyConsts) * b->bodiesPerArena, "alloc consts");
		DeviceAlloc(b->dCars, sizeof(GpuCarGameState) * numArenas * numCarsPerArena, "alloc cars");
		DeviceAlloc(b->dPads, sizeof(GpuPadState) * numArenas * numPads, "alloc pads");
		b->dSchedule = nullptr;
		DeviceAlloc(b->dDynProxies, sizeof(GpuDynProxy) * numArenas * b->bodiesPerArena,
			"alloc proxies");
		DeviceAlloc(b->dRegCounters, sizeof(int) * numArenas, "alloc regctr");
		DeviceAlloc(b->dBallStates, sizeof(GpuBallGameState) * numArenas, "alloc ballstates");
		CudaCheck(cudaMemset(b->dBallStates, 0, sizeof(GpuBallGameState) * numArenas),
			"zero ballstates");
		DeviceAlloc(b->dManifolds, sizeof(CuManifold) * numArenas * GPU_MAX_MANIFOLDS,
			"alloc manifolds");
		DeviceAlloc(b->dNumManifolds, sizeof(int) * numArenas, "alloc nummanifolds");
		CudaCheck(cudaMemset(b->dNumManifolds, 0, sizeof(int) * numArenas), "zero nummanifolds");
		DeviceAlloc(b->dPairs, sizeof(int2) * numArenas * 96, "alloc pairs");
		DeviceAlloc(b->dNumPairs, sizeof(int) * numArenas, "alloc numpairs");
		CudaCheck(cudaMemset(b->dNumPairs, 0, sizeof(int) * numArenas), "zero numpairs");
		DeviceAlloc(b->dSolveScratch, sizeof(CuSolveCtx) * numArenas, "alloc solve scratch");
		DeviceAlloc(b->dRngStates, sizeof(uint32_t) * numArenas, "alloc rng states");
		CudaCheck(cudaMemset(b->dRngStates, 1, sizeof(uint32_t) * numArenas), "init rng states");
		DeviceAlloc(b->dArenaDone, sizeof(unsigned) * numArenas, "alloc arena done flags");
		CudaCheck(cudaMemset(b->dArenaDone, 0, sizeof(unsigned) * numArenas),
			"zero arena done flags");
		DeviceAlloc(b->dBlockCycles, sizeof(unsigned) * 2 * b->numBlocks,
			"alloc block cycles");
		CudaCheck(cudaMemset(b->dBlockCycles, 0, sizeof(unsigned) * 2 * b->numBlocks),
			"zero block cycles");
	} catch (...) {
		DestroyBatch(b);
		throw;
	}
	CudaCheck(cudaStreamCreateWithFlags(&b->stream, cudaStreamNonBlocking), "create stream");
	CudaCheck(cudaEventCreateWithFlags(&b->downloadEvent, cudaEventDisableTiming), "create event");
	CudaCheck(cudaStreamCreateWithFlags(&b->resetStream, cudaStreamNonBlocking),
		"create reset stream");
	CudaCheck(cudaEventCreateWithFlags(&b->resetEvent, cudaEventDisableTiming),
		"create reset event");
	return b;
}

static cudaStream_t BatchStream(GpuBatch* b) {
	return b->stream;
}

void DestroyBatch(GpuBatch* b) {
	if (b->stream) {
		cudaStreamSynchronize(b->resetStream);
		cudaStreamSynchronize(b->stream);
		cudaEventDestroy(b->resetEvent);
		cudaStreamDestroy(b->resetStream);
		cudaEventDestroy(b->downloadEvent);
		cudaStreamDestroy(b->stream);
	}
	if (b->copySourceEvent)
		cudaEventDestroy(b->copySourceEvent);
	if (b->copyDoneEvent)
		cudaEventDestroy(b->copyDoneEvent);
	cudaFreeHost(b->copyPairs);
	cudaFree(b->dCopyPairs);
	cudaFree(b->dArenaDone);
	cudaFree(b->dBlockCycles);
	cudaFree(b->dBodies);
	cudaFree(b->dConsts);
	cudaFree(b->dCars);
	cudaFree(b->dPads);
	cudaFree(b->dSchedule);
	cudaFree(b->dPlanes);
	cudaFree(b->dMeshes);
	cudaFree(b->dTris);
	cudaFree(b->dMeshBvhs);
	cudaFree(b->dBvhNodes);
	cudaFree(b->dHullPoints);
	cudaFree(b->dConstControls);
	cudaFree(b->dTiles);
	cudaFree(b->dTileTables);
	cudaFree(b->dTileDamage);
	cudaFree(b->dTileNoResponse);
	cudaFree(b->dBvhSubtreeBoxes);
	cudaFree(b->dTriInfos);
	cudaFree(b->dTriCull);
	cudaFree(b->dMeshMargins);
	cudaFree(b->dCollideConsts);
	cudaFree(b->dStatics);
	cudaFree(b->dCellStaticCount);
	cudaFree(b->dCellStatics);
	cudaFree(b->dDynProxies);
	cudaFree(b->dRegCounters);
	cudaFree(b->dBallStates);
	cudaFree(b->dManifolds);
	cudaFree(b->dNumManifolds);
	cudaFree(b->dPairs);
	cudaFree(b->dNumPairs);
	cudaFree(b->dSolveScratch);
	cudaFree(b->dDynTemplates);
	cudaFree(b->dCarOrder);
	cudaFree(b->dRngStates);
	delete b;
}

__device__ float CuOrderedKeyFloat(unsigned key) {
	return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

__device__ bool CuRayRootAxisOverlaps(const GpuConstMesh& cm, int axis, bool maxSide, float p) {
	uint16_t q[3];
	CuRayQuantizeClamp(GV(cm.bvhAabbMin), GV(cm.bvhAabbMax), GV(cm.bvhQuantization), { p, p, p },
		maxSide ? 1 : 0, q);
	return maxSide ? q[axis] >= cm.root.qmin[axis] : q[axis] <= cm.root.qmax[axis];
}

__global__ void BuildRayMeshRecordsKernel(GpuConstMesh* meshes, int numMeshes,
	unsigned* unquantizeMismatches) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numMeshes * 6)
		return;
	GpuConstMesh& cm = meshes[i / 6];
	const int axis = i % 3;
	const bool maxSide = i % 6 >= 3;
	const float quantization = (&cm.bvhQuantization.x)[axis];
	const float recip = (&cm.bvhQuantizationRecip.x)[axis];
	const float bvhMin = (&cm.bvhAabbMin.x)[axis];
	constexpr unsigned HALF_RANGE = 1u << 15;
	unsigned mismatches = 0;
	for (unsigned q = maxSide ? HALF_RANGE : 0; q < (maxSide ? 2 * HALF_RANGE : HALF_RANGE); q++)
		mismatches += __float_as_uint(CuUnquantizeAxis(q, quantization, recip, bvhMin)) !=
			__float_as_uint((float)q / quantization + bvhMin);
	atomicAdd(unquantizeMismatches, mismatches);
	float* rootBound = maxSide ? &cm.rootBounds.mx.x : &cm.rootBounds.mn.x;
	const unsigned rootQ = maxSide ? cm.root.qmax[axis] : cm.root.qmin[axis];
	rootBound[axis] = (float)rootQ / quantization + bvhMin;
	constexpr unsigned NEG_INF_KEY = 0x007fffffu;
	constexpr unsigned POS_INF_KEY = 0xff800000u;
	const bool passLow = CuRayRootAxisOverlaps(cm, axis, maxSide, CuOrderedKeyFloat(NEG_INF_KEY));
	const bool passHigh = CuRayRootAxisOverlaps(cm, axis, maxSide, CuOrderedKeyFloat(POS_INF_KEY));
	float limit;
	if (passLow == passHigh) {
		limit = passLow ? CuOrderedKeyFloat(maxSide ? NEG_INF_KEY : POS_INF_KEY) : nanf("");
	} else {
		unsigned lo = NEG_INF_KEY, hi = POS_INF_KEY;
		while (hi - lo > 1) {
			const unsigned mid = lo + (hi - lo) / 2;
			if (CuRayRootAxisOverlaps(cm, axis, maxSide, CuOrderedKeyFloat(mid)) == passLow)
				lo = mid;
			else
				hi = mid;
		}
		limit = CuOrderedKeyFloat(maxSide ? hi : lo);
	}
	float* limits = maxSide ? &cm.rayMaxLimit.x : &cm.rayMinLimit.x;
	limits[axis] = limit;
}

__global__ void BuildTriCullKernel(const GpuVec3* tris, CuTriCull* out, int numTris) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numTris)
		return;
	out[i] = CuMakeTriCull(GV(tris[i * 3]), GV(tris[i * 3 + 1]), GV(tris[i * 3 + 2]));
}

void UploadMeshBvhs(GpuBatch* b,
	const GpuMeshBvh* bvhs, int numMeshes,
	const GpuBvhNode* nodes, int numNodes,
	const GpuBvhSubtree* subtrees, int numSubtrees,
	const GpuTriInfo* triInfos, int numTris,
	const float* meshMargins) {
	DeviceAlloc(b->dMeshBvhs, sizeof(GpuMeshBvh) * numMeshes, "alloc bvhs");
	DeviceAlloc(b->dBvhNodes, sizeof(GpuBvhNode) * numNodes, "alloc bvh nodes");
	DeviceAlloc(b->dTriInfos, sizeof(GpuTriInfo) * numTris, "alloc tri infos");
	DeviceAlloc(b->dMeshMargins, sizeof(float) * numMeshes, "alloc mesh margins");
	CudaCheck(cudaMemcpy(b->dMeshBvhs, bvhs, sizeof(GpuMeshBvh) * numMeshes, cudaMemcpyHostToDevice), "up bvhs");
	CudaCheck(cudaMemcpy(b->dBvhNodes, nodes, sizeof(GpuBvhNode) * numNodes, cudaMemcpyHostToDevice), "up bvh nodes");
	std::vector<uint4> boxes(numSubtrees);
	for (int s = 0; s < numSubtrees; s++) {
		const GpuBvhSubtree& st = subtrees[s];
		if (st.rootNodeIndex < 0 || st.rootNodeIndex >= (1 << CU_BVH_SUBTREE_SIZE_SHIFT) ||
			st.subtreeSize < 1 || st.subtreeSize >= (1 << (32 - CU_BVH_SUBTREE_SIZE_SHIFT))) {
			fprintf(stderr, "UploadMeshBvhs: subtree %d (root %d, size %d) does not fit the packed header\n",
				s, st.rootNodeIndex, st.subtreeSize);
			std::abort();
		}
		boxes[s] = { st.qmin[0] | (unsigned)st.qmin[1] << 16, st.qmin[2] | (unsigned)st.qmax[0] << 16,
			st.qmax[1] | (unsigned)st.qmax[2] << 16,
			(unsigned)st.rootNodeIndex | (unsigned)st.subtreeSize << CU_BVH_SUBTREE_SIZE_SHIFT };
	}
	DeviceAlloc(b->dBvhSubtreeBoxes, sizeof(uint4) * numSubtrees, "alloc bvh subtree boxes");
	CudaCheck(cudaMemcpy(b->dBvhSubtreeBoxes, boxes.data(), sizeof(uint4) * numSubtrees,
		cudaMemcpyHostToDevice), "up bvh subtree boxes");
	CudaCheck(cudaMemcpy(b->dTriInfos, triInfos, sizeof(GpuTriInfo) * numTris, cudaMemcpyHostToDevice), "up tri infos");
	CudaCheck(cudaMemcpy(b->dMeshMargins, meshMargins, sizeof(float) * numMeshes, cudaMemcpyHostToDevice), "up mesh margins");
	// the ray tests reach the BVHs through the world-geo descriptor
	b->geo.bvhs = b->dMeshBvhs;
	b->geo.bvhNodes = b->dBvhNodes;

	// constant-memory mesh records for the wheel ray test (UploadWorldGeo has
	// already placed the mesh infos on the device; read them back rather than
	// keeping a host copy)
	if (numMeshes > GPU_MAX_CONST_MESHES) {
		fprintf(stderr, "UploadMeshBvhs: %d meshes exceed GPU_MAX_CONST_MESHES (%d)\n",
			numMeshes, GPU_MAX_CONST_MESHES);
		abort();
	}
	if (numMeshes > 0) {
		std::vector<GpuMeshInfo> meshes(numMeshes);
		CudaCheck(cudaMemcpy(meshes.data(), b->dMeshes, sizeof(GpuMeshInfo) * numMeshes,
			cudaMemcpyDeviceToHost), "read back meshes");
		std::vector<GpuConstMesh> cm(numMeshes);
		for (int m = 0; m < numMeshes; m++) {
			cm[m].triStart = meshes[m].triStart;
			cm[m].group = meshes[m].group;
			cm[m].mask = meshes[m].mask;
			cm[m].nodeStart = bvhs[m].nodeStart;
			cm[m].numNodes = bvhs[m].numNodes;
			cm[m].bvhAabbMin = bvhs[m].bvhAabbMin;
			cm[m].bvhAabbMax = bvhs[m].bvhAabbMax;
			cm[m].bvhQuantization = bvhs[m].bvhQuantization;
			cm[m].bvhQuantizationRecip = { 1.0f / bvhs[m].bvhQuantization.x,
				1.0f / bvhs[m].bvhQuantization.y, 1.0f / bvhs[m].bvhQuantization.z };
			cm[m].root = nodes[bvhs[m].nodeStart];
		}
		GpuConstMesh* dConstMeshes;
		DeviceAlloc(dConstMeshes, sizeof(GpuConstMesh) * numMeshes,
			"alloc const mesh staging");
		CudaCheck(cudaMemcpy(dConstMeshes, cm.data(), sizeof(GpuConstMesh) * numMeshes,
			cudaMemcpyHostToDevice), "up const mesh staging");
		unsigned* dMismatches = nullptr;
		try {
			DeviceAlloc(dMismatches, sizeof(unsigned), "alloc unquantize mismatch count");
		} catch (...) {
			cudaFree(dConstMeshes);
			throw;
		}
		CudaCheck(cudaMemset(dMismatches, 0, sizeof(unsigned)), "clear unquantize mismatch count");
		BuildRayMeshRecordsKernel<<<(numMeshes * 6 + 63) / 64, 64>>>(dConstMeshes, numMeshes,
			dMismatches);
		CudaCheck(cudaGetLastError(), "ray mesh records launch");
		unsigned mismatches = 0;
		CudaCheck(cudaMemcpy(&mismatches, dMismatches, sizeof(unsigned), cudaMemcpyDeviceToHost),
			"read back unquantize mismatch count");
		CudaCheck(cudaMemcpy(cm.data(), dConstMeshes, sizeof(GpuConstMesh) * numMeshes,
			cudaMemcpyDeviceToHost), "read back ray mesh records");
		CudaCheck(cudaFree(dMismatches), "free unquantize mismatch count");
		CudaCheck(cudaFree(dConstMeshes), "free const mesh staging");
		if (mismatches != 0) {
			fprintf(stderr, "UploadMeshBvhs: reciprocal unquantization differs from division for "
				"%u quantized values; the ray walk cannot reproduce these meshes\n", mismatches);
			abort();
		}
		CudaCheck(cudaMemcpyToSymbol(g_constMeshes, cm.data(), sizeof(GpuConstMesh) * numMeshes),
			"upload const meshes");
	}
}

// dropshot tile shapes + neighbor tables (uploaded once) and per-arena tile state
void UploadTiles(GpuBatch* b, const GpuTileShape* tiles, int numTiles,
	const GpuTileTables& tables) {
	cudaFree(b->dTiles);
	cudaFree(b->dTileTables);
	cudaFree(b->dTileDamage);
	cudaFree(b->dTileNoResponse);
	DeviceAlloc(b->dTiles, sizeof(GpuTileShape) * numTiles, "alloc tiles");
	CudaCheck(cudaMemcpy(b->dTiles, tiles, sizeof(GpuTileShape) * numTiles,
		cudaMemcpyHostToDevice), "up tiles");
	DeviceAlloc(b->dTileTables, sizeof(GpuTileTables), "alloc tile tables");
	CudaCheck(cudaMemcpy(b->dTileTables, &tables, sizeof(GpuTileTables),
		cudaMemcpyHostToDevice), "up tile tables");
	size_t stateLen = (size_t)b->numArenas * numTiles;
	DeviceAlloc(b->dTileDamage, stateLen, "alloc tile damage");
	CudaCheck(cudaMemset(b->dTileDamage, 0, stateLen), "init tile damage");
	DeviceAlloc(b->dTileNoResponse, stateLen, "alloc tile resp");
	CudaCheck(cudaMemset(b->dTileNoResponse, 0, stateLen), "init tile resp");
	b->numTiles = numTiles;
	b->geo.tiles = b->dTiles;
	b->geo.tileTables = b->dTileTables;
	b->geo.numTiles = numTiles;
}

void DownloadTileState(GpuBatch* b, int arenaIdx, uint8_t* damageOut, uint8_t* noResponseOut) {
	CudaCheck(cudaMemcpy(damageOut, b->dTileDamage + (size_t)arenaIdx * b->numTiles,
		b->numTiles, cudaMemcpyDeviceToHost), "down tile damage");
	if (noResponseOut)
		CudaCheck(cudaMemcpy(noResponseOut, b->dTileNoResponse + (size_t)arenaIdx * b->numTiles,
			b->numTiles, cudaMemcpyDeviceToHost), "down tile resp");
}

// snowday puck hull points (uploaded once; geo carries the device pointer)
void UploadBallHull(GpuBatch* b, const GpuVec3* points, int numPoints) {
	cudaFree(b->dHullPoints);
	DeviceAlloc(b->dHullPoints, sizeof(GpuVec3) * numPoints, "alloc hull pts");
	CudaCheck(cudaMemcpy(b->dHullPoints, points, sizeof(GpuVec3) * numPoints,
		cudaMemcpyHostToDevice), "up hull pts");
	b->geo.hullPoints = b->dHullPoints;
	b->geo.numHullPoints = numPoints;
}

void UploadCollideData(GpuBatch* b, const GpuCollideConsts& consts,
	const GpuStaticObj* statics, int numStatics,
	const int* cellStaticStart, int numCells, const int* cellStaticList,
	const GpuGridParams& grid) {
	DeviceAlloc(b->dCollideConsts, sizeof(GpuCollideConsts), "alloc collide consts");
	DeviceAlloc(b->dStatics, sizeof(GpuStaticObj) * numStatics, "alloc statics");
	CudaCheck(cudaMemcpy(b->dCollideConsts, &consts, sizeof(GpuCollideConsts), cudaMemcpyHostToDevice), "up collide consts");
	CudaCheck(cudaMemcpy(b->dStatics, statics, sizeof(GpuStaticObj) * numStatics, cudaMemcpyHostToDevice), "up statics");
	int stride = 1;
	for (int c = 0; c < numCells; c++)
		if (cellStaticStart[c + 1] - cellStaticStart[c] > stride)
			stride = cellStaticStart[c + 1] - cellStaticStart[c];
	std::vector<int> cellCount(numCells);
	std::vector<CuCellStatic> cellStatics((size_t)numCells * stride);
	for (int c = 0; c < numCells; c++) {
		cellCount[c] = cellStaticStart[c + 1] - cellStaticStart[c];
		for (int i = 0; i < cellCount[c]; i++) {
			const int s = cellStaticList[cellStaticStart[c] + i];
			CuCellStatic& cs = cellStatics[(size_t)c * stride + i];
			cs.aabbMin = statics[s].aabbMin;
			cs.aabbMax = statics[s].aabbMax;
			cs.group = statics[s].group;
			cs.mask = statics[s].mask;
			cs.staticIdx = s;
		}
	}
	DeviceAlloc(b->dCellStaticCount, sizeof(int) * numCells, "alloc cell count");
	DeviceAlloc(b->dCellStatics, sizeof(CuCellStatic) * cellStatics.size(), "alloc cell statics");
	CudaCheck(cudaMemcpy(b->dCellStaticCount, cellCount.data(), sizeof(int) * numCells,
		cudaMemcpyHostToDevice), "up cell count");
	CudaCheck(cudaMemcpy(b->dCellStatics, cellStatics.data(), sizeof(CuCellStatic) * cellStatics.size(),
		cudaMemcpyHostToDevice), "up cell statics");
	b->cellStaticStride = stride;
	b->numStatics = numStatics;
	b->grid = grid;
	b->hasCollideData = true;
}

void UploadDynProxies(GpuBatch* b, int arenaIdx, const GpuDynProxy* proxies, int regCounter) {
	CudaCheck(cudaMemcpy(b->dDynProxies + (size_t)arenaIdx * b->bodiesPerArena, proxies,
		sizeof(GpuDynProxy) * b->bodiesPerArena, cudaMemcpyHostToDevice), "up proxies");
	CudaCheck(cudaMemcpy(b->dRegCounters + arenaIdx, &regCounter, sizeof(int),
		cudaMemcpyHostToDevice), "up regctr");
}

void UploadBallState(GpuBatch* b, int arenaIdx, const GpuBallGameState& state) {
	MemcpyOnBatchStream(b, b->dBallStates + arenaIdx, &state, sizeof(GpuBallGameState),
		cudaMemcpyHostToDevice, "up ball state");
}

void DownloadBallState(GpuBatch* b, int arenaIdx, GpuBallGameState* state) {
	MemcpyOnBatchStream(b, state, b->dBallStates + arenaIdx, sizeof(GpuBallGameState),
		cudaMemcpyDeviceToHost, "down ball state");
}

void SetTickCount(GpuBatch* b, uint64_t tickCount) {
	b->tickCount = tickCount;
}

void UploadTailData(GpuBatch* b, const GpuTailConsts& consts,
	const GpuPadConfig* padConfigs, int numPads,
	const GpuDynBody* dynTemplates) {
	if (numPads != b->numPads) {
		fprintf(stderr, "UploadTailData: %d pad configs for a batch of %d pads\n", numPads,
			b->numPads);
		std::abort();
	}
	DeviceAlloc(b->dDynTemplates, sizeof(GpuDynBody) * b->bodiesPerArena, "alloc dyn templates");
	DeviceAlloc(b->dCarOrder, sizeof(int) * b->numArenas * b->numCars, "alloc car order");
	b->stepConsts.tail = consts;
	for (int p = 0; p < numPads; p++)
		b->stepConsts.pads[p] = padConfigs[p];
	CudaCheck(cudaMemcpy(b->dDynTemplates, dynTemplates, sizeof(GpuDynBody) * b->bodiesPerArena, cudaMemcpyHostToDevice), "up dyn templates");
	b->hasTailData = true;
}

void UploadCarOrder(GpuBatch* b, int arenaIdx, const int* order) {
	CudaCheck(cudaMemcpy(b->dCarOrder + (size_t)arenaIdx * b->numCars, order,
		sizeof(int) * b->numCars, cudaMemcpyHostToDevice), "up car order");
}

void UploadRngState(GpuBatch* b, int arenaIdx, uint32_t state) {
	CudaCheck(cudaMemcpy(b->dRngStates + arenaIdx, &state, sizeof(uint32_t),
		cudaMemcpyHostToDevice), "up rng state");
}

void UploadCarConsts(GpuBatch* b, const GpuCarConsts& consts) {
	b->stepConsts.car = consts;
}

// ---- car pre-tick single-shot (compare harness) ----

__device__ __forceinline__ CuBodyDyn DynToBody(const GpuDynBody& d) {
	CuBodyDyn b;
	b.basis = GM(d.basis);
	b.origin = GV(d.origin);
	b.linVel = GV(d.linVel);
	b.angVel = GV(d.angVel);
	b.invInertiaWorld = GM(d.invInertiaWorld);
	b.invInertiaDiagLocal = GV(d.invInertiaDiagLocal);
	b.invMass = d.invMass;
	return b;
}

__global__ void CarPreTickOnceKernel(
	GpuWorldGeo geo, GpuGridParams grid,
	const __grid_constant__ GpuCarConsts cc, GpuCarTickState* state,
	const GpuDynBody* dyn,
	int numDyn, int selfDynIdx, float tickTime,
	GpuWheelScratchOut* wheelsOut) {

	CuCarTickIO io;
	io.body.basis = GM(state->basis);
	io.body.origin = GV(state->origin);
	io.body.linVel = GV(state->linVel);
	io.body.angVel = GV(state->angVel);
	io.body.invInertiaWorld = GM(state->invInertiaWorld);
	io.body.invInertiaDiagLocal = GV(state->invInertiaDiagLocal);
	io.body.invMass = state->invMass;
	io.totalForce = GV(state->totalForce);
	io.totalTorque = GV(state->totalTorque);
	io.st = state->st;
	for (int i = 0; i < 4; i++) io.wheels[i] = state->wheels[i];
	io.controls = state->controls;
	io.solverTimeStep = state->solverTimeStep;

	// dynamic body impulse views (ground bodies for friction)
	CuBodyDyn dynBodies[8];
	for (int i = 0; i < numDyn && i < 8; i++)
		dynBodies[i] = DynToBody(dyn[i]);

	CuWheelScratch W[4];
	CuWheelResult results[4];
	if (!io.st.isDemoed) {
		for (int i = 0; i < 4; i++) {
			CuCarPreTickWheel(io, cc, geo, grid, dyn, numDyn, selfDynIdx, dynBodies, i, W[i]);
			results[i] = CuWheelResultOf(io, cc, i, tickTime, W[i]);
		}
	}
	float dbgFwd = 0;
	CuCarPreTick(io, cc, tickTime, results, &dbgFwd);
	state->dbgForwardSpeedUU = dbgFwd;

	// write back
	for (int r = 0; r < 3; r++)
		state->basis.row[r] = { io.body.basis.r[r].x, io.body.basis.r[r].y, io.body.basis.r[r].z };
	state->origin = { io.body.origin.x, io.body.origin.y, io.body.origin.z };
	state->linVel = { io.body.linVel.x, io.body.linVel.y, io.body.linVel.z };
	state->angVel = { io.body.angVel.x, io.body.angVel.y, io.body.angVel.z };
	state->totalForce = { io.totalForce.x, io.totalForce.y, io.totalForce.z };
	state->totalTorque = { io.totalTorque.x, io.totalTorque.y, io.totalTorque.z };
	state->st = io.st;
	for (int i = 0; i < 4; i++) state->wheels[i] = io.wheels[i];
	state->controls = io.controls;

	for (int i = 0; i < 4; i++) {
		GpuWheelScratchOut& o = wheelsOut[i];
		for (int r = 0; r < 3; r++)
			o.worldBasis.row[r] = { W[i].worldBasis.r[r].x, W[i].worldBasis.r[r].y, W[i].worldBasis.r[r].z };
		o.hardPointWS = { W[i].hardPointWS.x, W[i].hardPointWS.y, W[i].hardPointWS.z };
		o.contactPointWS = { W[i].contactPointWS.x, W[i].contactPointWS.y, W[i].contactPointWS.z };
		o.contactNormalWS = { W[i].contactNormalWS.x, W[i].contactNormalWS.y, W[i].contactNormalWS.z };
		o.isInContact = W[i].isInContact;
		o.isInContactWithWorld = W[i].isInContactWithWorld;
		o.groundIdx = W[i].groundIdx;
		o.suspensionRelativeVelocity = W[i].suspensionRelativeVelocity;
		o.clippedInvContactDotSuspension = W[i].clippedInvContactDotSuspension;
		o.extraPushback = W[i].extraPushback;
		o.impulse = { W[i].impulse.x, W[i].impulse.y, W[i].impulse.z };
		o.dbgWheelTraceLen = W[i].dbgWheelTraceLen;
		o.dbgPushThresh = W[i].dbgPushThresh;
		o.dbgResolveRaw = W[i].dbgResolveRaw;
	}
}

void RunCarPreTickOnce(GpuBatch* b, const GpuCarTickState& in,
	const GpuDynBody* dynBodies, int numDyn, int selfDynIdx,
	const GpuGridParams& grid, float tickTime,
	GpuCarTickState& out, GpuWheelScratchOut wheelsOut[4]) {

	static GpuCarTickState* dState = nullptr;
	static GpuDynBody* dDyn = nullptr;
	static GpuWheelScratchOut* dWheels = nullptr;
	if (!dState) {
		DeviceAlloc(dState, sizeof(GpuCarTickState), "alloc tick state");
		DeviceAlloc(dDyn, sizeof(GpuDynBody) * 8, "alloc dyn8");
		DeviceAlloc(dWheels, sizeof(GpuWheelScratchOut) * 4, "alloc wheels out");
	}
	CudaCheck(cudaMemcpy(dState, &in, sizeof(in), cudaMemcpyHostToDevice), "upload tick state");
	CudaCheck(cudaMemcpy(dDyn, dynBodies, sizeof(GpuDynBody) * numDyn, cudaMemcpyHostToDevice), "upload dyn");

	CarPreTickOnceKernel<<<1, 1>>>(b->geo, grid, b->stepConsts.car, dState, dDyn,
		numDyn, selfDynIdx, tickTime, dWheels);
	CudaCheck(cudaGetLastError(), "car pretick launch");
	CudaCheck(cudaMemcpy(&out, dState, sizeof(out), cudaMemcpyDeviceToHost), "download tick state");
	CudaCheck(cudaMemcpy(wheelsOut, dWheels, sizeof(GpuWheelScratchOut) * 4, cudaMemcpyDeviceToHost), "download wheels");
}

void UploadWorldGeo(GpuBatch* b,
	const GpuPlane* planes, int numPlanes,
	const GpuMeshInfo* meshes, int numMeshes,
	const GpuVec3* tris, int numTris) {
	if (numPlanes > CU_MAX_CONST_PLANES) {
		fprintf(stderr, "UploadWorldGeo: %d planes exceed CU_MAX_CONST_PLANES (%d)\n",
			numPlanes, CU_MAX_CONST_PLANES);
		abort();
	}
	if (numPlanes > 0)
		CudaCheck(cudaMemcpyToSymbol(g_constPlanes, planes, sizeof(GpuPlane) * numPlanes),
			"upload const planes");
	DeviceAlloc(b->dPlanes, sizeof(GpuPlane) * numPlanes, "alloc planes");
	DeviceAlloc(b->dMeshes, sizeof(GpuMeshInfo) * numMeshes, "alloc meshes");
	DeviceAlloc(b->dTris, sizeof(GpuVec3) * 3 * numTris, "alloc tris");
	CudaCheck(cudaMemcpy(b->dPlanes, planes, sizeof(GpuPlane) * numPlanes, cudaMemcpyHostToDevice), "upload planes");
	CudaCheck(cudaMemcpy(b->dMeshes, meshes, sizeof(GpuMeshInfo) * numMeshes, cudaMemcpyHostToDevice), "upload meshes");
	CudaCheck(cudaMemcpy(b->dTris, tris, sizeof(GpuVec3) * 3 * numTris, cudaMemcpyHostToDevice), "upload tris");
	DeviceAlloc(b->dTriCull, sizeof(CuTriCull) * (numTris > 0 ? numTris : 1), "alloc tri cull");
	if (numTris > 0) {
		BuildTriCullKernel<<<(numTris + 255) / 256, 256>>>(b->dTris, b->dTriCull, numTris);
		CudaCheck(cudaGetLastError(), "tri cull launch");
		CudaCheck(cudaDeviceSynchronize(), "tri cull build");
	}
	b->geo.planes = b->dPlanes;
	b->geo.numPlanes = numPlanes;
	b->geo.meshes = b->dMeshes;
	b->geo.numMeshes = numMeshes;
	b->geo.meshTris = b->dTris;
}

__global__ void RayFuzzKernel(GpuWorldGeo geo, GpuGridParams grid, const GpuVec3* fromTo, int numRays,
	int cbGroup, int cbMask, const GpuDynBody* dyn, int numDyn, int ignoreIdx,
	GpuRayFuzzResult* results) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numRays) return;
	GpuRayHit hit;
	CuRayTestWorld(geo, grid, dyn, numDyn, ignoreIdx, GV(fromTo[i * 2]), GV(fromTo[i * 2 + 1]),
		cbGroup, cbMask, hit);
	results[i].hitKind = hit.hitKind;
	results[i].fraction = hit.fraction;
	results[i].normal = { hit.normalWorld.x, hit.normalWorld.y, hit.normalWorld.z };
}

void RunRayFuzz(GpuBatch* b, const GpuVec3* fromTo, int numRays,
	int cbGroup, int cbMask, const GpuDynBody* dynBodies, int numDyn, int ignoreIdx,
	const GpuGridParams& grid, GpuRayFuzzResult* results) {
	GpuVec3* dRays;
	GpuRayFuzzResult* dRes;
	GpuDynBody* dDyn = nullptr;
	DeviceAlloc(dRays, sizeof(GpuVec3) * 2 * numRays, "alloc rays");
	DeviceAlloc(dRes, sizeof(GpuRayFuzzResult) * numRays, "alloc ray results");
	CudaCheck(cudaMemcpy(dRays, fromTo, sizeof(GpuVec3) * 2 * numRays, cudaMemcpyHostToDevice), "upload rays");
	if (numDyn > 0) {
		DeviceAlloc(dDyn, sizeof(GpuDynBody) * numDyn, "alloc dyn");
		CudaCheck(cudaMemcpy(dDyn, dynBodies, sizeof(GpuDynBody) * numDyn, cudaMemcpyHostToDevice), "upload dyn");
	}
	RayFuzzKernel<<<(numRays + 127) / 128, 128>>>(b->geo, grid, dRays, numRays, cbGroup, cbMask,
		dDyn, numDyn, ignoreIdx, dRes);
	CudaCheck(cudaGetLastError(), "rayfuzz launch");
	CudaCheck(cudaMemcpy(results, dRes, sizeof(GpuRayFuzzResult) * numRays, cudaMemcpyDeviceToHost), "download ray results");
	cudaFree(dRays);
	cudaFree(dRes);
	cudaFree(dDyn);
}

void UploadArenaState(GpuBatch* b, int arenaIdx,
	const GpuBodyState* bodies, const GpuCarGameState* cars, const GpuPadState* pads) {
	MemcpyOnBatchStream(b, b->dBodies + (size_t)arenaIdx * b->bodiesPerArena, bodies,
		sizeof(GpuBodyState) * b->bodiesPerArena, cudaMemcpyHostToDevice, "upload bodies");
	MemcpyOnBatchStream(b, b->dCars + (size_t)arenaIdx * b->numCars, cars,
		sizeof(GpuCarGameState) * b->numCars, cudaMemcpyHostToDevice, "upload cars");
	MemcpyOnBatchStream(b, b->dPads + (size_t)arenaIdx * b->numPads, pads,
		sizeof(GpuPadState) * b->numPads, cudaMemcpyHostToDevice, "upload pads");
}

static_assert(sizeof(GpuBodyState) % 4 == 0 && sizeof(GpuCarGameState) % 4 == 0 &&
	sizeof(GpuPadState) % 4 == 0 && sizeof(GpuBallGameState) % 4 == 0 &&
	sizeof(GpuCarControls) % 4 == 0, "staged state is copied in whole words");

struct CuWordSpan {
	const unsigned* src;
	unsigned* dst;
	int words;
};

struct CuArenaSpans {
	CuWordSpan spans[4];
};

__device__ void CuAwaitArena(unsigned* arenaDone, int arenaIdx, unsigned gen) {
	cuda::atomic_ref<unsigned, cuda::thread_scope_device> done(arenaDone[arenaIdx]);
	while (done.load(cuda::memory_order_acquire) != gen)
		__nanosleep(500);
}

__global__ void GpuStageArenaKernel(unsigned* arenaDone, unsigned gen, int arenaIdx,
	CuArenaSpans staged) {
	CuAwaitArena(arenaDone, arenaIdx, gen);
	for (const CuWordSpan& s : staged.spans)
		for (int i = threadIdx.x; i < s.words; i += blockDim.x)
			s.dst[i] = s.src[i];
}

__global__ void GpuStageSubsetKernel(unsigned* arenaDone, unsigned gen, const int* arenaIdxs,
	int count, int* list, const GpuCarControls* staged, GpuCarControls* controls, int numCars) {
	const int words = numCars * (int)(sizeof(GpuCarControls) / 4);
	for (int i = threadIdx.x; i < count; i += blockDim.x) {
		const int a = arenaIdxs[i];
		CuAwaitArena(arenaDone, a, gen);
		const unsigned* src = reinterpret_cast<const unsigned*>(staged + (size_t)a * numCars);
		unsigned* dst = reinterpret_cast<unsigned*>(controls + (size_t)a * numCars);
		for (int w = 0; w < words; w++)
			dst[w] = src[w];
		list[i] = a;
	}
}

template <typename T>
static T* MappedDevicePtr(T* host, const char* what) {
	void* dev = nullptr;
	CudaCheck(cudaHostGetDevicePointer(&dev, host, 0), what);
	return static_cast<T*>(dev);
}

void SetArenaStaging(GpuBatch* b, const GpuArenaStaging& host) {
	b->staging.bodies = MappedDevicePtr(host.bodies, "map staged bodies");
	b->staging.cars = MappedDevicePtr(host.cars, "map staged cars");
	b->staging.pads = MappedDevicePtr(host.pads, "map staged pads");
	b->staging.balls = MappedDevicePtr(host.balls, "map staged balls");
	b->staging.controls = MappedDevicePtr(host.controls, "map staged controls");
	b->staging.slots = MappedDevicePtr(host.slots, "map staged slots");
}

static void OrderAfterResetWork(GpuBatch* b) {
	CudaCheck(cudaEventRecord(b->resetEvent, b->resetStream), "record reset event");
	CudaCheck(cudaStreamWaitEvent(b->stream, b->resetEvent, 0), "order after reset work");
}

template <typename T>
static CuWordSpan StagedSpan(const T* src, T* dst, int count) {
	return { reinterpret_cast<const unsigned*>(src), reinterpret_cast<unsigned*>(dst),
		count * (int)(sizeof(T) / 4) };
}

void UploadArenaStateAsync(GpuBatch* b, int arenaIdx) {
	if (!b->staging.bodies) {
		fprintf(stderr, "UploadArenaStateAsync: no arena staging registered\n");
		std::abort();
	}
	const size_t bodyAt = (size_t)arenaIdx * b->bodiesPerArena;
	const size_t carAt = (size_t)arenaIdx * b->numCars;
	const size_t padAt = (size_t)arenaIdx * b->numPads;
	CuArenaSpans staged = { {
		StagedSpan(b->staging.bodies + bodyAt, b->dBodies + bodyAt, b->bodiesPerArena),
		StagedSpan(b->staging.cars + carAt, b->dCars + carAt, b->numCars),
		StagedSpan(b->staging.pads + padAt, b->dPads + padAt, b->numPads),
		StagedSpan(b->staging.balls + arenaIdx, b->dBallStates + arenaIdx, 1),
	} };
	GpuStageArenaKernel<<<1, 128, 0, b->resetStream>>>(b->dArenaDone, b->launchGen, arenaIdx,
		staged);
	CudaCheck(cudaGetLastError(), "stage arena launch");
	OrderAfterResetWork(b);
}

void UploadBodyConsts(GpuBatch* b, const GpuBodyConsts* consts) {
	CudaCheck(cudaMemcpy(b->dConsts, consts, sizeof(GpuBodyConsts) * b->bodiesPerArena,
		cudaMemcpyHostToDevice), "upload consts");
}

void UploadControlSchedule(GpuBatch* b, const GpuCarControls* schedule, int numTicks) {
	size_t count = (size_t)b->numArenas * b->numCars * numTicks;
	cudaFree(b->dSchedule);
	DeviceAlloc(b->dSchedule, sizeof(GpuCarControls) * count, "alloc schedule");
	CudaCheck(cudaMemcpy(b->dSchedule, schedule, sizeof(GpuCarControls) * count,
		cudaMemcpyHostToDevice), "upload schedule");
	b->scheduleTicks = numTicks;
	cudaFree(b->dConstControls);
	b->dConstControls = nullptr; // schedule mode takes over
}

// Chunked-stepping controls: one CarControls per car, held constant for every tick
// of subsequent StepTicks calls (the RL usage: controls change between chunks, not
// within them). Overrides any uploaded schedule until UploadControlSchedule is called.
void UploadConstantControls(GpuBatch* b, const GpuCarControls* perCar) {
	size_t count = (size_t)b->numArenas * b->numCars;
	if (!b->dConstControls)
		DeviceAlloc(b->dConstControls, sizeof(GpuCarControls) * count,
			"alloc const controls");
	MemcpyOnBatchStream(b, b->dConstControls, perCar, sizeof(GpuCarControls) * count,
		cudaMemcpyHostToDevice, "upload const controls");
}

void UploadConstantControlsAsync(GpuBatch* b, const GpuCarControls* perCar) {
	size_t count = (size_t)b->numArenas * b->numCars;
	if (!b->dConstControls)
		DeviceAlloc(b->dConstControls, sizeof(GpuCarControls) * count,
			"alloc const controls");
	CudaCheck(cudaMemcpyAsync(b->dConstControls, perCar, sizeof(GpuCarControls) * count,
		cudaMemcpyHostToDevice, b->stream), "upload const controls async");
}

// ============================== DEVICE-SIDE PHASES ==============================
// (activation constants + CuIsActive + CuSetActivationState live in
// CudaCollideWorld.cuh - the collide and solve phases need them too)

// Phase 5: predictUnconstraintMotion (btDiscreteDynamicsWorld.cpp:1053).
// All our bodies are dynamic (non-static/kinematic). The interpolation transform it
// writes is write-only in RocketSim - skipped. Only damping has observable effect.
__device__ __forceinline__ void PhasePredict(GpuBodyState& s, const GpuBodyConsts& c) {
	CuVec3 lv = { s.linVel.x, s.linVel.y, s.linVel.z };
	CuVec3 av = { s.angVel.x, s.angVel.y, s.angVel.z };
	CuApplyDamping(lv, av, c.linDampFactor, c.angDampFactor);
	s.linVel = { lv.x, lv.y, lv.z };
	s.angVel = { av.x, av.y, av.z };
}

// Phase 8: integrateTransforms (btDiscreteDynamicsWorld.cpp:913). No CCD in RocketSim
// (thresholds are 0). proceedToTransform = setCenterOfMassTransform = worldTransform
// assignment + updateInertiaTensor.
__device__ __forceinline__ void PhaseIntegrate(GpuBodyState& s, const GpuBodyConsts& c, float dt) {
	if (!CuIsActive(s.activationState))
		return;
	CuMat3 basis = { { { s.basis.row[0].x, s.basis.row[0].y, s.basis.row[0].z },
	                   { s.basis.row[1].x, s.basis.row[1].y, s.basis.row[1].z },
	                   { s.basis.row[2].x, s.basis.row[2].y, s.basis.row[2].z } } };
	CuVec3 pos = { s.pos.x, s.pos.y, s.pos.z };
	CuVec3 lv = { s.linVel.x, s.linVel.y, s.linVel.z };
	CuVec3 av = { s.angVel.x, s.angVel.y, s.angVel.z };

	CuMat3 nb;
	CuVec3 np;
	CuIntegrateTransform(basis, pos, lv, av, dt, c.noRot != 0, nb, np);

	CuMat3 invLocal = CuUpdateInertiaTensor(nb, { c.invInertiaLocal.x, c.invInertiaLocal.y, c.invInertiaLocal.z });

	for (int i = 0; i < 3; i++) {
		s.basis.row[i] = { nb.r[i].x, nb.r[i].y, nb.r[i].z };
		s.invInertiaWorld.row[i] = { invLocal.r[i].x, invLocal.r[i].y, invLocal.r[i].z };
	}
	s.pos = { np.x, np.y, np.z };
}

// Phase 9: updateActivationState (btDiscreteDynamicsWorld.cpp:558) +
// btRigidBody::updateDeactivation / wantsSleeping (btRigidBody.h:512/529)
__device__ __forceinline__ void PhaseActivation(GpuBodyState& s, const GpuBodyConsts& c, float dt) {
	// updateDeactivation
	int st = s.activationState;
	if (!(st == CU_ISLAND_SLEEPING || st == CU_DISABLE_DEACTIVATION)) {
		CuVec3 lv = { s.linVel.x, s.linVel.y, s.linVel.z };
		CuVec3 av = { s.angVel.x, s.angVel.y, s.angVel.z };
		if (CuLength2(lv) < c.linSleepThreshSq && CuLength2(av) < c.angSleepThreshSq) {
			s.deactivationTime += dt;
		} else {
			s.deactivationTime = 0.0f;
			CuSetActivationState(s.activationState, 0);
		}
	}

	// wantsSleeping
	st = s.activationState;
	bool wants;
	if (st == CU_DISABLE_DEACTIVATION) wants = false;
	else if (st == CU_ISLAND_SLEEPING || st == CU_WANTS_DEACTIVATION) wants = true;
	else wants = s.deactivationTime > CU_DEACTIVATION_TIME;

	if (wants) {
		// non-static path of updateActivationState
		if (s.activationState == CU_ACTIVE_TAG)
			CuSetActivationState(s.activationState, CU_WANTS_DEACTIVATION);
		if (s.activationState == CU_ISLAND_SLEEPING) {
			s.angVel = { 0, 0, 0 };
			s.linVel = { 0, 0, 0 };
		}
	} else {
		if (s.activationState != CU_DISABLE_DEACTIVATION)
			CuSetActivationState(s.activationState, CU_ACTIVE_TAG);
	}
}

// ============================== MONOLITHIC KERNEL ==============================

struct GpuStateMirror {
	GpuBodyState* bodies;
	GpuCarGameState* cars;
	GpuPadState* pads;
	GpuBallGameState* balls;
	unsigned* arenaMirrored;
};

// Collision-phase buffers/constants, bundled to keep the kernel signature sane.
struct GpuCollideKernelArgs {
	int enabled;
	GpuDynProxy* proxies;            // [numArenas * bodiesPerArena]
	int* regCounters;                // [numArenas]
	GpuBallGameState* ballStates;    // [numArenas]
	const GpuCollideConsts* consts;
	const GpuStaticObj* statics;
	int numStatics;
	const int* cellStaticCount;
	const CuCellStatic* cellStatics;
	int cellStaticStride;
	GpuGridParams grid;
	const GpuMeshBvh* bvhs;
	const GpuBvhNode* bvhNodes;
	const uint4* bvhSubtreeBoxes;
	const GpuTriInfo* triInfos;
	const CuTriCull* triCull;
	const float* meshMargins;
	CuManifold* manifolds;           // [numArenas * GPU_MAX_MANIFOLDS]
	int* numManifolds;               // [numArenas]
	int2* pairs;                     // [numArenas * 96]
	int* numPairs;                   // [numArenas]
	CuSolveCtx* solveScratch;        // [numArenas]
	uint64_t tickBase;
	// game-tail (full-step mode)
	int tailEnabled;
	const GpuDynBody* dynTemplates;  // [bodiesPerArena]
	const int* carOrder;             // [numArenas * numCars]
	uint32_t* rngStates;             // [numArenas]
	uint8_t* tileDamage;             // [numArenas * numTiles] (dropshot)
	uint8_t* tileNoResponse;         // [numArenas * numTiles]
	int numTiles;
	const GpuCarControls* constControls; // non-null: per-car controls, schedule unused
	// 2026-06-11: diagnostic - non-null enables per-phase clock64 accumulation
	// ([GPU_PHASE_COUNT] device counters, summed over blocks). See --phase-prof.
	unsigned long long* profCycles;
	// 2026-06-11: non-null = step only these arena indices (grid = list length).
	// Arenas are independent, so a subset step is bit-identical to its slice of
	// a full step. Used to re-step reset arenas after an async chunk launch.
	const int* arenaList;
	GpuStateMirror mirror;
	unsigned* arenaDone;
	unsigned doneGen;
	unsigned* blockCycles;
};

__device__ __forceinline__ void CuMirrorBytes(void* dst, const void* src, size_t bytes) {
	size_t vectorBytes = 0;
	if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15) == 0) {
		const uint4* s = static_cast<const uint4*>(src);
		uint4* d = static_cast<uint4*>(dst);
		const size_t n = bytes / 16;
		const size_t stride = blockDim.x;
		size_t i = threadIdx.x;
		for (; i + 3 * stride < n; i += 4 * stride) {
			const uint4 a = s[i];
			const uint4 b = s[i + stride];
			const uint4 c = s[i + 2 * stride];
			const uint4 e = s[i + 3 * stride];
			d[i] = a;
			d[i + stride] = b;
			d[i + 2 * stride] = c;
			d[i + 3 * stride] = e;
		}
		for (; i < n; i += stride)
			d[i] = s[i];
		vectorBytes = n * 16;
	}
	const unsigned* s = static_cast<const unsigned*>(src);
	unsigned* d = static_cast<unsigned*>(dst);
	for (size_t i = vectorBytes / 4 + threadIdx.x; i < bytes / 4; i += blockDim.x)
		d[i] = s[i];
}

// 2026-06-11: 255 regs/thread + ~21KB stack is measured-optimal here; capping
// registers via __launch_bounds__ raises occupancy but the extra spill traffic
// makes the kernel slower (2026-06-11: 128 / ~170 regs 23-32% slower with one
// arena per block; 2026-09-02: __launch_bounds__(32, 11) = 168 regs fits the
// 2000-arena batch in one wave and cuts the kickoff-scenario bench 1.83 ->
// 1.58 ms, yet loses 1-3% end-to-end where arenas are contact-heavy). Do not
// add launch bounds without an end-to-end A/B; hashes stay identical either
// way (register allocation does not change FP results).
struct CuStepGroupLayout {
	int io, collideProxies, collideCand, collidePairs, collideMesh, dyn, dynBodies, raySrc, rayTgt, padGrant, flags;
	int bytes;
};

__host__ __device__ inline int CuStepPlace(int& field, int offset, size_t bytes) {
	offset = (offset + 15) & ~15;
	field = offset;
	return offset + (int)bytes;
}

__host__ __device__ inline CuStepGroupLayout CuStepGroupSmemLayout(int bodies, int lanes,
	int numPads) {
	const int carsPerBatch = lanes / 4;
	CuStepGroupLayout L;
	int off = 0;
	size_t ioBytes = sizeof(CuCarTickIO) * carsPerBatch;
	size_t proxyBytes = (sizeof(GpuDynProxy) * bodies + 15) & ~(size_t)15;
	size_t candBytes = (sizeof(int) * (1 + CU_STATIC_CAND_MAX) * bodies + 15) & ~(size_t)15;
	size_t collideBytes = proxyBytes + candBytes + sizeof(int2) * CU_MAX_PAIRS;
	size_t padCarBytes = (sizeof(CuPadGridCar) + sizeof(int)) * bodies;
	size_t aliasBytes = ioBytes > collideBytes ? ioBytes : collideBytes;
	aliasBytes = aliasBytes > padCarBytes ? aliasBytes : padCarBytes;
	off = CuStepPlace(L.io, off, aliasBytes);
	L.collideProxies = L.io;
	L.collideCand = L.io + (int)proxyBytes;
	L.collidePairs = L.collideCand + (int)candBytes;
	L.collideMesh = (L.collidePairs + (int)(sizeof(int2) * CU_MAX_PAIRS) + 15) & ~15;
	off = CuStepPlace(L.dyn, off, sizeof(GpuDynBody) * bodies);
	off = CuStepPlace(L.dynBodies, off, sizeof(CuBodyDyn) * bodies);
	off = CuStepPlace(L.raySrc, off, sizeof(CuVec3) * 4 * carsPerBatch);
	off = CuStepPlace(L.rayTgt, off, sizeof(CuVec3) * 4 * carsPerBatch);
	off = CuStepPlace(L.padGrant, off, sizeof(int) * (numPads > 0 ? numPads : 1));
	// [0..cpb) skip flags, [cpb..2cpb) car indices, then dep flag, ok flag
	off = CuStepPlace(L.flags, off, sizeof(int) * (2 * carsPerBatch + 2));
	const int solveBytes = CuSolveSharedBytes(bodies, GPU_MAX_MANIFOLDS);
	L.bytes = off > solveBytes ? off : solveBytes;
	const int meshEnd = L.collideMesh + CuMeshSmemBytes(lanes);
	L.bytes = L.bytes > meshEnd ? L.bytes : meshEnd;
	return L;
}

__device__ __forceinline__ CuWheelResult CuShflWheelResult(const CuWheelResult& r, int srcLane) {
	CuWheelResult o;
	o.contactPointWS = CuShflVec3(r.contactPointWS, srcLane);
	o.contactNormalWS = CuShflVec3(r.contactNormalWS, srcLane);
	o.impulse = CuShflVec3(r.impulse, srcLane);
	o.suspensionLengthNew = __shfl_sync(0xffffffffu, r.suspensionLengthNew, srcLane);
	o.extraPushback = __shfl_sync(0xffffffffu, r.extraPushback, srcLane);
	o.suspensionForce = __shfl_sync(0xffffffffu, r.suspensionForce, srcLane);
	o.latFriction = __shfl_sync(0xffffffffu, r.latFriction, srcLane);
	o.longFriction = __shfl_sync(0xffffffffu, r.longFriction, srcLane);
	const unsigned flags = __shfl_sync(0xffffffffu,
		(r.isInContact ? 1u : 0u) | (r.isInContactWithWorld ? 2u : 0u) | (r.hasGround ? 4u : 0u),
		srcLane);
	o.isInContact = (flags & 1u) != 0;
	o.isInContactWithWorld = (flags & 2u) != 0;
	o.hasGround = (flags & 4u) != 0;
	return o;
}

__device__ int CuDispatchGroup(const unsigned* __restrict__ lastCycles, int groups, int rank) {
	const int lane = threadIdx.x & 31;
	unsigned most = 0;
	for (int g = lane; g < groups; g += 32)
		most = max(most, lastCycles[g]);
	for (int offset = 16; offset > 0; offset >>= 1)
		most = max(most, __shfl_xor_sync(0xffffffffu, most, offset));
	const unsigned slowCycles = most / CU_SLOW_BLOCK_DIVISOR;
	int slowCount = 0;
	for (int base = 0; base < groups; base += 32) {
		const int g = base + lane;
		slowCount += __popc(__ballot_sync(0xffffffffu, g < groups && lastCycles[g] > slowCycles));
	}
	const bool wantSlow = rank < slowCount;
	if (!wantSlow)
		rank -= slowCount;
	for (int base = 0; base < groups; base += 32) {
		const int g = base + lane;
		const bool pick = g < groups && (lastCycles[g] > slowCycles) == wantSlow;
		const unsigned picks = __ballot_sync(0xffffffffu, pick);
		const int count = __popc(picks);
		if (rank < count) {
			const unsigned below = (1u << lane) - 1u;
			const unsigned hit = __ballot_sync(0xffffffffu, pick && __popc(picks & below) == rank);
			return base + __ffs(hit) - 1;
		}
		rank -= count;
	}
	__trap();
	__builtin_unreachable();
}

__global__ void GpuStepArenasKernel(
	GpuBodyState* bodies, const GpuBodyConsts* consts,
	GpuCarGameState* cars, GpuPadState* pads,
	const GpuCarControls* schedule, int scheduleTicks,
	int numCars, int numPads, int numTicks,
	int startPhase, int endPhase, float dt,
	int lanesPerArena, int numActive, int arenasPerBlock,
	GpuWorldGeo geo, GpuCollideKernelArgs ca, const __grid_constant__ CuStepConsts k) {

	const long long blockStart = clock64();
	const int group = ca.blockCycles
		? CuDispatchGroup(ca.blockCycles + (size_t)((ca.doneGen + 1) & 1) * gridDim.x, gridDim.x,
			blockIdx.x)
		: blockIdx.x;
	const int sub = threadIdx.x / lanesPerArena;
	const int lane = threadIdx.x - sub * lanesPerArena;
	const int slot = group * arenasPerBlock + sub;
	const bool active = slot < numActive;
	const int arenaIdx = active ? (ca.arenaList ? ca.arenaList[slot] : slot) : 0;
	const int bodiesPerArena = 1 + numCars;
	extern __shared__ __align__(16) unsigned char sStepSmem[];
	const CuStepGroupLayout sgL = CuStepGroupSmemLayout(bodiesPerArena, lanesPerArena, numPads);
	unsigned char* sgBase = sStepSmem + (size_t)sub * sgL.bytes;

	// per-phase cycle accumulation (diagnostic build only: the accumulators
	// and clock reads cost registers in a kernel pinned at the 255 cap)
#ifdef RS_GPU_PHASE_PROF
	const bool profOn = ca.profCycles != nullptr;
#else
	constexpr bool profOn = false;
#endif
	unsigned long long profAcc[GPU_PHASE_COUNT] = {};
	long long profT0 = 0;
	const bool fused = !profOn && ca.tailEnabled && startPhase == 0 &&
		endPhase == GPU_PHASE_COUNT - 1;

	GpuBodyState* arenaBodies = bodies + (size_t)arenaIdx * bodiesPerArena;
	GpuCarGameState* arenaCars = cars + (size_t)arenaIdx * numCars;
	GpuPadState* arenaPads = pads + (size_t)arenaIdx * numPads;

	for (int tick = 0; tick < numTicks; tick++) {
		if (profOn && lane == 0)
			profT0 = clock64();
		if (ca.tailEnabled && startPhase <= GPU_PHASE_GAME_PRE && GPU_PHASE_GAME_PRE <= endPhase) {
			CuCarTickIO* sIoAll = reinterpret_cast<CuCarTickIO*>(sgBase + sgL.io);
			GpuDynBody* sDyn = reinterpret_cast<GpuDynBody*>(sgBase + sgL.dyn);
			CuBodyDyn* sDynBodies = reinterpret_cast<CuBodyDyn*>(sgBase + sgL.dynBodies);
			CuVec3* sRaySrc = reinterpret_cast<CuVec3*>(sgBase + sgL.raySrc);
			CuVec3* sRayTgt = reinterpret_cast<CuVec3*>(sgBase + sgL.rayTgt);
			const int carsPerBatch = lanesPerArena >> 2;
			int* sSkip = reinterpret_cast<int*>(sgBase + sgL.flags);
			int* sCi = sSkip + carsPerBatch;
			int& sDep = sSkip[2 * carsPerBatch];
			int& sOk = sSkip[2 * carsPerBatch + 1];

			if (active && lane == 0) {
				// Arena::Step head: ball zero-vel sleeping
				CuBallSleepCheck(arenaBodies[0]);
			}
			uint64_t tickCount = ca.tickBase + tick;
#ifdef RS_GPU_PHASE_PROF
			long long subT = clock64();
			auto subLap = [&](int k) { if (profOn && lane == 0) { long long t = clock64(); atomicAdd(&ca.profCycles[GPU_PHASE_COUNT + 5 + k], (unsigned long long)(t - subT)); subT = t; } };
#else
			auto subLap = [](int) {};
#endif
			const int wheel = lane & 3;
			const int slot = lane >> 2; // car slot of this lane's wheel work
			for (int oc = 0; oc < numCars; oc += carsPerBatch) {
				const int nb = numCars - oc < carsPerBatch ? numCars - oc : carsPerBatch;
				for (int attempt = 0; attempt <= nb; attempt++) {
					const bool exact = attempt > 0;
					const int carsHere = exact ? 1 : nb;
					const int first = exact ? oc + attempt - 1 : oc;
					if (attempt == 1 && !__syncthreads_or(active && !sOk))
						break;
					if (lane == 0) {
						sDep = 0;
						if (!exact)
							sOk = 0;
					}
					__syncthreads();
					const bool work = active && (exact ? !sOk : true);
					const bool isLeader = wheel == 0 && slot < carsHere;

					// phase 1 (leaders): dependency check or demolition block,
					// then stage this car's tick IO
					if (work && isLeader) {
						const int ci = ca.carOrder[(size_t)arenaIdx * numCars + first + slot];
						GpuCarGameState& cst = arenaCars[ci];
						sCi[slot] = ci;
						int skip = 0;
						if (!exact) {
							if (cst.isDemoed || cst.noContactResponse != 0 ||
								arenaBodies[1 + ci].activationState != CU_ACTIVE_TAG)
								sDep = 1;
						} else {
							// Car::_PreTickUpdate head: demolition handling. RAW activation
							// writes (the CPU assigns m_activationState1 directly, so
							// non-demoed cars are forced awake every tick).
							if (cst.isDemoed) {
								float t = cst.demoRespawnTimer - dt;
								cst.demoRespawnTimer = t > 0.0f ? t : 0.0f; // RS_MAX
								if (cst.demoRespawnTimer == 0.0f)
									CuCarRespawn(arenaBodies[1 + ci], cst,
										ca.rngStates[arenaIdx], consts[1 + ci], k.tail);
								arenaBodies[1 + ci].activationState = CU_DISABLE_SIMULATION;
								cst.noContactResponse = 1;
								// CPU re-evaluates isDemoed: a just-respawned car continues
								// the pre-tick (with the disabled body + response flag still
								// set for this tick - quirk preserved)
								if (cst.isDemoed)
									skip = 1;
							} else {
								arenaBodies[1 + ci].activationState = CU_ACTIVE_TAG;
								cst.noContactResponse = 0;
							}
						}
						sSkip[slot] = skip;
						if (!skip) {
							// bench sets car->controls from the schedule before each tick;
							// RL chunked stepping holds per-car constant controls instead
							GpuCarControls ctrl = ca.constControls
								? ca.constControls[(size_t)arenaIdx * numCars + ci]
								: schedule[((size_t)arenaIdx * numCars + ci) * (size_t)scheduleTicks + (size_t)tickCount];
							CuCarTickIO& sIo = sIoAll[slot];
							const GpuBodyState& s = arenaBodies[1 + ci];
							sIo.body.basis = CuLoadBasis(s.basis);
							sIo.body.origin = GV(s.pos);
							sIo.body.linVel = GV(s.linVel);
							sIo.body.angVel = GV(s.angVel);
							sIo.body.invInertiaWorld = CuLoadBasis(s.invInertiaWorld);
							sIo.body.invInertiaDiagLocal = GV(consts[1 + ci].invInertiaLocal);
							sIo.body.invMass = consts[1 + ci].invMass;
							sIo.totalForce = GV(s.totalForce);
							sIo.totalTorque = GV(s.totalTorque);
							sIo.st = cst;
							for (int w = 0; w < 4; w++)
								sIo.wheels[w] = cst.wheels[w];
							sIo.controls = ctrl;
							// solverInfo.m_timeStep: 1/60 ctor default until the first
							// bullet step assigns the tick time
							sIo.solverTimeStep = tickCount == 0 ? (1.0f / 60.0f) : dt;
						}
					}
					__syncthreads();
					subLap(0);

					// phase 2: live dyn-body table (templates + committed state +
					// proxy aabbMin), one lane per body, after the demolition
					// block's response-flag writes; wheel lanes build their
					// wheel transform and suspension ray
					CuWheelScratch w;
					CuWheelResult wheelResult = {};
					const bool dep1 = sDep != 0;
					if (work && !dep1) {
						if (lane < bodiesPerArena) {
							const int d = lane;
							sDyn[d] = ca.dynTemplates[d];
							const GpuBodyState& s = arenaBodies[d];
							sDyn[d].basis = s.basis;
							sDyn[d].origin = s.pos;
							sDyn[d].linVel = s.linVel;
							sDyn[d].angVel = s.angVel;
							sDyn[d].invInertiaWorld = s.invInertiaWorld;
							sDyn[d].aabbMin = ca.proxies[(size_t)arenaIdx * bodiesPerArena + d].aabbMin;
							// live CF_NO_CONTACT_RESPONSE: cars demoed earlier in this
							// tick's iteration order are no-response for our wheel rays
							if (d >= 1)
								sDyn[d].hasContactResponse = arenaCars[d - 1].noContactResponse ? 0 : 1;
							sDynBodies[d] = DynToBody(sDyn[d]);
						}
						if (slot < carsHere && !sSkip[slot])
							CuCarPreTickWheelPre(sIoAll[slot], k.car, wheel, w,
								sRaySrc[lane], sRayTgt[lane]);
					}
					__syncthreads();

					const bool rayOwner = work && !dep1 && slot < carsHere && !sSkip[slot];
					GpuRayHit hit;
					CuRayTestStaticMeshesWarp(geo, rayOwner, rayOwner ? sRaySrc[lane] : CuVec3{},
						rayOwner ? sRayTgt[lane] : CuVec3{}, k.car.rayCastGroup, CU_FILTER_ALL, hit);
					subLap(1);

					if (rayOwner) {
						CuCarPreTickWheelPost(sIoAll[slot], k.car, geo, ca.grid, sDyn,
							bodiesPerArena, 1 + sCi[slot], sDynBodies, wheel,
							sRaySrc[lane], sRayTgt[lane], hit, w);
						wheelResult = CuWheelResultOf(sIoAll[slot], k.car, wheel, dt, w);
						if (!exact && w.groundIdx >= 0)
							sDep = 1;
					}
					__syncthreads();
					subLap(2);
#ifdef RS_GPU_PHASE_PROF
					if (profOn && lane == 0 && active && attempt == 0) {
						atomicAdd(&ca.profCycles[GPU_PHASE_COUNT + 10], 1ull); // speculative batches
						if (sDep) atomicAdd(&ca.profCycles[GPU_PHASE_COUNT + 11], 1ull); // fell back
					}
#endif

					CuWheelResult W[4];
					const int wheelBase = (threadIdx.x & 31) & ~3;
					for (int i = 0; i < 4; i++)
						W[i] = CuShflWheelResult(wheelResult, wheelBase + i);
					if (work && sDep == 0 && isLeader && !sSkip[slot]) {
						const int ci = sCi[slot];
						CuCarTickIO& sIo = sIoAll[slot];
						CuCarPreTick(sIo, k.car, dt, W);
						GpuBodyState& s = arenaBodies[1 + ci];
						for (int r = 0; r < 3; r++)
							s.basis.row[r] = { sIo.body.basis.r[r].x, sIo.body.basis.r[r].y, sIo.body.basis.r[r].z };
						s.pos = { sIo.body.origin.x, sIo.body.origin.y, sIo.body.origin.z };
						s.linVel = { sIo.body.linVel.x, sIo.body.linVel.y, sIo.body.linVel.z };
						s.angVel = { sIo.body.angVel.x, sIo.body.angVel.y, sIo.body.angVel.z };
						s.totalForce = { sIo.totalForce.x, sIo.totalForce.y, sIo.totalForce.z };
						s.totalTorque = { sIo.totalTorque.x, sIo.totalTorque.y, sIo.totalTorque.z };
						GpuCarGameState& cst = arenaCars[ci];
						cst = sIo.st;
						for (int w2 = 0; w2 < 4; w2++)
							cst.wheels[w2] = sIo.wheels[w2];
					}
					if (!exact && lane == 0)
						sOk = sDep == 0 ? 1 : 0;
					__syncthreads();
					subLap(3);
				}
			}
			if (active && !(fused && tick > 0))
				for (int p = lane; p < numPads; p += lanesPerArena)
					CuPadPreTick(arenaPads[p], dt);
			if (active && lane == 0) {
				// Ball::_PreTickUpdate (heatseeker steering / snowday ground-stick
				// reset / hoops+dropshot kickoff launch; no-op in soccar)
				if (ca.consts->gameMode == CU_GAMEMODE_HEATSEEKER)
					CuBallPreTickHS(arenaBodies[0], ca.ballStates[arenaIdx], *ca.consts, dt);
				else if (ca.consts->gameMode == CU_GAMEMODE_SNOWDAY)
					ca.ballStates[arenaIdx].groundStickApplied = 0;
				else if (ca.consts->gameMode == CU_GAMEMODE_HOOPS ||
					ca.consts->gameMode == CU_GAMEMODE_DROPSHOT)
					CuBallPreTickLaunch(arenaBodies[0], ca.ballStates[arenaIdx],
						*ca.consts, consts[0], dt);
			}
			if (!fused)
				__syncthreads();
			if (active && lane < bodiesPerArena) {
				CuApplyGravity(arenaBodies[lane], consts[lane]);
				if (fused)
					PhasePredict(arenaBodies[lane], consts[lane]);
			}
			if (!fused)
				__syncthreads();
			subLap(4);
		}
		if (!fused)
			__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_GAME_PRE] += t - profT0; profT0 = t; }

		if (!fused && startPhase <= GPU_PHASE_PREDICT && GPU_PHASE_PREDICT <= endPhase) {
			if (active && lane < bodiesPerArena)
				PhasePredict(arenaBodies[lane], consts[lane]);
		}
		__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_PREDICT] += t - profT0; profT0 = t; }

		if (ca.enabled && startPhase <= GPU_PHASE_COLLISION && GPU_PHASE_COLLISION <= endPhase) {
			GpuDynProxy* sProxies = reinterpret_cast<GpuDynProxy*>(sgBase + sgL.collideProxies);
			int* staticCand = reinterpret_cast<int*>(sgBase + sgL.collideCand);
			int2* sPairs = reinterpret_cast<int2*>(sgBase + sgL.collidePairs);
			{
				CuCollideIn cin;
				cin.bodies = arenaBodies;
				cin.bodyConsts = consts;
				cin.numBodies = bodiesPerArena;
				cin.cars = arenaCars;
				cin.ball = ca.ballStates + arenaIdx;
				cin.proxies = ca.proxies + (size_t)arenaIdx * bodiesPerArena;
				cin.regCounter = ca.regCounters + arenaIdx;
				cin.statics = ca.statics;
				cin.numStatics = ca.numStatics;
				cin.cellStaticCount = ca.cellStaticCount;
				cin.cellStatics = ca.cellStatics;
				cin.cellStaticStride = ca.cellStaticStride;
				cin.grid = ca.grid;
				cin.geo = geo;
				cin.bvhs = ca.bvhs;
				cin.bvhNodes = ca.bvhNodes;
				cin.bvhSubtreeBoxes = ca.bvhSubtreeBoxes;
				cin.triInfos = ca.triInfos;
				cin.triCull = ca.triCull;
				cin.meshMargins = ca.meshMargins;
				cin.cc = ca.consts;
				cin.tickCount = ca.tickBase + tick;
				cin.dt = dt;
				cin.manifolds = ca.manifolds + (size_t)arenaIdx * GPU_MAX_MANIFOLDS;
				cin.numManifolds = ca.numManifolds + arenaIdx;
				cin.pairsOut = ca.pairs + (size_t)arenaIdx * 96;
				cin.numPairsOut = ca.numPairs + arenaIdx;
				cin.tileDamage = ca.tileDamage
					? ca.tileDamage + (size_t)arenaIdx * ca.numTiles : nullptr;
				cin.tileNoResponse = ca.tileNoResponse
					? ca.tileNoResponse + (size_t)arenaIdx * ca.numTiles : nullptr;
#ifdef RS_GPU_PHASE_PROF
				cin.prof = profOn ? ca.profCycles + GPU_PHASE_COUNT : nullptr;
#endif
				const bool body = active && lane < bodiesPerArena;
				const int regBase = body ? *cin.regCounter : 0;
				GpuDynProxy proxy = {};
				if (body)
					proxy = cin.proxies[lane];
				const bool moved = body && CuCollideUpdateAabbBody(cin, lane, proxy);
				const unsigned ballot = __ballot_sync(0xffffffffu, moved);
				const int groupBase = (sub * lanesPerArena) & 31;
				const unsigned groupBits = (ballot >> groupBase) &
					(lanesPerArena >= 32 ? 0xffffffffu : ((1u << lanesPerArena) - 1u));
				if (moved)
					proxy.regSeq = regBase + __popc(groupBits & ((1u << lane) - 1u)) + 1;
				if (body && lane == 0 && groupBits)
					*cin.regCounter = regBase + __popc(groupBits);
				if (body) {
					cin.proxies[lane] = proxy;
					sProxies[lane] = proxy;
					CuCollideStaticCandidates(cin, proxy, lane,
						staticCand + lane * (1 + CU_STATIC_CAND_MAX));
				}
				__syncthreads();
				int numPairs = 0;
				if (active && lane == 0)
					numPairs = CuCollideBroadphase(cin, sProxies, staticCand, sPairs);
				__syncthreads();
				numPairs = __shfl_sync(0xffffffffu, numPairs, 0, lanesPerArena);
				CuCollideNarrowphase(cin, sPairs, numPairs, active, lane, lanesPerArena, groupBase,
					CuWarpMeshSmem{ sStepSmem, sgL.bytes, sgL.collideMesh, lanesPerArena });
			}
		} else {
			__syncthreads();
			__syncthreads();
		}
		__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_COLLISION] += t - profT0; profT0 = t; }

		if (ca.enabled && startPhase <= GPU_PHASE_SOLVE && GPU_PHASE_SOLVE <= endPhase) {
			if (active) {
				CuSolveTick(arenaBodies, consts, arenaCars, bodiesPerArena, ca.numStatics,
					ca.manifolds + (size_t)arenaIdx * GPU_MAX_MANIFOLDS,
					ca.numManifolds[arenaIdx],
					ca.pairs + (size_t)arenaIdx * 96,
					ca.numPairs[arenaIdx], dt, ca.solveScratch + arenaIdx,
					ca.statics,
					ca.tileNoResponse
						? ca.tileNoResponse + (size_t)arenaIdx * ca.numTiles : nullptr,
					sgBase, sgL.bytes, lane, lanesPerArena, (sub * lanesPerArena) & 31);
			}
		}
		__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_SOLVE] += t - profT0; profT0 = t; }

		if (startPhase <= GPU_PHASE_INTEGRATE && GPU_PHASE_INTEGRATE <= endPhase) {
			if (active && lane < bodiesPerArena) {
				PhaseIntegrate(arenaBodies[lane], consts[lane], dt);
				if (fused)
					PhaseActivation(arenaBodies[lane], consts[lane], dt);
			}
		}
		if (!fused)
			__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_INTEGRATE] += t - profT0; profT0 = t; }

		if (!fused && startPhase <= GPU_PHASE_ACTIVATION && GPU_PHASE_ACTIVATION <= endPhase) {
			if (active && lane < bodiesPerArena)
				PhaseActivation(arenaBodies[lane], consts[lane], dt);
		}
		if (!fused)
			__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_ACTIVATION] += t - profT0; profT0 = t; }

		if (ca.tailEnabled && startPhase <= GPU_PHASE_GAME_POST && GPU_PHASE_GAME_POST <= endPhase) {
			uint64_t tickCount = ca.tickBase + tick;
			CuPadGridCar* sPadCars = reinterpret_cast<CuPadGridCar*>(sgBase + sgL.io);
			int* sPadOrder = reinterpret_cast<int*>(sPadCars + numCars);
			int* sPadGrant = reinterpret_cast<int*>(sgBase + sgL.padGrant);
			if (active && lane < bodiesPerArena)
				CuClearForces(arenaBodies[lane]);
			if (active && lane >= 1 && lane < bodiesPerArena) {
				const int ci = lane - 1;
				GpuCarControls ctrl = ca.constControls
					? ca.constControls[(size_t)arenaIdx * numCars + ci]
					: schedule[((size_t)arenaIdx * numCars + ci) * (size_t)scheduleTicks + (size_t)tickCount];
				CuCarPostTick(arenaBodies[1 + ci], arenaCars[ci], k.tail, ctrl, dt);
				CuCarFinish(arenaBodies[1 + ci], arenaCars[ci], k.tail);
				CuPadGridCarStage(arenaBodies[1 + ci], arenaCars[ci], k.tail, *ca.consts,
					sPadCars[ci]);
				sPadOrder[ci] = ca.carOrder[(size_t)arenaIdx * numCars + ci];
			}
			if (active && lane == 0) {
				CuBallFinish(arenaBodies[0], ca.ballStates[arenaIdx], k.tail);
				if (ca.consts->gameMode == CU_GAMEMODE_DROPSHOT &&
					ca.ballStates[arenaIdx].dsHasDamaged &&
					ca.ballStates[arenaIdx].dsLastDamageTick == tickCount) {
					uint8_t* dmg = ca.tileDamage + (size_t)arenaIdx * ca.numTiles;
					uint8_t* nr = ca.tileNoResponse + (size_t)arenaIdx * ca.numTiles;
					for (int t = 0; t < ca.numTiles; t++)
						nr[t] = dmg[t] == GPU_DS_TILE_BROKEN ? 1 : 0;
				}
			}
			__syncthreads();
			int grants = 0;
			if (active) {
				for (int p = lane; p < numPads; p += lanesPerArena) {
					GpuPadState& pad = arenaPads[p];
					const GpuPadConfig& pc = k.pads[p];
					for (int oc = 0; oc < numCars; oc++) {
						const int ci = sPadOrder[oc];
						if (CuPadCollidesCar(pad, pc, sPadCars[ci], k.tail))
							pad.lockedCarIdx = ci;
					}
					sPadGrant[p] = CuPadPostTickPad(pad, pc, arenaCars, k.tail);
					grants |= sPadGrant[p] >= 0;
					if (fused && tick + 1 < numTicks)
						CuPadPreTick(pad, dt);
				}
			}
			grants = __syncthreads_or(grants);
			if (active && lane == 0 && grants)
				for (int p = 0; p < numPads; p++)
					if (sPadGrant[p] >= 0)
						CuPadGrantBoost(k.pads[p], arenaCars[sPadGrant[p]], k.tail);
		}
		if (!fused)
			__syncthreads();
		if (profOn && lane == 0)
			profAcc[GPU_PHASE_GAME_POST] += clock64() - profT0;
	}

	if (profOn && lane == 0 && active)
		for (int p = 0; p < GPU_PHASE_COUNT; p++)
			atomicAdd(&ca.profCycles[p], profAcc[p]);

	if (ca.blockCycles && threadIdx.x == 0) {
		const long long cycles = clock64() - blockStart;
		ca.blockCycles[(size_t)(ca.doneGen & 1) * gridDim.x + group] =
			(unsigned)min(cycles, (long long)UINT_MAX);
	}

	if (ca.mirror.bodies) {
		__syncthreads();
		const size_t first = (size_t)group * arenasPerBlock;
		const size_t n = (size_t)min(numActive - (int)first, arenasPerBlock);
		CuMirrorBytes(ca.mirror.bodies + first * bodiesPerArena, bodies + first * bodiesPerArena,
			sizeof(GpuBodyState) * bodiesPerArena * n);
		CuMirrorBytes(ca.mirror.cars + first * numCars, cars + first * numCars,
			sizeof(GpuCarGameState) * numCars * n);
		CuMirrorBytes(ca.mirror.pads + first * numPads, pads + first * numPads,
			sizeof(GpuPadState) * numPads * n);
		CuMirrorBytes(ca.mirror.balls + first, ca.ballStates + first,
			sizeof(GpuBallGameState) * n);
	}

	if (ca.arenaDone) {
		if (ca.mirror.arenaMirrored)
			__threadfence_system();
		else
			__threadfence();
		__syncthreads();
		if (active && lane == 0) {
			cuda::atomic_ref<unsigned, cuda::thread_scope_device>(ca.arenaDone[arenaIdx])
				.store(ca.doneGen, cuda::memory_order_release);
			if (ca.mirror.arenaMirrored)
				cuda::atomic_ref<unsigned, cuda::thread_scope_system>(
					ca.mirror.arenaMirrored[arenaIdx]).store(ca.doneGen, cuda::memory_order_release);
		}
	}
}

static void Launch(GpuBatch* b, cudaStream_t stream, int numTicks, int startPhase,
	int endPhase, const int* dArenaList = nullptr, int listCount = 0,
	long long tickBaseOverride = -1, const GpuStateMirror* mirror = nullptr) {
	constexpr float TICK_TIME = 1.0f / 120.0f;
	const int lanes = b->lanesPerArena;
	const int apb = b->arenasPerBlock;
	GpuCollideKernelArgs ca = {};
	ca.enabled = b->hasCollideData ? 1 : 0;
	if (b->hasCollideData) {
		ca.proxies = b->dDynProxies;
		ca.regCounters = b->dRegCounters;
		ca.ballStates = b->dBallStates;
		ca.consts = b->dCollideConsts;
		ca.statics = b->dStatics;
		ca.numStatics = b->numStatics;
		ca.cellStaticCount = b->dCellStaticCount;
		ca.cellStatics = b->dCellStatics;
		ca.cellStaticStride = b->cellStaticStride;
		ca.grid = b->grid;
		ca.bvhs = b->dMeshBvhs;
		ca.bvhNodes = b->dBvhNodes;
		ca.bvhSubtreeBoxes = b->dBvhSubtreeBoxes;
		ca.triInfos = b->dTriInfos;
		ca.triCull = b->dTriCull;
		ca.meshMargins = b->dMeshMargins;
		ca.manifolds = b->dManifolds;
		ca.numManifolds = b->dNumManifolds;
		ca.pairs = b->dPairs;
		ca.numPairs = b->dNumPairs;
		ca.solveScratch = b->dSolveScratch;
		ca.tickBase = b->tickCount;
		ca.tailEnabled = b->hasTailData ? 1 : 0;
		ca.dynTemplates = b->dDynTemplates;
		ca.carOrder = b->dCarOrder;
		ca.rngStates = b->dRngStates;
		ca.tileDamage = b->dTileDamage;
		ca.tileNoResponse = b->dTileNoResponse;
		ca.numTiles = b->numTiles;
		ca.constControls = b->dConstControls;
		ca.profCycles = b->dProfCycles;
		ca.arenaList = dArenaList;
		if (tickBaseOverride >= 0)
			ca.tickBase = (uint64_t)tickBaseOverride;
		if (mirror)
			ca.mirror = *mirror;
	}
	if (!dArenaList) {
		ca.arenaDone = b->dArenaDone;
		ca.doneGen = ++b->launchGen;
		ca.blockCycles = b->dBlockCycles;
	}
	const int count = dArenaList ? listCount : b->numArenas;
	const int grid = dArenaList ? (count + apb - 1) / apb : b->numBlocks;
	size_t smemBytes = (size_t)CuStepGroupSmemLayout(b->bodiesPerArena, lanes,
		b->numPads).bytes * (size_t)apb;
	GpuStepArenasKernel<<<grid, CU_STEP_BLOCK_THREADS, smemBytes, stream>>>(
		b->dBodies, b->dConsts, b->dCars, b->dPads, b->dSchedule, b->scheduleTicks,
		b->numCars, b->numPads, numTicks, startPhase, endPhase, TICK_TIME,
		lanes, count, apb, b->geo, ca, b->stepConsts);
	CudaCheck(cudaGetLastError(), "kernel launch");
}

void StepTicks(GpuBatch* b, int numTicks) {
	Launch(b, b->stream, numTicks, 0, GPU_PHASE_COUNT - 1);
	CudaCheck(cudaStreamSynchronize(b->stream), "kernel run");
	b->tickCount += numTicks;
}

// 2026-06-11 async stepping: launch without the device sync so the chunk can
// overlap CPU work; SyncBatch() completes it. StepTicksSubset re-steps ONLY the
// listed arenas over the same tick window (explicit tickBase) - used when a
// reset invalidates an arena's async chunk; bit-identical to its slice of a
// full chunk because arenas are independent.
void StepTicksAsync(GpuBatch* b, int numTicks) {
	Launch(b, b->stream, numTicks, 0, GPU_PHASE_COUNT - 1);
	b->tickCount += numTicks;
}

unsigned long long BatchTickCount(GpuBatch* b) {
	return (unsigned long long)b->tickCount;
}

void SyncBatch(GpuBatch* b) {
	CudaCheck(cudaStreamSynchronize(b->stream), "async kernel run");
}

void StepTicksSubset(GpuBatch* b, int numTicks, const int* arenaIdxs, int count,
	unsigned long long tickBase) {
	if (count <= 0)
		return;
	if (!b->dArenaList)
		DeviceAlloc(b->dArenaList, sizeof(int) * b->numArenas, "list alloc");
	MemcpyOnBatchStream(b, b->dArenaList, arenaIdxs, sizeof(int) * count,
		cudaMemcpyHostToDevice, "list up");
	Launch(b, b->stream, numTicks, 0, GPU_PHASE_COUNT - 1, b->dArenaList, count,
		(long long)tickBase);
	CudaCheck(cudaStreamSynchronize(b->stream), "kernel run");
	// tickCount not advanced: this re-covers a window already counted
}

void StepTicksSubsetAsync(GpuBatch* b, int numTicks, int count, unsigned long long tickBase) {
	if (count <= 0)
		return;
	if (!b->staging.slots) {
		fprintf(stderr, "StepTicksSubsetAsync: no arena staging registered\n");
		std::abort();
	}
	if (!b->dArenaList)
		DeviceAlloc(b->dArenaList, sizeof(int) * b->numArenas, "list alloc");
	if (!b->dConstControls)
		DeviceAlloc(b->dConstControls,
			sizeof(GpuCarControls) * (size_t)b->numArenas * b->numCars, "alloc const controls");
	GpuStageSubsetKernel<<<1, 256, 0, b->resetStream>>>(b->dArenaDone, b->launchGen,
		b->staging.slots, count, b->dArenaList, b->staging.controls, b->dConstControls,
		b->numCars);
	CudaCheck(cudaGetLastError(), "stage subset launch");
	Launch(b, b->resetStream, numTicks, 0, GPU_PHASE_COUNT - 1, b->dArenaList, count,
		(long long)tickBase);
	OrderAfterResetWork(b);
}

// 2026-06-11: per-phase cycle profiling (diagnostic; ~zero cost while disabled)
void EnablePhaseProfiling(GpuBatch* b) {
	if (b->dProfCycles)
		return;
#ifndef RS_GPU_PHASE_PROF
	fprintf(stderr, "EnablePhaseProfiling: built without RS_GPU_PHASE_PROF, phase cycles will read 0\n");
#endif
	DeviceAlloc(b->dProfCycles,
		sizeof(unsigned long long) * (GPU_PHASE_PROF_SLOTS), "prof alloc");
	CudaCheck(cudaMemset(b->dProfCycles, 0,
		sizeof(unsigned long long) * (GPU_PHASE_PROF_SLOTS)), "prof zero");
}

void GetPhaseCycles(GpuBatch* b, unsigned long long out[GPU_PHASE_PROF_SLOTS]) {
	if (!b->dProfCycles) {
		for (int i = 0; i < GPU_PHASE_PROF_SLOTS; i++) out[i] = 0;
		return;
	}
	CudaCheck(cudaMemcpy(out, b->dProfCycles,
		sizeof(unsigned long long) * (GPU_PHASE_PROF_SLOTS),
		cudaMemcpyDeviceToHost), "prof down");
}

// Batched full-state download: one device-to-host copy per array (the per-arena
// DownloadArenaState loop is prohibitively chatty at RL batch sizes)
void DownloadAllArenaStates(GpuBatch* b,
	GpuBodyState* bodies, GpuCarGameState* cars, GpuPadState* pads) {
	MemcpyOnBatchStream(b, bodies, b->dBodies,
		sizeof(GpuBodyState) * b->numArenas * b->bodiesPerArena,
		cudaMemcpyDeviceToHost, "down all bodies");
	MemcpyOnBatchStream(b, cars, b->dCars,
		sizeof(GpuCarGameState) * b->numArenas * b->numCars,
		cudaMemcpyDeviceToHost, "down all cars");
	if (b->numPads > 0)
		MemcpyOnBatchStream(b, pads, b->dPads,
			sizeof(GpuPadState) * b->numArenas * b->numPads,
			cudaMemcpyDeviceToHost, "down all pads");
}

// Batched ball game state download (callback side effects: hit info lives in the
// car states; this carries impulse caches + mode info)
void DownloadAllBallStates(GpuBatch* b, GpuBallGameState* balls) {
	MemcpyOnBatchStream(b, balls, b->dBallStates,
		sizeof(GpuBallGameState) * b->numArenas, cudaMemcpyDeviceToHost, "down all balls");
}

unsigned StepTicksAndMirrorAsync(GpuBatch* b, int numTicks, GpuBodyState* bodies,
	GpuCarGameState* cars, GpuPadState* pads, GpuBallGameState* balls, unsigned* arenaMirrored) {
	if (!b->hasCollideData) {
		fprintf(stderr, "StepTicksAndMirrorAsync: the batch has no collide data uploaded\n");
		std::abort();
	}
	GpuStateMirror m = {};
	m.bodies = MappedDevicePtr(bodies, "mirror bodies");
	m.cars = MappedDevicePtr(cars, "mirror cars");
	m.pads = b->numPads > 0 ? MappedDevicePtr(pads, "mirror pads") : nullptr;
	m.balls = MappedDevicePtr(balls, "mirror balls");
	m.arenaMirrored = MappedDevicePtr(arenaMirrored, "mirror arena flags");
	Launch(b, b->stream, numTicks, 0, GPU_PHASE_COUNT - 1, nullptr, 0, -1, &m);
	b->tickCount += numTicks;
	return b->launchGen;
}

void RecordDownloadEvent(GpuBatch* b) {
	CudaCheck(cudaEventRecord(b->downloadEvent, b->stream), "record download event");
}

bool DownloadEventDone(GpuBatch* b) {
	const cudaError_t status = cudaEventQuery(b->downloadEvent);
	if (status == cudaErrorNotReady)
		return false;
	CudaCheck(status, "query download event");
	return true;
}

void WaitDownloadEvent(GpuBatch* b) {
	CudaCheck(cudaEventSynchronize(b->downloadEvent), "wait download event");
}

struct CuArenaArrays {
	GpuBodyState* bodies;
	GpuDynProxy* proxies;
	GpuCarGameState* cars;
	int* carOrder;
	GpuPadState* pads;
	GpuBallGameState* balls;
	int* regCounters;
	uint32_t* rngStates;
	uint8_t* tileDamage;
	uint8_t* tileNoResponse;
};

static CuArenaArrays ArenaArrays(const GpuBatch* b) {
	return { b->dBodies, b->dDynProxies, b->dCars, b->dCarOrder, b->dPads, b->dBallStates,
		b->dRegCounters, b->dRngStates, b->dTileDamage, b->dTileNoResponse };
}

template <typename Unit, typename T>
__device__ void CuCopyArenaSlice(const T* src, T* dst, int perArena, int2 pair) {
	static_assert(sizeof(T) % sizeof(Unit) == 0, "arena slices are copied in whole units");
	const int units = perArena * (int)(sizeof(T) / sizeof(Unit));
	const Unit* from = reinterpret_cast<const Unit*>(src + (size_t)pair.x * perArena);
	Unit* to = reinterpret_cast<Unit*>(dst + (size_t)pair.y * perArena);
	for (int i = threadIdx.x; i < units; i += blockDim.x)
		to[i] = from[i];
}

__global__ void CopyArenaStatesKernel(CuArenaArrays src, CuArenaArrays dst, const int2* pairs,
	int bodiesPerArena, int numCars, int numPads, int numTiles) {
	const int2 pair = pairs[blockIdx.x];
	CuCopyArenaSlice<unsigned>(src.bodies, dst.bodies, bodiesPerArena, pair);
	CuCopyArenaSlice<unsigned>(src.proxies, dst.proxies, bodiesPerArena, pair);
	CuCopyArenaSlice<unsigned>(src.cars, dst.cars, numCars, pair);
	CuCopyArenaSlice<unsigned>(src.carOrder, dst.carOrder, numCars, pair);
	CuCopyArenaSlice<unsigned>(src.pads, dst.pads, numPads, pair);
	CuCopyArenaSlice<unsigned>(src.balls, dst.balls, 1, pair);
	CuCopyArenaSlice<unsigned>(src.regCounters, dst.regCounters, 1, pair);
	CuCopyArenaSlice<unsigned>(src.rngStates, dst.rngStates, 1, pair);
	if (numTiles > 0) {
		CuCopyArenaSlice<uint8_t>(src.tileDamage, dst.tileDamage, numTiles, pair);
		CuCopyArenaSlice<uint8_t>(src.tileNoResponse, dst.tileNoResponse, numTiles, pair);
	}
}

static cudaEvent_t CopyEvent(cudaEvent_t& event) {
	if (!event)
		CudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create copy event");
	return event;
}

void CopyArenaStates(GpuBatch* source, GpuBatch* dest, const int* sourceIdx, const int* destIdx,
	int count) {
	if (source == dest || source->numCars != dest->numCars || source->numPads != dest->numPads ||
		source->bodiesPerArena != dest->bodiesPerArena || source->numTiles != dest->numTiles ||
		!source->hasTailData || !dest->hasTailData) {
		fprintf(stderr, "CopyArenaStates: needs two distinct batches with game-tail data and equal "
			"cars (%d, %d), pads (%d, %d), bodies (%d, %d) and tiles (%d, %d) per arena\n",
			source->numCars, dest->numCars, source->numPads, dest->numPads, source->bodiesPerArena,
			dest->bodiesPerArena, source->numTiles, dest->numTiles);
		std::abort();
	}
	if (source->tickCount != dest->tickCount) {
		fprintf(stderr, "CopyArenaStates: source tick %llu differs from destination tick %llu\n",
			(unsigned long long)source->tickCount, (unsigned long long)dest->tickCount);
		std::abort();
	}
	if (count <= 0 || count > dest->numArenas) {
		fprintf(stderr, "CopyArenaStates: %d pairs for a destination of %d arenas\n", count,
			dest->numArenas);
		std::abort();
	}
	CudaCheck(cudaEventRecord(CopyEvent(source->copySourceEvent), source->stream),
		"record copy source event");
	CudaCheck(cudaStreamWaitEvent(dest->stream, source->copySourceEvent, 0),
		"order copy after source");
	CudaCheck(cudaEventSynchronize(CopyEvent(dest->copyDoneEvent)), "wait previous copy");
	if (!dest->copyPairs) {
		int2* hostPairs = nullptr;
		int2* devicePairs = nullptr;
		PinnedHostAlloc(hostPairs, sizeof(int2) * dest->numArenas, "alloc copy pairs");
		try {
			DeviceAlloc(devicePairs, sizeof(int2) * dest->numArenas, "alloc device copy pairs");
		} catch (...) {
			cudaFreeHost(hostPairs);
			throw;
		}
		dest->copyPairs = hostPairs;
		dest->dCopyPairs = devicePairs;
	}
	for (int i = 0; i < count; i++) {
		if (sourceIdx[i] < 0 || sourceIdx[i] >= source->numArenas || destIdx[i] < 0 ||
			destIdx[i] >= dest->numArenas) {
			fprintf(stderr, "CopyArenaStates: pair %d maps arena %d of %d to arena %d of %d\n", i,
				sourceIdx[i], source->numArenas, destIdx[i], dest->numArenas);
			std::abort();
		}
		dest->copyPairs[i] = make_int2(sourceIdx[i], destIdx[i]);
	}
	CudaCheck(cudaMemcpyAsync(dest->dCopyPairs, dest->copyPairs, sizeof(int2) * count,
		cudaMemcpyHostToDevice, dest->stream), "upload copy pairs");
	CopyArenaStatesKernel<<<count, 128, 0, dest->stream>>>(ArenaArrays(source), ArenaArrays(dest),
		dest->dCopyPairs, dest->bodiesPerArena, dest->numCars, dest->numPads, dest->numTiles);
	CudaCheck(cudaGetLastError(), "copy arenas launch");
	CudaCheck(cudaEventRecord(dest->copyDoneEvent, dest->stream), "record copy done event");
	CudaCheck(cudaStreamWaitEvent(source->stream, dest->copyDoneEvent, 0),
		"order source after copy");
}

void RunPhaseSegment(GpuBatch* b, int startPhase, int endPhase) {
	Launch(b, b->stream, 1, startPhase, endPhase);
	CudaCheck(cudaStreamSynchronize(b->stream), "kernel run");
}

void DownloadArenaState(GpuBatch* b, int arenaIdx,
	GpuBodyState* bodies, GpuCarGameState* cars, GpuPadState* pads) {
	MemcpyOnBatchStream(b, bodies, b->dBodies + (size_t)arenaIdx * b->bodiesPerArena,
		sizeof(GpuBodyState) * b->bodiesPerArena, cudaMemcpyDeviceToHost, "download bodies");
	MemcpyOnBatchStream(b, cars, b->dCars + (size_t)arenaIdx * b->numCars,
		sizeof(GpuCarGameState) * b->numCars, cudaMemcpyDeviceToHost, "download cars");
	MemcpyOnBatchStream(b, pads, b->dPads + (size_t)arenaIdx * b->numPads,
		sizeof(GpuPadState) * b->numPads, cudaMemcpyDeviceToHost, "download pads");
}

} // namespace rscuda
