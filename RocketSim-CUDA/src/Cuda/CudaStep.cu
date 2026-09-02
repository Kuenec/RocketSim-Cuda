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

struct GpuBatch;
static cudaStream_t BatchStream(GpuBatch* b);

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

	// 2026-07-16: all of this batch's runtime work (step kernel, per-step
	// control uploads, state downloads, reset re-uploads) runs on a private
	// non-blocking stream instead of the legacy default stream. The physics
	// itself is unchanged - same kernel, same launch shape, same copy
	// contents; only the stream the work is ordered on differs. This stops
	// the simulation from serializing against the trainer's ML kernels
	// (libtorch uses the legacy stream), letting the inference forward run
	// concurrently with the in-flight physics chunk. Blocking-call semantics
	// are preserved: every copy is async-on-stream + stream-sync.
	cudaStream_t stream = nullptr;
	cudaEvent_t downloadEvent = nullptr; // completion of the async state download
	unsigned long long* dProfCycles = nullptr; // [GPU_PHASE_COUNT], diagnostic
	int* dArenaList = nullptr; // [numArenas], subset-step index list (lazy alloc)

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

	GpuCarConsts* dCarConsts; // per-car-slot constants (uploaded once)

	// mesh BVHs + internal-edge data (uploaded once)
	GpuMeshBvh* dMeshBvhs;
	GpuBvhNode* dBvhNodes;
	GpuBvhNodeBounds* dBvhNodeBounds;
	GpuVec3* dHullPoints;  // snowday puck (null otherwise)
	GpuCarControls* dConstControls; // per-car constant controls (chunked RL stepping)
	GpuTileShape* dTiles;          // dropshot tiles (null otherwise)
	GpuTileTables* dTileTables;
	uint8_t* dTileDamage;          // [numArenas * numTiles]
	uint8_t* dTileNoResponse;      // [numArenas * numTiles]
	int numTiles;
	GpuBvhSubtree* dBvhSubtrees;
	GpuTriInfo* dTriInfos;
	float* dMeshMargins;

	// collision orchestration (uploaded once / persistent per arena)
	GpuCollideConsts* dCollideConsts;
	GpuStaticObj* dStatics;
	int numStatics;
	int* dCellStaticStart;
	int* dCellStaticList;
	GpuGridParams grid;
	GpuDynProxy* dDynProxies;     // [numArenas * bodiesPerArena]
	int* dRegCounters;            // [numArenas]
	GpuBallGameState* dBallStates;// [numArenas]
	CuManifold* dManifolds;       // [numArenas * GPU_MAX_MANIFOLDS]
	int* dNumManifolds;           // [numArenas]
	GpuManifoldOut* dManifoldsOut;// [numArenas * GPU_MAX_MANIFOLDS] (download mirror)
	int* dCollideDebug;           // [numArenas * (2*96+1)] pair-list diagnostic
	int2* dPairs;                 // [numArenas * 96] (collide -> solver islands)
	int* dNumPairs;               // [numArenas]
	CuSolveCtx* dSolveScratch;    // [numArenas] (too large for device stack)
	bool hasCollideData;

	// game-tail data (full-step mode)
	GpuTailConsts* dTailConsts;
	GpuPadConfig* dPadConfigs;
	GpuDynBody* dDynTemplates;    // [bodiesPerArena] shape/filter templates
	int* dCarOrder;               // [numArenas * numCars] CPU _cars iteration order
	uint32_t* dRngStates;         // [numArenas] minstd_rand0 state (demo respawns)
	bool hasTailData;
};
#define COLLIDE_DEBUG_INTS (2 * 96 + 1)

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
	// the kernel's parallel pad phase stages grants in a fixed 64-slot shared
	// array (soccar has the most pads at 34); fail loudly rather than truncate
	if (numPads > 64) {
		fprintf(stderr, "CreateBatch: numPads %d exceeds the kernel's 64-pad limit\n", numPads);
		std::abort();
	}
	b->numPads = numPads;
	b->bodiesPerArena = 1 + numCarsPerArena;
	b->scheduleTicks = 0;
	b->tickCount = 0;
	CudaCheck(cudaMalloc(&b->dBodies, sizeof(GpuBodyState) * numArenas * b->bodiesPerArena), "alloc bodies");
	CudaCheck(cudaMalloc(&b->dConsts, sizeof(GpuBodyConsts) * b->bodiesPerArena), "alloc consts");
	CudaCheck(cudaMalloc(&b->dCars, sizeof(GpuCarGameState) * numArenas * numCarsPerArena), "alloc cars");
	CudaCheck(cudaMalloc(&b->dPads, sizeof(GpuPadState) * numArenas * numPads), "alloc pads");
	b->dSchedule = nullptr;
	CudaCheck(cudaMalloc(&b->dDynProxies, sizeof(GpuDynProxy) * numArenas * b->bodiesPerArena), "alloc proxies");
	CudaCheck(cudaMalloc(&b->dRegCounters, sizeof(int) * numArenas), "alloc regctr");
	CudaCheck(cudaMalloc(&b->dBallStates, sizeof(GpuBallGameState) * numArenas), "alloc ballstates");
	CudaCheck(cudaMemset(b->dBallStates, 0, sizeof(GpuBallGameState) * numArenas), "zero ballstates");
	CudaCheck(cudaMalloc(&b->dManifolds, sizeof(CuManifold) * numArenas * GPU_MAX_MANIFOLDS), "alloc manifolds");
	CudaCheck(cudaMalloc(&b->dNumManifolds, sizeof(int) * numArenas), "alloc nummanifolds");
	CudaCheck(cudaMemset(b->dNumManifolds, 0, sizeof(int) * numArenas), "zero nummanifolds");
	CudaCheck(cudaMalloc(&b->dManifoldsOut, sizeof(GpuManifoldOut) * numArenas * GPU_MAX_MANIFOLDS), "alloc manifoldsout");
	CudaCheck(cudaMalloc(&b->dCollideDebug, sizeof(int) * numArenas * COLLIDE_DEBUG_INTS), "alloc collidedbg");
	CudaCheck(cudaMalloc(&b->dPairs, sizeof(int2) * numArenas * 96), "alloc pairs");
	CudaCheck(cudaMalloc(&b->dNumPairs, sizeof(int) * numArenas), "alloc numpairs");
	CudaCheck(cudaMemset(b->dNumPairs, 0, sizeof(int) * numArenas), "zero numpairs");
	CudaCheck(cudaMalloc(&b->dSolveScratch, sizeof(CuSolveCtx) * numArenas), "alloc solve scratch");
	CudaCheck(cudaMalloc(&b->dRngStates, sizeof(uint32_t) * numArenas), "alloc rng states");
	CudaCheck(cudaMemset(b->dRngStates, 1, sizeof(uint32_t) * numArenas), "init rng states");
	CudaCheck(cudaStreamCreateWithFlags(&b->stream, cudaStreamNonBlocking), "create stream");
	CudaCheck(cudaEventCreateWithFlags(&b->downloadEvent, cudaEventDisableTiming), "create event");
	return b;
}

static cudaStream_t BatchStream(GpuBatch* b) {
	return b->stream;
}

void DestroyBatch(GpuBatch* b) {
	if (b->stream) {
		cudaStreamSynchronize(b->stream);
		cudaEventDestroy(b->downloadEvent);
		cudaStreamDestroy(b->stream);
	}
	cudaFree(b->dBodies);
	cudaFree(b->dConsts);
	cudaFree(b->dCars);
	cudaFree(b->dPads);
	cudaFree(b->dSchedule);
	cudaFree(b->dPlanes);
	cudaFree(b->dMeshes);
	cudaFree(b->dTris);
	cudaFree(b->dCarConsts);
	cudaFree(b->dMeshBvhs);
	cudaFree(b->dBvhNodes);
	cudaFree(b->dBvhNodeBounds);
	cudaFree(b->dHullPoints);
	cudaFree(b->dConstControls);
	cudaFree(b->dTiles);
	cudaFree(b->dTileTables);
	cudaFree(b->dTileDamage);
	cudaFree(b->dTileNoResponse);
	cudaFree(b->dBvhSubtrees);
	cudaFree(b->dTriInfos);
	cudaFree(b->dMeshMargins);
	cudaFree(b->dCollideConsts);
	cudaFree(b->dStatics);
	cudaFree(b->dCellStaticStart);
	cudaFree(b->dCellStaticList);
	cudaFree(b->dDynProxies);
	cudaFree(b->dRegCounters);
	cudaFree(b->dBallStates);
	cudaFree(b->dManifolds);
	cudaFree(b->dNumManifolds);
	cudaFree(b->dManifoldsOut);
	cudaFree(b->dCollideDebug);
	cudaFree(b->dPairs);
	cudaFree(b->dNumPairs);
	cudaFree(b->dSolveScratch);
	cudaFree(b->dTailConsts);
	cudaFree(b->dPadConfigs);
	cudaFree(b->dDynTemplates);
	cudaFree(b->dCarOrder);
	cudaFree(b->dRngStates);
	delete b;
}

// Unquantized node bounds for one mesh's node span (same device expression the
// ray walk evaluated inline before 2026-09-02, so the table is bit-identical)
__global__ void BuildBvhNodeBoundsKernel(const GpuBvhNode* nodes, GpuBvhNodeBounds* out,
	int count, GpuVec3 quantization, GpuVec3 bvhMin) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count)
		return;
	CuVec3 bounds[2];
	CuBvhUnquantizeNode(nodes[i], GV(quantization), GV(bvhMin), bounds);
	out[i].mn = { bounds[0].x, bounds[0].y, bounds[0].z };
	out[i].mx = { bounds[1].x, bounds[1].y, bounds[1].z };
}

void UploadMeshBvhs(GpuBatch* b,
	const GpuMeshBvh* bvhs, int numMeshes,
	const GpuBvhNode* nodes, int numNodes,
	const GpuBvhSubtree* subtrees, int numSubtrees,
	const GpuTriInfo* triInfos, int numTris,
	const float* meshMargins) {
	CudaCheck(cudaMalloc(&b->dMeshBvhs, sizeof(GpuMeshBvh) * numMeshes), "alloc bvhs");
	CudaCheck(cudaMalloc(&b->dBvhNodes, sizeof(GpuBvhNode) * numNodes), "alloc bvh nodes");
	CudaCheck(cudaMalloc(&b->dBvhNodeBounds, sizeof(GpuBvhNodeBounds) * (numNodes > 0 ? numNodes : 1)),
		"alloc bvh node bounds");
	CudaCheck(cudaMalloc(&b->dBvhSubtrees, sizeof(GpuBvhSubtree) * numSubtrees), "alloc bvh subtrees");
	CudaCheck(cudaMalloc(&b->dTriInfos, sizeof(GpuTriInfo) * numTris), "alloc tri infos");
	CudaCheck(cudaMalloc(&b->dMeshMargins, sizeof(float) * numMeshes), "alloc mesh margins");
	CudaCheck(cudaMemcpy(b->dMeshBvhs, bvhs, sizeof(GpuMeshBvh) * numMeshes, cudaMemcpyHostToDevice), "up bvhs");
	CudaCheck(cudaMemcpy(b->dBvhNodes, nodes, sizeof(GpuBvhNode) * numNodes, cudaMemcpyHostToDevice), "up bvh nodes");
	CudaCheck(cudaMemcpy(b->dBvhSubtrees, subtrees, sizeof(GpuBvhSubtree) * numSubtrees, cudaMemcpyHostToDevice), "up bvh subtrees");
	CudaCheck(cudaMemcpy(b->dTriInfos, triInfos, sizeof(GpuTriInfo) * numTris, cudaMemcpyHostToDevice), "up tri infos");
	CudaCheck(cudaMemcpy(b->dMeshMargins, meshMargins, sizeof(float) * numMeshes, cudaMemcpyHostToDevice), "up mesh margins");
	// per-mesh node spans: [nodeStart, next mesh's nodeStart) in upload order
	for (int m = 0; m < numMeshes; m++) {
		int end = m + 1 < numMeshes ? bvhs[m + 1].nodeStart : numNodes;
		int count = end - bvhs[m].nodeStart;
		if (count <= 0)
			continue;
		BuildBvhNodeBoundsKernel<<<(count + 255) / 256, 256>>>(
			b->dBvhNodes + bvhs[m].nodeStart, b->dBvhNodeBounds + bvhs[m].nodeStart,
			count, bvhs[m].bvhQuantization, bvhs[m].bvhAabbMin);
		CudaCheck(cudaGetLastError(), "bvh node bounds launch");
	}
	CudaCheck(cudaDeviceSynchronize(), "bvh node bounds build");
	// the ray tests reach the BVHs through the world-geo descriptor
	b->geo.bvhs = b->dMeshBvhs;
	b->geo.bvhNodes = b->dBvhNodes;
	b->geo.bvhNodeBounds = b->dBvhNodeBounds;

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
			cm[m].root = nodes[bvhs[m].nodeStart];
			CudaCheck(cudaMemcpy(&cm[m].rootBounds, b->dBvhNodeBounds + bvhs[m].nodeStart,
				sizeof(GpuBvhNodeBounds), cudaMemcpyDeviceToHost), "read back root bounds");
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
	CudaCheck(cudaMalloc(&b->dTiles, sizeof(GpuTileShape) * numTiles), "alloc tiles");
	CudaCheck(cudaMemcpy(b->dTiles, tiles, sizeof(GpuTileShape) * numTiles,
		cudaMemcpyHostToDevice), "up tiles");
	CudaCheck(cudaMalloc(&b->dTileTables, sizeof(GpuTileTables)), "alloc tile tables");
	CudaCheck(cudaMemcpy(b->dTileTables, &tables, sizeof(GpuTileTables),
		cudaMemcpyHostToDevice), "up tile tables");
	size_t stateLen = (size_t)b->numArenas * numTiles;
	CudaCheck(cudaMalloc(&b->dTileDamage, stateLen), "alloc tile damage");
	CudaCheck(cudaMemset(b->dTileDamage, 0, stateLen), "init tile damage");
	CudaCheck(cudaMalloc(&b->dTileNoResponse, stateLen), "alloc tile resp");
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
	CudaCheck(cudaMalloc(&b->dHullPoints, sizeof(GpuVec3) * numPoints), "alloc hull pts");
	CudaCheck(cudaMemcpy(b->dHullPoints, points, sizeof(GpuVec3) * numPoints,
		cudaMemcpyHostToDevice), "up hull pts");
	b->geo.hullPoints = b->dHullPoints;
	b->geo.numHullPoints = numPoints;
}

void UploadCollideData(GpuBatch* b, const GpuCollideConsts& consts,
	const GpuStaticObj* statics, int numStatics,
	const int* cellStaticStart, int numCells,
	const int* cellStaticList, int listLen,
	const GpuGridParams& grid) {
	CudaCheck(cudaMalloc(&b->dCollideConsts, sizeof(GpuCollideConsts)), "alloc collide consts");
	CudaCheck(cudaMalloc(&b->dStatics, sizeof(GpuStaticObj) * numStatics), "alloc statics");
	CudaCheck(cudaMalloc(&b->dCellStaticStart, sizeof(int) * (numCells + 1)), "alloc cell start");
	CudaCheck(cudaMalloc(&b->dCellStaticList, sizeof(int) * listLen), "alloc cell list");
	CudaCheck(cudaMemcpy(b->dCollideConsts, &consts, sizeof(GpuCollideConsts), cudaMemcpyHostToDevice), "up collide consts");
	CudaCheck(cudaMemcpy(b->dStatics, statics, sizeof(GpuStaticObj) * numStatics, cudaMemcpyHostToDevice), "up statics");
	CudaCheck(cudaMemcpy(b->dCellStaticStart, cellStaticStart, sizeof(int) * (numCells + 1), cudaMemcpyHostToDevice), "up cell start");
	CudaCheck(cudaMemcpy(b->dCellStaticList, cellStaticList, sizeof(int) * listLen, cudaMemcpyHostToDevice), "up cell list");
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
	CudaCheck(cudaMalloc(&b->dTailConsts, sizeof(GpuTailConsts)), "alloc tail consts");
	CudaCheck(cudaMalloc(&b->dPadConfigs, sizeof(GpuPadConfig) * numPads), "alloc pad configs");
	CudaCheck(cudaMalloc(&b->dDynTemplates, sizeof(GpuDynBody) * b->bodiesPerArena), "alloc dyn templates");
	CudaCheck(cudaMalloc(&b->dCarOrder, sizeof(int) * b->numArenas * b->numCars), "alloc car order");
	CudaCheck(cudaMemcpy(b->dTailConsts, &consts, sizeof(GpuTailConsts), cudaMemcpyHostToDevice), "up tail consts");
	CudaCheck(cudaMemcpy(b->dPadConfigs, padConfigs, sizeof(GpuPadConfig) * numPads, cudaMemcpyHostToDevice), "up pad configs");
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

void DownloadCollideResults(GpuBatch* b, int arenaIdx,
	GpuManifoldOut* manifolds, int* numManifolds,
	GpuDynProxy* proxies, GpuBallGameState* ball) {
	CudaCheck(cudaMemcpy(numManifolds, b->dNumManifolds + arenaIdx, sizeof(int),
		cudaMemcpyDeviceToHost), "down nummanifolds");
	CudaCheck(cudaMemcpy(manifolds, b->dManifoldsOut + (size_t)arenaIdx * GPU_MAX_MANIFOLDS,
		sizeof(GpuManifoldOut) * GPU_MAX_MANIFOLDS, cudaMemcpyDeviceToHost), "down manifolds");
	CudaCheck(cudaMemcpy(proxies, b->dDynProxies + (size_t)arenaIdx * b->bodiesPerArena,
		sizeof(GpuDynProxy) * b->bodiesPerArena, cudaMemcpyDeviceToHost), "down proxies");
	CudaCheck(cudaMemcpy(ball, b->dBallStates + arenaIdx, sizeof(GpuBallGameState),
		cudaMemcpyDeviceToHost), "down ball state");
}

void DownloadCollideDebug(GpuBatch* b, int arenaIdx, int* debugOut) {
	CudaCheck(cudaMemcpy(debugOut, b->dCollideDebug + (size_t)arenaIdx * COLLIDE_DEBUG_INTS,
		sizeof(int) * COLLIDE_DEBUG_INTS, cudaMemcpyDeviceToHost), "down collide dbg");
}

void DownloadSolveDebug(GpuBatch* b, int arenaIdx,
	GpuSolverRowDl* contactRows, int* numContact,
	GpuSolverRowDl* frictionRows, int* numFriction) {
	static CuSolveCtx s; // host mirror (large)
	CudaCheck(cudaMemcpy(&s, b->dSolveScratch + arenaIdx, sizeof(CuSolveCtx),
		cudaMemcpyDeviceToHost), "down solve scratch");
	auto conv = [](const CuSolverRow& r, GpuSolverRowDl& o) {
		o.bodyA = r.bodyIdA;
		o.bodyB = r.bodyIdB;
		o.normal1 = { r.contactNormal1.x, r.contactNormal1.y, r.contactNormal1.z };
		o.relpos1CrossNormal = { r.relpos1CrossNormal.x, r.relpos1CrossNormal.y, r.relpos1CrossNormal.z };
		o.angularComponentA = { r.angularComponentA.x, r.angularComponentA.y, r.angularComponentA.z };
		o.jacDiagABInv = r.jacDiagABInv;
		o.rhs = r.rhs;
		o.rhsPenetration = r.rhsPenetration;
		o.lowerLimit = r.lowerLimit;
		o.upperLimit = r.upperLimit;
		o.appliedImpulse = r.appliedImpulse;
		o.appliedPushImpulse = r.appliedPushImpulse;
		o.friction = r.friction;
		o.frictionIndex = r.frictionIndex;
		o.isSpecial = r.isSpecial;
	};
	*numContact = s.numContactRows;
	for (int i = 0; i < s.numContactRows; i++)
		conv(s.contactRows[i], contactRows[i]);
	*numFriction = s.numFrictionRows;
	for (int i = 0; i < s.numFrictionRows; i++)
		conv(s.frictionRows[i], frictionRows[i]);
}

void UploadCarConsts(GpuBatch* b, const GpuCarConsts& consts) {
	if (!b->dCarConsts)
		CudaCheck(cudaMalloc(&b->dCarConsts, sizeof(GpuCarConsts)), "alloc car consts");
	CudaCheck(cudaMemcpy(b->dCarConsts, &consts, sizeof(GpuCarConsts), cudaMemcpyHostToDevice), "upload car consts");
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
	const GpuCarConsts* cc, GpuCarTickState* state,
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
	float dbgFwd = 0;
	CuCarPreTick(io, *cc, geo, grid, dyn, numDyn, selfDynIdx, dynBodies, tickTime, W, &dbgFwd);
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
		CudaCheck(cudaMalloc(&dState, sizeof(GpuCarTickState)), "alloc tick state");
		CudaCheck(cudaMalloc(&dDyn, sizeof(GpuDynBody) * 8), "alloc dyn8");
		CudaCheck(cudaMalloc(&dWheels, sizeof(GpuWheelScratchOut) * 4), "alloc wheels out");
	}
	CudaCheck(cudaMemcpy(dState, &in, sizeof(in), cudaMemcpyHostToDevice), "upload tick state");
	CudaCheck(cudaMemcpy(dDyn, dynBodies, sizeof(GpuDynBody) * numDyn, cudaMemcpyHostToDevice), "upload dyn");

	CarPreTickOnceKernel<<<1, 1>>>(b->geo, grid, b->dCarConsts, dState, dDyn,
		numDyn, selfDynIdx, tickTime, dWheels);
	CudaCheck(cudaGetLastError(), "car pretick launch");
	CudaCheck(cudaMemcpy(&out, dState, sizeof(out), cudaMemcpyDeviceToHost), "download tick state");
	CudaCheck(cudaMemcpy(wheelsOut, dWheels, sizeof(GpuWheelScratchOut) * 4, cudaMemcpyDeviceToHost), "download wheels");
}

void UploadWorldGeo(GpuBatch* b,
	const GpuPlane* planes, int numPlanes,
	const GpuMeshInfo* meshes, int numMeshes,
	const GpuVec3* tris, int numTris) {
	CudaCheck(cudaMalloc(&b->dPlanes, sizeof(GpuPlane) * numPlanes), "alloc planes");
	CudaCheck(cudaMalloc(&b->dMeshes, sizeof(GpuMeshInfo) * numMeshes), "alloc meshes");
	CudaCheck(cudaMalloc(&b->dTris, sizeof(GpuVec3) * 3 * numTris), "alloc tris");
	CudaCheck(cudaMemcpy(b->dPlanes, planes, sizeof(GpuPlane) * numPlanes, cudaMemcpyHostToDevice), "upload planes");
	CudaCheck(cudaMemcpy(b->dMeshes, meshes, sizeof(GpuMeshInfo) * numMeshes, cudaMemcpyHostToDevice), "upload meshes");
	CudaCheck(cudaMemcpy(b->dTris, tris, sizeof(GpuVec3) * 3 * numTris, cudaMemcpyHostToDevice), "upload tris");
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
	CudaCheck(cudaMalloc(&dRays, sizeof(GpuVec3) * 2 * numRays), "alloc rays");
	CudaCheck(cudaMalloc(&dRes, sizeof(GpuRayFuzzResult) * numRays), "alloc ray results");
	CudaCheck(cudaMemcpy(dRays, fromTo, sizeof(GpuVec3) * 2 * numRays, cudaMemcpyHostToDevice), "upload rays");
	if (numDyn > 0) {
		CudaCheck(cudaMalloc(&dDyn, sizeof(GpuDynBody) * numDyn), "alloc dyn");
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

void UploadArenaStateAsync(GpuBatch* b, int arenaIdx,
	const GpuBodyState* bodies, const GpuCarGameState* cars, const GpuPadState* pads) {
	CudaCheck(cudaMemcpyAsync(b->dBodies + (size_t)arenaIdx * b->bodiesPerArena, bodies,
		sizeof(GpuBodyState) * b->bodiesPerArena, cudaMemcpyHostToDevice, b->stream),
		"upload bodies async");
	CudaCheck(cudaMemcpyAsync(b->dCars + (size_t)arenaIdx * b->numCars, cars,
		sizeof(GpuCarGameState) * b->numCars, cudaMemcpyHostToDevice, b->stream),
		"upload cars async");
	CudaCheck(cudaMemcpyAsync(b->dPads + (size_t)arenaIdx * b->numPads, pads,
		sizeof(GpuPadState) * b->numPads, cudaMemcpyHostToDevice, b->stream),
		"upload pads async");
}

void UploadBallStateAsync(GpuBatch* b, int arenaIdx, const GpuBallGameState* state) {
	CudaCheck(cudaMemcpyAsync(b->dBallStates + arenaIdx, state, sizeof(GpuBallGameState),
		cudaMemcpyHostToDevice, b->stream), "up ball state async");
}

void UploadBodyConsts(GpuBatch* b, const GpuBodyConsts* consts) {
	CudaCheck(cudaMemcpy(b->dConsts, consts, sizeof(GpuBodyConsts) * b->bodiesPerArena,
		cudaMemcpyHostToDevice), "upload consts");
}

void UploadControlSchedule(GpuBatch* b, const GpuCarControls* schedule, int numTicks) {
	size_t count = (size_t)b->numArenas * b->numCars * numTicks;
	cudaFree(b->dSchedule);
	CudaCheck(cudaMalloc(&b->dSchedule, sizeof(GpuCarControls) * count), "alloc schedule");
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
		CudaCheck(cudaMalloc(&b->dConstControls, sizeof(GpuCarControls) * count),
			"alloc const controls");
	MemcpyOnBatchStream(b, b->dConstControls, perCar, sizeof(GpuCarControls) * count,
		cudaMemcpyHostToDevice, "upload const controls");
}

void UploadConstantControlsAsync(GpuBatch* b, const GpuCarControls* perCar) {
	size_t count = (size_t)b->numArenas * b->numCars;
	if (!b->dConstControls)
		CudaCheck(cudaMalloc(&b->dConstControls, sizeof(GpuCarControls) * count),
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

// Collision-phase buffers/constants, bundled to keep the kernel signature sane.
struct GpuCollideKernelArgs {
	int enabled;
	GpuDynProxy* proxies;            // [numArenas * bodiesPerArena]
	int* regCounters;                // [numArenas]
	GpuBallGameState* ballStates;    // [numArenas]
	const GpuCollideConsts* consts;
	const GpuStaticObj* statics;
	int numStatics;
	const int* cellStaticStart;
	const int* cellStaticList;
	GpuGridParams grid;
	const GpuMeshBvh* bvhs;
	const GpuBvhNode* bvhNodes;
	const GpuBvhSubtree* bvhSubtrees;
	const GpuTriInfo* triInfos;
	const float* meshMargins;
	CuManifold* manifolds;           // [numArenas * GPU_MAX_MANIFOLDS]
	int* numManifolds;               // [numArenas]
	GpuManifoldOut* manifoldsOut;    // [numArenas * GPU_MAX_MANIFOLDS]
	int* debug;                      // [numArenas * COLLIDE_DEBUG_INTS]
	int2* pairs;                     // [numArenas * 96]
	int* numPairs;                   // [numArenas]
	CuSolveCtx* solveScratch;        // [numArenas]
	uint64_t tickBase;
	// game-tail (full-step mode)
	int tailEnabled;
	const GpuTailConsts* tail;
	const GpuPadConfig* padConfigs;
	const GpuDynBody* dynTemplates;  // [bodiesPerArena]
	const int* carOrder;             // [numArenas * numCars]
	const GpuCarConsts* carConsts;
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
};

__device__ static void StoreVec(GpuVec3& dst, CuVec3 v) { dst = { v.x, v.y, v.z }; }

// Convert the device manifolds of one arena into the host download mirror.
__device__ static void EmitManifoldsOut(const CuManifold* src, int n, GpuManifoldOut* dst) {
	for (int i = 0; i < n; i++) {
		const CuManifold& m = src[i];
		GpuManifoldOut& o = dst[i];
		o.bodyA = m.bodyA;
		o.bodyB = m.bodyB;
		o.numContacts = m.numContacts;
		for (int c = 0; c < m.numContacts; c++) {
			const CuManifoldPoint& p = m.pts[c];
			GpuContactOut& po = o.pts[c];
			StoreVec(po.localPointA, p.localPointA);
			StoreVec(po.localPointB, p.localPointB);
			StoreVec(po.positionWorldOnA, p.positionWorldOnA);
			StoreVec(po.positionWorldOnB, p.positionWorldOnB);
			StoreVec(po.normalWorldOnB, p.normalWorldOnB);
			StoreVec(po.lateralFrictionDir1, p.lateralFrictionDir1);
			StoreVec(po.lateralFrictionDir2, p.lateralFrictionDir2);
			po.distance = p.distance1;
			po.combinedFriction = p.combinedFriction;
			po.combinedRestitution = p.combinedRestitution;
			po.partId0 = p.partId0;
			po.partId1 = p.partId1;
			po.index0 = p.index0;
			po.index1 = p.index1;
			po.isSpecial = p.isSpecial;
		}
	}
}

// 2026-06-11: 255 regs/thread + ~21KB stack is measured-optimal here; capping
// registers via __launch_bounds__ raises occupancy but the extra spill traffic
// makes the kernel slower (2026-06-11: 128 / ~170 regs 23-32% slower with one
// arena per block; 2026-09-02: __launch_bounds__(32, 11) = 168 regs fits the
// 2000-arena batch in one wave and cuts the kickoff-scenario bench 1.83 ->
// 1.58 ms, yet loses 1-3% end-to-end where arenas are contact-heavy). Do not
// add launch bounds without an end-to-end A/B; hashes stay identical either
// way (register allocation does not change FP results).
// Per-lane-group shared scratch for the car pre-tick / pad phases, carved
// from dynamic shared memory. The layout is sized by this batch's staged
// bodies, ray lanes per wheel and pads (host and device compute the same
// offsets), so a 1v1 batch does not pay the 8-body / 16-lane worst case:
// shared memory per SM is what the L1 cache has to give up.
struct CuStepGroupLayout {
	int io, dyn, dynBodies, raySrc, rayTgt, rayFrac, rayMesh, rayNrm, padGrant, flags;
	int bytes;
};

__host__ __device__ inline int CuStepPlace(int& field, int offset, size_t bytes) {
	offset = (offset + 15) & ~15;
	field = offset;
	return offset + (int)bytes;
}

// Per-arena shared scratch. Car pre-ticks run in batches of lanes/4 cars
// (one lane per wheel, see the kernel): each car slot has its own staged
// CuCarTickIO; every body is staged once per batch (the wheel ray tests visit
// all of them); the ray arrays hold one entry per (wheel slot, ray lane).
// Wheel scratch lives in the wheel lane's registers/local memory and reaches
// the car's leader lane by warp shuffle, so it costs no shared memory.
__host__ __device__ inline CuStepGroupLayout CuStepGroupSmemLayout(int bodies, int lanes,
	int numPads) {
	const int carsPerBatch = lanes / 4;
	CuStepGroupLayout L;
	int off = 0;
	// the collision phase aliases the IO staging for its per-body static-pair
	// candidate lists ([bodies][1 + CU_STATIC_CAND_MAX] ints)
	size_t ioBytes = sizeof(CuCarTickIO) * carsPerBatch;
	size_t candBytes = sizeof(int) * (1 + CU_STATIC_CAND_MAX) * bodies;
	off = CuStepPlace(L.io, off, ioBytes > candBytes ? ioBytes : candBytes);
	off = CuStepPlace(L.dyn, off, sizeof(GpuDynBody) * bodies);
	off = CuStepPlace(L.dynBodies, off, sizeof(CuBodyDyn) * bodies);
	off = CuStepPlace(L.raySrc, off, sizeof(CuVec3) * 4 * carsPerBatch);
	off = CuStepPlace(L.rayTgt, off, sizeof(CuVec3) * 4 * carsPerBatch);
	off = CuStepPlace(L.rayFrac, off, sizeof(float) * lanes);
	off = CuStepPlace(L.rayMesh, off, sizeof(int) * lanes);
	off = CuStepPlace(L.rayNrm, off, sizeof(CuVec3) * lanes);
	off = CuStepPlace(L.padGrant, off, sizeof(int) * (numPads > 0 ? numPads : 1));
	// [0..cpb) skip flags, [cpb..2cpb) car indices, then dep flag, ok flag
	off = CuStepPlace(L.flags, off, sizeof(int) * (2 * carsPerBatch + 2));
	L.bytes = off;
	return L;
}

__global__ void GpuStepArenasKernel(
	GpuBodyState* bodies, const GpuBodyConsts* consts,
	GpuCarGameState* cars, GpuPadState* pads,
	const GpuCarControls* schedule, int scheduleTicks,
	int numCars, int numPads, int numTicks,
	int startPhase, int endPhase, float dt,
	int lanesPerArena, int numActive, int arenasPerBlock,
	GpuWorldGeo geo, GpuCollideKernelArgs ca) {

	// 2026-09-02: arenasPerBlock arenas share one block as lane groups of lanesPerArena
	// threads (lane 0 of each group runs the order-sensitive serial phases);
	// every arena's computation is unchanged, the packing only raises the
	// number of arenas resident per SM. Barriers stay block-wide and
	// unconditional; an inactive group (past numActive) idles through them.
	const int sub = threadIdx.x / lanesPerArena;
	const int lane = threadIdx.x - sub * lanesPerArena;
	const int slot = blockIdx.x * arenasPerBlock + sub;
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

	GpuBodyState* arenaBodies = bodies + (size_t)arenaIdx * bodiesPerArena;
	GpuCarGameState* arenaCars = cars + (size_t)arenaIdx * numCars;
	GpuPadState* arenaPads = pads + (size_t)arenaIdx * numPads;

	for (int tick = 0; tick < numTicks; tick++) {
		if (profOn && lane == 0)
			profT0 = clock64();
		if (ca.tailEnabled && startPhase <= GPU_PHASE_GAME_PRE && GPU_PHASE_GAME_PRE <= endPhase) {
			// 2026-09-02: car pre-ticks run in batches of carsPerBatch = lanes/4
			// cars, one lane per wheel and one leader lane (wheel 0's) per car
			// for the order-sensitive Car::_PreTickUpdate body. The CPU runs
			// cars sequentially in _cars order and a later car can observe an
			// earlier one only through (a) the demolition block's activation /
			// contact-response writes and (b) wheel friction against a dynamic
			// ground body (the pre-tick never moves a body, and no helper writes
			// the staged body tables). A batch is first attempted speculatively
			// from the committed state: if any car in it is demoed, still
			// carries a respawn no-response flag, is not ACTIVE_TAG, or has a
			// wheel resting on a dynamic body, nothing is committed and the
			// batch is redone one car at a time by the same code (exact CPU
			// order, all lanes on that car's rays). Otherwise the cars' inputs
			// are identical to the sequential ones and their writes are
			// disjoint, so committing them together is bit-identical. Barrier
			// counts are uniform across the block's lane groups: every attempt
			// runs its barriers whether or not its work is predicated off.
			CuCarTickIO* sIoAll = reinterpret_cast<CuCarTickIO*>(sgBase + sgL.io);
			GpuDynBody* sDyn = reinterpret_cast<GpuDynBody*>(sgBase + sgL.dyn);
			CuBodyDyn* sDynBodies = reinterpret_cast<CuBodyDyn*>(sgBase + sgL.dynBodies);
			CuVec3* sRaySrc = reinterpret_cast<CuVec3*>(sgBase + sgL.raySrc);
			CuVec3* sRayTgt = reinterpret_cast<CuVec3*>(sgBase + sgL.rayTgt);
			float* sRayFrac = reinterpret_cast<float*>(sgBase + sgL.rayFrac);
			int* sRayMesh = reinterpret_cast<int*>(sgBase + sgL.rayMesh);
			CuVec3* sRayNrm = reinterpret_cast<CuVec3*>(sgBase + sgL.rayNrm);
			const int carsPerBatch = lanesPerArena >> 2;
			int* sSkip = reinterpret_cast<int*>(sgBase + sgL.flags);
			int* sCi = sSkip + carsPerBatch;
			int& sDep = sSkip[2 * carsPerBatch];
			int& sOk = sSkip[2 * carsPerBatch + 1];
			static_assert(sizeof(CuWheelScratch) % 4 == 0, "wheel scratch shuffles whole words");
			constexpr int kWheelWords = (int)(sizeof(CuWheelScratch) / 4);

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
				// attempt 0: speculative batch of nb cars; attempts 1..nb: exact
				// single-car passes, predicated off when attempt 0 committed
				for (int attempt = 0; attempt <= nb; attempt++) {
					const bool exact = attempt > 0;
					const int carsHere = exact ? 1 : nb;
					const int first = exact ? oc + attempt - 1 : oc;
					const int raysPerWheel = lanesPerArena / (4 * carsHere);
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
										ca.rngStates[arenaIdx], consts[1 + ci], *ca.tail);
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
							CuCarPreTickWheelPre(sIoAll[slot], *ca.carConsts, wheel, w,
								sRaySrc[lane], sRayTgt[lane]);
					}
					__syncthreads();

					// phase 3: static-mesh ray test, raysPerWheel lanes per wheel
					// slot; lane 0 of each wheel later lex-min-reduces (fraction,
					// mesh index), identical to the sequential strict-< /
					// first-visited tie-break for any lane count
					{
						const int wheelSlot = lane / raysPerWheel;
						const int rayLane = lane - wheelSlot * raysPerWheel;
						if (work && !dep1 && !sSkip[wheelSlot >> 2]) {
							float bf = 1.0f;
							int bm = 0x7fffffff;
							CuVec3 bn = { 0, 0, 0 };
							CuRayTestStaticMeshes(geo, sRaySrc[wheelSlot], sRayTgt[wheelSlot],
								ca.carConsts->rayCastGroup, CU_FILTER_ALL, rayLane, raysPerWheel, bf, bn, bm);
							sRayFrac[wheelSlot * raysPerWheel + rayLane] = bf;
							sRayMesh[wheelSlot * raysPerWheel + rayLane] = bm;
							sRayNrm[wheelSlot * raysPerWheel + rayLane] = bn;
						}
					}
					__syncthreads();
					subLap(1);

					// phase 4 (wheel lanes): reduce, planes, dynamic bodies,
					// suspension, friction; a wheel resting on a dynamic body
					// makes the speculative batch order-dependent
					if (work && !dep1 && slot < carsHere && !sSkip[slot]) {
						GpuRayHit hit;
						hit.fraction = 1.0f;
						hit.hitKind = -1;
						hit.normalWorld = { 0, 0, 0 };
						int bestMesh = 0x7fffffff;
						for (int l = 0; l < raysPerWheel; l++) {
							float f = sRayFrac[lane * raysPerWheel + l];
							int m = sRayMesh[lane * raysPerWheel + l];
							if (f < hit.fraction || (f == hit.fraction && m < bestMesh)) {
								hit.fraction = f;
								bestMesh = m;
								hit.normalWorld = sRayNrm[lane * raysPerWheel + l];
							}
						}
						if (bestMesh != 0x7fffffff)
							hit.hitKind = 0;
						CuCarPreTickWheelPost(sIoAll[slot], *ca.carConsts, geo, ca.grid, sDyn,
							bodiesPerArena, 1 + sCi[slot], sDynBodies, wheel,
							sRaySrc[lane], sRayTgt[lane], hit, w);
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

					// phase 5: leaders gather their four wheels' scratch by warp
					// shuffle (uniform: every lane shuffles), run the car body
					// and commit chassis + game state + wheels
					CuWheelScratch W[4];
					{
						const unsigned* src = reinterpret_cast<const unsigned*>(&w);
						const int base = (threadIdx.x & 31) & ~3;
						for (int i = 0; i < 4; i++) {
							unsigned* dst = reinterpret_cast<unsigned*>(&W[i]);
							for (int j = 0; j < kWheelWords; j++)
								dst[j] = __shfl_sync(0xffffffffu, src[j], base + i);
						}
					}
					const bool commit = work && sDep == 0;
					if (commit && isLeader && !sSkip[slot]) {
						const int ci = sCi[slot];
						CuCarTickIO& sIo = sIoAll[slot];
						float dbgFwd;
						CuCarPreTick(sIo, *ca.carConsts, geo, ca.grid, sDyn, bodiesPerArena,
							1 + ci, sDynBodies, dt, W, &dbgFwd, /*wheelsPrecomputed=*/true);
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
			// pad pre-ticks: independent per pad, strided over the lane group
			if (active)
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
			// stepSimulation head: applyGravity (active bodies). After the ball
			// pre-tick: the kickoff launch sets ACTIVE_TAG, which gravity reads.
			__syncthreads();
			if (active && lane < bodiesPerArena)
				CuApplyGravity(arenaBodies[lane], consts[lane]);
			__syncthreads();
			subLap(4);
		}
		__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_GAME_PRE] += t - profT0; profT0 = t; }

		if (startPhase <= GPU_PHASE_PREDICT && GPU_PHASE_PREDICT <= endPhase) {
			if (active && lane < bodiesPerArena)
				PhasePredict(arenaBodies[lane], consts[lane]);
		}
		__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_PREDICT] += t - profT0; profT0 = t; }

		if (ca.enabled && startPhase <= GPU_PHASE_COLLISION && GPU_PHASE_COLLISION <= endPhase) {
			// 2026-09-02: the aabb refresh and the static-pair candidate scan run
			// one lane per body (independent per body); regSeq - the arena-wide
			// registration counter the CPU increments in body order - is
			// assigned by rank among the bodies that changed cell (ballot), the
			// same sequence. Thread 0 then merges candidates in body order and
			// runs the rest of the collision tick. The candidate lists alias the
			// car pre-tick's IO staging, which is dead in this phase.
			int* staticCand = reinterpret_cast<int*>(sgBase + sgL.io);
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
				cin.cellStaticStart = ca.cellStaticStart;
				cin.cellStaticList = ca.cellStaticList;
				cin.grid = ca.grid;
				cin.geo = geo;
				cin.bvhs = ca.bvhs;
				cin.bvhNodes = ca.bvhNodes;
				cin.bvhSubtrees = ca.bvhSubtrees;
				cin.triInfos = ca.triInfos;
				cin.meshMargins = ca.meshMargins;
				cin.cc = *ca.consts;
				cin.tickCount = ca.tickBase + tick;
				cin.dt = dt;
				cin.manifolds = ca.manifolds + (size_t)arenaIdx * GPU_MAX_MANIFOLDS;
				cin.numManifolds = ca.numManifolds + arenaIdx;
				cin.pairsOut = ca.pairs + (size_t)arenaIdx * 96;
				cin.numPairsOut = ca.numPairs + arenaIdx;
				cin.debug = ca.debug ? ca.debug + (size_t)arenaIdx * (2 * 96 + 1) : nullptr;
				cin.tileDamage = ca.tileDamage
					? ca.tileDamage + (size_t)arenaIdx * ca.numTiles : nullptr;
				cin.tileNoResponse = ca.tileNoResponse
					? ca.tileNoResponse + (size_t)arenaIdx * ca.numTiles : nullptr;
#ifdef RS_GPU_PHASE_PROF
				cin.prof = profOn ? ca.profCycles + GPU_PHASE_COUNT : nullptr;
#endif
				const bool body = active && lane < bodiesPerArena;
				const int regBase = body ? *cin.regCounter : 0;
				const bool moved = body && CuCollideUpdateAabbBody(cin, lane);
				const unsigned ballot = __ballot_sync(0xffffffffu, moved);
				const int groupBase = (sub * lanesPerArena) & 31;
				const unsigned groupBits = (ballot >> groupBase) &
					(lanesPerArena >= 32 ? 0xffffffffu : ((1u << lanesPerArena) - 1u));
				if (moved)
					cin.proxies[lane].regSeq = regBase + __popc(groupBits & ((1u << lane) - 1u)) + 1;
				if (body && lane == 0 && groupBits)
					*cin.regCounter = regBase + __popc(groupBits);
				__syncthreads();
				if (body)
					CuCollideStaticCandidates(cin, lane, staticCand + lane * (1 + CU_STATIC_CAND_MAX));
				__syncthreads();
				if (active && lane == 0) {
					CuCollideTick(cin, staticCand);
					EmitManifoldsOut(cin.manifolds, *cin.numManifolds,
						ca.manifoldsOut + (size_t)arenaIdx * GPU_MAX_MANIFOLDS);
				}
			}
		} else {
			__syncthreads();
			__syncthreads();
		}
		__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_COLLISION] += t - profT0; profT0 = t; }

		if (ca.enabled && startPhase <= GPU_PHASE_SOLVE && GPU_PHASE_SOLVE <= endPhase) {
			if (active && lane == 0) {
				CuSolveTick(arenaBodies, consts, arenaCars, bodiesPerArena, ca.numStatics,
					ca.manifolds + (size_t)arenaIdx * GPU_MAX_MANIFOLDS,
					ca.numManifolds[arenaIdx],
					ca.pairs + (size_t)arenaIdx * 96,
					ca.numPairs[arenaIdx], dt, ca.solveScratch + arenaIdx,
					ca.statics,
					ca.tileNoResponse
						? ca.tileNoResponse + (size_t)arenaIdx * ca.numTiles : nullptr);
			}
		}
		__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_SOLVE] += t - profT0; profT0 = t; }

		if (startPhase <= GPU_PHASE_INTEGRATE && GPU_PHASE_INTEGRATE <= endPhase) {
			if (active && lane < bodiesPerArena)
				PhaseIntegrate(arenaBodies[lane], consts[lane], dt);
		}
		__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_INTEGRATE] += t - profT0; profT0 = t; }

		if (startPhase <= GPU_PHASE_ACTIVATION && GPU_PHASE_ACTIVATION <= endPhase) {
			if (active && lane < bodiesPerArena)
				PhaseActivation(arenaBodies[lane], consts[lane], dt);
		}
		__syncthreads();
		if (profOn && lane == 0) { long long t = clock64(); profAcc[GPU_PHASE_ACTIVATION] += t - profT0; profT0 = t; }

		if (ca.tailEnabled && startPhase <= GPU_PHASE_GAME_POST && GPU_PHASE_GAME_POST <= endPhase) {
			// 2026-06-11: clearForces and the car post-tick/finish pair touch only
			// their own body/car state (post-tick's body param is even const), so
			// they run one-thread-per-body. Only the boost pad grid checks contend
			// on shared pad state - those stay sequential in _cars order below.
			// Reordering all post/finish before all grid checks cannot change
			// results: on CPU, car K's post/finish never reads pads or another
			// car, and grid checks already ran after their own car's post/finish.
			{
				uint64_t tickCount = ca.tickBase + tick;
				// stepSimulation tail: clearForces (all nonstatic bodies)
				if (active && lane < bodiesPerArena)
					CuClearForces(arenaBodies[lane]);
				// car post-tick + finish, one car per lane (carOrder is a
				// permutation; per-car work is order-independent here)
				if (active && lane < numCars) {
					int ci = lane;
					GpuCarControls ctrl = ca.constControls
						? ca.constControls[(size_t)arenaIdx * numCars + ci]
						: schedule[((size_t)arenaIdx * numCars + ci) * (size_t)scheduleTicks + (size_t)tickCount];
					CuCarPostTick(arenaBodies[1 + ci], arenaCars[ci], *ca.tail, ctrl, dt);
					CuCarFinish(arenaBodies[1 + ci], arenaCars[ci], *ca.tail);
				}
			}
			__syncthreads();
			if (active && lane == 0) {
				uint64_t tickCount = ca.tickBase + tick;
				// boost pad grid checks, in _cars order (shared pad locks)
				for (int oc = 0; oc < numCars; oc++) {
					int ci = ca.carOrder[(size_t)arenaIdx * numCars + oc];
					CuPadGridCheck(arenaBodies[1 + ci], arenaCars[ci], ci,
						arenaPads, ca.padConfigs, *ca.tail, *ca.consts);
				}
			}
			__syncthreads();
			// pad post-ticks: per-pad phase in parallel, boost grants in pad order
			// (see CuPadPostTickPad; soccar has the most pads at 34 < 64)
			int* sPadGrant = reinterpret_cast<int*>(sgBase + sgL.padGrant);
			if (active)
				for (int p = lane; p < numPads; p += lanesPerArena)
					sPadGrant[p] = CuPadPostTickPad(arenaPads[p], ca.padConfigs[p],
						arenaCars, *ca.tail);
			__syncthreads();
			if (active && lane == 0) {
				uint64_t tickCount = ca.tickBase + tick;
				for (int p = 0; p < numPads; p++)
					if (sPadGrant[p] >= 0)
						CuPadGrantBoost(ca.padConfigs[p], arenaCars[sPadGrant[p]], *ca.tail);
				// ball finish (goal-score callback is null in the bench)
				CuBallFinish(arenaBodies[0], ca.ballStates[arenaIdx], *ca.tail);
				// dropshot: SetDropshotTilesState after a damage tick (deferred so the
				// ball doesn't fall through a tile broken THIS tick)
				if (ca.consts->gameMode == CU_GAMEMODE_DROPSHOT &&
					ca.ballStates[arenaIdx].dsHasDamaged &&
					ca.ballStates[arenaIdx].dsLastDamageTick == tickCount) {
					uint8_t* dmg = ca.tileDamage + (size_t)arenaIdx * ca.numTiles;
					uint8_t* nr = ca.tileNoResponse + (size_t)arenaIdx * ca.numTiles;
					for (int t = 0; t < ca.numTiles; t++)
						nr[t] = dmg[t] == GPU_DS_TILE_BROKEN ? 1 : 0;
				}
				// tickCount++ handled by the tick loop / host counter
			}
		}
		__syncthreads();
		if (profOn && lane == 0)
			profAcc[GPU_PHASE_GAME_POST] += clock64() - profT0;
	}

	if (profOn && lane == 0 && active)
		for (int p = 0; p < GPU_PHASE_COUNT; p++)
			atomicAdd(&ca.profCycles[p], profAcc[p]);
}

static void Launch(GpuBatch* b, int numTicks, int startPhase, int endPhase,
	bool sync = true, const int* dArenaList = nullptr, int listCount = 0,
	long long tickBaseOverride = -1) {
	constexpr float TICK_TIME = 1.0f / 120.0f;
	// 2026-09-02: lane group per arena = 16 threads (two arenas per 32-thread
	// block), widened to the next power of two when an arena has more bodies.
	// Measured on an RTX 5070 at 2000-4000 1v1 arenas: 16 lanes beats 8 and
	// 4 (their extra residency is eaten by the longer per-lane mesh walks)
	// and cuts the 7-tick chunk 42% versus one arena per block, with every
	// per-step state hash unchanged (gpusim_validate).
	int lanes = 8;
	while (lanes < b->bodiesPerArena)
		lanes *= 2;
	int threads = lanes > 32 ? 64 : 32;
	int apb = threads / lanes;
	GpuCollideKernelArgs ca = {};
	ca.enabled = b->hasCollideData ? 1 : 0;
	if (b->hasCollideData) {
		ca.proxies = b->dDynProxies;
		ca.regCounters = b->dRegCounters;
		ca.ballStates = b->dBallStates;
		ca.consts = b->dCollideConsts;
		ca.statics = b->dStatics;
		ca.numStatics = b->numStatics;
		ca.cellStaticStart = b->dCellStaticStart;
		ca.cellStaticList = b->dCellStaticList;
		ca.grid = b->grid;
		ca.bvhs = b->dMeshBvhs;
		ca.bvhNodes = b->dBvhNodes;
		ca.bvhSubtrees = b->dBvhSubtrees;
		ca.triInfos = b->dTriInfos;
		ca.meshMargins = b->dMeshMargins;
		ca.manifolds = b->dManifolds;
		ca.numManifolds = b->dNumManifolds;
		ca.manifoldsOut = b->dManifoldsOut;
		ca.debug = b->dCollideDebug;
		ca.pairs = b->dPairs;
		ca.numPairs = b->dNumPairs;
		ca.solveScratch = b->dSolveScratch;
		ca.tickBase = b->tickCount;
		ca.tailEnabled = b->hasTailData ? 1 : 0;
		ca.tail = b->dTailConsts;
		ca.padConfigs = b->dPadConfigs;
		ca.dynTemplates = b->dDynTemplates;
		ca.carOrder = b->dCarOrder;
		ca.carConsts = b->dCarConsts;
		ca.rngStates = b->dRngStates;
		ca.tileDamage = b->dTileDamage;
		ca.tileNoResponse = b->dTileNoResponse;
		ca.numTiles = b->numTiles;
		ca.constControls = b->dConstControls;
		ca.profCycles = b->dProfCycles;
		ca.arenaList = dArenaList;
		if (tickBaseOverride >= 0)
			ca.tickBase = (uint64_t)tickBaseOverride;
	}
	int count = dArenaList ? listCount : b->numArenas;
	int grid = (count + apb - 1) / apb;
	size_t smemBytes = (size_t)CuStepGroupSmemLayout(b->bodiesPerArena, lanes,
		b->numPads).bytes * (size_t)apb;
	GpuStepArenasKernel<<<grid, threads, smemBytes, b->stream>>>(
		b->dBodies, b->dConsts, b->dCars, b->dPads, b->dSchedule, b->scheduleTicks,
		b->numCars, b->numPads, numTicks, startPhase, endPhase, TICK_TIME,
		lanes, count, apb, b->geo, ca);
	CudaCheck(cudaGetLastError(), "kernel launch");
	if (sync)
		CudaCheck(cudaStreamSynchronize(b->stream), "kernel run");
}

void StepTicks(GpuBatch* b, int numTicks) {
	Launch(b, numTicks, 0, GPU_PHASE_COUNT - 1);
	b->tickCount += numTicks;
}

// 2026-06-11 async stepping: launch without the device sync so the chunk can
// overlap CPU work; SyncBatch() completes it. StepTicksSubset re-steps ONLY the
// listed arenas over the same tick window (explicit tickBase) - used when a
// reset invalidates an arena's async chunk; bit-identical to its slice of a
// full chunk because arenas are independent.
void StepTicksAsync(GpuBatch* b, int numTicks) {
	Launch(b, numTicks, 0, GPU_PHASE_COUNT - 1, /*sync=*/false);
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
		CudaCheck(cudaMalloc(&b->dArenaList, sizeof(int) * b->numArenas), "list alloc");
	MemcpyOnBatchStream(b, b->dArenaList, arenaIdxs, sizeof(int) * count,
		cudaMemcpyHostToDevice, "list up");
	Launch(b, numTicks, 0, GPU_PHASE_COUNT - 1, /*sync=*/true,
		b->dArenaList, count, (long long)tickBase);
	// tickCount not advanced: this re-covers a window already counted
}

void StepTicksSubsetAsync(GpuBatch* b, int numTicks, const int* arenaIdxs, int count,
	unsigned long long tickBase) {
	if (count <= 0)
		return;
	if (!b->dArenaList)
		CudaCheck(cudaMalloc(&b->dArenaList, sizeof(int) * b->numArenas), "list alloc");
	CudaCheck(cudaMemcpyAsync(b->dArenaList, arenaIdxs, sizeof(int) * count,
		cudaMemcpyHostToDevice, b->stream), "list up async");
	Launch(b, numTicks, 0, GPU_PHASE_COUNT - 1, /*sync=*/false,
		b->dArenaList, count, (long long)tickBase);
}

// 2026-06-11: per-phase cycle profiling (diagnostic; ~zero cost while disabled)
void EnablePhaseProfiling(GpuBatch* b) {
	if (b->dProfCycles)
		return;
#ifndef RS_GPU_PHASE_PROF
	fprintf(stderr, "EnablePhaseProfiling: built without RS_GPU_PHASE_PROF, phase cycles will read 0\n");
#endif
	CudaCheck(cudaMalloc(&b->dProfCycles,
		sizeof(unsigned long long) * (GPU_PHASE_PROF_SLOTS)), "prof alloc");
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

void DownloadAllArenaStatesAsync(GpuBatch* b,
	GpuBodyState* bodies, GpuCarGameState* cars, GpuPadState* pads) {
	CudaCheck(cudaMemcpyAsync(bodies, b->dBodies,
		sizeof(GpuBodyState) * b->numArenas * b->bodiesPerArena,
		cudaMemcpyDeviceToHost, b->stream), "down all bodies async");
	CudaCheck(cudaMemcpyAsync(cars, b->dCars,
		sizeof(GpuCarGameState) * b->numArenas * b->numCars,
		cudaMemcpyDeviceToHost, b->stream), "down all cars async");
	if (b->numPads > 0)
		CudaCheck(cudaMemcpyAsync(pads, b->dPads,
			sizeof(GpuPadState) * b->numArenas * b->numPads,
			cudaMemcpyDeviceToHost, b->stream), "down all pads async");
}

void DownloadAllBallStatesAsync(GpuBatch* b, GpuBallGameState* balls) {
	CudaCheck(cudaMemcpyAsync(balls, b->dBallStates,
		sizeof(GpuBallGameState) * b->numArenas, cudaMemcpyDeviceToHost, b->stream),
		"down all balls async");
}

void RecordDownloadEvent(GpuBatch* b) {
	CudaCheck(cudaEventRecord(b->downloadEvent, b->stream), "record download event");
}

void WaitDownloadEvent(GpuBatch* b) {
	CudaCheck(cudaEventSynchronize(b->downloadEvent), "wait download event");
}

void RunPhaseSegment(GpuBatch* b, int startPhase, int endPhase) {
	Launch(b, 1, startPhase, endPhase);
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
