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

namespace rscuda {

static void CudaCheck(cudaError_t err, const char* what) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
		exit(2);
	}
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
	// 1KB device stack overflows (proven in cuda_math_test).
	CudaCheck(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024), "set stack size");
	GpuBatch* b = new GpuBatch{};
	b->numArenas = numArenas;
	b->numCars = numCarsPerArena;
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
	return b;
}

void DestroyBatch(GpuBatch* b) {
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
	delete b;
}

void UploadMeshBvhs(GpuBatch* b,
	const GpuMeshBvh* bvhs, int numMeshes,
	const GpuBvhNode* nodes, int numNodes,
	const GpuBvhSubtree* subtrees, int numSubtrees,
	const GpuTriInfo* triInfos, int numTris,
	const float* meshMargins) {
	CudaCheck(cudaMalloc(&b->dMeshBvhs, sizeof(GpuMeshBvh) * numMeshes), "alloc bvhs");
	CudaCheck(cudaMalloc(&b->dBvhNodes, sizeof(GpuBvhNode) * numNodes), "alloc bvh nodes");
	CudaCheck(cudaMalloc(&b->dBvhSubtrees, sizeof(GpuBvhSubtree) * numSubtrees), "alloc bvh subtrees");
	CudaCheck(cudaMalloc(&b->dTriInfos, sizeof(GpuTriInfo) * numTris), "alloc tri infos");
	CudaCheck(cudaMalloc(&b->dMeshMargins, sizeof(float) * numMeshes), "alloc mesh margins");
	CudaCheck(cudaMemcpy(b->dMeshBvhs, bvhs, sizeof(GpuMeshBvh) * numMeshes, cudaMemcpyHostToDevice), "up bvhs");
	CudaCheck(cudaMemcpy(b->dBvhNodes, nodes, sizeof(GpuBvhNode) * numNodes, cudaMemcpyHostToDevice), "up bvh nodes");
	CudaCheck(cudaMemcpy(b->dBvhSubtrees, subtrees, sizeof(GpuBvhSubtree) * numSubtrees, cudaMemcpyHostToDevice), "up bvh subtrees");
	CudaCheck(cudaMemcpy(b->dTriInfos, triInfos, sizeof(GpuTriInfo) * numTris, cudaMemcpyHostToDevice), "up tri infos");
	CudaCheck(cudaMemcpy(b->dMeshMargins, meshMargins, sizeof(float) * numMeshes, cudaMemcpyHostToDevice), "up mesh margins");
	// the ray tests reach the BVHs through the world-geo descriptor
	b->geo.bvhs = b->dMeshBvhs;
	b->geo.bvhNodes = b->dBvhNodes;
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
	CudaCheck(cudaMemcpy(b->dBallStates + arenaIdx, &state, sizeof(GpuBallGameState),
		cudaMemcpyHostToDevice), "up ball state");
}

void DownloadBallState(GpuBatch* b, int arenaIdx, GpuBallGameState* state) {
	CudaCheck(cudaMemcpy(state, b->dBallStates + arenaIdx, sizeof(GpuBallGameState),
		cudaMemcpyDeviceToHost), "down ball state");
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
	CudaCheck(cudaMemcpy(b->dBodies + (size_t)arenaIdx * b->bodiesPerArena, bodies,
		sizeof(GpuBodyState) * b->bodiesPerArena, cudaMemcpyHostToDevice), "upload bodies");
	CudaCheck(cudaMemcpy(b->dCars + (size_t)arenaIdx * b->numCars, cars,
		sizeof(GpuCarGameState) * b->numCars, cudaMemcpyHostToDevice), "upload cars");
	CudaCheck(cudaMemcpy(b->dPads + (size_t)arenaIdx * b->numPads, pads,
		sizeof(GpuPadState) * b->numPads, cudaMemcpyHostToDevice), "upload pads");
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

__global__ void GpuStepArenasKernel(
	GpuBodyState* bodies, const GpuBodyConsts* consts,
	GpuCarGameState* cars, GpuPadState* pads,
	const GpuCarControls* schedule, int scheduleTicks,
	int numCars, int numPads, int numTicks,
	int startPhase, int endPhase, float dt,
	GpuWorldGeo geo, GpuCollideKernelArgs ca) {

	const int arenaIdx = blockIdx.x;
	const int bodiesPerArena = 1 + numCars;
	const int bodyIdx = threadIdx.x;

	GpuBodyState* arenaBodies = bodies + (size_t)arenaIdx * bodiesPerArena;
	GpuCarGameState* arenaCars = cars + (size_t)arenaIdx * numCars;
	GpuPadState* arenaPads = pads + (size_t)arenaIdx * numPads;

	for (int tick = 0; tick < numTicks; tick++) {
		if (ca.tailEnabled && startPhase <= GPU_PHASE_GAME_PRE && GPU_PHASE_GAME_PRE <= endPhase) {
			if (bodyIdx == 0) {
				uint64_t tickCount = ca.tickBase + tick;
				// Arena::Step head: ball zero-vel sleeping
				CuBallSleepCheck(arenaBodies[0]);
				// car pre-ticks, in the CPU's _cars iteration order; each car sees
				// the previous cars' post-pre-tick state (live body reads)
				for (int oc = 0; oc < numCars; oc++) {
					int ci = ca.carOrder[(size_t)arenaIdx * numCars + oc];
					GpuCarGameState& cst = arenaCars[ci];
					// bench sets car->controls from the schedule before each tick
					GpuCarControls ctrl = schedule[
						((size_t)arenaIdx * numCars + ci) * (size_t)scheduleTicks + (size_t)tickCount];
					CuCarTickIO io;
					{
						const GpuBodyState& s = arenaBodies[1 + ci];
						io.body.basis = CuLoadBasis(s.basis);
						io.body.origin = GV(s.pos);
						io.body.linVel = GV(s.linVel);
						io.body.angVel = GV(s.angVel);
						io.body.invInertiaWorld = CuLoadBasis(s.invInertiaWorld);
						io.body.invInertiaDiagLocal = GV(consts[1 + ci].invInertiaLocal);
						io.body.invMass = consts[1 + ci].invMass;
						io.totalForce = GV(s.totalForce);
						io.totalTorque = GV(s.totalTorque);
					}
					io.st = cst;
					for (int w = 0; w < 4; w++)
						io.wheels[w] = cst.wheels[w];
					io.controls = ctrl;
					// solverInfo.m_timeStep: 1/60 ctor default until the first
					// bullet step assigns the tick time
					io.solverTimeStep = tickCount == 0 ? (1.0f / 60.0f) : dt;

					// live dyn-body table (templates + current state + proxy aabbMin)
					GpuDynBody dyn[8];
					CuBodyDyn dynBodies[8];
					for (int d = 0; d < bodiesPerArena && d < 8; d++) {
						dyn[d] = ca.dynTemplates[d];
						const GpuBodyState& s = arenaBodies[d];
						dyn[d].basis = s.basis;
						dyn[d].origin = s.pos;
						dyn[d].linVel = s.linVel;
						dyn[d].angVel = s.angVel;
						dyn[d].invInertiaWorld = s.invInertiaWorld;
						dyn[d].aabbMin = ca.proxies[(size_t)arenaIdx * bodiesPerArena + d].aabbMin;
						dynBodies[d] = DynToBody(dyn[d]);
					}
					CuWheelScratch W[4];
					float dbgFwd;
					CuCarPreTick(io, *ca.carConsts, geo, ca.grid, dyn, bodiesPerArena,
						1 + ci, dynBodies, dt, W, &dbgFwd);
					// write back chassis + game state + wheels
					{
						GpuBodyState& s = arenaBodies[1 + ci];
						for (int r = 0; r < 3; r++)
							s.basis.row[r] = { io.body.basis.r[r].x, io.body.basis.r[r].y, io.body.basis.r[r].z };
						s.pos = { io.body.origin.x, io.body.origin.y, io.body.origin.z };
						s.linVel = { io.body.linVel.x, io.body.linVel.y, io.body.linVel.z };
						s.angVel = { io.body.angVel.x, io.body.angVel.y, io.body.angVel.z };
						s.totalForce = { io.totalForce.x, io.totalForce.y, io.totalForce.z };
						s.totalTorque = { io.totalTorque.x, io.totalTorque.y, io.totalTorque.z };
					}
					cst = io.st;
					for (int w = 0; w < 4; w++)
						cst.wheels[w] = io.wheels[w];
					// friction impulses on dynamic ground bodies (ball / other cars)
					for (int d = 0; d < bodiesPerArena && d < 8; d++) {
						if (d == 1 + ci)
							continue;
						GpuBodyState& s = arenaBodies[d];
						s.linVel = { dynBodies[d].linVel.x, dynBodies[d].linVel.y, dynBodies[d].linVel.z };
						s.angVel = { dynBodies[d].angVel.x, dynBodies[d].angVel.y, dynBodies[d].angVel.z };
					}
				}
				// pad pre-ticks
				for (int p = 0; p < numPads; p++)
					CuPadPreTick(arenaPads[p], dt);
				// Ball::_PreTickUpdate (heatseeker steering; no-op in soccar)
				if (ca.consts->gameMode == CU_GAMEMODE_HEATSEEKER)
					CuBallPreTickHS(arenaBodies[0], ca.ballStates[arenaIdx], *ca.consts, dt);
				// stepSimulation head: applyGravity (active bodies)
				for (int d = 0; d < bodiesPerArena; d++)
					CuApplyGravity(arenaBodies[d], consts[d]);
			}
		}
		__syncthreads();

		if (startPhase <= GPU_PHASE_PREDICT && GPU_PHASE_PREDICT <= endPhase) {
			if (bodyIdx < bodiesPerArena)
				PhasePredict(arenaBodies[bodyIdx], consts[bodyIdx]);
		}
		__syncthreads();

		if (ca.enabled && startPhase <= GPU_PHASE_COLLISION && GPU_PHASE_COLLISION <= endPhase) {
			if (bodyIdx == 0) {
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
				CuCollideTick(cin);
				EmitManifoldsOut(cin.manifolds, *cin.numManifolds,
					ca.manifoldsOut + (size_t)arenaIdx * GPU_MAX_MANIFOLDS);
			}
		}
		__syncthreads();

		if (ca.enabled && startPhase <= GPU_PHASE_SOLVE && GPU_PHASE_SOLVE <= endPhase) {
			if (bodyIdx == 0) {
				CuSolveTick(arenaBodies, consts, arenaCars, bodiesPerArena, ca.numStatics,
					ca.manifolds + (size_t)arenaIdx * GPU_MAX_MANIFOLDS,
					ca.numManifolds[arenaIdx],
					ca.pairs + (size_t)arenaIdx * 96,
					ca.numPairs[arenaIdx], dt, ca.solveScratch + arenaIdx);
			}
		}
		__syncthreads();

		if (startPhase <= GPU_PHASE_INTEGRATE && GPU_PHASE_INTEGRATE <= endPhase) {
			if (bodyIdx < bodiesPerArena)
				PhaseIntegrate(arenaBodies[bodyIdx], consts[bodyIdx], dt);
		}
		__syncthreads();

		if (startPhase <= GPU_PHASE_ACTIVATION && GPU_PHASE_ACTIVATION <= endPhase) {
			if (bodyIdx < bodiesPerArena)
				PhaseActivation(arenaBodies[bodyIdx], consts[bodyIdx], dt);
		}
		__syncthreads();

		if (ca.tailEnabled && startPhase <= GPU_PHASE_GAME_POST && GPU_PHASE_GAME_POST <= endPhase) {
			if (bodyIdx == 0) {
				uint64_t tickCount = ca.tickBase + tick;
				// stepSimulation tail: clearForces (all nonstatic bodies)
				for (int d = 0; d < bodiesPerArena; d++)
					CuClearForces(arenaBodies[d]);
				// car post-ticks + finish + boost pad grid, in _cars order
				for (int oc = 0; oc < numCars; oc++) {
					int ci = ca.carOrder[(size_t)arenaIdx * numCars + oc];
					GpuCarControls ctrl = schedule[
						((size_t)arenaIdx * numCars + ci) * (size_t)scheduleTicks + (size_t)tickCount];
					CuCarPostTick(arenaBodies[1 + ci], arenaCars[ci], *ca.tail, ctrl, dt);
					CuCarFinish(arenaBodies[1 + ci], arenaCars[ci], *ca.tail);
					CuPadGridCheck(arenaBodies[1 + ci], arenaCars[ci], ci,
						arenaPads, ca.padConfigs, *ca.tail, *ca.consts);
				}
				// pad post-ticks
				for (int p = 0; p < numPads; p++)
					CuPadPostTick(arenaPads[p], ca.padConfigs[p], arenaCars, *ca.tail);
				// ball finish (goal-score callback is null in the bench)
				CuBallFinish(arenaBodies[0], ca.ballStates[arenaIdx], *ca.tail);
				// tickCount++ handled by the tick loop / host counter
			}
		}
		__syncthreads();
	}
}

static void Launch(GpuBatch* b, int numTicks, int startPhase, int endPhase) {
	constexpr float TICK_TIME = 1.0f / 120.0f;
	int threads = b->bodiesPerArena > 32 ? 64 : 32;
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
	}
	GpuStepArenasKernel<<<b->numArenas, threads>>>(
		b->dBodies, b->dConsts, b->dCars, b->dPads, b->dSchedule, b->scheduleTicks,
		b->numCars, b->numPads, numTicks, startPhase, endPhase, TICK_TIME,
		b->geo, ca);
	CudaCheck(cudaGetLastError(), "kernel launch");
	CudaCheck(cudaDeviceSynchronize(), "kernel run");
}

void StepTicks(GpuBatch* b, int numTicks) {
	Launch(b, numTicks, 0, GPU_PHASE_COUNT - 1);
	b->tickCount += numTicks;
}

void RunPhaseSegment(GpuBatch* b, int startPhase, int endPhase) {
	Launch(b, 1, startPhase, endPhase);
}

void DownloadArenaState(GpuBatch* b, int arenaIdx,
	GpuBodyState* bodies, GpuCarGameState* cars, GpuPadState* pads) {
	CudaCheck(cudaMemcpy(bodies, b->dBodies + (size_t)arenaIdx * b->bodiesPerArena,
		sizeof(GpuBodyState) * b->bodiesPerArena, cudaMemcpyDeviceToHost), "download bodies");
	CudaCheck(cudaMemcpy(cars, b->dCars + (size_t)arenaIdx * b->numCars,
		sizeof(GpuCarGameState) * b->numCars, cudaMemcpyDeviceToHost), "download cars");
	CudaCheck(cudaMemcpy(pads, b->dPads + (size_t)arenaIdx * b->numPads,
		sizeof(GpuPadState) * b->numPads, cudaMemcpyDeviceToHost), "download pads");
}

} // namespace rscuda
