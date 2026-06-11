#pragma once
#include "GpuTypes.h"

// 2026-06-10: Host API for the monolithic GPU step. CUDA-free header (pimpl) so it can
// be included by code that builds without the toolkit; implementation lives in CudaStep.cu
// and is only compiled when CMake found CUDA (RS_CUDA=1).

namespace rscuda {

// True if a CUDA device is present and usable (always false in non-CUDA builds).
bool Available();

// One batch = N arenas with identical car counts, stepped together by one kernel launch.
struct GpuBatch;

GpuBatch* CreateBatch(int numArenas, int numCarsPerArena, int numPads);
void DestroyBatch(GpuBatch* batch);

// Upload full state for one arena slot. bodies: [ball, car0, car1, ...] (1 + numCars).
void UploadArenaState(GpuBatch* batch, int arenaIdx,
	const GpuBodyState* bodies, const GpuCarGameState* cars, const GpuPadState* pads);

// Per-body-slot constants (shared by all arenas), uploaded once after CreateBatch
void UploadBodyConsts(GpuBatch* batch, const GpuBodyConsts* consts);

// Static world geometry (identical for all arenas), uploaded once. Tris are 3 verts
// per triangle, mesh-relative indexing via GpuMeshInfo::triStart.
void UploadWorldGeo(GpuBatch* batch,
	const GpuPlane* planes, int numPlanes,
	const GpuMeshInfo* meshes, int numMeshes,
	const GpuVec3* tris, int numTris);

// Per-car-slot constants for the car pre-tick phase (same for all arenas/cars in gates)
void UploadCarConsts(GpuBatch* batch, const GpuCarConsts& consts);

// Dev/compare: run ONE car's full pre-tick from an uploaded state bundle against the
// uploaded world geometry + provided dynamic bodies. Returns the mutated state and the
// per-wheel scratch for mismatch localization.
void RunCarPreTickOnce(GpuBatch* batch, const GpuCarTickState& in,
	const GpuDynBody* dynBodies, int numDyn, int selfDynIdx,
	const GpuGridParams& grid, float tickTime,
	GpuCarTickState& out, GpuWheelScratchOut wheelsOut[4]);

// Mesh BVHs + triangle infos (internal-edge data), uploaded once. Arrays are shared
// across meshes; per-mesh views via GpuMeshBvh (nodeStart/subtreeStart) and triInfo
// indexed by (meshIdx triStart + triangleIndex).
void UploadMeshBvhs(GpuBatch* batch,
	const GpuMeshBvh* bvhs, int numMeshes,
	const GpuBvhNode* nodes, int numNodes,
	const GpuBvhSubtree* subtrees, int numSubtrees,
	const GpuTriInfo* triInfos, int numTris,
	const float* meshMargins);

// Collision-phase constants: static object table (broadphase handle order), static
// per-cell CSR lists, grid params, and the collide consts. Uploaded once.
void UploadCollideData(GpuBatch* batch, const GpuCollideConsts& consts,
	const GpuStaticObj* statics, int numStatics,
	const int* cellStaticStart, int numCells,
	const int* cellStaticList, int listLen,
	const GpuGridParams& grid);

// Per-arena persistent broadphase proxy state (initial values from the CPU proxies)
void UploadDynProxies(GpuBatch* batch, int arenaIdx,
	const GpuDynProxy* proxies, int regCounter);

void UploadBallState(GpuBatch* batch, int arenaIdx, const GpuBallGameState& state);
void DownloadBallState(GpuBatch* batch, int arenaIdx, GpuBallGameState* state);

// The arena tick counter the device uses (ball-hit extra-impulse guard etc.)
void SetTickCount(GpuBatch* batch, uint64_t tickCount);

// Game-tail data for the full GPU step: tail constants, pad configs (pad order =
// arena _boostPads order), dyn-body shape/filter templates per slot. Once.
void UploadTailData(GpuBatch* batch, const GpuTailConsts& consts,
	const GpuPadConfig* padConfigs, int numPads,
	const GpuDynBody* dynTemplates);

// Per-arena car pre/post-tick iteration order (the CPU's _cars unordered_set order)
void UploadCarOrder(GpuBatch* batch, int arenaIdx, const int* order);

// Per-arena minstd_rand0 state for demo respawn draws (post-.seed() engine state)
void UploadRngState(GpuBatch* batch, int arenaIdx, uint32_t state);

// Download the collide-phase results of one arena: manifolds in dispatcher order,
// proxy state, and the ball game state (callback side effects).
void DownloadCollideResults(GpuBatch* batch, int arenaIdx,
	GpuManifoldOut* manifolds, int* numManifolds,
	GpuDynProxy* proxies, GpuBallGameState* ball);

// Diagnostic: the collide phase's broadphase pair list ([0]=numPairs, then h0,h1
// per pair; handle < numStatics is a static). Buffer must hold 2*96+1 ints.
void DownloadCollideDebug(GpuBatch* batch, int arenaIdx, int* debugOut);

// Diagnostic: the solve phase's constraint rows (post-iterations scratch state).
// Buffers must hold >= 404 entries each.
struct GpuSolverRowDl {
	int bodyA, bodyB;            // solver body pool ids
	GpuVec3 normal1, relpos1CrossNormal, angularComponentA;
	float jacDiagABInv, rhs, rhsPenetration, lowerLimit, upperLimit;
	float appliedImpulse, appliedPushImpulse, friction;
	int frictionIndex;
	uint8_t isSpecial;
};
void DownloadSolveDebug(GpuBatch* batch, int arenaIdx,
	GpuSolverRowDl* contactRows, int* numContact,
	GpuSolverRowDl* frictionRows, int* numFriction);

// Diagnostic: closest-hit static-world raycast for ray-fuzz validation vs CPU rayTest.
struct GpuRayFuzzResult {
	int hitKind;       // -1 = none, 0 = static
	float fraction;
	GpuVec3 normal;
};
void RunRayFuzz(GpuBatch* batch, const GpuVec3* fromTo /*2 per ray*/, int numRays,
	int cbGroup, int cbMask, const GpuDynBody* dynBodies, int numDyn, int ignoreIdx,
	const GpuGridParams& grid, GpuRayFuzzResult* results);

// Dev/compare API: run ONE tick executing only phases [startPhase, endPhase].
// Production stepping uses StepTicks (full phase range, all ticks in one launch).
void RunPhaseSegment(GpuBatch* batch, int startPhase, int endPhase);

// Upload the full per-tick control schedule: layout [carGlobalIdx * numTicks + tick],
// carGlobalIdx = arenaIdx * numCarsPerArena + carIdx. One upload, zero per-tick traffic.
void UploadControlSchedule(GpuBatch* batch, const GpuCarControls* schedule, int numTicks);

// Run the monolithic kernel for numTicks ticks. The GPU result is THE simulation result;
// the CPU does no physics while this runs.
void StepTicks(GpuBatch* batch, int numTicks);

// Download full state for one arena slot (same layout as upload).
void DownloadArenaState(GpuBatch* batch, int arenaIdx,
	GpuBodyState* bodies, GpuCarGameState* cars, GpuPadState* pads);

} // namespace rscuda
