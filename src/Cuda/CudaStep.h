#pragma once
#include "GpuAllocationError.h"
#include "GpuTypes.h"

#include <cstddef>

// 2026-06-10: Host API for the monolithic GPU step. CUDA-free header (pimpl) so it can
// be included by code that builds without the toolkit; implementation lives in CudaStep.cu
// and is only compiled when CMake found CUDA (RS_CUDA=1).

namespace rscuda {

// True if a CUDA device is present and usable (always false in non-CUDA builds).
bool Available();

// One batch = N arenas with identical car counts, stepped together by one kernel launch.
struct GpuBatch;

GpuBatch* CreateBatch(int numArenas, int numCarsPerArena, int numPads);
[[noreturn]] void ThrowAllocationError(RocketSim::GpuAllocationKind kind, const char* buffer,
	size_t bytes, int cudaError);
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

// Snowday puck hull points (m_unscaledPoints, BT units), uploaded once
void UploadBallHull(GpuBatch* batch, const GpuVec3* points, int numPoints);

// Dropshot tile shapes + neighbor tables, uploaded once (also allocates and zeroes
// the per-arena tile damage/response state)
void UploadTiles(GpuBatch* batch, const GpuTileShape* tiles, int numTiles,
	const GpuTileTables& tables);

// Per-arena tile damage states (one byte per tile, DropshotTileState values) and,
// optionally, the per-tile no-contact-response flags (post-tick sync mirror)
void DownloadTileState(GpuBatch* batch, int arenaIdx, uint8_t* damageOut,
	uint8_t* noResponseOut = nullptr);

// Mesh BVHs + triangle infos (internal-edge data), uploaded once. Arrays are shared
// across meshes; per-mesh views via GpuMeshBvh (nodeStart/subtreeStart) and triInfo
// indexed by (meshIdx triStart + triangleIndex).
void UploadMeshBvhs(GpuBatch* batch,
	const GpuMeshBvh* bvhs, int numMeshes,
	const GpuBvhNode* nodes, int numNodes,
	const GpuBvhSubtree* subtrees, int numSubtrees,
	const GpuTriInfo* triInfos, int numTris,
	const float* meshMargins);

void UploadCollideData(GpuBatch* batch, const GpuCollideConsts& consts,
	const GpuStaticObj* statics, int numStatics,
	const int* cellStaticStart, int numCells, const int* cellStaticList,
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
// the CPU does no physics while this runs. Advances the batch tick counter, so
// chunked stepping (RL decision steps) keeps absolute tick semantics.
void StepTicks(GpuBatch* batch, int numTicks);

// 2026-06-11 async stepping (see CudaStep.cu): launch-without-sync + completion,
// and an arena-subset re-step over an explicit tick window (for arenas whose
// async chunk was invalidated by a CPU-side reset).
void StepTicksAsync(GpuBatch* batch, int numTicks);
unsigned long long BatchTickCount(GpuBatch* batch);
void SyncBatch(GpuBatch* batch);
struct GpuArenaStaging {
	GpuBodyState* bodies;
	GpuCarGameState* cars;
	GpuPadState* pads;
	GpuBallGameState* balls;
	GpuCarControls* controls;
	int* slots;
};
void SetArenaStaging(GpuBatch* batch, const GpuArenaStaging& mappedHost);
void UploadArenaStateAsync(GpuBatch* batch, int arenaIdx);
void UploadConstantControlsAsync(GpuBatch* batch, const GpuCarControls* perCar);
void StepTicksSubsetAsync(GpuBatch* batch, int numTicks, int count, unsigned long long tickBase);
void StepTicksSubset(GpuBatch* batch, int numTicks, const int* arenaIdxs, int count,
	unsigned long long tickBase);

// 2026-06-11: diagnostic per-phase cycle counters (see GpuPhase). Enable before
// stepping; Get downloads the totals (sum of per-block cycles per phase).
void EnablePhaseProfiling(GpuBatch* batch);
void GetPhaseCycles(GpuBatch* batch, unsigned long long out[GPU_PHASE_PROF_SLOTS]);

// Chunked-stepping controls: one CarControls per car ([arenaIdx * numCars + carIdx]),
// held constant across subsequent StepTicks calls. Overrides any schedule.
void UploadConstantControls(GpuBatch* batch, const GpuCarControls* perCar);

// Download full state for one arena slot (same layout as upload).
void DownloadArenaState(GpuBatch* batch, int arenaIdx,
	GpuBodyState* bodies, GpuCarGameState* cars, GpuPadState* pads);

// Batched downloads across ALL arenas (one device-to-host copy per array)
void DownloadAllArenaStates(GpuBatch* batch,
	GpuBodyState* bodies, GpuCarGameState* cars, GpuPadState* pads);
void DownloadAllBallStates(GpuBatch* batch, GpuBallGameState* balls);

unsigned StepTicksAndMirrorAsync(GpuBatch* batch, int numTicks, GpuBodyState* bodies,
	GpuCarGameState* cars, GpuPadState* pads, GpuBallGameState* balls, unsigned* arenaMirrored);
void RecordDownloadEvent(GpuBatch* batch);
bool DownloadEventDone(GpuBatch* batch);
void WaitDownloadEvent(GpuBatch* batch);

void CopyArenaStates(GpuBatch* source, GpuBatch* dest, const int* sourceIdx, const int* destIdx,
	int count);

} // namespace rscuda
