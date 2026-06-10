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

// Diagnostic: closest-hit static-world raycast for ray-fuzz validation vs CPU rayTest.
struct GpuRayFuzzResult {
	int hitKind;       // -1 = none, 0 = static
	float fraction;
	GpuVec3 normal;
};
void RunRayFuzz(GpuBatch* batch, const GpuVec3* fromTo /*2 per ray*/, int numRays,
	int cbGroup, int cbMask, const GpuDynBody* dynBodies, int numDyn, int ignoreIdx,
	const GpuGridParams& grid, GpuRayFuzzResult* results);

// Load the glibc trig correction table (see CudaMath.cuh). Must be called before any
// stepping; returns false if the file is missing/corrupt.
bool LoadTrigFix(const char* path);

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
