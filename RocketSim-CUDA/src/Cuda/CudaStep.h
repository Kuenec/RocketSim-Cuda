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
