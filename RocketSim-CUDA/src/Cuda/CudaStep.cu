// 2026-06-10: Monolithic GPU step - implementation scaffold.
//
// Architecture (per project requirements): ONE __global__ kernel executes the ENTIRE
// physics tick for all arenas in the batch, looping ticks inside the kernel so a whole
// gate run is a single launch with zero per-tick CPU work. Each physics phase is a
// __device__ function called from the kernel in CPU reference order:
//
//   per tick:
//     1. ball zero-vel sleep check        (Arena::Step head)
//     2. car pre-tick (vehicle+game)      (Car::_PreTickUpdate)
//     3. pad pre-tick                     (BoostPad::_PreTickUpdate)
//     4. ball pre-tick                    (Ball::_PreTickUpdate)
//     5. bullet: predict unconstrained    (btDiscreteDynamicsWorld)
//     6. bullet: collision detection      (broadphase + narrowphase + callbacks)
//     7. bullet: islands + solver         (btSequentialImpulseConstraintSolver)
//     8. bullet: integrate transforms
//     9. bullet: update activation
//    10. car post-tick + finish + pads    (Car::_PostTickUpdate/_FinishPhysicsTick)
//    11. pad post-tick, ball finish       (Arena::Step tail)
//
// Phases are being ported one at a time; unported phases are currently absent, so the
// kernel does NOT yet produce correct simulation output. CUDA% stays 0 until the full
// step matches the CPU reference.

#include "CudaStep.h"

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
	return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct GpuBatch {
	int numArenas;
	int numCars;        // per arena
	int numPads;        // per arena
	int bodiesPerArena; // 1 ball + numCars

	GpuBodyState* dBodies;      // [numArenas * bodiesPerArena]
	GpuCarGameState* dCars;     // [numArenas * numCars]
	GpuPadState* dPads;         // [numArenas * numPads]
	GpuCarControls* dSchedule;  // [numArenas * numCars * scheduleTicks]
	int scheduleTicks;
	uint64_t tickCount;         // ticks simulated so far (mirrors Arena::tickCount delta)
};

GpuBatch* CreateBatch(int numArenas, int numCarsPerArena, int numPads) {
	GpuBatch* b = new GpuBatch{};
	b->numArenas = numArenas;
	b->numCars = numCarsPerArena;
	b->numPads = numPads;
	b->bodiesPerArena = 1 + numCarsPerArena;
	b->scheduleTicks = 0;
	b->tickCount = 0;
	CudaCheck(cudaMalloc(&b->dBodies, sizeof(GpuBodyState) * numArenas * b->bodiesPerArena), "alloc bodies");
	CudaCheck(cudaMalloc(&b->dCars, sizeof(GpuCarGameState) * numArenas * numCarsPerArena), "alloc cars");
	CudaCheck(cudaMalloc(&b->dPads, sizeof(GpuPadState) * numArenas * numPads), "alloc pads");
	b->dSchedule = nullptr;
	return b;
}

void DestroyBatch(GpuBatch* b) {
	cudaFree(b->dBodies);
	cudaFree(b->dCars);
	cudaFree(b->dPads);
	cudaFree(b->dSchedule);
	delete b;
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

void UploadControlSchedule(GpuBatch* b, const GpuCarControls* schedule, int numTicks) {
	size_t count = (size_t)b->numArenas * b->numCars * numTicks;
	cudaFree(b->dSchedule);
	CudaCheck(cudaMalloc(&b->dSchedule, sizeof(GpuCarControls) * count), "alloc schedule");
	CudaCheck(cudaMemcpy(b->dSchedule, schedule, sizeof(GpuCarControls) * count,
		cudaMemcpyHostToDevice), "upload schedule");
	b->scheduleTicks = numTicks;
}

// ============================== DEVICE-SIDE PHASES ==============================
// Each phase is a __device__ function; the monolithic kernel below is the ONLY
// call site, and the only path that produces simulation output in --gpu-step mode.

// (phases land here as they are ported)

// ============================== MONOLITHIC KERNEL ==============================

__global__ void GpuStepArenasKernel(
	GpuBodyState* bodies, GpuCarGameState* cars, GpuPadState* pads,
	const GpuCarControls* schedule,
	int numCars, int numPads, int numTicks, uint64_t startTickCount) {

	const int arenaIdx = blockIdx.x;
	const int bodiesPerArena = 1 + numCars;

	GpuBodyState* arenaBodies = bodies + (size_t)arenaIdx * bodiesPerArena;
	GpuCarGameState* arenaCars = cars + (size_t)arenaIdx * numCars;
	GpuPadState* arenaPads = pads + (size_t)arenaIdx * numPads;
	(void)arenaBodies; (void)arenaPads;

	for (int tick = 0; tick < numTicks; tick++) {
		// Controls for this tick (input data, precomputed host-side once per run)
		if (threadIdx.x < (unsigned)numCars && schedule) {
			const GpuCarControls& c =
				schedule[((size_t)arenaIdx * numCars + threadIdx.x) * numTicks + tick];
			// lastControls handoff happens in car post-tick once that phase is ported;
			// stash current controls nowhere yet - no phase consumes them.
			(void)c; (void)arenaCars;
		}
		__syncthreads();

		// Phases 1-11 land here in CPU reference order as they are ported.

		__syncthreads();
	}
}

void StepTicks(GpuBatch* b, int numTicks) {
	GpuStepArenasKernel<<<b->numArenas, 64>>>(
		b->dBodies, b->dCars, b->dPads, b->dSchedule,
		b->numCars, b->numPads, numTicks, b->tickCount);
	CudaCheck(cudaGetLastError(), "kernel launch");
	CudaCheck(cudaDeviceSynchronize(), "kernel run");
	b->tickCount += numTicks;
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
