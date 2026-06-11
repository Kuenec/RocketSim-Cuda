#pragma once
// 2026-06-10: Host-side bridge between live RocketSim Arena objects and the GPU
// batch (CudaStep.h). Built for RL training integration: the CPU arenas stay the
// source of truth for observation/reward code, the GPU produces the physics.
// The extraction/snapshot/writeback logic is the bench driver's validated code,
// generalized from bench car slots to plain Car* lists.

#include "../Sim/Arena/Arena.h"

#ifdef RS_CUDA

#include "CudaStep.h"
#include <vector>

namespace rscuda {

// Steps a group of UNIFORM arenas (same game mode, same car count, default car
// config) on the GPU in lockstep, holding each car's controls constant within a
// chunk (the RL decision-step model). Car slot order within an arena is the
// arena's _cars iteration order, matching the CPU's own pre-tick order.
class GpuArenaBatch {
public:
	// All arenas must share game mode, car count and tick count.
	// rngSeed drives on-device demolition respawn draws (per-arena derived).
	GpuArenaBatch(const std::vector<RocketSim::Arena*>& arenas, uint64_t rngSeed);
	~GpuArenaBatch();
	GpuArenaBatch(const GpuArenaBatch&) = delete;
	GpuArenaBatch& operator=(const GpuArenaBatch&) = delete;

	// Re-upload one arena's full dynamic state (call after a state-set/reset)
	void UploadArena(int idx);

	// Gather car->controls from the arenas, hold them constant, simulate numTicks
	void Step(int numTicks);

	// Batched download + write results back into the CPU arenas (bodies, car
	// states incl. wheels and rotMat, ball, pads, tickCount advance)
	void Writeback();

	// Car game-state snapshots from the last two Writebacks, for deriving bump/
	// demo events: index = arenaIdx * NumCars() + slot
	const std::vector<GpuCarGameState>& CurCars() const { return curCars; }
	const std::vector<GpuCarGameState>& PrevCars() const { return prevCars; }
	const std::vector<std::vector<RocketSim::Car*>>& CarLists() const { return carLists; }

	int NumArenas() const { return (int)arenas.size(); }
	int NumCars() const { return numCars; }

private:
	std::vector<RocketSim::Arena*> arenas;
	std::vector<std::vector<RocketSim::Car*>> carLists; // _cars order per arena
	GpuBatch* batch = nullptr;
	int numCars = 0;
	int numPads = 0;
	int pendingTicks = 0; // simulated since the last Writeback
	std::vector<GpuBodyState> bodiesScratch;
	std::vector<GpuCarGameState> curCars, prevCars;
	std::vector<GpuPadState> padsScratch;
	std::vector<GpuBallGameState> ballsScratch;
	std::vector<GpuCarControls> controlsScratch;
};

} // namespace rscuda

#endif // RS_CUDA
