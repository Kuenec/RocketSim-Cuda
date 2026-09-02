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

	// Async form (kernel overlaps CPU work until Sync) + an arena-subset re-step
	// over the same window for reset arenas (see GpuArenaBridge.cpp)
	void StepAsync(int numTicks);
	void Sync();
	void StepSubset(const int* slots, int count, int numTicks);

	// 2026-09-02 asynchronous reset re-upload + subset re-step: queued on the
	// batch stream behind the in-flight chunk from pinned staging, so the small
	// subset kernel runs beside the trainer's inference instead of idling the
	// GPU at Sync time. Staging slots are per arena; call Sync before the next
	// reset touches the same slots.
	void UploadArenaAsync(int idx);
	void StepSubsetAsync(const int* slots, int count, int numTicks);

	// 2026-09-02: this step's second-half chunk, the state download and the
	// NEXT step's first-half chunk enqueued back-to-back on the batch stream
	// (both chunks run under the controls gathered here); the host waits only
	// for the download, so the writeback overlaps the in-flight chunk instead
	// of the GPU idling through the copy and the scatter. Bookkeeping matches
	// Step(numTicks) + Download() + StepAsync(nextTicks).
	void StepDownloadAndLaunch(int numTicks, int nextTicks);

	// Batched download + write results back into the CPU arenas (bodies, car
	// states incl. wheels and rotMat, ball, pads, tickCount advance).
	// Equivalent to Download() + WritebackArena(all) + FinishWriteback().
	void Writeback();

	// Split form, so callers can scatter arenas across worker threads:
	// Download() once (single-threaded), then WritebackArena(a) for each arena
	// (distinct arenas safe from different threads), then FinishWriteback().
	void Download();
	void WritebackArena(int a);
	void FinishWriteback();

	// 2026-06-11 GPU-direct GameState path: compose public states straight from
	// the download scratch (no Arena round-trip; mirrors WritebackCar/Ball +
	// GetState exactly), plus a minimal Arena sync for the consumers that still
	// read the Arena each step (ball body/state, car ballHitInfo, tickCount).
	void FillCarState(int a, int slot, RocketSim::CarState& out) const;
	void FillBallState(int a, RocketSim::BallState& out) const;
	void WritebackArenaLite(int a);
	const GpuPadState* PadsScratch() const { return padsScratch.data(); }
	int NumPads() const { return numPads; }

	// Car game-state snapshots from the last two Writebacks, for deriving bump/
	// demo events: index = arenaIdx * NumCars() + slot
	const std::vector<GpuCarGameState>& CurCars() const { return curCars; }
	const std::vector<GpuCarGameState>& PrevCars() const { return prevCars; }
	const std::vector<std::vector<RocketSim::Car*>>& CarLists() const { return carLists; }

	int NumArenas() const { return (int)arenas.size(); }
	int NumCars() const { return numCars; }

	// Diagnostic passthroughs to the kernel's per-phase cycle counters
	void EnablePhaseProfiling();
	void GetPhaseCycles(unsigned long long out[rscuda::GPU_PHASE_PROF_SLOTS]);

private:
	std::vector<RocketSim::Arena*> arenas;
	std::vector<std::vector<RocketSim::Car*>> carLists; // _cars order per arena
	GpuBatch* batch = nullptr;
	int numCars = 0;
	int numPads = 0;
	int pendingTicks = 0;  // downloaded-or-complete ticks since the last Writeback
	int inFlightTicks = 0; // async-launched ticks not yet Sync()ed
	std::vector<GpuBodyState> bodiesScratch;
	std::vector<GpuCarGameState> curCars, prevCars;
	std::vector<GpuPadState> padsScratch;
	std::vector<GpuBallGameState> ballsScratch;
	std::vector<GpuCarControls> controlsScratch;

	// pinned staging for the asynchronous reset path (cudaHostAlloc)
	GpuBodyState* stageBodies = nullptr;    // [numArenas * (1 + numCars)]
	GpuCarGameState* stageCars = nullptr;   // [numArenas * numCars]
	GpuPadState* stagePads = nullptr;       // [numArenas * numPads]
	GpuBallGameState* stageBalls = nullptr; // [numArenas]
	GpuCarControls* stageControls = nullptr;// [numArenas * numCars]
	int* stageSlots = nullptr;              // [numArenas]

	void SnapshotArena(int idx, GpuBodyState* bodies, GpuCarGameState* cars,
		GpuPadState* pads, GpuBallGameState& ball) const;
	void GatherControls(GpuCarControls* out) const;
};

} // namespace rscuda

#endif // RS_CUDA
