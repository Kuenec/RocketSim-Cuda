#pragma once
#include <chrono>
#include <cstdint>

// 2026-06-10: Phase timing counters for the CUDA port (work copy only, not in reference).
// Disabled by default; rocketsim_bench --profile enables them. When disabled the only
// overhead is one branch per scope.
namespace rsprof {

enum Phase : int {
	CAR_PRE_TICK,
	PAD_PRE_TICK,
	BALL_PRE_TICK,
	BT_PREDICT,
	BT_COLLISION,
	BT_ISLANDS,
	BT_SOLVE,
	BT_INTEGRATE,
	BT_VEHICLE,
	BT_ACTIVATION,
	CAR_POST_TICK,
	TICK_TAIL,
	PHASE_COUNT
};

extern bool enabled;
extern uint64_t totalNs[PHASE_COUNT];
extern const char* const PHASE_NAMES[PHASE_COUNT];

struct Scope {
	Phase p;
	bool on;
	std::chrono::steady_clock::time_point t0;
	explicit Scope(Phase p) : p(p), on(enabled) {
		if (on) t0 = std::chrono::steady_clock::now();
	}
	~Scope() {
		if (on) totalNs[p] += (uint64_t)(std::chrono::steady_clock::now() - t0).count();
	}
};

} // namespace rsprof
