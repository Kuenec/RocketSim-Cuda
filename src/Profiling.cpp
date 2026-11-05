#include "Profiling.h"

namespace rsprof {

bool enabled = false;
uint64_t totalNs[PHASE_COUNT] = {};
const char* const PHASE_NAMES[PHASE_COUNT] = {
	"car_pre_tick",
	"pad_pre_tick",
	"ball_pre_tick",
	"bt_predict",
	"bt_collision",
	"bt_islands",
	"bt_solve",
	"bt_integrate",
	"bt_vehicle",
	"bt_activation",
	"car_post_tick",
	"tick_tail",
};

} // namespace rsprof
