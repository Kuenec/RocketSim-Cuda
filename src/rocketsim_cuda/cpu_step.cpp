#include "rocketsim_cuda/cpu_step.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

#define RS_FMA_DISPATCH

#pragma clang attribute push(__attribute__((internal_linkage)),                \
                             apply_to = function)

#include "rsc/sim/arena.hpp"

#pragma clang attribute pop

namespace rocketsim_cuda {

void step_arena(const rsc::ArenaStatic &view, rsc::Arena &arena,
                rsc::StepScratch &scratch, uint32_t ticks) {
  rsc::arena_step_ticks(view, arena, scratch, ticks);
}

}
