#include "rocketsim_cuda/cpu_step.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include <xmmintrin.h>

#define RS_FMA_DISPATCH

#pragma clang attribute push(                                                  \
    __attribute__((target("avx2,fma"), internal_linkage)),                     \
    apply_to = function)

#include "rsc/sim/arena.hpp"

#pragma clang attribute pop

namespace rocketsim_cuda {

bool cpu_supports_avx2_step() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

void step_arena_avx2(const rsc::ArenaStatic &view, rsc::Arena &arena,
                     rsc::StepScratch &scratch, uint32_t ticks) {
  rsc::arena_step_ticks(view, arena, scratch, ticks);
}

}
