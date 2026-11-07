#pragma once

#include <cstdint>

namespace rsc {
struct ArenaStatic;
struct Arena;
struct StepScratch;
}

namespace rocketsim_cuda {

void step_arena(const rsc::ArenaStatic &view, rsc::Arena &arena,
                rsc::StepScratch &scratch, uint32_t ticks);

bool cpu_supports_avx2_step();

void step_arena_avx2(const rsc::ArenaStatic &view, rsc::Arena &arena,
                     rsc::StepScratch &scratch, uint32_t ticks);

}
