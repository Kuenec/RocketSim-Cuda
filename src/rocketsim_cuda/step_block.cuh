#pragma once

#include <cstdint>

#include "rsc/sim/arena.hpp"

namespace rsc {

__device__ __forceinline__ void
step_arena_block(const ArenaStatic *__restrict__ st, Arena *arenas,
                 StepScratch *scratch, TeamSatScratch *team_sat,
                 uint32_t num_arenas, uint32_t lanes_per_arena, uint32_t ticks,
                 uint32_t block, uint32_t slot, uint32_t block_mask) {
  uint32_t thread = block * blockDim.x + threadIdx.x;
  uint32_t i = thread / lanes_per_arena;
  uint32_t live = __ballot_sync(block_mask, i < num_arenas);
  if (i >= num_arenas) {
    return;
  }
  uint32_t team_lane = threadIdx.x % 32;
  uint32_t slot_thread = slot * blockDim.x + threadIdx.x;
  LaneGroup lanes{thread % lanes_per_arena,
                  lanes_per_arena,
                  live,
                  team_lane,
                  static_cast<uint32_t>(__popc(live)),
                  scratch + (slot_thread - team_lane) / lanes_per_arena,
                  team_sat + slot};
  Arena &a = arenas[i];
  if (lanes.leader()) {
    a.counters = ArenaEventCounters{};
  }
  for (uint32_t t = 0; t < ticks; ++t) {
    arena_step_tick(*st, a, scratch[slot_thread / lanes_per_arena], lanes);
  }
}

__global__ void persistent_step_kernel(
    const ArenaStatic *__restrict__ st, Arena *arenas, StepScratch *scratch,
    TeamSatScratch *team_sat, uint32_t num_arenas, uint32_t lanes_per_arena,
    uint32_t ticks, uint32_t blocks, const uint32_t *block_order,
    uint32_t *block_cycles, uint32_t *next_block);

}
