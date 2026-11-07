#include "rocketsim_cuda/step_block.cuh"

namespace rsc {

__global__ void persistent_step_kernel(
    const ArenaStatic *__restrict__ st, Arena *arenas, StepScratch *scratch,
    TeamSatScratch *team_sat, uint32_t num_arenas, uint32_t lanes_per_arena,
    uint32_t ticks, uint32_t blocks, const uint32_t *block_order,
    uint32_t *block_cycles, uint32_t *next_block) {
  uint32_t block_mask =
      blockDim.x >= 32 ? 0xffffffffu : (1u << blockDim.x) - 1u;
  uint32_t claim = blockIdx.x;
  while (claim < blocks) {
    uint32_t block = block_order[claim];
    if (threadIdx.x == 0) {
      block_cycles[block] = static_cast<uint32_t>(clock());
    }
    step_arena_block(st, arenas, scratch, team_sat, num_arenas, lanes_per_arena,
                     ticks, block, blockIdx.x, block_mask);
    if (threadIdx.x == 0) {
      block_cycles[block] =
          static_cast<uint32_t>(clock()) - block_cycles[block];
      claim = gridDim.x + atomicAdd(next_block, 1u);
    }
    claim = __shfl_sync(block_mask, claim, 0);
  }
}

}
