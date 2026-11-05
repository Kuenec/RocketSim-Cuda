#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "rocketsim_cuda/rocketsim_cuda.hpp"

namespace rsc {

struct Arena;
struct ArenaOp;
class HostStatic;

class CudaBatch {
public:
  CudaBatch(const HostStatic &st, uint32_t num_arenas);
  ~CudaBatch();
  CudaBatch(const CudaBatch &) = delete;
  CudaBatch &operator=(const CudaBatch &) = delete;

  void upload(uint32_t first, uint32_t count, const Arena *src);
  void download(uint32_t arena, Arena &dst);
  void read(uint32_t arena, size_t offset, size_t size, void *dst);
  void apply_op(uint32_t arena, const ArenaOp &op);
  void apply_staged_controls(const rocketsim_cuda::CarControls *controls,
                             const uint8_t *mask, uint32_t cars_per_arena);
  void set_controls_device(const rocketsim_cuda::CarControls *controls,
                           uint32_t cars_per_arena);
  void reset_kickoff_masked(const uint8_t *mask, const uint64_t *seeds,
                            rocketsim_cuda::MemorySpace space);
  void export_state(const rocketsim_cuda::StateBuffers &out, uint32_t first,
                    uint32_t count, uint32_t cars_per_arena,
                    uint32_t pads_per_arena, rocketsim_cuda::MemorySpace space);
  void step(uint32_t ticks);
  void sync();
  void *stream() const;
  void copy_from(const CudaBatch &src, const uint32_t *src_arenas,
                 const uint32_t *dst_arenas, uint32_t count);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}
