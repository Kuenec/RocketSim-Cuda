#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "rocketsim_cuda/rocketsim_cuda.hpp"

namespace rsc {

struct Arena;
struct ArenaOp;
class HostStatic;

void *alloc_pinned(size_t bytes);
void free_pinned(void *pointer);
void release_pinned(void *pointer) noexcept;

template <typename T> class PinnedRows {
public:
  PinnedRows() = default;
  ~PinnedRows() { release_pinned(data_); }
  PinnedRows(const PinnedRows &) = delete;
  PinnedRows &operator=(const PinnedRows &) = delete;

  T *reserve(size_t count) {
    if (count > capacity_) {
      free_pinned(data_);
      data_ = nullptr;
      capacity_ = 0;
      data_ = static_cast<T *>(alloc_pinned(sizeof(T) * count));
      std::uninitialized_default_construct_n(data_, count);
      capacity_ = count;
    }
    return data_;
  }

private:
  T *data_ = nullptr;
  size_t capacity_ = 0;
};

struct StepTiming {
  float total_ms;
  uint32_t launches;
};

class CudaBatch {
public:
  CudaBatch(const HostStatic &st, uint32_t num_arenas, uint32_t num_cars);
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
  void set_states_masked(const uint8_t *mask,
                         const rocketsim_cuda::BallState *balls,
                         const rocketsim_cuda::CarReset *cars,
                         uint32_t cars_per_arena,
                         rocketsim_cuda::MemorySpace space);
  void export_state(const rocketsim_cuda::StateBuffers &out, uint32_t first,
                    uint32_t count, const uint8_t *mask,
                    uint32_t cars_per_arena, uint32_t pads_per_arena,
                    rocketsim_cuda::MemorySpace space);
  void step(uint32_t ticks);
  void set_active(uint32_t count);
  StepTiming take_step_timing();
  void transfer(void *dst, const void *src, size_t bytes);
  void sync();
  void *stream() const;
  void copy_from(const CudaBatch &src, const uint32_t *src_arenas,
                 const uint32_t *dst_arenas, uint32_t count);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}
