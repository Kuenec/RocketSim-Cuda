#pragma once

#include <cstdint>
#include <memory>

namespace rsc {

struct Arena;
struct CarControls;
class HostStatic;

class CudaBatch {
public:
  CudaBatch(const HostStatic &st, uint32_t num_arenas);
  ~CudaBatch();
  CudaBatch(const CudaBatch &) = delete;
  CudaBatch &operator=(const CudaBatch &) = delete;

  void upload(uint32_t arena, const Arena &src);
  void download(uint32_t arena, Arena &dst);
  void apply_controls(const CarControls *controls, const uint8_t *mask,
                      uint32_t cars_per_arena);
  void step(uint32_t ticks);
  void sync();
  void copy_from(const CudaBatch &src, const uint32_t *src_arenas,
                 const uint32_t *dst_arenas, uint32_t count);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}
