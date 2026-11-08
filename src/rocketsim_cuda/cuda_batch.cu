#include "rocketsim_cuda/cuda_batch.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "rocketsim_cuda/convert.hpp"
#include "rocketsim_cuda/step_block.cuh"
#include "rsc/host/builder.hpp"

namespace rsc {

namespace {

constexpr uint32_t k_max_block_size = 32;
constexpr uint32_t k_target_blocks_per_sm = 4;
constexpr size_t k_callee_stack_headroom = 1024;
constexpr uint32_t k_max_lanes_per_arena = 32;
constexpr uint32_t k_timed_launches = 64;
constexpr uint32_t k_cost_buckets = 64;
constexpr uint32_t k_copy_block_size = 256;

uint32_t pow2_lanes(uint32_t wanted) {
  uint32_t lanes = 1;
  while (lanes < wanted && lanes < k_max_lanes_per_arena) {
    lanes *= 2;
  }
  return lanes;
}

void check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA ") + what +
                             " failed: " + cudaGetErrorString(err));
  }
}

template <typename T> class DeviceArray {
public:
  DeviceArray() = default;
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  ~DeviceArray() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  void allocate(size_t count) {
    if (ptr_) {
      check(cudaFree(ptr_), "cudaFree");
      ptr_ = nullptr;
    }
    if (count == 0) {
      return;
    }
    check(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc");
  }

  void upload(const T *src, size_t count) {
    allocate(count);
    if (count != 0) {
      check(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpy to device");
    }
  }

  T *get() const { return ptr_; }

private:
  T *ptr_ = nullptr;
};

struct DeviceMesh {
  DeviceArray<TriangleShape> tris;
  DeviceArray<TriangleInfo> infos;
  DeviceArray<BvhWideNode> nodes;
  DeviceArray<BvhWideLeaf> leaves;
};

__device__ uint32_t cost_bucket(uint32_t cycles, uint64_t top) {
  return k_cost_buckets - 1 -
         static_cast<uint32_t>(static_cast<uint64_t>(cycles) * k_cost_buckets /
                               top);
}

__global__ void order_blocks_kernel(const uint32_t *block_cycles,
                                    uint32_t blocks, uint32_t *block_order,
                                    uint32_t *next_block) {
  __shared__ uint32_t cursor[k_cost_buckets];
  uint32_t lane = threadIdx.x;
  if (lane == 0) {
    *next_block = 0;
  }
  uint32_t most = 0;
  for (uint32_t b = lane; b < blocks; b += 32) {
    most = max(most, block_cycles[b]);
  }
  most = __reduce_max_sync(0xffffffffu, most);
  uint64_t top = static_cast<uint64_t>(most) + 1;
  for (uint32_t k = lane; k < k_cost_buckets; k += 32) {
    cursor[k] = 0;
  }
  __syncwarp();
  for (uint32_t b = lane; b < blocks; b += 32) {
    atomicAdd(&cursor[cost_bucket(block_cycles[b], top)], 1u);
  }
  __syncwarp();
  if (lane == 0) {
    uint32_t sum = 0;
    for (uint32_t k = 0; k < k_cost_buckets; ++k) {
      uint32_t count = cursor[k];
      cursor[k] = sum;
      sum += count;
    }
  }
  __syncwarp();
  for (uint32_t base = 0; base < blocks; base += 32) {
    uint32_t b = base + lane;
    bool valid = b < blocks;
    uint32_t bucket =
        valid ? cost_bucket(block_cycles[b], top) : k_cost_buckets;
    uint32_t peers = __match_any_sync(0xffffffffu, bucket);
    uint32_t first = __ffs(peers) - 1;
    uint32_t start = 0;
    if (valid && lane == first) {
      start = atomicAdd(&cursor[bucket], static_cast<uint32_t>(__popc(peers)));
    }
    start = __shfl_sync(0xffffffffu, start, first);
    if (valid) {
      block_order[start + __popc(peers & ((1u << lane) - 1))] = b;
    }
    __syncwarp();
  }
}

__global__ void resident_step_kernel(const ArenaStatic *__restrict__ st,
                                     Arena *arenas, StepScratch *scratch,
                                     TeamSatScratch *team_sat,
                                     uint32_t num_arenas,
                                     uint32_t lanes_per_arena, uint32_t ticks,
                                     bool reset_counters) {
  step_arena_block(st, arenas, scratch, team_sat, num_arenas, lanes_per_arena,
                   ticks, reset_counters, blockIdx.x, blockIdx.x,
                   __activemask());
}

static_assert(sizeof(Arena) % sizeof(uint64_t) == 0);
static_assert(alignof(Arena) >= alignof(uint64_t));

__global__ void copy_arenas_kernel(const Arena *from, Arena *to,
                                   const uint32_t *from_arenas,
                                   const uint32_t *to_arenas) {
  const auto *src = reinterpret_cast<const unsigned char *>(
      from + (from_arenas ? from_arenas[blockIdx.x] : blockIdx.x));
  auto *dst = reinterpret_cast<unsigned char *>(
      to + (to_arenas ? to_arenas[blockIdx.x] : blockIdx.x));
  for (size_t offset = threadIdx.x * sizeof(uint64_t); offset < sizeof(Arena);
       offset += blockDim.x * sizeof(uint64_t)) {
    uint64_t word;
    memcpy(&word, __builtin_assume_aligned(src + offset, alignof(uint64_t)),
           sizeof(word));
    memcpy(__builtin_assume_aligned(dst + offset, alignof(uint64_t)), &word,
           sizeof(word));
  }
}

__global__ void controls_kernel(Arena *arenas,
                                const rocketsim_cuda::CarControls *controls,
                                const uint8_t *mask, uint32_t cars_per_arena,
                                uint32_t total) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total || (mask && mask[i] == 0)) {
    return;
  }
  arenas[i / cars_per_arena].cars[i % cars_per_arena].state.controls =
      rocketsim_cuda::convert::to_internal(controls[i]);
}

__global__ void op_kernel(const ArenaStatic *__restrict__ st, Arena *arena,
                          ArenaOp op) {
  arena_apply_op(*st, *arena, op);
}

__global__ void kickoff_kernel(const ArenaStatic *__restrict__ st,
                               Arena *arenas, uint32_t num_arenas,
                               const uint8_t *mask, const uint64_t *seeds) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_arenas || mask[i] == 0) {
    return;
  }
  arena_reset_to_random_kickoff(*st, arenas[i], seeds != nullptr,
                                seeds ? seeds[i] : 0);
}

__global__ void set_state_kernel(const ArenaStatic *__restrict__ st,
                                 Arena *arenas, uint32_t num_arenas,
                                 const uint8_t *mask,
                                 const rocketsim_cuda::BallState *balls,
                                 const rocketsim_cuda::CarReset *cars,
                                 uint32_t cars_per_arena) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_arenas || mask[i] == 0) {
    return;
  }
  CarState cores[k_max_cars];
  bool on_ground[k_max_cars];
  for (uint32_t c = 0; c < cars_per_arena; ++c) {
    const rocketsim_cuda::CarReset &reset =
        cars[static_cast<size_t>(i) * cars_per_arena + c];
    cores[c] = rocketsim_cuda::convert::core_to_internal(reset.core);
    on_ground[c] = reset.on_ground;
  }
  arena_set_state(*st, arenas[i],
                  rocketsim_cuda::convert::to_internal(balls[i]), cores,
                  on_ground);
}

__global__ void export_kernel(const ArenaStatic *__restrict__ st,
                              const Arena *arenas, uint32_t first,
                              uint32_t count, const uint8_t *mask,
                              rocketsim_cuda::StateBuffers out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || (mask && !mask[first + i])) {
    return;
  }
  rocketsim_cuda::convert::export_arena(*st, arenas[first + i], i, out);
}

uint32_t grid_for(uint32_t count, uint32_t block) {
  return (count + block - 1) / block;
}

template <typename T>
T *staged(DeviceArray<T> &array, size_t &capacity, size_t count) {
  if (count > capacity) {
    array.allocate(count);
    capacity = count;
  }
  return array.get();
}

template <typename T>
void copy_out(T *host, const T *device, size_t count, cudaStream_t stream) {
  if (host && count != 0) {
    check(cudaMemcpyAsync(host, device, sizeof(T) * count,
                          cudaMemcpyDeviceToHost, stream),
          "state download");
  }
}
}

struct CudaBatch::Impl {
  uint32_t num_arenas;
  uint32_t lanes_per_arena;
  std::unique_ptr<DeviceMesh[]> meshes;
  DeviceArray<MeshView> mesh_views;
  DeviceArray<RigidBody> static_bodies;
  DeviceArray<BroadphaseProxy> static_proxies;
  DeviceArray<Vec3A> hull_points;
  DeviceArray<uint32_t> cell_masks;
  DeviceArray<float> sin_table;
  DeviceArray<BoostPadConfig> pad_configs;
  DeviceArray<Aabb> pad_aabbs;
  DeviceArray<BvhWideNode> pad_nodes;
  DeviceArray<BvhWideLeaf> pad_leaves;
  DeviceArray<Arena> arenas;
  DeviceArray<StepScratch> scratch;
  size_t scratch_capacity = 0;
  DeviceArray<TeamSatScratch> team_sat;
  DeviceArray<uint32_t> next_block;
  DeviceArray<rocketsim_cuda::CarControls> controls;
  size_t controls_capacity = 0;
  DeviceArray<rocketsim_cuda::CarControls> queued_controls;
  size_t queued_capacity = 0;
  uint32_t queued_cars_per_arena = 0;
  uint32_t queued_after = 0;
  bool has_queue = false;
  DeviceArray<uint8_t> mask;
  size_t mask_capacity = 0;
  DeviceArray<uint64_t> seeds;
  size_t seeds_capacity = 0;
  DeviceArray<rocketsim_cuda::BallState> in_balls;
  size_t in_balls_capacity = 0;
  DeviceArray<rocketsim_cuda::CarReset> in_cars;
  size_t in_cars_capacity = 0;
  DeviceArray<rocketsim_cuda::ArenaInfo> out_infos;
  size_t out_infos_capacity = 0;
  DeviceArray<rocketsim_cuda::BallState> out_balls;
  size_t out_balls_capacity = 0;
  DeviceArray<rocketsim_cuda::CarState> out_cars;
  size_t out_cars_capacity = 0;
  DeviceArray<rocketsim_cuda::PadStateView> out_pads;
  size_t out_pads_capacity = 0;
  DeviceArray<rocketsim_cuda::TileStates> out_tiles;
  size_t out_tiles_capacity = 0;
  DeviceArray<rocketsim_cuda::EventCounts> out_counts;
  size_t out_counts_capacity = 0;
  DeviceArray<uint32_t> copy_from_arenas;
  size_t copy_from_capacity = 0;
  DeviceArray<uint32_t> copy_to_arenas;
  size_t copy_to_capacity = 0;
  ArenaStatic view;
  DeviceArray<ArenaStatic> device_view;
  cudaStream_t stream = nullptr;
  uint32_t block_size = 1;
  uint32_t blocks = 0;
  uint32_t slot_blocks = 0;
  uint32_t car_lanes = 1;
  uint32_t resident_threads = 0;
  uint32_t sm_count = 0;
  size_t team_sat_capacity = 0;
  DeviceArray<uint32_t> block_order;
  DeviceArray<uint32_t> block_cycles;
  size_t block_capacity = 0;
  cudaEvent_t launch_events[2 * k_timed_launches] = {};
  uint32_t timed_launches = 0;

  void configure(uint32_t active) {
    num_arenas = active;
    lanes_per_arena = k_max_lanes_per_arena;
    while (lanes_per_arena > car_lanes &&
           active * lanes_per_arena > resident_threads) {
      lanes_per_arena /= 2;
    }
    uint32_t threads = active * lanes_per_arena;
    uint32_t per_block = threads / (k_target_blocks_per_sm * sm_count);
    block_size = lanes_per_arena;
    while (block_size * 2 <= per_block && block_size < k_max_block_size) {
      block_size *= 2;
    }
    blocks = (threads + block_size - 1) / block_size;
    int fitting_blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &fitting_blocks, persistent_step_kernel,
              static_cast<int>(block_size), 0),
          "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    slot_blocks =
        std::min(blocks, static_cast<uint32_t>(fitting_blocks) * sm_count);
    if (slot_blocks > team_sat_capacity) {
      team_sat.allocate(slot_blocks);
      team_sat_capacity = slot_blocks;
    }
    size_t slots =
        static_cast<size_t>(slot_blocks) * (block_size / lanes_per_arena);
    if (slots > scratch_capacity) {
      scratch.allocate(slots);
      check(cudaMemsetAsync(scratch.get(), 0, sizeof(StepScratch) * slots,
                            stream),
            "scratch reset");
      scratch_capacity = slots;
    }
    if (blocks > block_capacity) {
      block_order.allocate(blocks);
      block_cycles.allocate(blocks);
      block_capacity = blocks;
    }
    if (blocks != 0) {
      check(cudaMemsetAsync(block_cycles.get(), 0, sizeof(uint32_t) * blocks,
                            stream),
            "block cycles reset");
    }
  }

  Impl(const HostStatic &st, uint32_t n, uint32_t cars)
      : num_arenas(n), lanes_per_arena(1) {
    cudaFuncAttributes resident_attrs{};
    check(cudaFuncGetAttributes(&resident_attrs, resident_step_kernel),
          "cudaFuncGetAttributes");
    cudaFuncAttributes persistent_attrs{};
    check(cudaFuncGetAttributes(&persistent_attrs, persistent_step_kernel),
          "cudaFuncGetAttributes");
    size_t stack_bytes = std::max(resident_attrs.localSizeBytes,
                                  persistent_attrs.localSizeBytes) +
                         k_callee_stack_headroom;
    size_t current_stack = 0;
    check(cudaDeviceGetLimit(&current_stack, cudaLimitStackSize),
          "cudaDeviceGetLimit");
    if (current_stack < stack_bytes) {
      check(cudaDeviceSetLimit(cudaLimitStackSize, stack_bytes),
            "cudaDeviceSetLimit");
    }
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreate");
    view = st.view();
    std::vector<MeshView> views = st.mesh_views;
    meshes = std::make_unique<DeviceMesh[]>(views.size());
    for (size_t i = 0; i < views.size(); ++i) {
      const MeshView &mv = st.mesh_views[i];
      const BvhStorage &bvh = *st.mesh_bvhs[i];
      DeviceMesh &dm = meshes[i];
      dm.tris.upload(mv.tris, mv.num_tris);
      dm.infos.upload(mv.infos, mv.num_tris);
      dm.nodes.upload(bvh.nodes.data(), bvh.nodes.size());
      dm.leaves.upload(bvh.leaves.data(), bvh.leaves.size());
      views[i].tris = dm.tris.get();
      views[i].infos = dm.infos.get();
      views[i].bvh.nodes = dm.nodes.get();
      views[i].bvh.leaves = dm.leaves.get();
    }
    mesh_views.upload(views.data(), views.size());
    static_bodies.upload(st.static_bodies.data(), st.static_bodies.size());
    static_proxies.upload(st.static_proxies.data(), st.static_proxies.size());
    hull_points.upload(st.hull_points.data(), st.hull_points.size());
    cell_masks.upload(st.cell_masks.data(), st.cell_masks.size());
    sin_table.upload(st.sin_table.data(), st.sin_table.size());
    view.world.meshes = mesh_views.get();
    view.world.static_bodies = static_bodies.get();
    view.world.static_proxies = static_proxies.get();
    view.world.hull_points = hull_points.get();
    view.world.cell_static_masks = cell_masks.get();
    view.world.sin_table = sin_table.get();
    if (view.has_pads) {
      pad_configs.upload(st.pad_configs.data(), st.pad_configs.size());
      pad_aabbs.upload(st.pad_aabbs.data(), st.pad_aabbs.size());
      pad_nodes.upload(st.pad_bvh.nodes.data(), st.pad_bvh.nodes.size());
      pad_leaves.upload(st.pad_bvh.leaves.data(), st.pad_bvh.leaves.size());
      view.pads.configs = pad_configs.get();
      view.pads.aabbs = pad_aabbs.get();
      view.pads.bvh.nodes = pad_nodes.get();
      view.pads.bvh.leaves = pad_leaves.get();
    }
    device_view.upload(&view, 1);
    arenas.allocate(n);
    next_block.allocate(1);
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    int device_sms = 0;
    check(cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount,
                                 device),
          "cudaDeviceGetAttribute");
    int blocks_per_sm = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks_per_sm, resident_step_kernel,
              static_cast<int>(k_max_block_size), 0),
          "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    sm_count = static_cast<uint32_t>(device_sms);
    resident_threads =
        static_cast<uint32_t>(blocks_per_sm) * k_max_block_size * sm_count;
    car_lanes = pow2_lanes(cars);
    for (cudaEvent_t &e : launch_events) {
      check(cudaEventCreate(&e), "cudaEventCreate");
    }
    configure(n);
  }

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
    for (cudaEvent_t e : launch_events) {
      if (e) {
        cudaEventDestroy(e);
      }
    }
  }
};

void *alloc_pinned(size_t bytes) {
  void *pointer = nullptr;
  check(cudaMallocHost(&pointer, bytes), "cudaMallocHost");
  return pointer;
}

void free_pinned(void *pointer) {
  if (pointer) {
    check(cudaFreeHost(pointer), "cudaFreeHost");
  }
}

void release_pinned(void *pointer) noexcept {
  if (pointer) {
    cudaFreeHost(pointer);
  }
}

CudaBatch::CudaBatch(const HostStatic &st, uint32_t num_arenas,
                     uint32_t num_cars)
    : impl_(std::make_unique<Impl>(st, num_arenas, num_cars)) {}

CudaBatch::~CudaBatch() = default;

void CudaBatch::upload(uint32_t first, uint32_t count, const Arena *src) {
  check(cudaMemcpyAsync(impl_->arenas.get() + first, src, sizeof(Arena) * count,
                        cudaMemcpyHostToDevice, impl_->stream),
        "arena upload");
}

void CudaBatch::download(uint32_t arena, Arena &dst) {
  read(arena, 0, sizeof(Arena), &dst);
}

void CudaBatch::read(uint32_t arena, size_t offset, size_t size, void *dst) {
  const char *src =
      reinterpret_cast<const char *>(impl_->arenas.get() + arena) + offset;
  check(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, impl_->stream),
        "arena read");
  sync();
}

void CudaBatch::apply_op(uint32_t arena, const ArenaOp &op) {
  op_kernel<<<1, 1, 0, impl_->stream>>>(impl_->device_view.get(),
                                        impl_->arenas.get() + arena, op);
  check(cudaGetLastError(), "arena op kernel launch");
}

void CudaBatch::apply_staged_controls(
    const rocketsim_cuda::CarControls *controls, const uint8_t *mask,
    uint32_t cars_per_arena) {
  uint32_t total = impl_->num_arenas * cars_per_arena;
  if (total == 0) {
    return;
  }
  rocketsim_cuda::CarControls *d_controls =
      staged(impl_->controls, impl_->controls_capacity, total);
  uint8_t *d_mask = staged(impl_->mask, impl_->mask_capacity, total);
  check(cudaMemcpyAsync(d_controls, controls,
                        sizeof(rocketsim_cuda::CarControls) * total,
                        cudaMemcpyHostToDevice, impl_->stream),
        "controls upload");
  check(cudaMemcpyAsync(d_mask, mask, total, cudaMemcpyHostToDevice,
                        impl_->stream),
        "controls mask upload");
  controls_kernel<<<grid_for(total, 256), 256, 0, impl_->stream>>>(
      impl_->arenas.get(), d_controls, d_mask, cars_per_arena, total);
  check(cudaGetLastError(), "controls kernel launch");
}

void CudaBatch::set_controls_device(const rocketsim_cuda::CarControls *controls,
                                    uint32_t cars_per_arena) {
  uint32_t total = impl_->num_arenas * cars_per_arena;
  if (total == 0) {
    return;
  }
  controls_kernel<<<grid_for(total, 256), 256, 0, impl_->stream>>>(
      impl_->arenas.get(), controls, nullptr, cars_per_arena, total);
  check(cudaGetLastError(), "controls kernel launch");
}

void CudaBatch::queue_controls_device(
    const rocketsim_cuda::CarControls *controls, uint32_t cars_per_arena,
    uint32_t after_ticks) {
  Impl &im = *impl_;
  uint32_t total = im.num_arenas * cars_per_arena;
  if (total == 0) {
    return;
  }
  rocketsim_cuda::CarControls *queued =
      staged(im.queued_controls, im.queued_capacity, total);
  check(cudaMemcpyAsync(queued, controls,
                        sizeof(rocketsim_cuda::CarControls) * total,
                        cudaMemcpyDeviceToDevice, im.stream),
        "queued controls copy");
  im.queued_cars_per_arena = cars_per_arena;
  im.queued_after = after_ticks;
  im.has_queue = true;
}

void CudaBatch::reset_kickoff_masked(const uint8_t *mask, const uint64_t *seeds,
                                     rocketsim_cuda::MemorySpace space) {
  uint32_t n = impl_->num_arenas;
  if (n == 0) {
    return;
  }
  const uint8_t *d_mask = mask;
  const uint64_t *d_seeds = seeds;
  if (space == rocketsim_cuda::MemorySpace::Host) {
    uint8_t *m = staged(impl_->mask, impl_->mask_capacity, n);
    check(cudaMemcpyAsync(m, mask, n, cudaMemcpyHostToDevice, impl_->stream),
          "reset mask upload");
    d_mask = m;
    if (seeds) {
      uint64_t *sd = staged(impl_->seeds, impl_->seeds_capacity, n);
      check(cudaMemcpyAsync(sd, seeds, sizeof(uint64_t) * n,
                            cudaMemcpyHostToDevice, impl_->stream),
            "reset seed upload");
      d_seeds = sd;
    }
  }
  kickoff_kernel<<<grid_for(n, 64), 64, 0, impl_->stream>>>(
      impl_->device_view.get(), impl_->arenas.get(), n, d_mask, d_seeds);
  check(cudaGetLastError(), "kickoff kernel launch");
}

void CudaBatch::set_states_masked(const uint8_t *mask,
                                  const rocketsim_cuda::BallState *balls,
                                  const rocketsim_cuda::CarReset *cars,
                                  uint32_t cars_per_arena,
                                  rocketsim_cuda::MemorySpace space) {
  Impl &im = *impl_;
  uint32_t n = im.num_arenas;
  if (n == 0) {
    return;
  }
  const uint8_t *d_mask = mask;
  const rocketsim_cuda::BallState *d_balls = balls;
  const rocketsim_cuda::CarReset *d_cars = cars;
  if (space == rocketsim_cuda::MemorySpace::Host) {
    uint8_t *m = staged(im.mask, im.mask_capacity, n);
    check(cudaMemcpyAsync(m, mask, n, cudaMemcpyHostToDevice, im.stream),
          "state mask upload");
    rocketsim_cuda::BallState *b = staged(im.in_balls, im.in_balls_capacity, n);
    check(cudaMemcpyAsync(b, balls, sizeof(rocketsim_cuda::BallState) * n,
                          cudaMemcpyHostToDevice, im.stream),
          "ball state upload");
    size_t car_rows = static_cast<size_t>(n) * cars_per_arena;
    rocketsim_cuda::CarReset *c =
        staged(im.in_cars, im.in_cars_capacity, car_rows);
    if (car_rows != 0) {
      check(cudaMemcpyAsync(c, cars,
                            sizeof(rocketsim_cuda::CarReset) * car_rows,
                            cudaMemcpyHostToDevice, im.stream),
            "car state upload");
    }
    d_mask = m;
    d_balls = b;
    d_cars = c;
  }
  set_state_kernel<<<grid_for(n, 64), 64, 0, im.stream>>>(
      im.device_view.get(), im.arenas.get(), n, d_mask, d_balls, d_cars,
      cars_per_arena);
  check(cudaGetLastError(), "set state kernel launch");
}

void CudaBatch::export_state(const rocketsim_cuda::StateBuffers &out,
                             uint32_t first, uint32_t count,
                             const uint8_t *mask, uint32_t cars_per_arena,
                             uint32_t pads_per_arena,
                             rocketsim_cuda::MemorySpace space) {
  if (count == 0) {
    return;
  }
  Impl &im = *impl_;
  rocketsim_cuda::StateBuffers target = out;
  bool host = space == rocketsim_cuda::MemorySpace::Host;
  size_t num_cars = static_cast<size_t>(count) * cars_per_arena;
  size_t num_pads = static_cast<size_t>(count) * pads_per_arena;
  if (host) {
    target.infos = out.infos
                       ? staged(im.out_infos, im.out_infos_capacity, count)
                       : nullptr;
    target.balls = out.balls
                       ? staged(im.out_balls, im.out_balls_capacity, count)
                       : nullptr;
    target.cars = out.cars && num_cars
                      ? staged(im.out_cars, im.out_cars_capacity, num_cars)
                      : nullptr;
    target.pads = out.pads && num_pads
                      ? staged(im.out_pads, im.out_pads_capacity, num_pads)
                      : nullptr;
    target.tiles = out.tiles
                       ? staged(im.out_tiles, im.out_tiles_capacity, count)
                       : nullptr;
    target.event_counts =
        out.event_counts ? staged(im.out_counts, im.out_counts_capacity, count)
                         : nullptr;
  }
  export_kernel<<<grid_for(count, 64), 64, 0, im.stream>>>(
      im.device_view.get(), im.arenas.get(), first, count, mask, target);
  check(cudaGetLastError(), "export kernel launch");
  if (!host) {
    return;
  }
  copy_out(out.infos, target.infos, count, im.stream);
  copy_out(out.balls, target.balls, count, im.stream);
  copy_out(out.cars, target.cars, num_cars, im.stream);
  copy_out(out.pads, target.pads, num_pads, im.stream);
  copy_out(out.tiles, target.tiles, count, im.stream);
  copy_out(out.event_counts, target.event_counts, count, im.stream);
  sync();
}

static void launch_ticks(CudaBatch::Impl &im, uint32_t ticks,
                         bool reset_counters) {
  bool persistent = im.blocks > im.slot_blocks;
  if (persistent) {
    order_blocks_kernel<<<1, 32, 0, im.stream>>>(
        im.block_cycles.get(), im.blocks, im.block_order.get(),
        im.next_block.get());
    check(cudaGetLastError(), "order kernel launch");
    persistent_step_kernel<<<im.slot_blocks, im.block_size, 0, im.stream>>>(
        im.device_view.get(), im.arenas.get(), im.scratch.get(),
        im.team_sat.get(), im.num_arenas, im.lanes_per_arena, ticks,
        reset_counters, im.blocks, im.block_order.get(), im.block_cycles.get(),
        im.next_block.get());
  } else {
    resident_step_kernel<<<im.blocks, im.block_size, 0, im.stream>>>(
        im.device_view.get(), im.arenas.get(), im.scratch.get(),
        im.team_sat.get(), im.num_arenas, im.lanes_per_arena, ticks,
        reset_counters);
  }
  check(cudaGetLastError(), "step kernel launch");
}

void CudaBatch::step(uint32_t ticks) {
  Impl &im = *impl_;
  bool queued = im.has_queue;
  im.has_queue = false;
  if (im.num_arenas == 0) {
    return;
  }
  bool timed = im.timed_launches < k_timed_launches;
  if (timed) {
    check(cudaEventRecord(im.launch_events[2 * im.timed_launches], im.stream),
          "launch start event");
  }
  if (queued) {
    uint32_t total = im.num_arenas * im.queued_cars_per_arena;
    launch_ticks(im, im.queued_after, true);
    controls_kernel<<<grid_for(total, 256), 256, 0, im.stream>>>(
        im.arenas.get(), im.queued_controls.get(), nullptr,
        im.queued_cars_per_arena, total);
    check(cudaGetLastError(), "queued controls kernel launch");
    if (ticks > im.queued_after) {
      launch_ticks(im, ticks - im.queued_after, false);
    }
  } else {
    launch_ticks(im, ticks, true);
  }
  if (timed) {
    check(
        cudaEventRecord(im.launch_events[2 * im.timed_launches + 1], im.stream),
        "launch stop event");
    ++im.timed_launches;
  }
}

void CudaBatch::set_active(uint32_t count) { impl_->configure(count); }

StepTiming CudaBatch::take_step_timing() {
  Impl &im = *impl_;
  StepTiming out{0.0f, im.timed_launches};
  for (uint32_t i = 0; i < im.timed_launches; ++i) {
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, im.launch_events[2 * i],
                               im.launch_events[2 * i + 1]),
          "cudaEventElapsedTime");
    out.total_ms += ms;
  }
  im.timed_launches = 0;
  return out;
}

void CudaBatch::transfer(void *dst, const void *src, size_t bytes) {
  check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, impl_->stream),
        "stream transfer");
}

void CudaBatch::sync() {
  check(cudaStreamSynchronize(impl_->stream), "stream synchronize");
}

void *CudaBatch::stream() const { return impl_->stream; }

void CudaBatch::copy_from(const CudaBatch &src, const uint32_t *src_arenas,
                          const uint32_t *dst_arenas, uint32_t count) {
  if (count == 0) {
    return;
  }
  check(cudaStreamSynchronize(src.impl_->stream), "source stream synchronize");
  Impl &im = *impl_;
  std::vector<std::pair<uint32_t, uint32_t>> moves;
  moves.reserve(count);
  for (uint32_t k = 0; k < count; ++k) {
    moves.push_back({dst_arenas[k], k});
  }
  std::sort(moves.begin(), moves.end());
  std::vector<uint32_t> from;
  std::vector<uint32_t> to;
  for (size_t k = 0; k < moves.size(); ++k) {
    if (k + 1 < moves.size() && moves[k + 1].first == moves[k].first) {
      continue;
    }
    from.push_back(src_arenas[moves[k].second]);
    to.push_back(moves[k].first);
  }
  bool overlap = false;
  if (&src == this) {
    std::vector<uint32_t> sources = from;
    std::sort(sources.begin(), sources.end());
    overlap = std::any_of(to.begin(), to.end(), [&](uint32_t arena) {
      return std::binary_search(sources.begin(), sources.end(), arena);
    });
  }
  uint32_t kept = static_cast<uint32_t>(to.size());
  uint32_t *from_device =
      staged(im.copy_from_arenas, im.copy_from_capacity, kept);
  uint32_t *to_device = staged(im.copy_to_arenas, im.copy_to_capacity, kept);
  check(cudaMemcpyAsync(from_device, from.data(), sizeof(uint32_t) * kept,
                        cudaMemcpyHostToDevice, im.stream),
        "copy source upload");
  check(cudaMemcpyAsync(to_device, to.data(), sizeof(uint32_t) * kept,
                        cudaMemcpyHostToDevice, im.stream),
        "copy destination upload");
  if (!overlap) {
    copy_arenas_kernel<<<kept, k_copy_block_size, 0, im.stream>>>(
        src.impl_->arenas.get(), im.arenas.get(), from_device, to_device);
    check(cudaGetLastError(), "arena copy kernel launch");
    sync();
    return;
  }
  DeviceArray<Arena> staging;
  staging.allocate(kept);
  copy_arenas_kernel<<<kept, k_copy_block_size, 0, im.stream>>>(
      im.arenas.get(), staging.get(), from_device, nullptr);
  check(cudaGetLastError(), "arena gather kernel launch");
  copy_arenas_kernel<<<kept, k_copy_block_size, 0, im.stream>>>(
      staging.get(), im.arenas.get(), nullptr, to_device);
  check(cudaGetLastError(), "arena scatter kernel launch");
  sync();
}

}
