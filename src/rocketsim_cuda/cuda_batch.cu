#include "rocketsim_cuda/cuda_batch.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "rsc/host/builder.hpp"

namespace rsc {

namespace {

constexpr uint32_t k_block_size = 64;
constexpr size_t k_device_stack_bytes = 16384;

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

__global__ void step_kernel(ArenaStatic st, Arena *arenas, StepScratch *scratch,
                            uint32_t num_arenas, uint32_t ticks) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_arenas) {
    return;
  }
  Arena &a = arenas[i];
  a.counters = ArenaEventCounters{};
  for (uint32_t t = 0; t < ticks; ++t) {
    arena_step_tick(st, a, scratch[i]);
  }
}

__global__ void controls_kernel(Arena *arenas, const CarControls *controls,
                                const uint8_t *mask, uint32_t cars_per_arena,
                                uint32_t total) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total || mask[i] == 0) {
    return;
  }
  arenas[i / cars_per_arena].cars[i % cars_per_arena].state.controls =
      controls[i];
}

}

struct CudaBatch::Impl {
  uint32_t num_arenas;
  std::unique_ptr<DeviceMesh[]> meshes;
  DeviceArray<MeshView> mesh_views;
  DeviceArray<RigidBody> static_bodies;
  DeviceArray<BroadphaseProxy> static_proxies;
  DeviceArray<Vec3A> hull_points;
  DeviceArray<uint32_t> cell_offsets;
  DeviceArray<uint16_t> cell_handles;
  DeviceArray<float> sin_table;
  DeviceArray<BoostPadConfig> pad_configs;
  DeviceArray<Aabb> pad_aabbs;
  DeviceArray<BvhWideNode> pad_nodes;
  DeviceArray<BvhWideLeaf> pad_leaves;
  DeviceArray<Arena> arenas;
  DeviceArray<StepScratch> scratch;
  DeviceArray<CarControls> controls;
  DeviceArray<uint8_t> controls_mask;
  DeviceArray<Arena> copy_staging;
  size_t copy_capacity = 0;
  ArenaStatic view;
  cudaStream_t stream = nullptr;

  Impl(const HostStatic &st, uint32_t n) : num_arenas(n) {
    size_t stack_bytes = 0;
    check(cudaDeviceGetLimit(&stack_bytes, cudaLimitStackSize),
          "cudaDeviceGetLimit");
    if (stack_bytes < k_device_stack_bytes) {
      check(cudaDeviceSetLimit(cudaLimitStackSize, k_device_stack_bytes),
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
    cell_offsets.upload(st.cell_offsets.data(), st.cell_offsets.size());
    cell_handles.upload(st.cell_handles.data(), st.cell_handles.size());
    sin_table.upload(st.sin_table.data(), st.sin_table.size());
    view.world.meshes = mesh_views.get();
    view.world.static_bodies = static_bodies.get();
    view.world.static_proxies = static_proxies.get();
    view.world.hull_points = hull_points.get();
    view.world.cell_static_offsets = cell_offsets.get();
    view.world.cell_static_handles = cell_handles.get();
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
    arenas.allocate(n);
    scratch.allocate(n);
  }

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }
};

CudaBatch::CudaBatch(const HostStatic &st, uint32_t num_arenas)
    : impl_(std::make_unique<Impl>(st, num_arenas)) {}

CudaBatch::~CudaBatch() = default;

void CudaBatch::upload(uint32_t arena, const Arena &src) {
  check(cudaMemcpyAsync(impl_->arenas.get() + arena, &src, sizeof(Arena),
                        cudaMemcpyHostToDevice, impl_->stream),
        "arena upload");
  sync();
}

void CudaBatch::download(uint32_t arena, Arena &dst) {
  check(cudaMemcpyAsync(&dst, impl_->arenas.get() + arena, sizeof(Arena),
                        cudaMemcpyDeviceToHost, impl_->stream),
        "arena download");
  sync();
}

void CudaBatch::download_all(Arena *dst) {
  check(cudaMemcpyAsync(dst, impl_->arenas.get(),
                        sizeof(Arena) * impl_->num_arenas,
                        cudaMemcpyDeviceToHost, impl_->stream),
        "arena download");
  sync();
}

void CudaBatch::apply_controls(const CarControls *controls, const uint8_t *mask,
                               uint32_t cars_per_arena) {
  uint32_t total = impl_->num_arenas * cars_per_arena;
  if (total == 0) {
    return;
  }
  if (!impl_->controls.get()) {
    impl_->controls.allocate(total);
    impl_->controls_mask.allocate(total);
  }
  check(cudaMemcpyAsync(impl_->controls.get(), controls,
                        sizeof(CarControls) * total, cudaMemcpyHostToDevice,
                        impl_->stream),
        "controls upload");
  check(cudaMemcpyAsync(impl_->controls_mask.get(), mask, total,
                        cudaMemcpyHostToDevice, impl_->stream),
        "controls mask upload");
  uint32_t blocks = (total + 255) / 256;
  controls_kernel<<<blocks, 256, 0, impl_->stream>>>(
      impl_->arenas.get(), impl_->controls.get(), impl_->controls_mask.get(),
      cars_per_arena, total);
  check(cudaGetLastError(), "controls kernel launch");
  sync();
}

void CudaBatch::step(uint32_t ticks) {
  uint32_t blocks = (impl_->num_arenas + k_block_size - 1) / k_block_size;
  step_kernel<<<blocks, k_block_size, 0, impl_->stream>>>(
      impl_->view, impl_->arenas.get(), impl_->scratch.get(), impl_->num_arenas,
      ticks);
  check(cudaGetLastError(), "step kernel launch");
}

void CudaBatch::sync() {
  check(cudaStreamSynchronize(impl_->stream), "stream synchronize");
}

void CudaBatch::copy_from(const CudaBatch &src, const uint32_t *src_arenas,
                          const uint32_t *dst_arenas, uint32_t count) {
  if (count == 0) {
    return;
  }
  check(cudaStreamSynchronize(src.impl_->stream), "source stream synchronize");
  if (count > impl_->copy_capacity) {
    impl_->copy_staging.allocate(count);
    impl_->copy_capacity = count;
  }
  for (uint32_t k = 0; k < count; ++k) {
    check(cudaMemcpyAsync(impl_->copy_staging.get() + k,
                          src.impl_->arenas.get() + src_arenas[k],
                          sizeof(Arena), cudaMemcpyDeviceToDevice,
                          impl_->stream),
          "arena copy");
  }
  for (uint32_t k = 0; k < count; ++k) {
    check(cudaMemcpyAsync(impl_->arenas.get() + dst_arenas[k],
                          impl_->copy_staging.get() + k, sizeof(Arena),
                          cudaMemcpyDeviceToDevice, impl_->stream),
          "arena copy");
  }
  sync();
}

}
