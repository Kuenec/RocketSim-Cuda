#include "rocketsim_cuda/c_api.hpp"

#include <algorithm>
#include <exception>
#include <string>

#ifdef RSC_HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace rocketsim_cuda;

struct RscMeshSet {
  std::shared_ptr<const MeshSet> meshes;
};

struct RscBatch {
  std::unique_ptr<ArenaBatch> batch;
};

namespace {

thread_local std::string g_last_error;

template <typename F> int guarded(F &&f) {
  try {
    f();
    return 0;
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return -1;
  }
}

}

extern "C" {

const char *rsc_last_error() { return g_last_error.c_str(); }

RscMeshSet *rsc_meshes_load(const char *path) {
  RscMeshSet *out = nullptr;
  guarded([&] { out = new RscMeshSet{MeshSet::load_directory(path)}; });
  return out;
}

void rsc_meshes_free(RscMeshSet *meshes) { delete meshes; }

int rsc_default_mutators(uint8_t game_mode, MutatorConfig *out) {
  return guarded(
      [&] { *out = default_mutators(static_cast<GameMode>(game_mode)); });
}

int rsc_car_body_config(uint8_t body, CarBodyConfig *out) {
  return guarded([&] { *out = car_body_config(static_cast<CarBody>(body)); });
}

int rsc_car_core_default(CarCore *out) {
  return guarded([&] { *out = car_core_default(); });
}

RscBatch *rsc_batch_create(const RscMeshSet *meshes, const RscBatchDesc *desc) {
  RscBatch *out = nullptr;
  guarded([&] {
    BatchConfig config;
    config.game_mode = static_cast<GameMode>(desc->game_mode);
    config.mem_weight_mode = static_cast<MemWeightMode>(desc->mem_weight_mode);
    config.min_pos = desc->min_pos;
    config.max_pos = desc->max_pos;
    config.max_aabb_len = desc->max_aabb_len;
    config.no_ball_rot = desc->no_ball_rot;
    config.cpu_threads = desc->cpu_threads;
    config.mutators = desc->mutators;
    config.cars.assign(desc->cars, desc->cars + desc->num_cars);
    config.rng_seeds.assign(desc->rng_seeds,
                            desc->rng_seeds + desc->num_arenas);
    if (desc->num_custom_pads >= 0) {
      config.custom_boost_pads.emplace(
          desc->custom_pads, desc->custom_pads + desc->num_custom_pads);
    }
    out = new RscBatch{std::make_unique<ArenaBatch>(
        static_cast<Backend>(desc->backend), meshes->meshes, config)};
  });
  return out;
}

void rsc_batch_free(RscBatch *batch) { delete batch; }

int rsc_batch_step(RscBatch *batch, uint32_t ticks) {
  return guarded([&] { batch->batch->step(ticks); });
}

int rsc_batch_step_async(RscBatch *batch, uint32_t ticks) {
  return guarded([&] { batch->batch->step_async(ticks); });
}

int rsc_batch_sync(RscBatch *batch) {
  return guarded([&] { batch->batch->sync(); });
}

int rsc_batch_set_controls(RscBatch *batch, uint32_t arena, uint32_t car,
                           const CarControls *controls) {
  return guarded(
      [&] { batch->batch->set_car_controls(arena, car, *controls); });
}

int rsc_abi_sizes(uint32_t *out, uint32_t count) {
  return guarded([&] {
    const uint32_t sizes[] = {sizeof(Vec3),
                              sizeof(PhysState),
                              sizeof(CarControls),
                              sizeof(RaycastHitInfo),
                              sizeof(CarCache),
                              sizeof(CarState),
                              sizeof(CarCore),
                              sizeof(CarBodyConfig),
                              sizeof(CarInfo),
                              sizeof(BallState),
                              sizeof(BoostPadConfig),
                              sizeof(TileStates),
                              sizeof(MutatorConfig),
                              sizeof(ArenaEvent),
                              sizeof(EventCounts),
                              sizeof(CarSpec),
                              sizeof(RscBatchDesc),
                              sizeof(RscPadState),
                              sizeof(RscArenaScalars),
                              sizeof(ArenaInfo),
                              sizeof(PadStateView),
                              sizeof(StateBuffers)};
    if (count != sizeof(sizes) / sizeof(sizes[0])) {
      throw std::invalid_argument("ABI size table length mismatch");
    }
    for (uint32_t i = 0; i < count; ++i) {
      out[i] = sizes[i];
    }
  });
}

int rsc_batch_set_all_controls(RscBatch *batch, const CarControls *controls,
                               uint32_t count) {
  return guarded([&] { batch->batch->set_all_controls({controls, count}); });
}

int rsc_batch_export(RscBatch *batch, uint8_t space, const StateBuffers *out) {
  return guarded([&] {
    batch->batch->export_state(*out, static_cast<MemorySpace>(space));
  });
}

int rsc_batch_set_all_controls_device(RscBatch *batch,
                                      const CarControls *device_controls) {
  return guarded(
      [&] { batch->batch->set_all_controls_device(device_controls); });
}

int rsc_batch_reset_kickoff_masked(RscBatch *batch, uint8_t space,
                                   const uint8_t *mask, const uint64_t *seeds) {
  return guarded([&] {
    batch->batch->reset_to_random_kickoff_masked(
        mask, seeds, static_cast<MemorySpace>(space));
  });
}

int rsc_device_alloc(size_t size, void **out) {
  return guarded([&] {
#ifdef RSC_HAS_CUDA
    cudaError_t err = cudaMalloc(out, size);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
#else
    *out = nullptr;
    throw std::runtime_error(
        "this build of rocketsim_cuda has no CUDA backend for " +
        std::to_string(size) + " device bytes");
#endif
  });
}

int rsc_device_free(void *pointer) {
  return guarded([&] {
#ifdef RSC_HAS_CUDA
    cudaError_t err = cudaFree(pointer);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
#else
    if (pointer != nullptr) {
      throw std::runtime_error(
          "this build of rocketsim_cuda has no CUDA backend");
    }
#endif
  });
}

int rsc_device_upload(void *device, const void *host, size_t size) {
  return guarded([&] {
#ifdef RSC_HAS_CUDA
    cudaError_t err = cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
      err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
#else
    throw std::runtime_error(
        "this build of rocketsim_cuda has no CUDA backend to copy " +
        std::to_string(size) + " bytes from " +
        std::to_string(reinterpret_cast<uintptr_t>(host)) + " to " +
        std::to_string(reinterpret_cast<uintptr_t>(device)));
#endif
  });
}

int rsc_device_download(void *host, const void *device, size_t size) {
  return guarded([&] {
#ifdef RSC_HAS_CUDA
    cudaError_t err = cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
#else
    throw std::runtime_error(
        "this build of rocketsim_cuda has no CUDA backend to copy " +
        std::to_string(size) + " bytes from " +
        std::to_string(reinterpret_cast<uintptr_t>(device)) + " to " +
        std::to_string(reinterpret_cast<uintptr_t>(host)));
#endif
  });
}

int rsc_batch_get_state(RscBatch *batch, uint32_t arena,
                        RscArenaScalars *scalars, BallState *ball,
                        TileStates *tiles, CarInfo *car_infos, CarState *cars,
                        uint32_t car_capacity, RscPadState *pads,
                        uint32_t pad_capacity) {
  return guarded([&] {
    ArenaState s = batch->batch->get_arena_state(arena);
    if (s.cars.size() > car_capacity || s.boost_pads.size() > pad_capacity) {
      throw std::length_error("state buffers are too small");
    }
    ArenaInfo info = batch->batch->get_arena_info(arena);
    scalars->tick_count = info.tick_count;
    scalars->rng_state = info.rng_state;
    scalars->error_flags = info.error_flags;
    scalars->ball_scored = info.ball_scored;
    scalars->has_tiles = s.tile_states.has_value();
    scalars->num_cars = static_cast<uint32_t>(s.cars.size());
    scalars->num_pads = static_cast<uint32_t>(s.boost_pads.size());
    *ball = s.ball;
    if (s.tile_states) {
      *tiles = *s.tile_states;
    }
    for (size_t c = 0; c < s.cars.size(); ++c) {
      car_infos[c] = s.car_infos[c];
      cars[c] = s.cars[c];
    }
    for (size_t p = 0; p < s.boost_pads.size(); ++p) {
      pads[p].config = s.boost_pad_configs[p];
      pads[p].cooldown = s.boost_pads[p].cooldown;
      pads[p].num_touching =
          static_cast<uint32_t>(s.boost_pads[p].touching_car_idxs.size());
      for (uint32_t i = 0; i < pads[p].num_touching; ++i) {
        pads[p].touching[i] = s.boost_pads[p].touching_car_idxs[i];
      }
    }
  });
}

int rsc_batch_get_car_state(RscBatch *batch, uint32_t arena, uint32_t car,
                            CarState *out) {
  return guarded([&] { *out = batch->batch->get_car_state(arena, car); });
}

int rsc_batch_get_ball_state(RscBatch *batch, uint32_t arena, BallState *out) {
  return guarded([&] { *out = batch->batch->get_ball_state(arena); });
}

int rsc_batch_get_pad_state(RscBatch *batch, uint32_t arena, uint32_t pad,
                            float *cooldown, uint32_t *touching,
                            uint32_t *num_touching) {
  return guarded([&] {
    BoostPadState s = batch->batch->get_boost_pad_state(arena, pad);
    if (s.touching_car_idxs.size() > k_max_cars) {
      throw std::length_error("pad membership exceeds the car capacity");
    }
    *cooldown = s.cooldown;
    *num_touching = static_cast<uint32_t>(s.touching_car_idxs.size());
    for (size_t i = 0; i < s.touching_car_idxs.size(); ++i) {
      touching[i] = s.touching_car_idxs[i];
    }
  });
}

int rsc_batch_get_tile_states(RscBatch *batch, uint32_t arena,
                              TileStates *out) {
  return guarded([&] { *out = batch->batch->get_tile_states(arena); });
}

int rsc_batch_save(RscBatch *batch, uint32_t arena, uint8_t *out,
                   size_t capacity, size_t *size) {
  return guarded([&] {
    std::vector<uint8_t> bytes = batch->batch->save_arena(arena);
    *size = bytes.size();
    if (bytes.size() > capacity) {
      throw std::length_error("snapshot buffer is too small");
    }
    std::copy(bytes.begin(), bytes.end(), out);
  });
}

int rsc_batch_load(RscBatch *batch, uint32_t arena, const uint8_t *data,
                   size_t size) {
  return guarded([&] { batch->batch->load_arena(arena, {data, size}); });
}

int rsc_batch_get_events(RscBatch *batch, uint32_t arena, ArenaEvent *out,
                         uint32_t capacity, uint32_t *count) {
  return guarded([&] {
    std::vector<ArenaEvent> events = batch->batch->get_last_step_events(arena);
    *count = static_cast<uint32_t>(events.size());
    if (events.size() > capacity) {
      throw std::length_error("event buffer is too small");
    }
    for (size_t i = 0; i < events.size(); ++i) {
      out[i] = events[i];
    }
  });
}

int rsc_batch_get_event_counts(RscBatch *batch, uint32_t arena,
                               EventCounts *out) {
  return guarded([&] { *out = batch->batch->get_event_counts(arena); });
}

int rsc_batch_get_mutators(RscBatch *batch, uint32_t arena,
                           MutatorConfig *out) {
  return guarded([&] { *out = batch->batch->get_mutator_config(arena); });
}

int rsc_batch_set_car_state(RscBatch *batch, uint32_t arena, uint32_t car,
                            const CarCore *core) {
  return guarded([&] { batch->batch->set_car_state(arena, car, *core); });
}

int rsc_batch_set_car_state_full(RscBatch *batch, uint32_t arena, uint32_t car,
                                 const CarState *state) {
  return guarded([&] { batch->batch->set_car_state_full(arena, car, *state); });
}

int rsc_batch_set_ball_state(RscBatch *batch, uint32_t arena,
                             const BallState *state) {
  return guarded([&] { batch->batch->set_ball_state(arena, *state); });
}

int rsc_batch_set_pad_state(RscBatch *batch, uint32_t arena, uint32_t pad,
                            float cooldown, const uint32_t *touching,
                            uint32_t num_touching) {
  return guarded([&] {
    BoostPadState s;
    s.cooldown = cooldown;
    s.touching_car_idxs.assign(touching, touching + num_touching);
    batch->batch->set_boost_pad_state(arena, pad, s);
  });
}

int rsc_batch_set_tiles(RscBatch *batch, uint32_t arena,
                        const TileStates *states) {
  return guarded([&] { batch->batch->set_tile_states(arena, *states); });
}

int rsc_batch_set_mutators(RscBatch *batch, uint32_t arena,
                           const MutatorConfig *mutators) {
  return guarded([&] { batch->batch->set_mutator_config(arena, *mutators); });
}

int rsc_batch_reset_kickoff(RscBatch *batch, uint32_t arena, bool has_seed,
                            uint64_t seed) {
  return guarded([&] {
    batch->batch->reset_to_random_kickoff(
        arena, has_seed ? std::optional<uint64_t>(seed) : std::nullopt);
  });
}

int rsc_batch_copy(RscBatch *dst, const RscBatch *src,
                   const uint32_t *src_arenas, const uint32_t *dst_arenas,
                   uint32_t count) {
  return guarded([&] {
    dst->batch->copy_arenas(*src->batch, {src_arenas, count},
                            {dst_arenas, count});
  });
}
}
