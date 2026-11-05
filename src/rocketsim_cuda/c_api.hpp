#pragma once

#include <cstddef>
#include <cstdint>

#include "rocketsim_cuda/rocketsim_cuda.hpp"

extern "C" {

struct RscMeshSet;
struct RscBatch;

struct RscBatchDesc {
  uint8_t backend;
  uint8_t game_mode;
  uint8_t mem_weight_mode;
  bool no_ball_rot;
  rocketsim_cuda::Vec3 min_pos;
  rocketsim_cuda::Vec3 max_pos;
  float max_aabb_len;
  rocketsim_cuda::MutatorConfig mutators;
  const rocketsim_cuda::CarSpec *cars;
  uint32_t num_cars;
  const uint64_t *rng_seeds;
  uint32_t num_arenas;
  const rocketsim_cuda::BoostPadConfig *custom_pads;
  int32_t num_custom_pads;
};

struct RscArenaScalars {
  uint64_t tick_count;
  uint64_t rng_state;
  uint32_t error_flags;
  bool ball_scored;
  bool has_tiles;
  uint32_t num_cars;
  uint32_t num_pads;
};

struct RscPadState {
  rocketsim_cuda::BoostPadConfig config;
  float cooldown;
  uint32_t num_touching;
  uint32_t touching[rocketsim_cuda::k_max_cars];
};

const char *rsc_last_error();

RscMeshSet *rsc_meshes_load(const char *path);
void rsc_meshes_free(RscMeshSet *meshes);

int rsc_default_mutators(uint8_t game_mode, rocketsim_cuda::MutatorConfig *out);
int rsc_car_body_config(uint8_t body, rocketsim_cuda::CarBodyConfig *out);
int rsc_car_core_default(rocketsim_cuda::CarCore *out);

RscBatch *rsc_batch_create(const RscMeshSet *meshes, const RscBatchDesc *desc);
void rsc_batch_free(RscBatch *batch);

int rsc_batch_step(RscBatch *batch, uint32_t ticks);
int rsc_batch_step_async(RscBatch *batch, uint32_t ticks);
int rsc_batch_sync(RscBatch *batch);
int rsc_batch_set_controls(RscBatch *batch, uint32_t arena, uint32_t car,
                           const rocketsim_cuda::CarControls *controls);
int rsc_batch_set_all_controls(RscBatch *batch,
                               const rocketsim_cuda::CarControls *controls,
                               uint32_t count);

int rsc_abi_sizes(uint32_t *out, uint32_t count);
int rsc_batch_export(RscBatch *batch, uint8_t space,
                     const rocketsim_cuda::StateBuffers *out);
int rsc_batch_set_all_controls_device(
    RscBatch *batch, const rocketsim_cuda::CarControls *device_controls);
int rsc_batch_reset_kickoff_masked(RscBatch *batch, uint8_t space,
                                   const uint8_t *mask, const uint64_t *seeds);
int rsc_device_alloc(size_t size, void **out);
int rsc_device_free(void *pointer);
int rsc_device_upload(void *device, const void *host, size_t size);
int rsc_device_download(void *host, const void *device, size_t size);
int rsc_batch_get_state(RscBatch *batch, uint32_t arena,
                        RscArenaScalars *scalars,
                        rocketsim_cuda::BallState *ball,
                        rocketsim_cuda::TileStates *tiles,
                        rocketsim_cuda::CarInfo *car_infos,
                        rocketsim_cuda::CarState *cars, uint32_t car_capacity,
                        RscPadState *pads, uint32_t pad_capacity);
int rsc_batch_get_car_state(RscBatch *batch, uint32_t arena, uint32_t car,
                            rocketsim_cuda::CarState *out);
int rsc_batch_get_ball_state(RscBatch *batch, uint32_t arena,
                             rocketsim_cuda::BallState *out);
int rsc_batch_get_pad_state(RscBatch *batch, uint32_t arena, uint32_t pad,
                            float *cooldown, uint32_t *touching,
                            uint32_t *num_touching);
int rsc_batch_get_tile_states(RscBatch *batch, uint32_t arena,
                              rocketsim_cuda::TileStates *out);
int rsc_batch_save(RscBatch *batch, uint32_t arena, uint8_t *out,
                   size_t capacity, size_t *size);
int rsc_batch_load(RscBatch *batch, uint32_t arena, const uint8_t *data,
                   size_t size);
int rsc_batch_get_events(RscBatch *batch, uint32_t arena,
                         rocketsim_cuda::ArenaEvent *out, uint32_t capacity,
                         uint32_t *count);
int rsc_batch_get_event_counts(RscBatch *batch, uint32_t arena,
                               rocketsim_cuda::EventCounts *out);
int rsc_batch_get_mutators(RscBatch *batch, uint32_t arena,
                           rocketsim_cuda::MutatorConfig *out);

int rsc_batch_set_car_state(RscBatch *batch, uint32_t arena, uint32_t car,
                            const rocketsim_cuda::CarCore *core);
int rsc_batch_set_car_state_full(RscBatch *batch, uint32_t arena, uint32_t car,
                                 const rocketsim_cuda::CarState *state);
int rsc_batch_set_ball_state(RscBatch *batch, uint32_t arena,
                             const rocketsim_cuda::BallState *state);
int rsc_batch_set_pad_state(RscBatch *batch, uint32_t arena, uint32_t pad,
                            float cooldown, const uint32_t *touching,
                            uint32_t num_touching);
int rsc_batch_set_tiles(RscBatch *batch, uint32_t arena,
                        const rocketsim_cuda::TileStates *states);
int rsc_batch_set_mutators(RscBatch *batch, uint32_t arena,
                           const rocketsim_cuda::MutatorConfig *mutators);
int rsc_batch_reset_kickoff(RscBatch *batch, uint32_t arena, bool has_seed,
                            uint64_t seed);
int rsc_batch_copy(RscBatch *dst, const RscBatch *src,
                   const uint32_t *src_arenas, const uint32_t *dst_arenas,
                   uint32_t count);
}
