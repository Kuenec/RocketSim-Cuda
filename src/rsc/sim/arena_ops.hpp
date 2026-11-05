#pragma once

#include <cstdint>

#include "rsc/sim/arena.hpp"

namespace rsc {

RS_HD void arena_set_ball_state(const ArenaStatic &st, Arena &a,
                                const BallState &state) {
  World w = arena_world(st, a);
  ball_set_state(a.ball, w.dyn_body(a.ball.rigid_body_idx), state);
}

RS_HD void arena_set_car_state_full(const ArenaStatic &st, Arena &a,
                                    uint32_t car_idx, const CarState &state) {
  World w = arena_world(st, a);
  Car &car = a.cars[car_idx];
  car_set_state(car, w.dyn_body(car.rigid_body_idx), state);
}

RS_HD void arena_set_car_state(const ArenaStatic &st, Arena &a,
                               uint32_t car_idx, const CarState &core) {
  arena_set_car_state_full(st, a, car_idx, car_state_from_core(core));
  car_reset_transient_contacts(a.cars[car_idx]);
}

RS_HD void arena_set_tile_states(Arena &a, const TileStates &tiles) {
  a.tiles = tiles;
  a.has_tiles = true;
  a.ball_only = false;
  arena_update_tile_flags(a);
}

RS_HD void arena_reset_pads(const ArenaStatic &st, Arena &a) {
  for (uint32_t i = 0; i < st.pads.num_pads; ++i) {
    a.pads[i].has_gave_tick = false;
    a.pads[i].gave_tick = 0;
    a.pads[i].num_touching = 0;
  }
}

RS_HD void arena_reset_to_random_kickoff(const ArenaStatic &st, Arena &a,
                                         bool has_seed, uint64_t seed) {
  using namespace consts;
  const CarSpawnPos *kickoff_locs;
  int num_kickoff = car::spawn::kickoff_locations(st.mode, kickoff_locs);
  const CarSpawnPos *respawn_locs;
  int num_respawn = car::spawn::respawn_locations(st.mode, respawn_locs);
  uint64_t perm[5] = {0, 1, 2, 3, 4};
  Rng seeded{seed};
  Rng &rng = has_seed ? seeded : a.rng;
  rng.shuffle(perm, static_cast<uint64_t>(num_kickoff));
  uint32_t num_blue = 0;
  uint32_t num_orange = 0;
  for (uint32_t c = 0; c < a.num_cars; ++c) {
    if (a.cars[c].info.team == Team::Blue) {
      ++num_blue;
    } else {
      ++num_orange;
    }
  }
  uint32_t at_respawn[4] = {0, 0, 0, 0};
  uint32_t amount = num_blue > num_orange ? num_blue : num_orange;
  World w = arena_world(st, a);
  for (uint32_t i = 0; i < amount; ++i) {
    CarSpawnPos sp;
    if (i < static_cast<uint32_t>(num_kickoff)) {
      sp = kickoff_locs[perm[i]];
    } else {
      uint32_t ri = (i - static_cast<uint32_t>(num_kickoff)) %
                    static_cast<uint32_t>(num_respawn);
      sp = respawn_locs[ri];
      sp.y += 250.0f * static_cast<float>(at_respawn[ri]);
      at_respawn[ri] += 1;
    }
    CarState spawn = car_state_from_core(
        car_core_grounded({{sp.x, sp.y, car::spawn::rest_z},
                           Mat3A::identity(),
                           Vec3A::zero(),
                           Vec3A::zero()},
                          a.mutators.car_spawn_boost_amount));
    spawn.cache.is_on_ground = true;
    for (int t = 0; t < 2; ++t) {
      Team team = t == 0 ? Team::Blue : Team::Orange;
      bool is_blue = t == 0;
      uint32_t team_count = 0;
      int32_t car_idx = -1;
      for (uint32_t c = 0; c < a.num_cars; ++c) {
        if (a.cars[c].info.team == team) {
          ++team_count;
          if (team_count == i + 1) {
            car_idx = static_cast<int32_t>(a.cars[c].info.idx);
            break;
          }
        }
      }
      if (car_idx < 0) {
        continue;
      }
      float yaw;
      if (is_blue) {
        yaw = sp.yaw_ang;
      } else {
        spawn.phys.pos *= Vec3A{-1.0f, -1.0f, 1.0f};
        yaw = sp.yaw_ang + k_pi;
      }
      spawn.phys.rot_mat = mat3a_from_euler_zyx(yaw, 0.0f, 0.0f);
      Car &car = a.cars[car_idx];
      car_set_state(car, w.dyn_body(car.rigid_body_idx), spawn);
      car.vehicle.reset_transient_contacts();
      car.sticky_gate_prev = false;
    }
  }
  BallState ball_state = ball_state_default();
  if (st.mode == GameMode::Heatseeker) {
    bool next_rand = rng.gen_bool();
    float y_sign = static_cast<float>((next_rand ? 1 : 0) * 2 - 1);
    Vec3A scale{-y_sign, y_sign, 1.0f};
    ball_state.phys.pos = heatseeker::ball_start_pos() * scale;
    ball_state.phys.vel = heatseeker::ball_start_vel() * scale;
  } else if (st.mode == GameMode::Snowday) {
    ball_state.phys.vel.z = k_f32_epsilon;
  } else if (st.mode == GameMode::Dropshot) {
    TileStates fresh{};
    arena_set_tile_states(a, fresh);
  }
  arena_set_ball_state(st, a, ball_state);
  if (st.has_pads) {
    arena_reset_pads(st, a);
  }
}

}
