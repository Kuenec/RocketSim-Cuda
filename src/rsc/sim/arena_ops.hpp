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
  a.pad_lists_unchecked = false;
  a.touched_pads = 0;
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

RS_HD void world_update_body_properties(World &w, uint32_t body_idx,
                                        const RigidBodyUpdateInfo &props) {
  rigid_body_update_physical_properties(w.dyn_body(body_idx), props);
  if (props.invalidate_contacts) {
    dispatcher_clear(*w.a);
  }
  collision_update_aabbs(w);
}

RS_HD void make_ball_collision_shape(GameMode mode,
                                     const MutatorConfig &mutators,
                                     ArenaPhysics &phys, Shape &shape,
                                     Vec3A &inertia) {
  using namespace consts;
  if (mode == GameMode::Snowday) {
    float ang_step =
        k_tau / static_cast<float>(snowday::puck_circle_point_amount);
    float cur_ang = 0.0f;
    const Vec3A neg_z{1.0f, 1.0f, -1.0f};
    uint32_t n = 0;
    for (int i = 0; i < snowday::puck_circle_point_amount; ++i) {
      Vec3A point = Vec3A{cosf_glibc(cur_ang) * mutators.ball_radius,
                          sinf_glibc(cur_ang) * mutators.ball_radius,
                          snowday::puck_height / 2.0f} *
                    uu_to_bt;
      cur_ang += ang_step;
      phys.puck_hull_points[n++] = point;
      phys.puck_hull_points[n++] = point * neg_z;
    }
    phys.puck_hull_count = n;
    HullView view{phys.puck_hull_points, n};
    ConvexHullShape hull = ConvexHullShape::make(HullSource::Arena, 0, view);
    inertia = ConvexHullShape::calculate_local_inertia(view, hull.margin,
                                                       mutators.ball_mass);
    shape = Shape::make_hull(hull);
    return;
  }
  SphereShape sphere{mutators.ball_radius * uu_to_bt};
  inertia = sphere.calculate_local_inertia(mutators.ball_mass);
  shape = Shape::make_sphere(sphere.radius);
}

RS_HD Shape make_car_shape(const CarBodyConfig &config, float car_mass,
                           Vec3A &inertia) {
  BoxShape child = BoxShape::make(config.hitbox_size * consts::uu_to_bt * 0.5f);
  inertia = child.calculate_local_inertia(car_mass);
  Affine3A offset{Mat3A::identity(),
                  config.hitbox_pos_offset * consts::uu_to_bt};
  return Shape::make_compound(CompoundShape::make(child, offset));
}

RS_HD void apply_suspension_strengths(Car &car, float inv_mass) {
  const CarBodyConfig &config = car.info.config;
  float strengths[4];
  car_suspension_strengths(
      inv_mass,
      config.front_wheels.connection_point_offset.x * consts::uu_to_bt,
      config.back_wheels.connection_point_offset.x * consts::uu_to_bt,
      strengths);
  for (int i = 0; i < k_num_wheels; ++i) {
    car.vehicle.wheels[i].suspension_force_scale = strengths[i];
  }
}

struct MutatorOp {
  MutatorConfig mutators;
  float ball_damping_factor;
  float car_damping_factor;
};

struct PadOp {
  float cooldown;
  uint32_t num_touching;
  uint8_t touching[k_max_cars];
};

struct KickoffOp {
  bool has_seed;
  uint64_t seed;
};

enum class ArenaOpKind : uint8_t {
  SetCarCore,
  SetCarFull,
  SetBall,
  SetPad,
  SetTiles,
  SetMutators,
  Kickoff,
  SetControls
};

struct ArenaOp {
  ArenaOpKind kind;
  uint32_t index;
  union {
    CarState car;
    BallState ball;
    PadOp pad;
    TileStates tiles;
    MutatorOp mutators;
    KickoffOp kickoff;
    CarControls controls;
  };
};

RS_HD void arena_set_mutator_config(const ArenaStatic &st, Arena &a,
                                    const MutatorOp &op) {
  using namespace consts;
  const MutatorConfig &mutators = op.mutators;
  const MutatorConfig old = a.mutators;
  bool car_mass_changed = old.car_mass != mutators.car_mass;
  bool ball_properties_changed = old.ball_radius != mutators.ball_radius ||
                                 old.ball_mass != mutators.ball_mass ||
                                 old.ball_drag != mutators.ball_drag;
  bool geometry_changed = old.ball_radius != mutators.ball_radius ||
                          old.ball_mass != mutators.ball_mass ||
                          car_mass_changed;
  World w = arena_world(st, a);
  if (old.gravity != mutators.gravity) {
    a.phys.gravity = mutators.gravity * uu_to_bt;
  }
  if (car_mass_changed) {
    for (uint32_t c = 0; c < a.num_cars; ++c) {
      Car &car = a.cars[c];
      RigidBodyUpdateInfo props;
      props.shape = make_car_shape(car.info.config, mutators.car_mass,
                                   props.local_inertia);
      props.mass = mutators.car_mass;
      props.friction = car::base_coefs.friction;
      props.restitution = car::base_coefs.restitution;
      props.linear_damping = 0.0f;
      props.damping_factor = op.car_damping_factor;
      props.invalidate_contacts = geometry_changed;
      world_update_body_properties(w, car.rigid_body_idx, props);
      apply_suspension_strengths(car, 1.0f / mutators.car_mass);
      car.vel_impulse_cache = Vec3A::zero();
    }
  }
  if (ball_properties_changed) {
    RigidBodyUpdateInfo props;
    make_ball_collision_shape(st.mode, mutators, a.phys, props.shape,
                              props.local_inertia);
    props.mass = mutators.ball_mass;
    if (st.mode == GameMode::Snowday) {
      props.friction = snowday::puck_coefs.friction;
      props.restitution = snowday::puck_coefs.restitution;
    } else {
      props.friction = ball::coefs.friction;
      props.restitution = ball::coefs.restitution;
    }
    props.linear_damping = mutators.ball_drag;
    props.damping_factor = op.ball_damping_factor;
    props.invalidate_contacts = geometry_changed;
    world_update_body_properties(w, a.ball.rigid_body_idx, props);
  }
  a.mutators = mutators;
}

RS_HD void arena_set_pad_state(const ArenaStatic &st, Arena &a, uint32_t idx,
                               const PadOp &op) {
  PadState &pad = a.pads[idx];
  a.pad_lists_unchecked = true;
  if (op.num_touching > 0) {
    a.touched_pads |= pad_bit(idx);
  } else {
    a.touched_pads &= ~pad_bit(idx);
  }
  pad.num_touching = op.num_touching;
  for (uint32_t i = 0; i < op.num_touching; ++i) {
    pad.touching[i] = op.touching[i];
  }
  if (op.cooldown > 0.0f) {
    float max_cooldown =
        pad_max_cooldown(st.pads.configs[idx].is_big, a.mutators);
    float since = max_f32_c(max_cooldown - op.cooldown, 0.0f);
    int64_t ticks =
        sat_f32_to_i64(round_half_away_f32(since * consts::tick_rate));
    pad.has_gave_tick = true;
    pad.gave_tick = static_cast<int64_t>(a.tick_count) - ticks;
  } else {
    pad.has_gave_tick = false;
    pad.gave_tick = 0;
  }
}

RS_HD void arena_apply_op(const ArenaStatic &st, Arena &a, const ArenaOp &op) {
  switch (op.kind) {
  case ArenaOpKind::SetCarCore:
    arena_set_car_state(st, a, op.index, op.car);
    break;
  case ArenaOpKind::SetCarFull:
    arena_set_car_state_full(st, a, op.index, op.car);
    break;
  case ArenaOpKind::SetBall:
    arena_set_ball_state(st, a, op.ball);
    break;
  case ArenaOpKind::SetPad:
    arena_set_pad_state(st, a, op.index, op.pad);
    break;
  case ArenaOpKind::SetTiles:
    arena_set_tile_states(a, op.tiles);
    break;
  case ArenaOpKind::SetMutators:
    arena_set_mutator_config(st, a, op.mutators);
    break;
  case ArenaOpKind::Kickoff:
    arena_reset_to_random_kickoff(st, a, op.kickoff.has_seed, op.kickoff.seed);
    break;
  case ArenaOpKind::SetControls:
    a.cars[op.index].state.controls = op.controls;
    break;
  }
}

}
