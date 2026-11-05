#pragma once

#include <cstdint>

#include "rocketsim_cuda/rocketsim_cuda.hpp"
#include "rsc/sim/arena_ops.hpp"

namespace rocketsim_cuda::convert {

RS_HD rsc::Vec3A to_internal(Vec3 v) { return {v.x, v.y, v.z}; }

RS_HD Vec3 to_public(rsc::Vec3A v) { return {v.x, v.y, v.z}; }

RS_HD rsc::Mat3A to_internal(const RotMat &m) {
  return rsc::Mat3A::from_cols(to_internal(m.forward), to_internal(m.right),
                               to_internal(m.up));
}

RS_HD RotMat to_public(const rsc::Mat3A &m) {
  return {to_public(m.x_axis), to_public(m.y_axis), to_public(m.z_axis)};
}

RS_HD rsc::GameMode to_internal(GameMode m) {
  return static_cast<rsc::GameMode>(static_cast<uint8_t>(m));
}

RS_HD rsc::Team to_internal(Team t) {
  return t == Team::Blue ? rsc::Team::Blue : rsc::Team::Orange;
}

RS_HD Team to_public(rsc::Team t) {
  return t == rsc::Team::Blue ? Team::Blue : Team::Orange;
}

RS_HD rsc::PhysState to_internal(const PhysState &p) {
  return {to_internal(p.pos), to_internal(p.rot_mat), to_internal(p.vel),
          to_internal(p.ang_vel)};
}

RS_HD PhysState to_public(const rsc::PhysState &p) {
  return {to_public(p.pos), to_public(p.rot_mat), to_public(p.vel),
          to_public(p.ang_vel)};
}

RS_HD rsc::CarControls to_internal(const CarControls &c) {
  return {c.throttle, c.steer, c.pitch, c.yaw,
          c.roll,     c.jump,  c.boost, c.handbrake};
}

RS_HD CarControls to_public(const rsc::CarControls &c) {
  return {c.throttle, c.steer, c.pitch, c.yaw,
          c.roll,     c.jump,  c.boost, c.handbrake};
}

RS_HD rsc::RaycastHitInfo to_internal(const RaycastHitInfo &h) {
  return {to_internal(h.hit_point), to_internal(h.hit_normal), h.hit_fraction,
          static_cast<rsc::UserInfoType>(static_cast<uint8_t>(h.user_info))};
}

RS_HD RaycastHitInfo to_public(const rsc::RaycastHitInfo &h) {
  return {to_public(h.hit_point), to_public(h.hit_normal), h.hit_fraction,
          static_cast<UserInfoType>(static_cast<uint8_t>(h.user_info))};
}

RS_HD int32_t optional_index(bool has, uint32_t idx) {
  return has ? static_cast<int32_t>(idx) : -1;
}

RS_HD rsc::CarState to_internal(const CarState &s) {
  rsc::CarState r;
  r.phys = to_internal(s.phys);
  r.controls = to_internal(s.controls);
  r.has_jumped = s.has_jumped;
  r.has_double_jumped = s.has_double_jumped;
  r.has_flipped = s.has_flipped;
  r.flip_rel_torque = to_internal(s.flip_rel_torque);
  r.jump_ticks = s.jump_ticks;
  r.flip_time = s.flip_time;
  r.is_flipping = s.is_flipping;
  r.is_jumping = s.is_jumping;
  r.jump_hold_broken = s.jump_hold_broken;
  r.air_time = s.air_time;
  r.air_time_since_jump = s.air_time_since_jump;
  r.boost = s.boost;
  r.handbrake_val = s.handbrake_val;
  r.is_supersonic = s.is_supersonic;
  r.supersonic_grace_timer = s.supersonic_grace_timer;
  r.time_since_boosted = s.time_since_boosted;
  r.boosting_time = s.boosting_time;
  r.is_boosting = s.is_boosting;
  r.is_auto_flipping = s.is_auto_flipping;
  r.auto_flip_timer = s.auto_flip_timer;
  r.auto_flip_torque_scale = s.auto_flip_torque_scale;
  r.bump_cooldown_timer = s.bump_cooldown_timer;
  r.bump_cooldown_victim =
      optional_index(s.has_bump_cooldown_victim, s.bump_cooldown_victim);
  r.has_last_extra_hit_tick = s.has_last_extra_hit_tick;
  r.last_extra_hit_tick = s.last_extra_hit_tick;
  r.is_demoed = s.is_demoed;
  r.demo_respawn_timer = s.demo_respawn_timer;
  r.cache.is_on_ground = s.cache.is_on_ground;
  for (int i = 0; i < 4; ++i) {
    r.cache.wheel_has_contact[i] = s.cache.has_wheel_contact[i];
    r.cache.wheels_with_contact[i] =
        to_internal(s.cache.wheels_with_contact[i]);
  }
  r.cache.has_world_contact_normal = s.cache.has_world_contact_normal;
  r.cache.world_contact_normal = to_internal(s.cache.world_contact_normal);
  r.cache.prev_controls = to_internal(s.cache.prev_controls);
  r.cache.world_wheels_prev = s.cache.world_wheels_prev;
  r.cache.demo_tick = s.cache.demo_tick;
  r.cache.demoer_idx = optional_index(s.cache.has_demoer, s.cache.demoer_idx);
  return r;
}

RS_HD CarState to_public(const rsc::CarState &r) {
  CarState s{};
  s.phys = to_public(r.phys);
  s.controls = to_public(r.controls);
  s.has_jumped = r.has_jumped;
  s.has_double_jumped = r.has_double_jumped;
  s.has_flipped = r.has_flipped;
  s.flip_rel_torque = to_public(r.flip_rel_torque);
  s.jump_ticks = r.jump_ticks;
  s.flip_time = r.flip_time;
  s.is_flipping = r.is_flipping;
  s.is_jumping = r.is_jumping;
  s.jump_hold_broken = r.jump_hold_broken;
  s.air_time = r.air_time;
  s.air_time_since_jump = r.air_time_since_jump;
  s.boost = r.boost;
  s.handbrake_val = r.handbrake_val;
  s.is_supersonic = r.is_supersonic;
  s.supersonic_grace_timer = r.supersonic_grace_timer;
  s.time_since_boosted = r.time_since_boosted;
  s.boosting_time = r.boosting_time;
  s.is_boosting = r.is_boosting;
  s.is_auto_flipping = r.is_auto_flipping;
  s.auto_flip_timer = r.auto_flip_timer;
  s.auto_flip_torque_scale = r.auto_flip_torque_scale;
  s.bump_cooldown_timer = r.bump_cooldown_timer;
  s.has_bump_cooldown_victim = r.bump_cooldown_victim >= 0;
  s.bump_cooldown_victim = r.bump_cooldown_victim >= 0
                               ? static_cast<uint32_t>(r.bump_cooldown_victim)
                               : 0;
  s.has_last_extra_hit_tick = r.has_last_extra_hit_tick;
  s.last_extra_hit_tick = r.last_extra_hit_tick;
  s.is_demoed = r.is_demoed;
  s.demo_respawn_timer = r.demo_respawn_timer;
  s.cache.is_on_ground = r.cache.is_on_ground;
  for (int i = 0; i < 4; ++i) {
    s.cache.has_wheel_contact[i] = r.cache.wheel_has_contact[i];
    s.cache.wheels_with_contact[i] = to_public(r.cache.wheels_with_contact[i]);
  }
  s.cache.has_world_contact_normal = r.cache.has_world_contact_normal;
  s.cache.world_contact_normal = to_public(r.cache.world_contact_normal);
  s.cache.prev_controls = to_public(r.cache.prev_controls);
  s.cache.world_wheels_prev = r.cache.world_wheels_prev;
  s.cache.demo_tick = r.cache.demo_tick;
  s.cache.has_demoer = r.cache.demoer_idx >= 0;
  s.cache.demoer_idx =
      r.cache.demoer_idx >= 0 ? static_cast<uint32_t>(r.cache.demoer_idx) : 0;
  return s;
}

RS_HD rsc::CarState core_to_internal(const CarCore &c) {
  rsc::CarState r = rsc::car_state_default();
  r.phys = to_internal(c.phys);
  r.controls = to_internal(c.controls);
  r.has_jumped = c.has_jumped;
  r.has_double_jumped = c.has_double_jumped;
  r.has_flipped = c.has_flipped;
  r.flip_rel_torque = to_internal(c.flip_rel_torque);
  r.jump_ticks = c.jump_ticks;
  r.flip_time = c.flip_time;
  r.is_flipping = c.is_flipping;
  r.is_jumping = c.is_jumping;
  r.jump_hold_broken = c.jump_hold_broken;
  r.air_time_since_jump = c.air_time_since_jump;
  r.boost = c.boost;
  r.is_auto_flipping = c.is_auto_flipping;
  r.auto_flip_timer = c.auto_flip_timer;
  r.auto_flip_torque_scale = c.auto_flip_torque_scale;
  r.bump_cooldown_timer = c.bump_cooldown_timer;
  r.bump_cooldown_victim =
      optional_index(c.has_bump_cooldown_victim, c.bump_cooldown_victim);
  r.has_last_extra_hit_tick = c.has_last_extra_hit_tick;
  r.last_extra_hit_tick = c.last_extra_hit_tick;
  r.is_demoed = c.is_demoed;
  r.demo_respawn_timer = c.demo_respawn_timer;
  return r;
}

RS_HD rsc::BallState to_internal(const BallState &b) {
  rsc::BallState r;
  r.phys = to_internal(b.phys);
  r.hs_info = {b.hs_info.y_target_dir, b.hs_info.cur_target_speed,
               b.hs_info.time_since_hit};
  r.ds_info = {b.ds_info.charge_level, b.ds_info.accumulated_hit_force,
               b.ds_info.y_target_dir, b.ds_info.has_last_damage_tick,
               b.ds_info.last_damage_tick};
  r.tick_count_since_kickoff = b.tick_count_since_kickoff;
  return r;
}

RS_HD BallState to_public(const rsc::BallState &r) {
  BallState b{};
  b.phys = to_public(r.phys);
  b.hs_info = {r.hs_info.y_target_dir, r.hs_info.cur_target_speed,
               r.hs_info.time_since_hit};
  b.ds_info = {r.ds_info.charge_level, r.ds_info.accumulated_hit_force,
               r.ds_info.y_target_dir, r.ds_info.has_last_damage_tick,
               r.ds_info.last_damage_tick};
  b.tick_count_since_kickoff = r.tick_count_since_kickoff;
  return b;
}

RS_HD rsc::MutatorConfig to_internal(const MutatorConfig &m) {
  rsc::MutatorConfig r;
  r.gravity = to_internal(m.gravity);
  r.car_mass = m.car_mass;
  r.ball_mass = m.ball_mass;
  r.ball_max_speed = m.ball_max_speed;
  r.ball_drag = m.ball_drag;
  r.jump_accel = m.jump_accel;
  r.jump_immediate_force = m.jump_immediate_force;
  r.boost_accel_ground = m.boost_accel_ground;
  r.boost_accel_air = m.boost_accel_air;
  r.boost_used_per_second = m.boost_used_per_second;
  r.respawn_delay = m.respawn_delay;
  r.bump_cooldown_time = m.bump_cooldown_time;
  r.car_max_boost_amount = m.car_max_boost_amount;
  r.car_spawn_boost_amount = m.car_spawn_boost_amount;
  r.boost_pad_amount_small = m.boost_pad_amount_small;
  r.boost_pad_amount_big = m.boost_pad_amount_big;
  r.boost_pad_cooldown_big = m.boost_pad_cooldown_big;
  r.boost_pad_cooldown_small = m.boost_pad_cooldown_small;
  r.ball_hit_extra_force_scale = m.ball_hit_extra_force_scale;
  r.bump_force_scale = m.bump_force_scale;
  r.ball_radius = m.ball_radius;
  r.unlimited_flips = m.unlimited_flips;
  r.unlimited_double_jumps = m.unlimited_double_jumps;
  r.recharge_boost_enabled = m.recharge_boost_enabled;
  r.recharge_boost_per_second = m.recharge_boost_per_second;
  r.recharge_boost_delay = m.recharge_boost_delay;
  r.demo_mode = static_cast<rsc::DemoMode>(static_cast<uint8_t>(m.demo_mode));
  r.enable_team_demos = m.enable_team_demos;
  r.goal_base_threshold_y = m.goal_base_threshold_y;
  r.boost_pads_enabled = m.boost_pads_enabled;
  return r;
}

RS_HD MutatorConfig to_public(const rsc::MutatorConfig &r) {
  MutatorConfig m{};
  m.gravity = to_public(r.gravity);
  m.car_mass = r.car_mass;
  m.ball_mass = r.ball_mass;
  m.ball_max_speed = r.ball_max_speed;
  m.ball_drag = r.ball_drag;
  m.jump_accel = r.jump_accel;
  m.jump_immediate_force = r.jump_immediate_force;
  m.boost_accel_ground = r.boost_accel_ground;
  m.boost_accel_air = r.boost_accel_air;
  m.boost_used_per_second = r.boost_used_per_second;
  m.respawn_delay = r.respawn_delay;
  m.bump_cooldown_time = r.bump_cooldown_time;
  m.car_max_boost_amount = r.car_max_boost_amount;
  m.car_spawn_boost_amount = r.car_spawn_boost_amount;
  m.boost_pad_amount_small = r.boost_pad_amount_small;
  m.boost_pad_amount_big = r.boost_pad_amount_big;
  m.boost_pad_cooldown_big = r.boost_pad_cooldown_big;
  m.boost_pad_cooldown_small = r.boost_pad_cooldown_small;
  m.ball_hit_extra_force_scale = r.ball_hit_extra_force_scale;
  m.bump_force_scale = r.bump_force_scale;
  m.ball_radius = r.ball_radius;
  m.unlimited_flips = r.unlimited_flips;
  m.unlimited_double_jumps = r.unlimited_double_jumps;
  m.recharge_boost_enabled = r.recharge_boost_enabled;
  m.recharge_boost_per_second = r.recharge_boost_per_second;
  m.recharge_boost_delay = r.recharge_boost_delay;
  m.demo_mode = static_cast<DemoMode>(static_cast<uint8_t>(r.demo_mode));
  m.enable_team_demos = r.enable_team_demos;
  m.goal_base_threshold_y = r.goal_base_threshold_y;
  m.boost_pads_enabled = r.boost_pads_enabled;
  return m;
}

template <int N>
RS_HD rsc::LinearPieceCurve<N>
curve_to_internal(const CurvePoint (&points)[N]) {
  float raw[N][2];
  for (int i = 0; i < N; ++i) {
    raw[i][0] = points[i].x;
    raw[i][1] = points[i].y;
  }
  return rsc::LinearPieceCurve<N>::make(raw);
}

template <int N>
RS_HD void curve_to_public(const rsc::LinearPieceCurve<N> &c,
                           CurvePoint (&out)[N]) {
  for (int i = 0; i < N; ++i) {
    out[i] = {c.curve[i].max_x, c.curve[i].max_y};
  }
}

RS_HD rsc::WheelPairConfig to_internal(const WheelPairConfig &w) {
  return {w.wheel_radius, w.suspension_rest_length,
          to_internal(w.connection_point_offset)};
}

RS_HD WheelPairConfig to_public(const rsc::WheelPairConfig &w) {
  return {w.wheel_radius, w.suspension_rest_length,
          to_public(w.connection_point_offset)};
}

RS_HD rsc::CarBodyConfig to_internal(const CarBodyConfig &c) {
  rsc::CarBodyConfig r;
  r.hitbox_size = to_internal(c.hitbox_size);
  r.hitbox_pos_offset = to_internal(c.hitbox_pos_offset);
  r.front_wheels = to_internal(c.front_wheels);
  r.back_wheels = to_internal(c.back_wheels);
  r.three_wheels = c.three_wheels;
  r.dodge_deadzone = c.dodge_deadzone;
  r.steer_angle_curve = curve_to_internal(c.steer_angle_curve);
  r.powerslide_steer_angle_curve =
      curve_to_internal(c.powerslide_steer_angle_curve);
  return r;
}

RS_HD CarBodyConfig to_public(const rsc::CarBodyConfig &r) {
  CarBodyConfig c{};
  c.hitbox_size = to_public(r.hitbox_size);
  c.hitbox_pos_offset = to_public(r.hitbox_pos_offset);
  c.front_wheels = to_public(r.front_wheels);
  c.back_wheels = to_public(r.back_wheels);
  c.three_wheels = r.three_wheels;
  c.dodge_deadzone = r.dodge_deadzone;
  curve_to_public(r.steer_angle_curve, c.steer_angle_curve);
  curve_to_public(r.powerslide_steer_angle_curve,
                  c.powerslide_steer_angle_curve);
  return c;
}

RS_HD TileStates to_public(const rsc::TileStates &t) {
  TileStates r{};
  for (int team = 0; team < 2; ++team) {
    for (uint32_t i = 0; i < k_tiles_per_team; ++i) {
      r.states[team][i] = static_cast<TileDamageState>(t.states[team][i]);
    }
  }
  return r;
}

RS_HD rsc::TileStates to_internal(const TileStates &t) {
  rsc::TileStates r{};
  for (int team = 0; team < 2; ++team) {
    for (uint32_t i = 0; i < k_tiles_per_team; ++i) {
      r.states[team][i] = static_cast<uint8_t>(t.states[team][i]);
    }
  }
  return r;
}

RS_HD EventCounts to_public(const rsc::ArenaEventCounters &c) {
  EventCounts r{};
  r.ball_hit_world = c.ball_hit_world;
  r.goal_scored = c.goal_scored;
  r.goal_tick = c.goal_tick;
  for (uint32_t i = 0; i < k_max_cars; ++i) {
    const rsc::CarEventCounters &s = c.cars[i];
    r.cars[i] = {s.hit_ball,      s.hit_world,      s.bumps_dealt,
                 s.demos_dealt,   s.bumps_received, s.demos_received,
                 s.boost_pickups, s.landed};
  }
  return r;
}

RS_HD ArenaInfo export_info(const rsc::ArenaStatic &st, const rsc::Arena &a) {
  return {a.tick_count, a.rng.state, a.phys.error_flags, a.num_events,
          rsc::arena_is_ball_scored(st, a)};
}

RS_HD PadStateView export_pad(const rsc::ArenaStatic &st, const rsc::Arena &a,
                              uint32_t pad) {
  const rsc::PadState &p = a.pads[pad];
  PadStateView v{};
  v.cooldown = 0.0f;
  if (p.has_gave_tick) {
    float max_cooldown =
        rsc::pad_max_cooldown(st.pads.configs[pad].is_big, a.mutators);
    float since =
        static_cast<float>(static_cast<int64_t>(a.tick_count) - p.gave_tick) *
        rsc::consts::tick_time;
    v.cooldown = rsc::max_f32_c(max_cooldown - since, 0.0f);
  }
  v.num_touching = p.num_touching;
  for (uint32_t i = 0; i < p.num_touching; ++i) {
    v.touching[i] = p.touching[i];
  }
  return v;
}

RS_HD void export_arena(const rsc::ArenaStatic &st, const rsc::Arena &a,
                        uint32_t arena, const StateBuffers &out) {
  uint32_t num_cars = a.num_cars;
  uint32_t num_pads = st.has_pads ? st.pads.num_pads : 0;
  if (out.infos) {
    out.infos[arena] = export_info(st, a);
  }
  if (out.balls) {
    out.balls[arena] = to_public(a.ball.state);
  }
  if (out.cars) {
    for (uint32_t c = 0; c < num_cars; ++c) {
      out.cars[static_cast<size_t>(arena) * num_cars + c] =
          to_public(a.cars[c].state);
    }
  }
  if (out.pads) {
    for (uint32_t p = 0; p < num_pads; ++p) {
      out.pads[static_cast<size_t>(arena) * num_pads + p] =
          export_pad(st, a, p);
    }
  }
  if (out.tiles && a.has_tiles) {
    out.tiles[arena] = to_public(a.tiles);
  }
  if (out.event_counts) {
    out.event_counts[arena] = to_public(a.counters);
  }
}

RS_HD ArenaEvent to_public(const rsc::ArenaEvent &e,
                           const rsc::LandedWheels &landed) {
  ArenaEvent r{};
  r.kind = static_cast<ArenaEventKind>(static_cast<uint8_t>(e.kind));
  switch (e.kind) {
  case rsc::ArenaEventKind::BallHitWorld:
    r.contact_point = to_public(e.point);
    r.contact_normal = to_public(e.vec);
    break;
  case rsc::ArenaEventKind::CarHitBall:
    r.car_idx = e.car_idx;
    r.contact_point = to_public(e.point);
    r.extra_hit_vel = to_public(e.vec);
    break;
  case rsc::ArenaEventKind::CarHitCar:
    r.car_idx = e.car_idx;
    r.victim_car_idx = e.other_idx;
    r.is_demo = e.is_demo;
    r.contact_point = to_public(e.point);
    break;
  case rsc::ArenaEventKind::CarHitWorld:
    r.car_idx = e.car_idx;
    r.contact_point = to_public(e.point);
    r.contact_normal = to_public(e.vec);
    break;
  case rsc::ArenaEventKind::CarPickupBoost:
    r.car_idx = e.car_idx;
    r.boost_pad_idx = e.other_idx;
    break;
  case rsc::ArenaEventKind::CarLanded:
    r.car_idx = e.car_idx;
    for (int w = 0; w < 4; ++w) {
      r.has_wheel_contact[w] = landed.has_contact[w];
      r.wheels[w] = to_public(landed.wheels[w]);
    }
    break;
  }
  return r;
}

}
