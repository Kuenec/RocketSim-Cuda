#pragma once

#include <cstdint>

#include "rsc/bullet/vehicle.hpp"
#include "rsc/bullet/world_types.hpp"
#include "rsc/math/rng.hpp"
#include "rsc/sim/consts.hpp"
#include "rsc/sim/quantize.hpp"

namespace rsc {

struct CarControls {
  float throttle, steer, pitch, yaw, roll;
  bool jump, boost, handbrake;

  RS_HD CarControls clamped() const {
    CarControls c = *this;
    c.throttle = clamp_f32(c.throttle, -1.0f, 1.0f);
    c.steer = clamp_f32(c.steer, -1.0f, 1.0f);
    c.pitch = clamp_f32(c.pitch, -1.0f, 1.0f);
    c.yaw = clamp_f32(c.yaw, -1.0f, 1.0f);
    c.roll = clamp_f32(c.roll, -1.0f, 1.0f);
    return c;
  }

  RS_HDI Vec3A pyr() const { return {pitch, yaw, roll}; }
};

struct WheelPairConfig {
  float wheel_radius;
  float suspension_rest_length;
  Vec3A connection_point_offset;
};

struct CarBodyConfig {
  Vec3A hitbox_size;
  Vec3A hitbox_pos_offset;
  WheelPairConfig front_wheels;
  WheelPairConfig back_wheels;
  bool three_wheels;
  float dodge_deadzone;
  LinearPieceCurve<6> steer_angle_curve;
  LinearPieceCurve<2> powerslide_steer_angle_curve;

  RS_HD float wheel_ray_min_hit_dist_sq() const {
    float d = (hitbox_size.z * 0.5f - 2.25f) * consts::uu_to_bt;
    return d * d;
  }
};

struct RaycastHitInfo {
  Vec3A hit_point;
  Vec3A hit_normal;
  float hit_fraction;
  UserInfoType user_info;
};

struct PhysState {
  Vec3A pos;
  Mat3A rot_mat;
  Vec3A vel;
  Vec3A ang_vel;
};

struct CarCache {
  bool is_on_ground;
  bool wheel_has_contact[4];
  RaycastHitInfo wheels_with_contact[4];
  bool has_world_contact_normal;
  Vec3A world_contact_normal;
  CarControls prev_controls;
  uint8_t world_wheels_prev;
  uint64_t demo_tick;
  int32_t demoer_idx;
};

struct CarState {
  PhysState phys;
  CarControls controls;
  bool has_jumped;
  bool has_double_jumped;
  bool has_flipped;
  Vec3A flip_rel_torque;
  uint32_t jump_ticks;
  float flip_time;
  bool is_flipping;
  bool is_jumping;
  bool jump_hold_broken;
  float air_time;
  float air_time_since_jump;
  float boost;
  float handbrake_val;
  bool is_supersonic;
  float supersonic_grace_timer;
  float time_since_boosted;
  float boosting_time;
  bool is_boosting;
  bool is_auto_flipping;
  float auto_flip_timer;
  float auto_flip_torque_scale;
  float bump_cooldown_timer;
  int32_t bump_cooldown_victim;
  bool has_last_extra_hit_tick;
  uint64_t last_extra_hit_tick;
  bool is_demoed;
  float demo_respawn_timer;
  CarCache cache;

  RS_HD int num_wheels_in_contact() const {
    int n = 0;
    for (int i = 0; i < 4; ++i) {
      n += cache.wheel_has_contact[i] ? 1 : 0;
    }
    return n;
  }

  RS_HDI Vec3A forward_dir() const { return phys.rot_mat.x_axis; }
  RS_HDI Vec3A right_dir() const { return phys.rot_mat.y_axis; }
  RS_HDI Vec3A up_dir() const { return phys.rot_mat.z_axis; }
};

RS_HD CarControls car_controls_default() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false, false, false};
}

RS_HD CarState car_state_default() {
  CarState s;
  s.phys = {{0.0f, 0.0f, consts::car::spawn::rest_z},
            Mat3A::identity(),
            Vec3A::zero(),
            Vec3A::zero()};
  s.controls = car_controls_default();
  s.has_jumped = false;
  s.has_double_jumped = false;
  s.has_flipped = false;
  s.flip_rel_torque = Vec3A::zero();
  s.jump_ticks = 0;
  s.flip_time = 0.0f;
  s.is_flipping = false;
  s.is_jumping = false;
  s.jump_hold_broken = false;
  s.air_time = 0.0f;
  s.air_time_since_jump = 0.0f;
  s.boost = consts::car::boost::spawn_amount;
  s.handbrake_val = 0.0f;
  s.is_supersonic = false;
  s.supersonic_grace_timer = 0.0f;
  s.time_since_boosted = 0.0f;
  s.boosting_time = 0.0f;
  s.is_boosting = false;
  s.is_auto_flipping = false;
  s.auto_flip_timer = 0.0f;
  s.auto_flip_torque_scale = 0.0f;
  s.bump_cooldown_timer = 0.0f;
  s.bump_cooldown_victim = -1;
  s.has_last_extra_hit_tick = false;
  s.last_extra_hit_tick = 0;
  s.is_demoed = false;
  s.demo_respawn_timer = 0.0f;
  s.cache.is_on_ground = true;
  for (int i = 0; i < 4; ++i) {
    s.cache.wheel_has_contact[i] = false;
    s.cache.wheels_with_contact[i] = {Vec3A::zero(), Vec3A::zero(), 0.0f,
                                      UserInfoType::None};
  }
  s.cache.has_world_contact_normal = false;
  s.cache.world_contact_normal = Vec3A::zero();
  s.cache.prev_controls = car_controls_default();
  s.cache.world_wheels_prev = 0;
  s.cache.demo_tick = UINT64_MAX;
  s.cache.demoer_idx = -1;
  return s;
}

RS_HD CarState car_state_from_core(CarState core) {
  CarState s = core;
  s.boost = clamp_f32(s.boost, 0.0f, consts::car::boost::max);
  if (s.is_jumping) {
    s.has_jumped = true;
  }
  if (s.is_flipping) {
    s.has_flipped = true;
  }
  constexpr float start = consts::car::supersonic::start_speed;
  s.air_time = 0.0f;
  s.handbrake_val = 0.0f;
  s.is_supersonic = s.phys.vel.length_squared() >= start * start;
  s.supersonic_grace_timer = 0.0f;
  s.time_since_boosted = 0.0f;
  s.boosting_time = 0.0f;
  s.is_boosting = false;
  s.cache.is_on_ground = false;
  for (int i = 0; i < 4; ++i) {
    s.cache.wheel_has_contact[i] = false;
    s.cache.wheels_with_contact[i] = {Vec3A::zero(), Vec3A::zero(), 0.0f,
                                      UserInfoType::None};
  }
  s.cache.has_world_contact_normal = false;
  s.cache.world_contact_normal = Vec3A::zero();
  s.cache.prev_controls = s.controls;
  s.cache.world_wheels_prev = 0;
  s.cache.demo_tick = UINT64_MAX;
  s.cache.demoer_idx = -1;
  return s;
}

RS_HD CarState car_core_grounded(PhysState phys, float boost) {
  CarState s = car_state_default();
  s.phys = phys;
  s.boost = clamp_f32(boost, 0.0f, consts::car::boost::max);
  return s;
}

struct MutatorConfig {
  Vec3A gravity;
  float car_mass;
  float ball_mass;
  float ball_max_speed;
  float ball_drag;
  float jump_accel;
  float jump_immediate_force;
  float boost_accel_ground;
  float boost_accel_air;
  float boost_used_per_second;
  float respawn_delay;
  float bump_cooldown_time;
  float car_max_boost_amount;
  float car_spawn_boost_amount;
  float boost_pad_amount_small;
  float boost_pad_amount_big;
  float boost_pad_cooldown_big;
  float boost_pad_cooldown_small;
  float ball_hit_extra_force_scale;
  float bump_force_scale;
  float ball_radius;
  bool unlimited_flips;
  bool unlimited_double_jumps;
  bool recharge_boost_enabled;
  float recharge_boost_per_second;
  float recharge_boost_delay;
  DemoMode demo_mode;
  bool enable_team_demos;
  float goal_base_threshold_y;
  bool boost_pads_enabled;
};

struct CarInfo {
  uint32_t idx;
  Team team;
  CarBodyConfig config;
};

struct Car {
  CarInfo info;
  VehicleRL vehicle;
  uint32_t rigid_body_idx;
  Vec3A vel_impulse_cache;
  CarState state;
  bool sticky_gate_prev;
};

RS_HD void car_suspension_strengths(float inv_mass, float front_x_bt,
                                    float rear_x_bt, float *out) {
  float mass = 1.0f / inv_mass;
  float front_dist = front_x_bt;
  float rear_dist = -rear_x_bt;
  if (front_dist < 0.0f || rear_dist < 0.0f) {
    float half = mass * 0.5f;
    float quarter = half * 0.5f;
    out[0] = out[1] = out[2] = out[3] = quarter;
    return;
  }
  float total = rear_dist + front_dist;
  float front_frac = front_dist / total;
  float front_axle = front_frac * mass;
  float rear_axle = mass - front_axle;
  float front_wheel = rear_axle * 0.5f;
  float rear_wheel = front_axle * 0.5f;
  out[0] = out[1] = front_wheel;
  out[2] = out[3] = rear_wheel;
}

RS_HD void car_sync_state_to_body(Car &car, RigidBody &rb) {
  rb.set_world_trans(
      {car.state.phys.rot_mat, car.state.phys.pos * consts::uu_to_bt});
  rb.interp_world_trans = rb.world_trans;
  rb.lin_vel = car.state.phys.vel * consts::uu_to_bt;
  rb.ang_vel = car.state.phys.ang_vel;
  rb.clear_accum_vels();
  rb.update_inertia_tensor();
  car.vel_impulse_cache = Vec3A::zero();
}

RS_HD void car_set_state(Car &car, RigidBody &rb, const CarState &state) {
  car.state = state;
  car_sync_state_to_body(car, rb);
}

RS_HD void car_reset_transient_contacts(Car &car) {
  car.vehicle.reset_transient_contacts();
  car.sticky_gate_prev = false;
  car.state.cache.world_wheels_prev = 0;
  car.state.cache.is_on_ground = false;
  for (int i = 0; i < 4; ++i) {
    car.state.cache.wheel_has_contact[i] = false;
    car.state.cache.wheels_with_contact[i] = {Vec3A::zero(), Vec3A::zero(),
                                              0.0f, UserInfoType::None};
  }
  car.state.cache.has_world_contact_normal = false;
  car.state.cache.world_contact_normal = Vec3A::zero();
}

RS_HD void car_respawn(Car &car, RigidBody &rb, Rng &rng, GameMode mode,
                       float boost_amount) {
  const CarSpawnPos *locs;
  int n = consts::car::spawn::respawn_locations(mode, locs);
  uint64_t idx = rng.usize_below(static_cast<uint64_t>(n));
  CarSpawnPos sp = locs[idx];
  float yaw = sp.yaw_ang + (car.info.team == Team::Blue ? 0.0f : k_pi);
  PhysState phys{
      {sp.x, sp.y * -team_y_dir(car.info.team), consts::car::spawn::respawn_z},
      mat3a_from_euler_zyx(yaw, 0.0f, 0.0f),
      Vec3A::zero(),
      Vec3A::zero()};
  CarState ns = car_state_from_core(car_core_grounded(phys, boost_amount));
  ns.cache.is_on_ground = true;
  car_set_state(car, rb, ns);
  car.vehicle.reset_transient_contacts();
  car.sticky_gate_prev = false;
  car.state.cache.world_wheels_prev = 0;
  car.state.cache.demo_tick = UINT64_MAX;
  car.state.cache.demoer_idx = -1;
}

RS_HD void car_update_contact_state(Car &car, const World &w) {
  int num_in_contact = 0;
  bool any_world = false;
  uint8_t world_count = 0;
  for (int i = 0; i < 4; ++i) {
    const WheelInfo &wheel = car.vehicle.wheels[i];
    if (!wheel.has_raycast) {
      car.state.cache.wheel_has_contact[i] = false;
      car.state.cache.wheels_with_contact[i] = {Vec3A::zero(), Vec3A::zero(),
                                                0.0f, UserInfoType::None};
      continue;
    }
    const RaycastInfo &info = wheel.raycast_info;
    if (info.is_in_contact_with_world) {
      any_world = true;
      ++world_count;
    }
    float trace_len = (wheel.hard_point - info.contact_point).length();
    car.state.cache.wheel_has_contact[i] = true;
    car.state.cache.wheels_with_contact[i] = {
        info.contact_point * consts::bt_to_uu, info.contact_normal,
        trace_len / wheel.real_ray_length,
        w.body(info.ground_body_idx).user_idx};
    ++num_in_contact;
  }
  car.state.cache.is_on_ground = num_in_contact >= 3;
  car.sticky_gate_prev = any_world;
  car.state.cache.world_wheels_prev = world_count;
}

RS_HD void car_update_wheels(Car &car, RigidBody &rb, float forward_speed_uu,
                             float quant_throttle, const MutatorConfig &mut,
                             bool &has_cached_up, Vec3A &cached_up) {
  using namespace consts;
  CarState &s = car.state;
  float handbrake_delta =
      (s.controls.handbrake ? car::drive::powerslide_rise_rate
                            : -car::drive::powerslide_fall_rate) *
      tick_time;
  s.handbrake_val = clamp_f32(s.handbrake_val + handbrake_delta, 0.0f, 1.0f);
  float real_brake = 0.0f;
  float raw_throttle = s.controls.throttle;
  float proc_throttle;
  float proc_steer;
  if (is_nan_f32(raw_throttle) || is_nan_f32(s.controls.steer)) {
    Vec3A q = quantize_axis_inputs({raw_throttle, s.controls.steer, 0.0f});
    proc_throttle = q.x;
    proc_steer = q.y;
  } else {
    proc_throttle = quant_throttle;
    proc_steer = quantize_axis_input(s.controls.steer);
  }
  bool all_wheels_contact = s.num_wheels_in_contact() == k_num_wheels;
  float real_throttle =
      (s.controls.boost && s.boost > 0.0f && all_wheels_contact)
          ? 1.0f
          : proc_throttle;
  float abs_speed = abs_f32(forward_speed_uu);
  float engine_throttle = real_throttle;
  if (!s.controls.handbrake) {
    if (abs_f32(real_throttle) > car::drive::throttle_deadzone) {
      if (abs_speed > car::drive::stopping_forward_vel &&
          signum_f32(real_throttle) != signum_f32(forward_speed_uu)) {
        real_brake = 1.0f;
        if (abs_speed > car::drive::braking_no_throttle_speed_thresh) {
          engine_throttle = 0.0f;
        }
      }
    } else {
      engine_throttle = 0.0f;
      real_brake = abs_speed < car::drive::stopping_forward_vel
                       ? 1.0f
                       : car::drive::coasting_brake_factor;
    }
  }
  float drive_speed_scale =
      curves::drive_speed_torque_factor().get_output(abs_speed);
  constexpr float engine_amount = car::drive::throttle_torque_amount * uu_to_bt;
  constexpr float brake_amount = car::drive::brake_torque_amount * uu_to_bt;
  float drive_engine_force =
      engine_throttle * engine_amount * drive_speed_scale;
  float drive_brake_force = real_brake * brake_amount;
  for (int i = 0; i < k_num_wheels; ++i) {
    car.vehicle.wheels[i].engine_force = drive_engine_force;
    car.vehicle.wheels[i].brake = drive_brake_force;
  }
  float steer_angle = car.info.config.steer_angle_curve.get_output(abs_speed);
  if (s.handbrake_val != 0.0f) {
    float ps =
        car.info.config.powerslide_steer_angle_curve.get_output(abs_speed);
    steer_angle += (ps - steer_angle) * s.handbrake_val;
  }
  steer_angle *= proc_steer;
  car.vehicle.wheels[0].steer_angle = steer_angle;
  car.vehicle.wheels[1].steer_angle = steer_angle;
  if (car.sticky_gate_prev) {
    if (!has_cached_up) {
      cached_up = car.vehicle.get_upwards_dir_from_wheel_contacts(rb);
      has_cached_up = true;
    }
    Vec3A upwards_dir = cached_up;
    bool full_stick =
        raw_throttle != 0.0f || abs_speed > car::drive::stopping_forward_vel;
    float sticky_force_scale =
        (car.info.config.three_wheels ? 0.0f : 1.0f) * 0.5f;
    if (full_stick) {
      sticky_force_scale += 1.0f - abs_f32(upwards_dir.z);
    }
    rb.add_linear_impulse(upwards_dir * sticky_force_scale *
                              -abs_f32(mut.gravity.z) * tick_time * uu_to_bt,
                          false, true);
  }
}

RS_HD void car_update_air_torque(Car &car, RigidBody &rb, int num_in_contact,
                                 float quant_throttle, bool prev_is_flipping,
                                 float prev_flip_time) {
  using namespace consts;
  CarState &s = car.state;
  bool allow_dodge = num_in_contact < 3;
  bool allow_air = num_in_contact == 0;
  if (s.is_flipping && allow_dodge && s.flip_rel_torque != Vec3A::zero()) {
    Vec3A rel = s.flip_rel_torque;
    float pitch_scale = 1.0f;
    if (rel.y != 0.0f && s.controls.pitch != 0.0f &&
        signum_f32(rel.y) == signum_f32(s.controls.pitch) &&
        prev_flip_time >= car::flip::pitch_cancel_gate_min_time) {
      pitch_scale =
          1.0f -
          min_f32_c(abs_f32(quantize_axis_input(-s.controls.pitch)), 1.0f);
    }
    Vec3A dodge_torque = round_ties_even_vec3a(
        {rel.x * car::flip::torque_x, rel.y * car::flip::torque_y, 0.0f});
    dodge_torque.y *= pitch_scale;
    dodge_torque *= tick_time;
    rb.add_angular_impulse(rb.world_trans.matrix3 * dodge_torque, false, true);
  }
  if (allow_air) {
    Vec3A inputs = quantize_axis_inputs(s.controls.pyr());
    float pitch_torque_scale = 1.0f;
    if (inputs != Vec3A::zero() &&
        (prev_is_flipping || s.is_flipping ||
         prev_flip_time < car::flip::pitchlock_extra_time)) {
      pitch_torque_scale = 0.0f;
    }
    Mat3A dirs = Mat3A::from_cols(-s.right_dir(), s.up_dir(), -s.forward_dir());
    Vec3A scaled = inputs * Vec3A{pitch_torque_scale, 1.0f, 1.0f};
    Vec3A torque_const{car::air_control::torque_x, car::air_control::torque_y,
                       car::air_control::torque_z};
    Vec3A damping_const{car::air_control::damping_x,
                        car::air_control::damping_y,
                        car::air_control::damping_z};
    Vec3A torque = inputs != Vec3A::zero() ? dirs * (scaled * torque_const)
                                           : Vec3A::zero();
    Vec3A retain = (Vec3A::one() - scaled.abs()).with_z(1.0f);
    Vec3A damping = dirs * (dirs.transpose_mul_vec3a(s.phys.ang_vel) *
                            damping_const * retain);
    Vec3A rb_torque =
        (torque - damping) * car::air_control::torque_apply_scale * tick_time;
    rb.add_angular_impulse(rb_torque, false, true);
  }
  if (allow_air) {
    float throttle_scale = s.is_boosting ? 1.0f : quant_throttle;
    if (throttle_scale != 0.0f) {
      Vec3A force = s.forward_dir() * throttle_scale *
                    car::drive::throttle_air_accel * uu_to_bt * tick_time;
      rb.add_linear_impulse(force, false, true);
    }
  }
}

RS_HD void car_update_jump(Car &car, RigidBody &rb, const MutatorConfig &mut,
                           bool jump_pressed) {
  using namespace consts;
  CarState &s = car.state;
  Vec3A up_dir = s.up_dir();
  if (!s.has_jumped && s.cache.is_on_ground && jump_pressed) {
    s.is_jumping = true;
    s.has_jumped = true;
    s.jump_ticks = 0;
    s.jump_hold_broken = false;
  }
  s.jump_ticks += 1;
  if (s.is_jumping) {
    if (!s.controls.jump) {
      s.jump_hold_broken = true;
    }
    s.is_jumping = s.jump_ticks <= car::jump::min_ticks ||
                   (!s.jump_hold_broken && s.controls.jump &&
                    s.jump_ticks <= car::jump::max_ticks);
    if (!s.is_jumping) {
      s.jump_ticks = 1;
    }
  }
  if (s.is_jumping) {
    if (s.jump_ticks == 1) {
      rb.add_linear_impulse(up_dir * mut.jump_immediate_force * uu_to_bt, false,
                            false);
      rb.limit_vels(car::max_speed * uu_to_bt, car::max_ang_speed);
    }
    constexpr float scale = uu_to_bt * tick_time;
    rb.add_linear_impulse(up_dir * mut.jump_accel * scale, false, true);
  }
}

RS_HD bool
car_update_double_jump_or_flip(Car &car, RigidBody &rb, const World &w,
                               const MutatorConfig &mut, bool jump_pressed,
                               float forward_speed_uu, bool was_is_jumping) {
  using namespace consts;
  CarState &s = car.state;
  if (s.cache.is_on_ground) {
    s.has_double_jumped = false;
    s.has_flipped = false;
    s.air_time = 0.0f;
    s.air_time_since_jump = 0.0f;
    s.flip_time += tick_time;
    return false;
  }
  s.air_time += tick_time;
  if (s.has_jumped && !was_is_jumping && !s.is_jumping) {
    s.air_time_since_jump += tick_time;
  } else {
    s.air_time_since_jump = 0.0f;
  }
  if (jump_pressed && s.air_time_since_jump < car::jump::doublejump_max_delay) {
    Vec3A dodge_input =
        quantize_axis_inputs(s.controls.pyr() * Vec3A{-1.0f, 1.0f, 1.0f});
    float input_magnitude = abs_f32(dodge_input.x) + abs_f32(dodge_input.y) +
                            abs_f32(dodge_input.z);
    bool is_flip_input = input_magnitude >= car.info.config.dodge_deadzone;
    bool can_use =
        (!s.is_auto_flipping && !s.has_double_jumped && !s.has_flipped) ||
        (is_flip_input ? mut.unlimited_flips : mut.unlimited_double_jumps);
    if (can_use) {
      s.has_jumped = true;
      if (is_flip_input) {
        s.flip_time = 0.0f;
        s.has_flipped = true;
        s.is_flipping = true;
        Vec3A dodge_dir = dodge_input;
        dodge_dir.y += dodge_dir.z;
        if (abs_f32(dodge_dir.x) < 0.1f && abs_f32(dodge_dir.y) < 0.1f) {
          dodge_dir = Vec3A::zero();
        } else {
          dodge_dir = dodge_dir.with_z(0.0f).normalize_or_zero();
        }
        Vec3A deadzone_dir =
            Vec3A::select(dodge_dir.abs().cmplt(Vec3A::splat(0.1f)),
                          Vec3A::zero(), dodge_dir);
        if (deadzone_dir.length_squared() > k_f32_epsilon * k_f32_epsilon) {
          s.flip_rel_torque = {-dodge_dir.y, dodge_dir.x, 0.0f};
          bool backwards =
              abs_f32(forward_speed_uu) < 100.0f
                  ? is_sign_negative_f32(deadzone_dir.x)
                  : signum_f32(deadzone_dir.x) != signum_f32(forward_speed_uu);
          float max_speed_scale_x =
              backwards ? car::flip::backward_impulse_max_speed_scale
                        : car::flip::forward_impulse_max_speed_scale;
          float ratio = abs_f32(forward_speed_uu) / car::max_speed;
          Vec3A initial = deadzone_dir * car::flip::initial_vel_scale;
          initial.x *= ((max_speed_scale_x - 1.0f) * ratio) + 1.0f;
          initial.y *=
              ((car::flip::side_impulse_max_speed_scale - 1.0f) * ratio) + 1.0f;
          if (backwards) {
            initial.x *= car::flip::backward_impulse_scale_x;
          }
          Vec3A forward = s.forward_dir();
          int32_t dodge_yaw = dodge_yaw_units(forward.y, forward.x);
          const float *table = w.s->sin_table;
          Vec3A forward_2d{sin_tab_angle(table, dodge_yaw + 0x4000),
                           sin_tab_angle(table, dodge_yaw), 0.0f};
          Vec3A right_2d{-forward_2d.y, forward_2d.x, 0.0f};
          Vec3A final_delta = initial.x * forward_2d + initial.y * right_2d;
          rb.add_linear_impulse(final_delta * uu_to_bt, false, false);
          rb.limit_vels(car::max_speed * uu_to_bt, car::max_ang_speed);
        }
      } else {
        rb.add_linear_impulse(s.up_dir() * mut.jump_immediate_force * uu_to_bt,
                              false, false);
        rb.limit_vels(car::max_speed * uu_to_bt, car::max_ang_speed);
        s.has_double_jumped = true;
      }
    }
  }
  if (s.is_flipping) {
    float pre = s.flip_time;
    bool still = s.has_flipped && pre < car::flip::torque_time;
    s.is_flipping = still;
    s.flip_time = still ? pre + tick_time : tick_time;
    return !still;
  }
  s.flip_time += tick_time;
  return false;
}

RS_HD void car_update_auto_roll(Car &car, RigidBody &rb, int num_in_contact,
                                bool &has_cached_up, Vec3A &cached_up) {
  using namespace consts;
  CarState &s = car.state;
  if (!car.sticky_gate_prev && !s.cache.has_world_contact_normal) {
    return;
  }
  Vec3A ground_up;
  if (car.sticky_gate_prev && num_in_contact > 0) {
    if (!has_cached_up) {
      cached_up = car.vehicle.get_upwards_dir_from_wheel_contacts(rb);
      has_cached_up = true;
    }
    ground_up = cached_up;
  } else {
    ground_up = s.cache.world_contact_normal;
  }
  Vec3A ground_down = -ground_up;
  Vec3A fwd = s.forward_dir();
  Vec3A right = s.right_dir();
  Vec3A cross_right = ground_up.cross(fwd);
  Vec3A cross_forward = ground_down.cross(cross_right);
  float right_factor = 1.0f - clamp_f32(right.dot(cross_right), 0.0f, 1.0f);
  float forward_factor = 1.0f - clamp_f32(fwd.dot(cross_forward), 0.0f, 1.0f);
  Vec3A torque_dir_right = fwd * -signum_f32(right.dot(ground_up));
  Vec3A torque_dir_forward = right * signum_f32(fwd.dot(ground_up));
  Vec3A torque_right = torque_dir_right * right_factor;
  Vec3A torque_forward = torque_dir_forward * forward_factor;
  constexpr float force_scale = car::autoroll::force * uu_to_bt * tick_time;
  constexpr float torque_scale = car::autoroll::torque * tick_time;
  rb.add_linear_impulse(ground_down * force_scale, false, true);
  rb.add_angular_impulse((torque_forward + torque_right) * torque_scale, false,
                         true);
}

RS_HD void car_update_boost(Car &car, RigidBody &rb, const MutatorConfig &mut) {
  using namespace consts;
  CarState &s = car.state;
  auto apply_accel = [&]() {
    float accel =
        s.cache.is_on_ground ? mut.boost_accel_ground : mut.boost_accel_air;
    rb.add_linear_impulse(accel * s.forward_dir() * (uu_to_bt * tick_time),
                          false, true);
  };
  auto recharge = [&]() {
    if (mut.recharge_boost_enabled &&
        s.time_since_boosted >= mut.recharge_boost_delay) {
      s.boost += mut.recharge_boost_per_second * tick_time;
    }
  };
  if (s.is_boosting) {
    float cost = mut.boost_used_per_second * tick_time;
    s.boost = max_f32_c(s.boost - cost, 0.0f);
    bool depleted = s.boost == 0.0f;
    bool latch_expired =
        !s.controls.boost && s.boosting_time >= car::boost::min_time;
    if (depleted || latch_expired) {
      s.is_boosting = false;
      s.boosting_time = 0.0f;
      s.time_since_boosted += tick_time;
      recharge();
    } else {
      s.is_boosting = true;
      s.boosting_time += tick_time;
      s.time_since_boosted = 0.0f;
      apply_accel();
    }
  } else if (s.controls.boost && s.boost > 0.0f) {
    s.is_boosting = true;
    s.boosting_time += tick_time;
    s.time_since_boosted = 0.0f;
    apply_accel();
  } else {
    s.is_boosting = false;
    s.boosting_time = 0.0f;
    s.time_since_boosted += tick_time;
    recharge();
  }
  s.boost = clamp_f32(s.boost, 0.0f, mut.car_max_boost_amount);
}

RS_HD void car_update_auto_flip(Car &car, RigidBody &rb, const World &w,
                                bool jump_pressed) {
  using namespace consts;
  CarState &s = car.state;
  if (jump_pressed && s.cache.has_world_contact_normal &&
      s.cache.world_contact_normal.z > car::autoflip::norm_z_thresh) {
    Vec3A wcn = s.cache.world_contact_normal;
    const Mat3A &rot = s.phys.rot_mat;
    float y_z = rot.y_axis.z;
    float z_z = rot.z_axis.z;
    float cy = sqrt_f32(z_z * z_z + y_z * y_z);
    float roll = cy > 16.0f * k_f32_epsilon ? atan2f_glibc(y_z, z_z) : 0.0f;
    float abs_roll = abs_f32(roll);
    if (wcn.dot(s.up_dir()) < car::autoflip::up_dot_thresh) {
      s.auto_flip_timer = car::autoflip::time * (abs_roll / k_pi);
      s.auto_flip_torque_scale = autoflip_impulse_sign(w.s->sin_table, rot);
      s.is_auto_flipping = true;
      constexpr float scale = car::autoflip::impulse * uu_to_bt;
      rb.add_linear_impulse(-s.up_dir() * scale, false, false);
    }
  }
  if (s.is_auto_flipping) {
    if (s.auto_flip_timer <= 0.0f) {
      s.is_auto_flipping = false;
      s.auto_flip_timer = 0.0f;
    } else {
      rb.ang_vel += s.forward_dir() * car::autoflip::ang_vel_delta *
                    s.auto_flip_torque_scale;
      rb.limit_vels(car::max_speed * uu_to_bt, car::max_ang_speed);
      s.auto_flip_timer -= tick_time;
    }
  }
}

RS_HD void car_apply_flip_zdamp_prepass(Car &car, World &w) {
  using namespace consts;
  const CarState &s = car.state;
  if (s.is_demoed || s.cache.is_on_ground || !s.is_flipping) {
    return;
  }
  if (!(s.flip_time >= car::flip::z_damp_start &&
        s.flip_time <= car::flip::torque_time)) {
    return;
  }
  RigidBody &rb = w.dyn_body(car.rigid_body_idx);
  if (rb.lin_vel.z < 0.0f || s.flip_time < car::flip::z_damp_end) {
    rb.lin_vel.z *= 1.0f - car::flip::z_damp_120;
  }
}

RS_HDN bool car_pre_tick_update(Car &car, World &w, Rng &rng, GameMode mode,
                                const MutatorConfig &mut,
                                const VehicleHit *precomputed_hits,
                                float *deferred_throttle) {
  using namespace consts;
  RigidBody &rb = w.dyn_body(car.rigid_body_idx);
  CarState &s = car.state;
  if (s.is_demoed) {
    s.demo_respawn_timer = max_f32_c(s.demo_respawn_timer - tick_time, 0.0f);
    if (s.demo_respawn_timer == 0.0f) {
      car_respawn(car, rb, rng, mode, mut.car_spawn_boost_amount);
    }
    rb.set_activation_state(ActivationState::DisableSimulation);
    rb.collision_flags |= k_flag_no_contact_response;
    return false;
  }
  rb.force_activate();
  rb.collision_flags =
      static_cast<uint8_t>(rb.collision_flags & ~k_flag_no_contact_response);
  s.controls = s.controls.clamped();
  float quant_throttle = quantize_axis_input(s.controls.throttle);
  float forward_speed_uu = rb.get_forward_speed() * bt_to_uu;
  bool jump_pressed = s.controls.jump && !s.cache.prev_controls.jump;
  int num_in_contact = s.num_wheels_in_contact();
  bool has_cached_up = false;
  Vec3A cached_up = Vec3A::zero();
  car_update_wheels(car, rb, forward_speed_uu, quant_throttle, mut,
                    has_cached_up, cached_up);
  if (s.cache.is_on_ground) {
    if (s.is_flipping) {
      s.flip_time = 0.0f;
    }
    s.is_flipping = false;
  }
  bool was_is_jumping = s.is_jumping;
  car_update_jump(car, rb, mut, jump_pressed);
  car_update_auto_flip(car, rb, w, jump_pressed);
  bool prev_is_flipping = s.is_flipping;
  float prev_flip_time = s.flip_time;
  bool flip_ended = car_update_double_jump_or_flip(
      car, rb, w, mut, jump_pressed, forward_speed_uu, was_is_jumping);
  if (!s.cache.is_on_ground) {
    car_update_air_torque(car, rb, num_in_contact, quant_throttle,
                          prev_is_flipping, prev_flip_time);
  }
  int world_wheels = s.cache.world_wheels_prev;
  if (quant_throttle != 0.0f && !s.is_flipping && !flip_ended &&
      world_wheels < 4 &&
      (world_wheels > 0 || s.cache.has_world_contact_normal)) {
    car_update_auto_roll(car, rb, world_wheels, has_cached_up, cached_up);
  }
  s.cache.has_world_contact_normal = false;
  s.cache.world_contact_normal = Vec3A::zero();
  car_update_boost(car, rb, mut);
  float real_throttle =
      (s.controls.boost && s.boost > 0.0f) ? 1.0f : quant_throttle;
  if (deferred_throttle) {
    *deferred_throttle = real_throttle;
    return true;
  }
  car.vehicle.update(w, tick_time, s.handbrake_val, real_throttle,
                     car.info.config.three_wheels, precomputed_hits);
  car_update_contact_state(car, w);
  return true;
}

RS_HDN void car_post_tick_update(Car &car, World &w) {
  using namespace consts;
  constexpr float start_sq =
      car::supersonic::start_speed * car::supersonic::start_speed;
  constexpr float maintain_sq =
      car::supersonic::maintain_min_speed * car::supersonic::maintain_min_speed;
  const RigidBody &rb = w.body(car.rigid_body_idx);
  CarState &s = car.state;
  if (s.is_demoed) {
    return;
  }
  s.phys.rot_mat = rb.world_trans.matrix3;
  float speed_sq = (rb.lin_vel * bt_to_uu).length_squared();
  if (s.is_supersonic) {
    if (speed_sq >= start_sq) {
      s.supersonic_grace_timer = 0.0f;
    } else if (speed_sq < maintain_sq) {
      s.is_supersonic = false;
      s.supersonic_grace_timer = 0.0f;
    } else {
      s.supersonic_grace_timer += tick_time;
      if (s.supersonic_grace_timer >= car::supersonic::maintain_max_time) {
        s.is_supersonic = false;
        s.supersonic_grace_timer = 0.0f;
      }
    }
  } else if (speed_sq >= start_sq) {
    s.is_supersonic = true;
    s.supersonic_grace_timer = 0.0f;
  } else {
    s.supersonic_grace_timer = 0.0f;
  }
  if (s.has_jumped && !s.is_jumping && s.cache.is_on_ground &&
      s.air_time > 0.0f) {
    int num_contacts = 0;
    bool all_extending = true;
    for (int i = 0; i < k_num_wheels; ++i) {
      const WheelInfo &wheel = car.vehicle.wheels[i];
      if (!wheel.has_raycast) {
        continue;
      }
      ++num_contacts;
      if (!(wheel.raycast_info.suspension_relative_vel > 1.0f)) {
        all_extending = false;
        break;
      }
    }
    bool leaving = num_contacts != 0 && all_extending;
    if (!leaving) {
      s.has_jumped = false;
    }
  } else if (s.has_jumped && !s.is_jumping && s.cache.is_on_ground &&
             s.jump_ticks > car::jump::settled_rearm_ticks) {
    s.has_jumped = false;
  }
  s.bump_cooldown_timer = max_f32_c(s.bump_cooldown_timer - tick_time, 0.0f);
  s.cache.prev_controls = s.controls;
}

RS_HD void car_finish_physics_tick(Car &car, RigidBody &rb) {
  using namespace consts;
  if (car.state.is_demoed) {
    return;
  }
  if (car.vel_impulse_cache != Vec3A::zero()) {
    rb.lin_vel += car.vel_impulse_cache;
    car.vel_impulse_cache = Vec3A::zero();
  }
  car.state.phys.pos = rb.world_trans.translation * bt_to_uu;
  car.state.phys.vel = rb.lin_vel * bt_to_uu;
  car.state.phys.ang_vel = rb.ang_vel;
}

}
