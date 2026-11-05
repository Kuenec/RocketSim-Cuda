#pragma once

#include <cstdint>

#include "rsc/bullet/world_types.hpp"
#include "rsc/sim/car.hpp"
#include "rsc/sim/consts.hpp"
#include "rsc/sim/dropshot.hpp"
#include "rsc/sim/quantize.hpp"

namespace rsc {

struct HeatseekerInfo {
  int8_t y_target_dir;
  float cur_target_speed;
  float time_since_hit;
};

struct DropshotInfo {
  uint8_t charge_level;
  float accumulated_hit_force;
  int8_t y_target_dir;
  bool has_last_damage_tick;
  uint64_t last_damage_tick;
};

struct BallState {
  PhysState phys;
  HeatseekerInfo hs_info;
  DropshotInfo ds_info;
  uint64_t tick_count_since_kickoff;
};

RS_HD BallState ball_state_default() {
  BallState s;
  s.phys = {{0.0f, 0.0f, consts::ball::rest_z},
            Mat3A::identity(),
            Vec3A::zero(),
            Vec3A::zero()};
  s.hs_info = {0, consts::heatseeker::initial_target_speed, 0.0f};
  s.ds_info = {1, 0.0f, 0, false, 0};
  s.tick_count_since_kickoff = 0;
  return s;
}

struct Ball {
  BallState state;
  uint32_t rigid_body_idx;
  bool ground_stick_applied;
};

RS_HD void ball_sync_state_to_body(Ball &ball, RigidBody &rb) {
  rb.set_world_trans(
      {ball.state.phys.rot_mat, ball.state.phys.pos * consts::uu_to_bt});
  ball.state.phys.rot_mat = rb.world_trans.matrix3;
  rb.interp_world_trans = rb.world_trans;
  rb.lin_vel = ball.state.phys.vel * consts::uu_to_bt;
  rb.ang_vel = ball.state.phys.ang_vel;
  rb.clear_accum_vels();
  rb.update_inertia_tensor();
  rb.force_activate();
}

RS_HD void ball_set_state(Ball &ball, RigidBody &rb, const BallState &state) {
  ball.state = state;
  ball_sync_state_to_body(ball, rb);
}

RS_HD void ball_pre_tick_update(Ball &ball, RigidBody &rb, GameMode mode) {
  using namespace consts;
  BallState &s = ball.state;
  if (mode == GameMode::Heatseeker) {
    if (s.hs_info.y_target_dir == 0) {
      return;
    }
    Angle vel_angle = Angle::from_forward(s.phys.vel);
    Vec3A goal_target{
        0.0f, heatseeker::target_y * static_cast<float>(s.hs_info.y_target_dir),
        heatseeker::target_z};
    Angle angle_to_goal = Angle::from_forward(goal_target - s.phys.pos);
    Angle delta = angle_to_goal - vel_angle;
    float cur_speed = s.phys.vel.length();
    float speed_ratio = cur_speed / heatseeker::max_speed;
    Angle na = vel_angle;
    float base = speed_ratio * tick_time;
    na.yaw += delta.yaw * base * heatseeker::horizontal_blend;
    na.pitch += delta.pitch * base * heatseeker::vertical_blend;
    na.yaw = rem_euclid_f32(na.yaw, k_tau);
    na.pitch = clamp_f32(na.pitch, -heatseeker::max_turn_pitch,
                         heatseeker::max_turn_pitch);
    na.normalize_fix();
    na.pitch = clamp_f32(na.pitch, -heatseeker::max_turn_pitch,
                         heatseeker::max_turn_pitch);
    na = na.round_ue3();
    float new_speed = cur_speed + (s.hs_info.cur_target_speed - cur_speed) *
                                      heatseeker::speed_blend;
    Vec3A new_dir = na.get_forward_vec();
    rb.lin_vel = new_dir * new_speed * uu_to_bt;
    s.hs_info.time_since_hit += tick_time;
    return;
  }
  if (mode == GameMode::Snowday) {
    ball.ground_stick_applied = false;
    return;
  }
  if (mode == GameMode::Dropshot || mode == GameMode::Hoops) {
    bool is_dropshot = mode == GameMode::Dropshot;
    float launch_delay =
        is_dropshot ? dropshot::ball_launch_delay : ball::hoops_launch_delay;
    float cur = static_cast<float>(s.tick_count_since_kickoff) * tick_time;
    float prev = cur - tick_time;
    if (prev < launch_delay && cur >= launch_delay) {
      if (s.phys.vel == Vec3A::zero() && s.phys.ang_vel == Vec3A::zero() &&
          s.phys.pos.x == 0.0f && s.phys.pos.y == 0.0f) {
        float launch_z = is_dropshot ? dropshot::ball_launch_z_vel
                                     : ball::hoops_launch_z_vel;
        rb.add_linear_impulse(Vec3A{0.0f, 0.0f, launch_z} * uu_to_bt, false,
                              false);
        rb.set_activation_state(ActivationState::Active);
      }
    }
  }
}

RS_HD void ball_finish_physics_tick(Ball &ball, const RigidBody &rb) {
  using namespace consts;
  ball.state.phys.vel = rb.lin_vel * bt_to_uu;
  ball.state.phys.ang_vel = rb.ang_vel;
  ball.state.phys.pos = rb.world_trans.translation * bt_to_uu;
  ball.state.phys.rot_mat = rb.world_trans.matrix3;
  ball.state.tick_count_since_kickoff += 1;
}

RS_HDN void ball_on_hit(Ball &ball, Car &car, GameMode mode,
                       const MutatorConfig &mut, uint64_t tick_count,
                       RigidBody &rb, const float *sin_table) {
  using namespace consts;
  BallState &s = ball.state;
  Vec3A car_forward = rotator_forward(sin_table, car.state.phys.rot_mat.x_axis);
  Vec3A rel_pos = s.phys.pos - car.state.phys.pos;
  Vec3A rel_vel = s.phys.vel - car.state.phys.vel;
  float rel_speed =
      min_f32_c(rel_vel.length(), ball::car_hit_impulse::max_delta_vel_uu);
  bool can_accel = !car.state.has_last_extra_hit_tick ||
                   car.state.last_extra_hit_tick + 1 < tick_count;
  if (rel_speed > 0.0f && can_accel) {
    bool extra_z_scale =
        mode == GameMode::Hoops && car.state.cache.is_on_ground &&
        car.state.phys.rot_mat.z_axis.z >
            ball::car_hit_impulse::z_scale_hoops_normal_z_thresh;
    float z_scale = extra_z_scale ? ball::car_hit_impulse::z_scale_hoops_ground
                                  : ball::car_hit_impulse::z_scale_normal;
    Vec3A hit_dir = (rel_pos * Vec3A{1.0f, 1.0f, z_scale}).normalize_or_zero();
    constexpr float adjust = 1.0f - ball::car_hit_impulse::forward_scale;
    Vec3A forward_adjustment = car_forward * hit_dir.dot(car_forward) * adjust;
    hit_dir = (hit_dir - forward_adjustment).normalize_or_zero();
    Vec3A added =
        hit_dir * rel_speed *
        curves::ball_car_extra_impulse_factor().get_output(rel_speed) *
        mut.ball_hit_extra_force_scale;
    rb.add_linear_impulse(added * uu_to_bt, false, false);
    car.state.has_last_extra_hit_tick = true;
    car.state.last_extra_hit_tick = tick_count;
  }
  if (mode == GameMode::Heatseeker) {
    bool can_increase =
        s.hs_info.time_since_hit > heatseeker::min_speedup_interval ||
        s.hs_info.y_target_dir == 0;
    int8_t new_dir =
        static_cast<int8_t>(static_cast<int>(car.info.team) * -2 + 1);
    if (can_increase && new_dir != s.hs_info.y_target_dir) {
      s.hs_info.time_since_hit = 0.0f;
      s.hs_info.cur_target_speed = min_f32(
          heatseeker::max_speed,
          s.hs_info.cur_target_speed + heatseeker::target_speed_increment);
    }
    s.hs_info.y_target_dir = new_dir;
  } else if (mode == GameMode::Dropshot) {
    Vec3A dir_from_car = (s.phys.pos - car.state.phys.pos).normalize_or_zero();
    Vec3A rel_vel_from_car = car.state.phys.vel - s.phys.vel;
    float vel_info = dir_from_car.dot(rel_vel_from_car);
    if (vel_info >= dropshot::min_charge_hit_speed) {
      s.ds_info.accumulated_hit_force += vel_info;
      if (s.ds_info.accumulated_hit_force >=
          dropshot::min_absorbed_force_for_supercharge) {
        s.ds_info.charge_level = 3;
      } else if (s.ds_info.accumulated_hit_force >=
                 dropshot::min_absorbed_force_for_charge) {
        s.ds_info.charge_level = 2;
      }
    }
    s.ds_info.y_target_dir =
        static_cast<int8_t>(static_cast<int>(car.info.team) * -2 + 1);
  }
}

RS_HD void ball_on_world_hit(Ball &ball, RigidBody &rb, GameMode mode,
                             Vec3A normal) {
  using namespace consts;
  BallState &s = ball.state;
  if (mode == GameMode::Heatseeker) {
    constexpr float arena_extent_y = 5120.0f;
    if (s.hs_info.y_target_dir == 0) {
      return;
    }
    float y_target_dir = static_cast<float>(s.hs_info.y_target_dir);
    float rel_normal_y = normal.y * y_target_dir;
    float rel_y = s.phys.pos.y * y_target_dir;
    if (rel_normal_y <= -heatseeker::wall_bounce_change_y_normal &&
        rel_y >= arena_extent_y - heatseeker::wall_bounce_change_y_thresh) {
      s.hs_info.y_target_dir = static_cast<int8_t>(s.hs_info.y_target_dir * -1);
      Vec3A goal_target{0.0f, heatseeker::target_y * y_target_dir,
                        heatseeker::target_z};
      Vec3A dir_to_goal = (goal_target - s.phys.pos).normalize_or_zero();
      Vec3A bounce_dir =
          dir_to_goal * (1.0f - heatseeker::wall_bounce_up_frac) +
          Vec3A{0.0f, 0.0f, 1.0f} * heatseeker::wall_bounce_up_frac;
      Vec3A bounce_impulse = bounce_dir * s.phys.vel.length() *
                             heatseeker::wall_bounce_force_scale;
      rb.add_linear_impulse(bounce_impulse * uu_to_bt, false, true);
    }
    return;
  }
  if (mode == GameMode::Snowday && !ball.ground_stick_applied) {
    Vec3A force = -normal * snowday::puck_ground_stick_force * tick_time;
    rb.add_linear_impulse(force, true, true);
    ball.ground_stick_applied = true;
  }
}

RS_HD void ball_on_dropshot_tile_collision(Ball &ball, TileStates &tiles,
                                           uint32_t tile_total_index,
                                           uint64_t tick_count) {
  using namespace consts;
  BallState &s = ball.state;
  uint32_t team_idx = tile_total_index / dropshot::num_tiles_per_team;
  uint32_t tile_idx = tile_total_index % dropshot::num_tiles_per_team;
  uint8_t tile_state = tiles.states[team_idx][tile_idx];
  Vec3A tile_pos = dropshot_tile_pos(static_cast<Team>(team_idx), tile_idx);
  if (s.ds_info.y_target_dir == 0) {
    return;
  }
  if (tile_state == k_tile_broken) {
    return;
  }
  if (s.ds_info.has_last_damage_tick) {
    float since =
        static_cast<float>(tick_count - s.ds_info.last_damage_tick) * tick_time;
    if (since <= dropshot::min_damage_interval) {
      return;
    }
  }
  if (s.phys.vel.z > -dropshot::min_downward_speed_to_damage) {
    return;
  }
  if (signum_f32(tile_pos.y) != static_cast<float>(s.ds_info.y_target_dir)) {
    return;
  }
  uint8_t indices[19];
  int count = dropshot_break_indices(tile_idx, s.ds_info.charge_level, indices);
  for (int i = 0; i < count; ++i) {
    uint8_t &st = tiles.states[team_idx][indices[i]];
    if (st == k_tile_full) {
      st = k_tile_damaged;
    } else if (st == k_tile_damaged) {
      st = k_tile_broken;
    }
  }
  s.ds_info.has_last_damage_tick = true;
  s.ds_info.last_damage_tick = tick_count;
  s.ds_info.accumulated_hit_force = 0.0f;
  s.ds_info.charge_level = 1;
  s.ds_info.y_target_dir = 0;
}

}
