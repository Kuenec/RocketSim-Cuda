#pragma once

#include <cstdint>

#include "rsc/math/hd.hpp"
#include "rsc/math/shared.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

enum class GameMode : uint8_t {
  Soccar,
  Hoops,
  Heatseeker,
  Snowday,
  Dropshot,
  TheVoid
};

enum class Team : uint8_t { Blue, Orange };

RS_HDI float team_y_dir(Team t) {
  return static_cast<float>(static_cast<int8_t>(t) * 2 - 1);
}

enum class DemoMode : uint8_t { Normal, OnContact, Disabled };

struct PhysicsCoefs {
  float friction;
  float restitution;
};

struct CarSpawnPos {
  float x, y, yaw_ang;
};

namespace consts {

inline constexpr float bt_to_uu = 50.0f;
inline constexpr float uu_to_bt = 1.0f / 50.0f;
inline constexpr float tick_rate = 120.0f;
inline constexpr float tick_time = 1.0f / tick_rate;
inline constexpr float gravity_z = -650.0f;

namespace arena {
inline constexpr PhysicsCoefs base_coefs{0.6f, 0.3f};
RS_HD Aabb get_aabb(GameMode mode) {
  float max_x = 4096.0f, max_y = 5120.0f, max_z = 2048.0f;
  if (mode == GameMode::Hoops) {
    max_x = 8900.0f / 3.0f;
    max_y = 3581.0f;
    max_z = 1820.0f;
  } else if (mode == GameMode::Dropshot) {
    max_x = 5075.0f;
    max_y = 4592.0f;
    max_z = 2024.0f;
  }
  float floor = mode == GameMode::Dropshot ? 1.5f : 0.0f;
  return {{-max_x, -max_y, floor}, {max_x, max_y, max_z}};
}
}

namespace car {
inline constexpr float mass_bt = 180.0f;
inline constexpr PhysicsCoefs base_coefs{0.3f, 0.1f};
inline constexpr PhysicsCoefs hit_ball_coefs{2.0f, 0.0f};
inline constexpr PhysicsCoefs hit_world_coefs{0.3f, 0.3f};
inline constexpr PhysicsCoefs hit_car_coefs{0.09f, 0.1f};
inline constexpr float max_speed = 2300.0f;
inline constexpr float max_ang_speed = 5.5f;
namespace boost {
inline constexpr float max = 100.0f;
inline constexpr float used_per_second = 33.3f;
inline constexpr float min_time = 0.1f;
inline constexpr float accel_ground = 2975.0f / 3.0f;
inline constexpr float accel_air = accel_ground;
inline constexpr float spawn_amount = max / 3.0f;
inline constexpr float recharge_per_second = 10.0f;
inline constexpr float recharge_delay = 0.25f;
}
namespace supersonic {
inline constexpr float start_speed = 2200.0f;
inline constexpr float maintain_min_speed = start_speed - 100.0f;
inline constexpr float maintain_max_time = 1.0f;
}
namespace demo {
inline constexpr float yaw_limit_deg = 45.572994f;
inline constexpr float pitch_limit_deg = 36.869896f;
}
namespace drive {
inline constexpr float throttle_torque_amount = mass_bt * 400.0f;
inline constexpr float brake_torque_amount = mass_bt * (14.25f + (1.0f / 3.0f));
inline constexpr float stopping_forward_vel = 25.0f;
inline constexpr float coasting_brake_factor = 0.15f;
inline constexpr float braking_no_throttle_speed_thresh = 0.01f;
inline constexpr float throttle_deadzone = 0.01f;
inline constexpr float throttle_air_accel = 200.0f / 3.0f;
inline constexpr float powerslide_rise_rate = 5.0f;
inline constexpr float powerslide_fall_rate = 2.0f;
}
namespace jump {
inline constexpr float accel = 4375.0f / 3.0f;
inline constexpr float immediate_force = 875.0f / 3.0f;
inline constexpr uint32_t min_ticks = 3;
inline constexpr uint32_t max_ticks = 24;
inline constexpr uint32_t settled_rearm_ticks = 15;
inline constexpr float doublejump_max_delay = 1.25f;
}
namespace flip {
inline constexpr float z_damp_120 = 0.35f;
inline constexpr float z_damp_start = 0.15f;
inline constexpr float z_damp_end = 0.21f;
inline constexpr float torque_time = 0.65f;
inline constexpr float torque_min_time = 0.41f;
inline constexpr float pitch_cancel_gate_min_time = 0.041f;
inline constexpr float pitchlock_time = 1.0f;
inline constexpr float pitchlock_extra_time = 0.3f;
inline constexpr float initial_vel_scale = 500.0f;
inline constexpr float torque_x = 260.0f;
inline constexpr float torque_y = 224.0f;
inline constexpr float forward_impulse_max_speed_scale = 1.0f;
inline constexpr float side_impulse_max_speed_scale = 1.9f;
inline constexpr float backward_impulse_max_speed_scale = 2.5f;
inline constexpr float backward_impulse_scale_x = 16.0f / 15.0f;
}
namespace air_control {
inline constexpr float torque_x = 130.0f;
inline constexpr float torque_y = 95.0f;
inline constexpr float torque_z = 400.0f;
inline constexpr float damping_x = 30.0f;
inline constexpr float damping_y = 20.0f;
inline constexpr float damping_z = 50.0f;
inline constexpr float torque_apply_scale =
    2.0f * k_pi / static_cast<float>(1 << 16) * 1000.0f;
}
namespace autoflip {
inline constexpr float impulse = 200.0f;
inline constexpr float ang_vel_delta = 1000.0f;
inline constexpr float time = 0.4f;
inline constexpr float norm_z_thresh = 0.707f;
inline constexpr float roll_thresh = 2.8f;
inline constexpr float up_dot_thresh = -0.966f;
}
namespace autoroll {
inline constexpr float force = 100.0f;
inline constexpr float torque = 80.0f;
}
namespace bump {
inline constexpr float cooldown_time = 0.25f;
inline constexpr float yaw_limit_deg = 70.0f;
inline constexpr float pitch_limit_deg = 36.869896f;
}
namespace spawn {
inline constexpr float rest_z = 17.0f;
inline constexpr float respawn_z = 36.0f;
inline constexpr float respawn_time = 3.0f;

RS_HD int kickoff_locations(GameMode mode, const CarSpawnPos *&out) {
  static constexpr CarSpawnPos soccar[5] = {
      {-2048.0f, -2560.0f, k_frac_pi_4 * 1.0f},
      {2048.0f, -2560.0f, k_frac_pi_4 * 3.0f},
      {-256.0f, -3840.0f, k_frac_pi_4 * 2.0f},
      {256.0f, -3840.0f, k_frac_pi_4 * 2.0f},
      {0.0f, -4608.0f, k_frac_pi_4 * 2.0f}};
  static constexpr CarSpawnPos hoops[5] = {
      {-1536.0f, -3072.0f, k_frac_pi_4 * 2.0f},
      {1536.0f, -3072.0f, k_frac_pi_4 * 2.0f},
      {-256.0f, -2816.0f, k_frac_pi_4 * 2.0f},
      {256.0f, -2816.0f, k_frac_pi_4 * 2.0f},
      {0.0f, -3200.0f, k_frac_pi_4 * 2.0f}};
  static constexpr CarSpawnPos dropshot[5] = {
      {-1867.0f, -2380.0f, k_frac_pi_4 * 1.0f},
      {1867.0f, -2380.0f, k_frac_pi_4 * 3.0f},
      {-256.0f, -3576.0f, k_frac_pi_4 * 2.0f},
      {256.0f, -3576.0f, k_frac_pi_4 * 2.0f},
      {0.0f, -4088.0f, k_frac_pi_4 * 2.0f}};
  static constexpr CarSpawnPos heatseeker[4] = {
      {-1000.0f, -4620.0f, k_frac_pi_2},
      {1000.0f, -4620.0f, k_frac_pi_2},
      {-2000.0f, -4620.0f, k_frac_pi_2},
      {2000.0f, -4620.0f, k_frac_pi_2}};
  switch (mode) {
  case GameMode::Hoops:
    out = hoops;
    return 5;
  case GameMode::Dropshot:
    out = dropshot;
    return 5;
  case GameMode::Heatseeker:
    out = heatseeker;
    return 4;
  default:
    out = soccar;
    return 5;
  }
}

RS_HD int respawn_locations(GameMode mode, const CarSpawnPos *&out) {
  static constexpr CarSpawnPos soccar[4] = {{-2304.0f, -4608.0f, k_frac_pi_2},
                                            {-2688.0f, -4608.0f, k_frac_pi_2},
                                            {2304.0f, -4608.0f, k_frac_pi_2},
                                            {2688.0f, -4608.0f, k_frac_pi_2}};
  static constexpr CarSpawnPos hoops[4] = {{-1920.0f, -3072.0f, k_frac_pi_2},
                                           {-1152.0f, -3072.0f, k_frac_pi_2},
                                           {1920.0f, -3072.0f, k_frac_pi_2},
                                           {1152.0f, -3072.0f, k_frac_pi_2}};
  static constexpr CarSpawnPos dropshot[4] = {{-2176.0f, -3410.0f, k_frac_pi_2},
                                              {-1152.0f, -3100.0f, k_frac_pi_2},
                                              {2176.0f, -3410.0f, k_frac_pi_2},
                                              {1152.0f, -3100.0f, k_frac_pi_2}};
  switch (mode) {
  case GameMode::Hoops:
    out = hoops;
    return 4;
  case GameMode::Dropshot:
    out = dropshot;
    return 4;
  default:
    out = soccar;
    return 4;
  }
}
}
}

namespace ball {
inline constexpr float mass_bt = 30.0f;
inline constexpr float rest_z = 93.15f;
inline constexpr float max_ang_speed = 6.0f;
inline constexpr float drag = 0.03f;
inline constexpr PhysicsCoefs coefs{0.35f, 0.6f};
inline constexpr float hoops_launch_z_vel = 1000.0f;
inline constexpr float hoops_launch_delay = 0.265f;
inline constexpr float max_speed = 6000.0f;
namespace car_hit_impulse {
inline constexpr float forward_scale = 0.65f;
inline constexpr float max_delta_vel_uu = 4600.0f;
inline constexpr float z_scale_normal = 0.35f;
inline constexpr float z_scale_hoops_ground = z_scale_normal * 1.55f;
inline constexpr float z_scale_hoops_normal_z_thresh = 0.1f;
}
}

namespace snowday {
inline constexpr float puck_radius = 114.25f;
inline constexpr float puck_height = 62.5f;
inline constexpr int puck_circle_point_amount = 20;
inline constexpr float puck_mass_bt = 50.0f;
inline constexpr float puck_ground_stick_force = 70.0f;
inline constexpr PhysicsCoefs puck_coefs{0.1f, 0.3f};
}

namespace ball {
RS_HDI float get_radius(GameMode mode) {
  switch (mode) {
  case GameMode::Hoops:
    return 96.3831f;
  case GameMode::Dropshot:
    return 100.2565f;
  case GameMode::Snowday:
    return snowday::puck_radius;
  default:
    return 91.25f;
  }
}
}

namespace goal {
inline constexpr float soccar_goal_score_base_threshold_y = 5124.25f;
inline constexpr float hoops_goal_score_threshold_z = 270.0f;
}

namespace heatseeker {
inline constexpr float initial_target_speed = 2900.0f;
inline constexpr float target_speed_increment = 85.0f;
inline constexpr float min_speedup_interval = 1.0f;
inline constexpr float target_y = 5120.0f;
inline constexpr float target_z = 320.0f;
inline constexpr float horizontal_blend = 1.45f;
inline constexpr float vertical_blend = 0.78f;
inline constexpr float speed_blend = 0.3f;
inline constexpr float max_turn_pitch =
    7000.0f * k_pi / static_cast<float>(1 << 15);
inline constexpr float max_speed = 4600.0f;
inline constexpr float wall_bounce_change_y_thresh = 300.0f;
inline constexpr float wall_bounce_change_y_normal = 0.5f;
inline constexpr float wall_bounce_force_scale = 1.0f / 3.0f;
inline constexpr float wall_bounce_up_frac = 0.3f;
RS_HDI Vec3A ball_start_pos() { return {-1000.0f, -2220.0f, 92.75f}; }
RS_HDI Vec3A ball_start_vel() { return {0.0f, -65.0f, 650.0f}; }
}

namespace dropshot {
inline constexpr float ball_launch_z_vel = 985.0f;
inline constexpr float ball_launch_delay = 0.26f;
inline constexpr float min_downward_speed_to_damage = 250.0f;
inline constexpr float min_charge_hit_speed = 500.0f;
inline constexpr float min_absorbed_force_for_charge = 2500.0f;
inline constexpr float min_absorbed_force_for_supercharge = 11000.0f;
inline constexpr float min_damage_interval = 0.1f;
inline constexpr float tile_hex_x = 7.6643f;
inline constexpr float tile_hex_y = 8.85f;
inline constexpr float tile_hex_half_y = 4.425f;
inline constexpr float tile_width_x = tile_hex_x * 2.0f * bt_to_uu;
inline constexpr float row_offset_y = (tile_hex_y + tile_hex_half_y) * bt_to_uu;
inline constexpr float tile_offset_y = 2.54736f * bt_to_uu;
inline constexpr int num_tiles_per_team = 70;
inline constexpr int tiles_in_first_row = 13;
inline constexpr int tiles_in_last_row = 7;
inline constexpr int num_tile_rows = tiles_in_first_row - tiles_in_last_row + 1;
}

namespace boost_pads {
inline constexpr float cyl_height = 95.0f;
inline constexpr float cyl_rad_big = 208.0f;
inline constexpr float cyl_rad_small = 176.0f;
inline constexpr float box_height = 64.0f;
inline constexpr float box_rad_big = 160.0f;
inline constexpr float box_rad_small = 96.0f;
inline constexpr float cooldown_big = 10.0f;
inline constexpr float cooldown_small = 4.0f;
inline constexpr float boost_amount_big = 100.0f;
inline constexpr float boost_amount_small = 12.0f;
}

namespace quantize {
inline constexpr float pos_scale = 100.0f;
inline constexpr float vel_scale = 100.0f;
inline constexpr float ang_vel_scale = 10000.0f;
}

namespace curves {
RS_HD const LinearPieceCurve<6> &steer_angle_from_speed() {
  static constexpr float pts[6][2] = {{0.0f, 0.53356f},    {500.0f, 0.31930f},
                                      {1000.0f, 0.18203f}, {1500.0f, 0.10570f},
                                      {1750.0f, 0.08507f}, {3000.0f, 0.03454f}};
  static constexpr LinearPieceCurve<6> c = LinearPieceCurve<6>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<6> &steer_angle_from_speed_threewheel() {
  static constexpr float pts[6][2] = {
      {0.0f, 0.342473f},    {2300.0f, 0.034837f}, {2300.0f, 0.034837f},
      {2300.0f, 0.034837f}, {2300.0f, 0.034837f}, {2300.0f, 0.034837f}};
  static constexpr LinearPieceCurve<6> c = LinearPieceCurve<6>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<6> &plank_steer_angle_from_speed() {
  static constexpr float pts[6][2] = {
      {0.0f, 0.529582f},    {500.0f, 0.317619f},  {1000.0f, 0.181425f},
      {1500.0f, 0.105480f}, {1750.0f, 0.084926f}, {3000.0f, 0.034519f}};
  static constexpr LinearPieceCurve<6> c = LinearPieceCurve<6>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<2> &powerslide_steer_angle_from_speed() {
  static constexpr float pts[2][2] = {{0.0f, 0.39235f}, {2500.0f, 0.12610f}};
  static constexpr LinearPieceCurve<2> c = LinearPieceCurve<2>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<2> &plank_powerslide_steer_angle_from_speed() {
  static constexpr float pts[2][2] = {{0.0f, 0.389930f}, {2500.0f, 0.125785f}};
  static constexpr LinearPieceCurve<2> c = LinearPieceCurve<2>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<3> &drive_speed_torque_factor() {
  static constexpr float pts[3][2] = {
      {0.0f, 1.0f}, {1400.0f, 0.1f}, {1410.0f, 0.0f}};
  static constexpr LinearPieceCurve<3> c = LinearPieceCurve<3>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<5> &ball_car_extra_impulse_factor() {
  static constexpr float pts[5][2] = {{0.0f, 0.65f},
                                      {500.0f, 0.65f},
                                      {1400.0f, 0.60f},
                                      {2300.0f, 0.55f},
                                      {4600.0f, 0.30f}};
  static constexpr LinearPieceCurve<5> c = LinearPieceCurve<5>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<3> &bump_vel_amount_ground() {
  static constexpr float pts[3][2] = {{0.0f, 150.0f / 180.0f},
                                      {1400.0f, 200000.0f / 180.0f},
                                      {2200.0f, 275000.0f / 180.0f}};
  static constexpr LinearPieceCurve<3> c = LinearPieceCurve<3>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<3> &bump_vel_amount_air() {
  static constexpr float pts[3][2] = {{0.0f, 150.0f / 180.0f},
                                      {1400.0f, 250000.0f / 180.0f},
                                      {2200.0f, 350000.0f / 180.0f}};
  static constexpr LinearPieceCurve<3> c = LinearPieceCurve<3>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<3> &bump_upward_vel_amount() {
  static constexpr float pts[3][2] = {{0.0f, 50.0f / 180.0f},
                                      {1400.0f, 50000.0f / 180.0f},
                                      {2200.0f, 75000.0f / 180.0f}};
  static constexpr LinearPieceCurve<3> c = LinearPieceCurve<3>::make(pts);
  return c;
}
}

}

}
