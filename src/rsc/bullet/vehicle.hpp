#pragma once

#include <cstdint>

#include "rsc/bullet/raycast.hpp"
#include "rsc/bullet/solver.hpp"
#include "rsc/bullet/world_types.hpp"
#include "rsc/math/shared.hpp"

namespace rsc {

inline constexpr int k_num_wheels = 4;
inline constexpr float k_bt_to_uu = 50.0f;
inline constexpr float k_uu_to_bt = 1.0f / 50.0f;

namespace vehicle_consts {
inline constexpr float suspension_stiffness = 500.0f;
inline constexpr float wheels_damping_compression = 25.0f;
inline constexpr float wheels_damping_relaxation = 40.0f;
inline constexpr float max_suspension_travel = 12.0f;
inline constexpr float suspension_subtraction = 0.05f;
}

namespace vehicle_curves {
RS_HD const LinearPieceCurve<2> &lat_friction() {
  static constexpr float pts[2][2] = {{0.0f, 1.0f}, {1.0f, 0.2f}};
  static constexpr LinearPieceCurve<2> c = LinearPieceCurve<2>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<2> &lat_friction_threewheel() {
  static constexpr float pts[2][2] = {{0.0f, 0.30f}, {1.0f, 0.25f}};
  static constexpr LinearPieceCurve<2> c = LinearPieceCurve<2>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<1> &handbrake_lat_friction_factor() {
  static constexpr float pts[1][2] = {{0.0f, 0.1f}};
  static constexpr LinearPieceCurve<1> c = LinearPieceCurve<1>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<2> &handbrake_long_friction_factor() {
  static constexpr float pts[2][2] = {{0.0f, 0.5f}, {1.0f, 0.9f}};
  static constexpr LinearPieceCurve<2> c = LinearPieceCurve<2>::make(pts);
  return c;
}
RS_HD const LinearPieceCurve<3> &non_sticky_friction_factor() {
  static constexpr float pts[3][2] = {
      {0.0f, 0.1f}, {0.707f, 0.5f}, {1.0f, 1.0f}};
  static constexpr LinearPieceCurve<3> c = LinearPieceCurve<3>::make(pts);
  return c;
}
}

struct RaycastInfo {
  Vec3A contact_normal;
  Vec3A contact_point;
  uint32_t ground_body_idx;
  float suspension_length;
  Vec3A impulse;
  bool has_ground_stick;
  Vec3A ground_stick;
  bool is_in_contact_with_world;
  float clipped_inv_contact_dot_suspension;
  float suspension_relative_vel;
};

struct VehicleHit {
  bool valid;
  Vec3A point;
  Vec3A normal;
  uint32_t body_idx;
};

RS_HD float resolve_single_collision(const RigidBody &body1,
                                     const RigidBody &body2,
                                     Vec3A contact_pos_world,
                                     Vec3A contact_normal_on_b, float time_step,
                                     float distance) {
  Vec3A rel_pos1 = contact_pos_world - body1.world_trans.translation;
  Vec3A rel_pos2 = contact_pos_world - body2.world_trans.translation;
  Vec3A vel1 = body1.get_vel_in_local_point(rel_pos1);
  Vec3A vel2 = body2.get_vel_in_local_point(rel_pos2);
  Vec3A vel = vel1 - vel2;
  float rel_vel = contact_normal_on_b.dot(vel);
  float positional_error = k_wheel_pushback_erp * -distance / time_step;
  float vel_error = -rel_vel;
  float denom0 =
      body1.compute_impulse_denominator(contact_pos_world, contact_normal_on_b);
  float denom1 =
      body2.compute_impulse_denominator(contact_pos_world, contact_normal_on_b);
  float jac_diag_ab_inv = 1.0f / (denom0 + denom1);
  float penetration_impulse = positional_error * jac_diag_ab_inv;
  float vel_impulse = vel_error * jac_diag_ab_inv;
  return max_f32_c(penetration_impulse + vel_impulse, 0.0f);
}

RS_HD float resolve_single_bilateral_fake_ground(const RigidBody &body1,
                                                 Vec3A pos, Vec3A normal) {
  constexpr float contact_damping = -0.2f;
  const Affine3A &t = body1.world_trans;
  Vec3A rel_pos1 = pos - t.translation;
  Vec3A vel1 = body1.get_vel_in_local_point(rel_pos1);
  Vec3A a_j = t.matrix3.transpose_mul_vec3a(rel_pos1.cross(normal));
  float jac_diag_ab = body1.inv_mass + (body1.inv_inertia_local * a_j).dot(a_j);
  float jac_diag_ab_inv = 1.0f / jac_diag_ab;
  float rel_vel = normal.dot(vel1);
  return contact_damping * rel_vel * jac_diag_ab_inv;
}

struct WheelInfo {
  bool has_raycast;
  RaycastInfo raycast_info;
  Vec3A hard_point;
  Vec3A axle_dir;
  Vec3A chassis_connection_point_cs;
  float suspension_rest_length_1;
  float wheels_radius;
  float engine_force;
  float brake;
  float steer_angle;
  Vec3A vel_at_contact_point;
  float lat_friction;
  float long_friction;
  float suspension_force_scale;
  float extra_pushback;
  float real_ray_length;

  RS_HD static WheelInfo make_default() {
    WheelInfo w;
    w.has_raycast = false;
    w.raycast_info = RaycastInfo{};
    w.hard_point = Vec3A::zero();
    w.axle_dir = Vec3A::zero();
    w.chassis_connection_point_cs = Vec3A::zero();
    w.suspension_rest_length_1 = 0.0f;
    w.wheels_radius = 0.0f;
    w.engine_force = 0.0f;
    w.brake = 0.0f;
    w.steer_angle = 0.0f;
    w.vel_at_contact_point = Vec3A::zero();
    w.lat_friction = 1.0f;
    w.long_friction = 1.0f;
    w.suspension_force_scale = 1.0f;
    w.extra_pushback = 0.0f;
    w.real_ray_length = 0.0f;
    return w;
  }

  RS_HD void set_params(Vec3A connection, float rest_length, float radius,
                        float force_scale) {
    chassis_connection_point_cs = connection;
    suspension_rest_length_1 = rest_length;
    wheels_radius = radius;
    suspension_force_scale = force_scale;
    float travel = vehicle_consts::max_suspension_travel * k_uu_to_bt;
    real_ray_length = suspension_rest_length_1 + travel + wheels_radius;
  }

  RS_HD void prepare_for_raycast(const Affine3A &trans, Vec3A &source,
                                 Vec3A &target) {
    hard_point = trans.transform_point3a(chassis_connection_point_cs);
    target = hard_point - (trans.matrix3.z_axis * real_ray_length);
    source = hard_point;
  }

  RS_HD void reset_wheel_suspension() {
    extra_pushback = 0.0f;
    has_raycast = false;
  }

  RS_HD void apply_ray_cast(const World &w, const RigidBody &chassis,
                            const Affine3A &trans, Vec3A front_axle_dir,
                            const VehicleHit &hit, float time_step,
                            bool front) {
    const RigidBody &hit_body = w.body(hit.body_idx);
    Vec3A contact_point = hit.point;
    Vec3A contact_normal = hit.normal;
    bool in_world = hit_body.is_static_obj();
    axle_dir = front ? front_axle_dir : trans.matrix3.y_axis;
    Vec3A up = trans.matrix3.z_axis;
    float wheel_trace_len_sq = (hard_point - contact_point).dot(up);
    float travel = vehicle_consts::max_suspension_travel * k_uu_to_bt;
    float max_suspension_len = suspension_rest_length_1 + travel;
    Vec3A rel_pos = contact_point - trans.translation;
    vel_at_contact_point = chassis.get_vel_in_local_point(rel_pos);
    float proj_vel = contact_normal.dot(vel_at_contact_point);
    float denom = contact_normal.dot(up);
    float suspension_relative_vel;
    float clipped;
    if (denom > 0.1f) {
      float inv = 1.0f / denom;
      suspension_relative_vel = proj_vel * inv;
      clipped = inv;
    } else {
      suspension_relative_vel = 0.0f;
      clipped = 10.0f;
    }
    float suspension_length =
        min_f32(wheel_trace_len_sq - wheels_radius, max_suspension_len);
    extra_pushback = 0.0f;
    bool is_car_hit = hit_body.user_idx == UserInfoType::Car;
    if (in_world || is_car_hit) {
      float thresh = suspension_rest_length_1 + wheels_radius -
                     vehicle_consts::suspension_subtraction;
      if (wheel_trace_len_sq < thresh) {
        float delta = wheel_trace_len_sq - thresh;
        float result = resolve_single_collision(chassis, hit_body, hit.point,
                                                hit.normal, time_step, delta);
        extra_pushback = result / static_cast<float>(k_num_wheels);
      }
    }
    has_raycast = true;
    raycast_info.contact_normal = contact_normal;
    raycast_info.contact_point = contact_point;
    raycast_info.ground_body_idx = hit.body_idx;
    raycast_info.suspension_length = suspension_length;
    raycast_info.impulse = Vec3A::zero();
    raycast_info.has_ground_stick = !in_world;
    raycast_info.ground_stick = in_world ? Vec3A::zero() : -contact_normal;
    raycast_info.is_in_contact_with_world = in_world;
    raycast_info.clipped_inv_contact_dot_suspension = clipped;
    raycast_info.suspension_relative_vel = suspension_relative_vel;
  }

  RS_HD void refresh_friction_curves(const RigidBody &chassis,
                                     Vec3A chassis_translation,
                                     Vec3A contact_normal, float handbrake_val,
                                     float real_throttle, bool three_wheels,
                                     bool is_dynamic_hit) {
    Vec3A lat_dir = axle_dir;
    Vec3A long_dir = lat_dir.cross(contact_normal);
    Vec3A wheel_delta = hard_point - chassis_translation;
    Vec3A cross_vec =
        (chassis.ang_vel.cross(wheel_delta) + chassis.lin_vel) * k_bt_to_uu;
    float base_friction = abs_f32(cross_vec.dot(lat_dir));
    float curve_input =
        base_friction > 5.0f
            ? base_friction / (abs_f32(cross_vec.dot(long_dir)) + base_friction)
            : 0.0f;
    float lat =
        three_wheels
            ? vehicle_curves::lat_friction_threewheel().get_output(curve_input)
            : vehicle_curves::lat_friction().get_output(curve_input);
    float lng = 1.0f;
    if (handbrake_val != 0.0f) {
      lat *= 1.0f + (vehicle_curves::handbrake_lat_friction_factor().get_output(
                         curve_input) -
                     1.0f) *
                        handbrake_val;
      lng *=
          1.0f + (vehicle_curves::handbrake_long_friction_factor().get_output(
                      curve_input) -
                  1.0f) *
                     handbrake_val;
    }
    if (real_throttle == 0.0f || is_dynamic_hit) {
      float scale = vehicle_curves::non_sticky_friction_factor().get_output(
          contact_normal.z);
      lat *= scale;
      lng *= scale;
    }
    lat_friction = lat;
    long_friction = lng;
  }

  RS_HD Vec3A calc_friction_impulses(const RigidBody &chassis,
                                     Vec3A contact_normal, Vec3A contact_point,
                                     float time_step, float friction_scale) {
    Vec3A axle = axle_dir.normalize_or_zero();
    Vec3A forward_dir = contact_normal.cross(axle).normalize_or_zero();
    float side_impulse =
        resolve_single_bilateral_fake_ground(chassis, contact_point, axle);
    float rolling_friction;
    if (engine_force == 0.0f) {
      if (brake == 0.0f) {
        rolling_friction = 0.0f;
      } else {
        constexpr float rolling_friction_scale = 113.73963f;
        float rel_vel = vel_at_contact_point.dot(forward_dir);
        if (time_step > 1.0f / 80.0f) {
          float threshold = 0.8f - (1.0f / (time_step * 150.0f));
          if (abs_f32(rel_vel) < threshold) {
            rel_vel = 0.0f;
          }
        }
        rolling_friction =
            clamp_f32(-rel_vel * rolling_friction_scale, -brake, brake);
      }
    } else {
      rolling_friction = -engine_force / friction_scale;
    }
    Vec3A total = forward_dir * rolling_friction * long_friction +
                  axle * side_impulse * lat_friction;
    return total * friction_scale;
  }

  RS_HD void update_friction_impulse(const RigidBody &chassis, float time_step,
                                     float friction_scale) {
    if (!has_raycast) {
      return;
    }
    raycast_info.impulse = calc_friction_impulses(
        chassis, raycast_info.contact_normal, raycast_info.contact_point,
        time_step, friction_scale);
  }

  RS_HD void update_suspension(RigidBody &cb, Vec3A chassis_translation,
                               float delta_time) {
    if (!has_raycast) {
      return;
    }
    const RaycastInfo &ri = raycast_info;
    float force = (suspension_rest_length_1 - ri.suspension_length) *
                  vehicle_consts::suspension_stiffness *
                  ri.clipped_inv_contact_dot_suspension;
    float damping_vel_scale = ri.suspension_relative_vel < 0.0f
                                  ? vehicle_consts::wheels_damping_compression
                                  : vehicle_consts::wheels_damping_relaxation;
    float wheels_suspension_force =
        force - (damping_vel_scale * ri.suspension_relative_vel);
    wheels_suspension_force *= suspension_force_scale;
    if (wheels_suspension_force <= 0.0f) {
      return;
    }
    float base_force_scale =
        wheels_suspension_force * delta_time + extra_pushback;
    Vec3A offset = ri.contact_point - chassis_translation;
    cb.add_linear_rel_pos_impulse(ri.contact_normal * base_force_scale, offset,
                                  true, false);
  }

  RS_HD void apply_friction_impulses(RigidBody &cb, const Affine3A &trans,
                                     float time_step) const {
    if (!has_raycast) {
      return;
    }
    Vec3A offset = raycast_info.contact_point - trans.translation;
    float up_dot = trans.matrix3.z_axis.dot(offset);
    Vec3A rel_pos = offset - trans.matrix3.z_axis * up_dot;
    cb.add_linear_rel_pos_impulse(raycast_info.impulse * time_step, rel_pos,
                                  true, false);
  }
};

struct VehicleRL {
  float min_hit_dist_sq;
  uint32_t chassis_body_idx;
  WheelInfo wheels[k_num_wheels];

  RS_HDN void cast_rays(const World &w, const Vec3A *from, const Vec3A *to,
                       VehicleHit *hits) const {
    QuadRayResult r =
        QuadRayResult::make(from, to, static_cast<int>(chassis_body_idx));
    r.collision_filter_group |= k_group_dropshot_floor;
    r.min_hit_dist_sq = min_hit_dist_sq;
    grid_ray_test(w, from, to, r);
    for (int i = 0; i < k_num_wheels; ++i) {
      hits[i].valid = false;
      int idx = r.collision_obj_idx[i];
      if (idx >= 0 && w.has_contact_response(static_cast<uint32_t>(idx))) {
        hits[i].valid = true;
        hits[i].body_idx = static_cast<uint32_t>(idx);
        hits[i].point = r.hit_point_world[i];
        hits[i].normal = r.hit_normal_world[i].normalize_or_zero();
      }
    }
  }

  RS_HD void reset_transient_contacts() {
    for (int i = 0; i < k_num_wheels; ++i) {
      WheelInfo &wh = wheels[i];
      wh.has_raycast = false;
      wh.hard_point = Vec3A::zero();
      wh.axle_dir = Vec3A::zero();
      wh.engine_force = 0.0f;
      wh.brake = 0.0f;
      wh.steer_angle = 0.0f;
      wh.vel_at_contact_point = Vec3A::zero();
      wh.lat_friction = 1.0f;
      wh.long_friction = 1.0f;
      wh.extra_pushback = 0.0f;
    }
  }

  RS_HD Vec3A get_upwards_dir_from_wheel_contacts(const RigidBody &cb) const {
    Vec3A sum = Vec3A::zero();
    for (int i = 0; i < k_num_wheels; ++i) {
      if (wheels[i].has_raycast &&
          wheels[i].raycast_info.is_in_contact_with_world) {
        sum += wheels[i].raycast_info.contact_normal;
      }
    }
    Vec3A out;
    if (sum.try_normalize(out)) {
      return out;
    }
    return cb.get_up_vector();
  }

  RS_HD int num_world_contacts() const {
    int n = 0;
    for (int i = 0; i < k_num_wheels; ++i) {
      if (wheels[i].has_raycast &&
          wheels[i].raycast_info.is_in_contact_with_world) {
        ++n;
      }
    }
    return n;
  }

  RS_HD Vec3A front_axle(const Affine3A &trans, float steer_angle, bool &cached,
                         float &cached_angle, Vec3A &cached_axle) const {
    if (cached && cached_angle == steer_angle) {
      return cached_axle;
    }
    Quat q = quat_from_axis_angle_simd(trans.matrix3.z_axis, steer_angle);
    Vec3A axle = quat_mul_vec3a(q, trans.matrix3.y_axis);
    cached = true;
    cached_angle = steer_angle;
    cached_axle = axle;
    return axle;
  }

  RS_HD bool refresh_wheel_contacts(const World &w, const RigidBody &chassis,
                                    float time_step) {
    Affine3A trans = chassis.world_trans;
    Vec3A sources[k_num_wheels];
    Vec3A targets[k_num_wheels];
    for (int i = 0; i < k_num_wheels; ++i) {
      wheels[i].prepare_for_raycast(trans, sources[i], targets[i]);
    }
    VehicleHit hits[k_num_wheels];
    cast_rays(w, sources, targets, hits);
    bool cached = false;
    float cached_angle = 0.0f;
    Vec3A cached_axle = Vec3A::zero();
    for (int i = 0; i < k_num_wheels; ++i) {
      bool front = i < 2;
      if (hits[i].valid) {
        Vec3A axle = front ? front_axle(trans, wheels[i].steer_angle, cached,
                                        cached_angle, cached_axle)
                           : trans.matrix3.y_axis;
        wheels[i].apply_ray_cast(w, chassis, trans, axle, hits[i], time_step,
                                 front);
      } else {
        wheels[i].reset_wheel_suspension();
      }
    }
    for (int i = 0; i < k_num_wheels; ++i) {
      if (wheels[i].has_raycast &&
          wheels[i].raycast_info.is_in_contact_with_world) {
        return true;
      }
    }
    return false;
  }

  RS_HD void apply_single_wheel_dynamic_effects(World &w,
                                                const WheelInfo &wheel) {
    if (!wheel.has_raycast) {
      return;
    }
    const RaycastInfo &info = wheel.raycast_info;
    if (info.has_ground_stick) {
      RigidBody &ground = w.dyn_body(info.ground_body_idx);
      Vec3A offset = info.contact_point - ground.world_trans.translation;
      ground.add_linear_rel_pos_impulse(info.ground_stick, offset, true, false);
    }
    if (wheel.extra_pushback <= 0.0f) {
      return;
    }
    if (info.ground_body_idx < w.num_static()) {
      return;
    }
    RigidBody &victim = w.dyn_body(info.ground_body_idx);
    if (victim.user_idx != UserInfoType::Car) {
      return;
    }
    float full = wheel.extra_pushback * static_cast<float>(k_num_wheels);
    Vec3A offset = info.contact_point - victim.world_trans.translation;
    victim.add_linear_rel_pos_impulse(-info.contact_normal * full, offset, true,
                                      false);
  }

  RS_HDN void update(World &w, float time_step, float handbrake_val,
                    float real_throttle, bool three_wheels) {
    Affine3A trans = w.body(chassis_body_idx).world_trans;
    Vec3A chassis_translation = trans.translation;
    float friction_scale = w.body(chassis_body_idx).get_mass() / 3.0f;
    Vec3A sources[k_num_wheels];
    Vec3A targets[k_num_wheels];
    for (int i = 0; i < k_num_wheels; ++i) {
      wheels[i].prepare_for_raycast(trans, sources[i], targets[i]);
    }
    VehicleHit hits[k_num_wheels];
    cast_rays(w, sources, targets, hits);
    bool cached = false;
    float cached_angle = 0.0f;
    Vec3A cached_axle = Vec3A::zero();
    int num_in_contact = 0;
    for (int i = 0; i < k_num_wheels; ++i) {
      WheelInfo &wheel = wheels[i];
      if (!hits[i].valid) {
        wheel.reset_wheel_suspension();
        continue;
      }
      const RigidBody &chassis = w.body(chassis_body_idx);
      const RigidBody &hit_body = w.body(hits[i].body_idx);
      ++num_in_contact;
      bool front = i < 2;
      Vec3A axle = front ? front_axle(trans, wheel.steer_angle, cached,
                                      cached_angle, cached_axle)
                         : trans.matrix3.y_axis;
      wheel.apply_ray_cast(w, chassis, trans, axle, hits[i], time_step, front);
      wheel.refresh_friction_curves(
          chassis, chassis_translation, hits[i].normal, handbrake_val,
          real_throttle, three_wheels, !hit_body.is_static_obj());
      apply_single_wheel_dynamic_effects(w, wheel);
    }
    if (num_in_contact < 3) {
      for (int i = 0; i < k_num_wheels; ++i) {
        wheels[i].engine_force /= 4.0f;
      }
    }
    if (num_in_contact == 0) {
      return;
    }
    RigidBody &chassis = w.dyn_body(chassis_body_idx);
    for (int i = 0; i < k_num_wheels; ++i) {
      wheels[i].update_suspension(chassis, chassis_translation, time_step);
    }
    for (int i = 0; i < k_num_wheels; ++i) {
      wheels[i].update_friction_impulse(chassis, time_step, friction_scale);
    }
    for (int i = 0; i < k_num_wheels; ++i) {
      wheels[i].apply_friction_impulses(chassis, trans, time_step);
    }
  }
};

}
