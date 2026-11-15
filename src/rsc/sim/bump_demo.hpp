#pragma once

#include <cstdint>

#include "rsc/bullet/car_toi.hpp"
#include "rsc/bullet/game_rotation.hpp"
#include "rsc/math/libm.hpp"
#include "rsc/sim/car.hpp"
#include "rsc/sim/consts.hpp"

namespace rsc {

namespace bump {

RS_HD Vec3A normalize_for_cone(Vec3A v) {
  float len_sq = v.length_squared();
  if (len_sq != 1.0f) {
    if (len_sq < 1e-8f) {
      return Vec3A::zero();
    }
    return v * (1.0f / sqrt_f32(len_sq));
  }
  return v;
}

#ifdef RSC_NATIVE_MATH

inline constexpr float k_rotator_pi = 3.14159274f;

RS_HD float cone_rotator_sin(int32_t angle) {
  float j = static_cast<float>((angle >> 2) & 0x3fff);
  return ucrt_sinf(((j + j) * k_rotator_pi) * (1.0f / 16384.0f));
}

RS_HD int32_t cone_rotator_units(float y, float x) {
  float units = (game_atan2f(y, x) * 32768.0f) / k_rotator_pi;
  return sat_f32_to_i32(round_ties_even(units + units + 0.5f)) >> 1;
}

RS_HD QuatAxes body_axes(const CarState &state) {
  constexpr int32_t quarter_turn = 0x4000;
  Vec3A forward = state.phys.rot_mat.x_axis;
  Vec3A right = state.phys.rot_mat.y_axis;
  Vec3A up = state.phys.rot_mat.z_axis;
  int32_t yaw = cone_rotator_units(forward.y, forward.x);
  float forward_xy = static_cast<float>(sqrt_f64(
      static_cast<double>(forward.y * forward.y + forward.x * forward.x)));
  int32_t pitch = cone_rotator_units(forward.z, forward_xy);
  float sy = cone_rotator_sin(yaw);
  float cy = cone_rotator_sin(yaw + quarter_turn);
  float z_dot = (up.y * cy + up.x * -sy) + up.z * -0.0f;
  float y_dot = (right.y * cy + right.x * -sy) + right.z * -0.0f;
  int32_t roll = cone_rotator_units(z_dot, y_dot);
  float sp = cone_rotator_sin(pitch);
  float cp = cone_rotator_sin(pitch + quarter_turn);
  float sr = cone_rotator_sin(roll);
  float cr = cone_rotator_sin(roll + quarter_turn);
  float sp_sr = sp * sr;
  float cr_sp = cr * sp;
  return {{cy * cp, cp * sy, sp},
          {sp_sr * cy - cr * sy, cy * cr + sp_sr * sy, -sr * cp},
          {-(cr_sp * cy + sy * sr), cy * sr - cr_sp * sy, cp * cr}};
}

RS_HD Quat sweep_slerp(Quat q1, Quat q2, float t) {
  float raw = ((q1.x * q2.x + q1.y * q2.y) + q1.z * q2.z) + q1.w * q2.w;
  float cosom = raw >= 0.0f ? raw : -raw;
  float scale0;
  float scale1;
  if (cosom < 0.9999f) {
    float c = -1.0f > cosom ? -1.0f : (cosom < 1.0f ? cosom : 1.0f);
    float omega = game_acosf(c);
    float inv_sin = 1.0f / game_sinf(omega);
    scale0 = game_sinf((1.0f - t) * omega) * inv_sin;
    scale1 = game_sinf(omega * t) * inv_sin;
  } else {
    scale0 = 1.0f - t;
    scale1 = t;
  }
  scale1 = raw >= 0.0f ? scale1 : -scale1;
  return {scale1 * q2.x + q1.x * scale0, scale1 * q2.y + q1.y * scale0,
          scale1 * q2.z + q1.z * scale0, scale1 * q2.w + q1.w * scale0};
}

RS_HD Vec3A sweep_angular_velocity(const Mat3A &from, const Mat3A &to) {
  float m00 = from.x_axis.x, m01 = from.y_axis.x, m02 = from.z_axis.x;
  float m10 = from.x_axis.y, m11 = from.y_axis.y, m12 = from.z_axis.y;
  float m20 = from.x_axis.z, m21 = from.y_axis.z, m22 = from.z_axis.z;
  float co0 = m11 * m22 - m12 * m21;
  float co1 = m12 * m20 - m10 * m22;
  float co2 = m10 * m21 - m11 * m20;
  float s = 1.0f / ((m00 * co0 + m01 * co1) + m02 * co2);
  float i00 = co0 * s;
  float i01 = (m02 * m21 - m01 * m22) * s;
  float i02 = (m01 * m12 - m02 * m11) * s;
  float i10 = co1 * s;
  float i11 = (m00 * m22 - m02 * m20) * s;
  float i12 = (m02 * m10 - m00 * m12) * s;
  float i20 = co2 * s;
  float i21 = (m01 * m20 - m00 * m21) * s;
  float i22 = (m00 * m11 - m01 * m10) * s;
  auto row = [&](float b0, float b1, float b2) {
    return Vec3A{(i00 * b0 + i10 * b1) + i20 * b2,
                 (i01 * b0 + i11 * b1) + i21 * b2,
                 (i02 * b0 + i12 * b1) + i22 * b2};
  };
  Vec3A d0 = row(to.x_axis.x, to.y_axis.x, to.z_axis.x);
  Vec3A d1 = row(to.x_axis.y, to.y_axis.y, to.z_axis.y);
  Vec3A d2 = row(to.x_axis.z, to.y_axis.z, to.z_axis.z);
  Quat q = game_quat_normalize_scalar(
      game_get_rotation_scalar({d0.x, d1.x, d2.x}, {d0.y, d1.y, d2.y},
                               {d0.z, d1.z, d2.z}),
      -1.0f);
  float w = q.w;
  w = -1.0f <= w ? w : -1.0f;
  w = w <= 1.0f ? w : 1.0f;
  float half = game_acosf(w);
  float angle = half + half;
  float len2 = (q.x * q.x + q.y * q.y) + q.z * q.z;
  Vec3A axis{1.0f, 0.0f, 0.0f};
  if (len2 >= 1.4210855e-14f) {
    float inv = 1.0f / sqrt_f32(len2);
    axis = {inv * q.x, q.y * inv, inv * q.z};
  }
  return {axis.x * angle, axis.y * angle, axis.z * angle};
}

#else

RS_HD QuatAxes body_axes(const CarState &state) {
  return {state.phys.rot_mat.x_axis, state.phys.rot_mat.y_axis,
          state.phys.rot_mat.z_axis};
}

RS_HD float glam_acos_approx(float v) {
  bool nonnegative = v >= 0.0f;
  float x = abs_f32(v);
  float omx = 1.0f - x;
  if (omx < 0.0f) {
    omx = 0.0f;
  }
  float root = sqrt_f32(omx);
  float result =
      ((((((-0.0012624911f * x + 0.00667009f) * x - 0.017088126f) * x +
          0.03089188f) *
             x -
         0.050174303f) *
            x +
        0.08897899f) *
           x -
       0.2145988f) *
          x +
      1.5707963f;
  result *= root;
  return nonnegative ? result : k_pi - result;
}

RS_HD float glam_sin(float v) {
  float turns = glam_round_ties_even(v * 0.15915494f);
  float x = v - 6.2831855f * turns;
  float c = copysign_f32(k_pi, x);
  float reflected = c - x;
  x = abs_f32(x) <= 1.5707964f ? x : reflected;
  float x2 = x * x;
  float result = -2.3889859e-8f * x2 + 2.7525562e-6f;
  result = result * x2 + -0.00019840874f;
  result = result * x2 + 0.008333331f;
  result = result * x2 + -0.16666667f;
  result = result * x2 + 1.0f;
  return result * x;
}

RS_HD Quat sweep_slerp(Quat start, Quat end, float s) {
  float dot =
      (start.x * end.x + start.z * end.z) + (start.y * end.y + start.w * end.w);
  if (dot < 0.0f) {
    end = {end.x * -1.0f, end.y * -1.0f, end.z * -1.0f, end.w * -1.0f};
    dot = -dot;
  }
  if (dot > 1.0f - k_f32_epsilon) {
    float keep = 1.0f - s;
    return Quat{start.x * keep + end.x * s, start.y * keep + end.y * s,
                start.z * keep + end.z * s, start.w * keep + end.w * s}
        .normalize();
  }
  float theta = glam_acos_approx(dot);
  float scale1 = glam_sin(theta * (1.0f - s));
  float scale2 = glam_sin(theta * s);
  float theta_sin = glam_sin(theta * 1.0f);
  return {(end.x * scale2 + start.x * scale1) / theta_sin,
          (end.y * scale2 + start.y * scale1) / theta_sin,
          (end.z * scale2 + start.z * scale1) / theta_sin,
          (end.w * scale2 + start.w * scale1) / theta_sin};
}

RS_HD Mat3A glam_mat3a_inverse(const Mat3A &m) {
  Vec3A tmp0 = m.y_axis.cross(m.z_axis);
  Vec3A tmp1 = m.z_axis.cross(m.x_axis);
  Vec3A tmp2 = m.x_axis.cross(m.y_axis);
  Vec3A inv_det = Vec3A::splat(1.0f / m.x_axis.dot(tmp0));
  return Mat3A::from_cols(tmp0 * inv_det, tmp1 * inv_det, tmp2 * inv_det)
      .transpose();
}

RS_HD Vec3A sweep_angular_velocity(const Mat3A &from, const Mat3A &to) {
  Mat3A relative = glam_mat3a_inverse(from) * to;
  Quat q = game_quat_normalize_scalar(game_get_rotation_scalar(relative.x_axis,
                                                               relative.y_axis,
                                                               relative.z_axis),
                                      -1.0f);
  float length = sqrt_f32((q.x * q.x + q.y * q.y) + q.z * q.z);
  if (length >= 1.0e-8f) {
    float angle = 2.0f * atan2f_glibc(length, q.w);
    return Vec3A{q.x / length, q.y / length, q.z / length} * angle;
  }
  return Vec3A{1.0f, 0.0f, 0.0f} * 0.0f;
}

#endif

RS_HD Vec3A forward_axis(const CarState &state) {
  return body_axes(state).forward;
}

RS_HD Vec3A hitbox_origin(Vec3A pos, const QuatAxes &axes, Vec3A offset) {
  return pos + Vec3A{axes.forward.dot(offset), axes.right.dot(offset),
                     axes.up.dot(offset)};
}

RS_HD float cone_angle_deg(Vec3A delta, Vec3A forward, Vec3A axis) {
  float remove = delta.dot(axis);
  remove = remove >= 0.0f ? remove / 1.01f : remove * 1.01f;
  Vec3A projected = normalize_for_cone(delta - axis * remove);
  return game_acosf(clamp_f32(forward.dot(projected), -1.0f, 1.0f)) * 57.29578f;
}

RS_HD bool is_within_forward_cone(const CarState &attacker,
                                  const CarBodyConfig &attacker_config,
                                  const CarState &victim,
                                  const CarBodyConfig &victim_config,
                                  bool reverse, float max_yaw_deg,
                                  float max_pitch_deg) {
  QuatAxes attacker_axes = body_axes(attacker);
  QuatAxes victim_axes = body_axes(victim);
  Vec3A attacker_origin = hitbox_origin(attacker.phys.pos, attacker_axes,
                                        attacker_config.hitbox_pos_offset);
  Vec3A victim_origin = hitbox_origin(victim.phys.pos, victim_axes,
                                      victim_config.hitbox_pos_offset);
  Vec3A delta = normalize_for_cone(victim_origin - attacker_origin);
  Vec3A forward = reverse ? -attacker_axes.forward : attacker_axes.forward;
  if (cone_angle_deg(delta, forward, attacker_axes.right) > max_pitch_deg) {
    return false;
  }
  return cone_angle_deg(delta, forward, attacker_axes.up) <= max_yaw_deg;
}

struct CarToiTransform {
  Mat3A basis;
  Vec3A origin;

  RS_HD Affine3A affine() const { return {basis, origin}; }

  RS_HD CarToiTransform integrate(Vec3A lin, Vec3A ang, float dt) const {
    CarToiTransform out = *this;
    out.origin = {dt * lin.x + origin.x, dt * lin.y + origin.y,
                  dt * lin.z + origin.z};
    Quat q_mirror = Quat::identity();
    game_integrate_orientation_basis_columns(out.basis, q_mirror, ang, dt);
    return out;
  }
};

RS_HD CarToiTransform sweep_start(const CarState &state) {
  const Mat3A &rot = state.phys.rot_mat;
  return {game_set_rotation_scalar_columns(
              game_get_rotation_scalar(rot.x_axis, rot.y_axis, rot.z_axis)),
          {state.phys.pos.x * consts::uu_to_bt,
           state.phys.pos.y * consts::uu_to_bt,
           state.phys.pos.z * consts::uu_to_bt}};
}

RS_HD CarToiTransform hitbox_transform(const CarToiTransform &transform,
                                       const CarBodyConfig &config) {
  Vec3A offset{config.hitbox_pos_offset.x * consts::uu_to_bt,
               config.hitbox_pos_offset.y * consts::uu_to_bt,
               config.hitbox_pos_offset.z * consts::uu_to_bt};
  return {transform.basis, transform.affine().transform_point3a(offset)};
}

RS_HD float hitbox_half_extent_bt(float size) {
  float published = size - 2.5f;
  float half_with_margin = (published * 0.02f) * 0.5f + 0.025f;
  float size_uu = (half_with_margin * 50.0f) * 2.0f;
  return (size_uu * 0.5f) * consts::uu_to_bt;
}

RS_HD CarToiBox hitbox_toi_box(const CarBodyConfig &config) {
  return CarToiBox::make({hitbox_half_extent_bt(config.hitbox_size.x),
                          hitbox_half_extent_bt(config.hitbox_size.y),
                          hitbox_half_extent_bt(config.hitbox_size.z)});
}

RS_HD float interpolated_origin_axis(float from, float to, float fraction) {
  return (to * 50.0f - from * 50.0f) * fraction + from * 50.0f;
}

struct CarSweep {
  CarToiBox box;
  CarToiTransform from;
  CarToiTransform to;
  CarToiTransform hitbox_from;
  Vec3A lin;
  Vec3A ang;

  RS_HD static CarSweep make(const CarState &state, const CarBodyConfig &config,
                             Vec3A vel, float tick_dt) {
    CarSweep sweep;
    sweep.box = hitbox_toi_box(config);
    CarToiTransform start = sweep_start(state);
    Vec3A lin{vel.x * consts::uu_to_bt, vel.y * consts::uu_to_bt,
              vel.z * consts::uu_to_bt};
    Vec3A ang = state.phys.ang_vel;
    sweep.from = start.integrate({-lin.x, -lin.y, -lin.z},
                                 {-ang.x, -ang.y, -ang.z}, tick_dt);
    sweep.to = start.integrate(lin, ang, tick_dt);
    sweep.hitbox_from = hitbox_transform(sweep.from, config);
    CarToiTransform hitbox_to = hitbox_transform(sweep.to, config);
    sweep.lin = hitbox_to.origin - sweep.hitbox_from.origin;
    sweep.ang =
        sweep_angular_velocity(sweep.hitbox_from.basis, hitbox_to.basis);
    return sweep;
  }

  RS_HD void pose_at(float fraction, CarState &state) const {
    state.phys.pos = {
        interpolated_origin_axis(from.origin.x, to.origin.x, fraction),
        interpolated_origin_axis(from.origin.y, to.origin.y, fraction),
        interpolated_origin_axis(from.origin.z, to.origin.z, fraction)};
    Quat rot = game_quat_normalize_scalar(
        sweep_slerp(game_get_rotation_scalar(from.basis.x_axis,
                                             from.basis.y_axis,
                                             from.basis.z_axis),
                    game_get_rotation_scalar(to.basis.x_axis, to.basis.y_axis,
                                             to.basis.z_axis),
                    fraction),
        1.0e-8f);
    state.phys.rot_mat = game_set_rotation_scalar_columns(rot);
  }
};

RS_HD bool time_of_impact_fraction(const CarSweep &a, const CarSweep &b,
                                   float &fraction) {
  constexpr float allowed_penetration = 0.04f;
  constexpr float min_distance = 0.001f;
  constexpr uint32_t max_iterations = 64;
  float max_ang = a.ang.length() * a.box.angular_motion_disc() +
                  b.ang.length() * b.box.angular_motion_disc();
  Vec3A rel = b.lin - a.lin;
  if (rel.length() + max_ang == 0.0f) {
    return false;
  }
  CarToiContact contact;
  if (!box_sweep_closest_points(a.box, a.hitbox_from.affine(), b.box,
                                b.hitbox_from.affine(), contact)) {
    return false;
  }
  float distance = contact.distance + allowed_penetration;
  if (rel.dot(contact.normal) + max_ang <= k_f32_epsilon) {
    return false;
  }
  float lambda = 0.0f;
  fraction = 0.0f;
  uint32_t iter = 0;
  while (distance > min_distance) {
    float projection = rel.dot(contact.normal) + max_ang;
    if (projection <= k_f32_epsilon) {
      return false;
    }
    fraction = lambda + distance / projection;
    if (fraction > 1.0f || fraction < 0.0f || fraction <= lambda) {
      return false;
    }
    CarToiTransform ia = a.hitbox_from.integrate(a.lin, a.ang, fraction);
    CarToiTransform ib = b.hitbox_from.integrate(b.lin, b.ang, fraction);
    if (!box_sweep_closest_points(a.box, ia.affine(), b.box, ib.affine(),
                                  contact)) {
      return false;
    }
    distance = contact.distance + allowed_penetration;
    iter += 1;
    if (iter > max_iterations) {
      return false;
    }
    lambda = fraction;
  }
  return fraction < 1.0f;
}

RS_HD void get_time_of_impact_poses(CarState &a, const CarBodyConfig &ac,
                                    Vec3A a_vel, CarState &b,
                                    const CarBodyConfig &bc, Vec3A b_vel,
                                    float tick_time) {
  CarSweep sweep_a = CarSweep::make(a, ac, a_vel, tick_time);
  CarSweep sweep_b = CarSweep::make(b, bc, b_vel, tick_time);
  float fraction;
  if (!time_of_impact_fraction(sweep_a, sweep_b, fraction)) {
    return;
  }
  sweep_a.pose_at(fraction, a);
  sweep_b.pose_at(fraction, b);
}

RS_HD bool should_demolish(const CarState &attacker, const CarBodyConfig &ac,
                           const CarState &victim, const CarBodyConfig &vc,
                           float tick_time) {
  using namespace consts::car;
  if (!attacker.is_supersonic) {
    return false;
  }
  Vec3A start_vel = attacker.phys.vel;
  float forward_speed = start_vel.dot(forward_axis(attacker));
  if (abs_f32(forward_speed) < supersonic::maintain_min_speed) {
    return false;
  }
  CarState ap = attacker;
  CarState vp = victim;
  get_time_of_impact_poses(ap, ac, start_vel, vp, vc, victim.phys.vel,
                           tick_time);
  return is_within_forward_cone(ap, ac, vp, vc, forward_speed < 0.0f,
                                demo::yaw_limit_deg, demo::pitch_limit_deg);
}

}

}
