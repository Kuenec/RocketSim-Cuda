#pragma once

#include <cstdint>

#include "rsc/bullet/rigid_body.hpp"
#include "rsc/math/glibc_atan2_hypot.hpp"
#include "rsc/math/libm.hpp"
#include "rsc/math/shared.hpp"
#include "rsc/math/vec.hpp"
#include "rsc/sim/consts.hpp"

namespace rsc {

inline constexpr int k_sin_table_size = 16384;

RS_HDI float sin_tab_angle(const float *table, int32_t angle) {
  return table[(angle >> 2) & 0x3fff];
}

RS_HD int32_t rotator_units(float y, float x) {
  float units = (atan2f_glibc(y, x) * 32768.0f) / k_pi;
  return sat_f32_to_i32(round_ties_even(units + units + 0.5f)) >> 1;
}

RS_HD int32_t dodge_yaw_units(float y, float x) {
  constexpr float scale = 32768.0f / k_pi;
  float units = atan2f_glibc(y, x) * scale;
  return sat_f32_to_i32(round_ties_even(units));
}

RS_HD Vec3A rotator_forward(const float *table, Vec3A forward) {
  int32_t yaw = rotator_units(forward.y, forward.x);
  float xy = static_cast<float>(sqrt_f64(
      static_cast<double>(forward.y * forward.y + forward.x * forward.x)));
  int32_t pitch = rotator_units(forward.z, xy);
  float cos_pitch = sin_tab_angle(table, pitch + 0x4000);
  return {cos_pitch * sin_tab_angle(table, yaw + 0x4000),
          cos_pitch * sin_tab_angle(table, yaw), sin_tab_angle(table, pitch)};
}

RS_HD int32_t axis_angle_to_int(float a0, float a1) {
  double ang = atan2_glibc(static_cast<double>(a0), static_cast<double>(a1));
  float u = (static_cast<float>(ang) * 32768.0f) / k_pi;
  float v = u * 2.0f + 0.5f;
  return sat_f32_to_i32(round_ties_even(v)) >> 1;
}

RS_HD float autoflip_impulse_sign(const float *table, const Mat3A &rot) {
  Vec3A col0 = rot.x_axis;
  Vec3A col1 = rot.y_axis;
  Vec3A col2 = rot.z_axis;
  int32_t yaw = axis_angle_to_int(col0.y, col0.x);
  float len = static_cast<float>(
      hypot_glibc(static_cast<double>(col0.x), static_cast<double>(col0.y)));
  int32_t pitch = axis_angle_to_int(col0.z, len);
  float tp = sin_tab_angle(table, pitch);
  float ty = sin_tab_angle(table, yaw);
  float tr = sin_tab_angle(table, 0);
  float tpq = sin_tab_angle(table, pitch + 0x4000);
  float tyq = sin_tab_angle(table, yaw + 0x4000);
  float trq = sin_tab_angle(table, 0x4000);
  Vec3A r1{tp * tr * tyq - trq * ty, tyq * trq + tp * tr * ty, -tr * tpq};
  int32_t roll = axis_angle_to_int(col2.dot(r1), col1.dot(r1));
  int16_t roll16 = static_cast<int16_t>(static_cast<uint16_t>(roll));
  return roll16 > 0 ? 1.0f : -1.0f;
}

RS_HD Vec3A quantize_axis_inputs(Vec3A ctrls) {
  Vec3A upper = Vec3A::splat(128.0f);
  Vec3A lower = Vec3A::splat(127.0f);
  Vec3A clamped = ctrls.clamp(Vec3A::splat(-1.0f), Vec3A::one());
  Vec3A scale = Vec3A::select(clamped.cmplt(Vec3A::zero()), upper, lower);
  Vec3A biased = clamped * scale + upper;
  Vec3A w = biased + biased + Vec3A::splat(0.5f);
  Vec3A r = w.round();
  Vec3A byte{static_cast<float>((sat_f32_to_i32(r.x) >> 1) & 0xff),
             static_cast<float>((sat_f32_to_i32(r.y) >> 1) & 0xff),
             static_cast<float>((sat_f32_to_i32(r.z) >> 1) & 0xff)};
  Vec3A s = byte - upper;
  return Vec3A::select(s.cmplt(Vec3A::zero()), s * (1.0f / upper), s / lower);
}

RS_HD float quantize_axis_input(float x) {
  float clamped = clamp_f32(x, -1.0f, 1.0f);
  float y = clamped < 0.0f ? max_f32_c(clamped * 128.0f, -128.0f)
                           : min_f32_c(clamped * 127.0f, 127.0f);
  float w = ((y + 128.0f) + (y + 128.0f)) + 0.5f;
  int32_t eax = sat_f32_to_i32(round_ties_even(w));
  uint8_t byte = static_cast<uint8_t>((eax >> 1) & 0xff);
  float s = static_cast<float>(byte) - 128.0f;
  if (byte < 0x80) {
    return s * (1.0f / 128.0f);
  }
  return s / 127.0f;
}

namespace quant {

RS_HD Vec3A sign3(Vec3A v) {
  return Vec3A::select(v.cmpeq(Vec3A::zero()), Vec3A::zero(), v.signum());
}

RS_HD Vec3A quantize_position(Vec3A vec, float scale) {
  return (vec * scale + 0.5f).floor() * (1.0f / scale);
}

RS_HD Vec3A quantize_velocity(Vec3A vec, float scale) {
  float inv_scale = 1.0f / scale;
  Vec3A scaled = vec * scale;
  Vec3A rounded;
  if (scaled.abs().cmplt(Vec3A::splat(2147483648.0f)).all()) {
    rounded = scaled.trunc() * inv_scale;
  } else {
    rounded = {static_cast<float>(sat_f32_to_i32(scaled.x)) * inv_scale,
               static_cast<float>(sat_f32_to_i32(scaled.y)) * inv_scale,
               static_cast<float>(sat_f32_to_i32(scaled.z)) * inv_scale};
  }
  float offset_mag = 0.1f * inv_scale;
  return rounded + (sign3(rounded) * offset_mag);
}

RS_HD Mat3A set_rotation_scalar(Quat q) {
  float d = ((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w;
  float s = 2.0f / d;
  float xs = q.x * s, ys = q.y * s, zs = q.z * s;
  float wx = q.w * xs, wy = q.w * ys, wz = q.w * zs;
  float xx = q.x * xs, xy = q.x * ys, xz = q.x * zs;
  float yy = q.y * ys, yz = q.y * zs, zz = q.z * zs;
  return Mat3A::from_cols({1.0f - (yy + zz), xy + wz, xz - wy},
                          {xy - wz, 1.0f - (xx + zz), yz + wx},
                          {xz + wy, yz - wx, 1.0f - (xx + yy)});
}

RS_HD Quat normalize_scalar(Quat q, float min_len_sq) {
  float len_sq = ((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w;
  if (!(len_sq > min_len_sq)) {
    return Quat::identity();
  }
  float s = 1.0f / sqrt_f32(len_sq);
  return {q.x * s, q.y * s, q.z * s, q.w * s};
}

RS_HD float quantize_component(float c) {
  constexpr int32_t max = (1 << 18) - 1;
  float max_f = static_cast<float>(max);
  float v = ((c * k_sqrt_2) * 0.5f + 0.5f) * max_f;
  v = (v + v) + 0.5f;
  int32_t k = sat_f32_to_i32(round_ties_even(v)) >> 1;
  if (k < 0) {
    k = 0;
  }
  if (k > max) {
    k = max;
  }
  float r = static_cast<float>(k) / max_f - 0.5f;
  return (r + r) * 0.70710677f;
}

struct QuatSplit {
  int big;
  float d[3];
};

RS_HD QuatSplit split_quat(Quat q_in) {
  Quat q = normalize_scalar(q_in, 1e-8f);
  float c[4] = {q.x, q.y, q.z, q.w};
  float a[4] = {abs_f32(c[0]), abs_f32(c[1]), abs_f32(c[2]), abs_f32(c[3])};
  int i01 = a[0] < a[1] ? 1 : 0;
  int i23 = a[2] < a[3] ? 3 : 2;
  int big = a[i01] >= a[i23] ? i01 : i23;
  float sign = c[big] < 0.0f ? -1.0f : 1.0f;
  return {big,
          {(big == 0 ? c[1] : c[0]) * sign, (big <= 1 ? c[2] : c[1]) * sign,
           (big <= 2 ? c[3] : c[2]) * sign}};
}

RS_HD void quantize_split(QuatSplit &s) {
  for (float &v : s.d) {
    v = quantize_component(v);
  }
}

RS_HD Quat join_quat(const QuatSplit &s) {
  float rest = ((1.0f - s.d[0] * s.d[0]) - s.d[1] * s.d[1]) - s.d[2] * s.d[2];
  float dropped = sqrt_f32(rest);
  return normalize_scalar(
      {s.big == 0 ? dropped : s.d[0],
       s.big == 1 ? dropped : (s.big == 0 ? s.d[0] : s.d[1]),
       s.big == 2 ? dropped : (s.big <= 1 ? s.d[1] : s.d[2]),
       s.big == 3 ? dropped : s.d[2]},
      1e-8f);
}

RS_HD Quat quantize_quat(Quat q_in) {
  QuatSplit s = split_quat(q_in);
  quantize_split(s);
  return join_quat(s);
}

RS_HD void quantize_quats(Quat *q, uint32_t n) {
  QuatSplit s[k_max_dyn_bodies];
  for (uint32_t i = 0; i < n; ++i) {
    s[i] = split_quat(q[i]);
  }
  for (uint32_t i = 0; i < n; ++i) {
    quantize_split(s[i]);
  }
  for (uint32_t i = 0; i < n; ++i) {
    q[i] = join_quat(s[i]);
  }
}

}

RS_HD void quantize_body_motion(RigidBody &body) {
  Vec3A new_pos =
      quant::quantize_position(body.world_trans.translation * consts::bt_to_uu,
                               consts::quantize::pos_scale) *
      consts::uu_to_bt;
  Vec3A new_vel = quant::quantize_velocity(body.lin_vel * consts::bt_to_uu,
                                           consts::quantize::vel_scale) *
                  consts::uu_to_bt;
  Vec3A new_ang_vel =
      quant::quantize_velocity(body.ang_vel, consts::quantize::ang_vel_scale);
  body.set_world_pos(new_pos);
  body.lin_vel = new_vel;
  body.ang_vel = new_ang_vel;
}

RS_HD void quantize_body_rotation(RigidBody &body, Quat rot) {
  Affine3A trans = body.world_trans;
  trans.matrix3 = quant::set_rotation_scalar(rot);
  body.set_world_trans_and_rot(trans, rot);
}

RS_HD void quantize_body(RigidBody &body) {
  quantize_body_motion(body);
  if (body.collision_flags & k_flag_no_angular_motion) {
    return;
  }
  quantize_body_rotation(body, quant::quantize_quat(body.world_quat));
}

}
