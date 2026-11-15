#pragma once

#include <cstdint>

#include "rsc/bullet/rigid_body.hpp"
#include "rsc/bullet/world_types.hpp"
#include "rsc/math/libm.hpp"
#include "rsc/math/shared.hpp"
#include "rsc/math/vec.hpp"
#include "rsc/sim/consts.hpp"

namespace rsc {

struct QuatAxes {
  Vec3A forward;
  Vec3A right;
  Vec3A up;
};

#ifdef RSC_NATIVE_MATH

inline constexpr int k_sin_table_size = 16384;
inline constexpr int32_t k_rotator_quarter_turn = 0x4000;

struct Rotator {
  int32_t pitch;
  int32_t yaw;
  int32_t roll;
};

RS_HDI float rotator_sin(const StaticWorldData &st, int32_t angle) {
  return st.sin_table[(angle >> 2) & 0x3fff];
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

RS_HD Vec3A rotator_forward(const StaticWorldData &st, Vec3A forward) {
  int32_t yaw = rotator_units(forward.y, forward.x);
  float xy = static_cast<float>(sqrt_f64(
      static_cast<double>(forward.y * forward.y + forward.x * forward.x)));
  int32_t pitch = rotator_units(forward.z, xy);
  float cos_pitch = rotator_sin(st, pitch + k_rotator_quarter_turn);
  return {cos_pitch * rotator_sin(st, yaw + k_rotator_quarter_turn),
          cos_pitch * rotator_sin(st, yaw), rotator_sin(st, pitch)};
}

RS_HD Vec3A dodge_direction(const StaticWorldData &st, Vec3A forward) {
  int32_t yaw = dodge_yaw_units(forward.y, forward.x);
  return {rotator_sin(st, yaw + k_rotator_quarter_turn), rotator_sin(st, yaw),
          0.0f};
}

RS_HD Vec3A rotator_up_axis(const StaticWorldData &st, Rotator r) {
  float sy = rotator_sin(st, r.yaw);
  float cy = rotator_sin(st, r.yaw + k_rotator_quarter_turn);
  float sp = rotator_sin(st, r.pitch);
  float cp = rotator_sin(st, r.pitch + k_rotator_quarter_turn);
  float sr = rotator_sin(st, r.roll);
  float cr = rotator_sin(st, r.roll + k_rotator_quarter_turn);
  float cr_sp = cr * sp;
  return {-(cr_sp * cy + sy * sr), cy * sr - cr_sp * sy, cp * cr};
}

RS_HD Rotator rotator_from_axes(const StaticWorldData &st, Vec3A forward,
                                Vec3A right, Vec3A up) {
  int32_t yaw = rotator_units(forward.y, forward.x);
  float forward_xy = static_cast<float>(sqrt_f64(
      static_cast<double>(forward.y * forward.y + forward.x * forward.x)));
  int32_t pitch = rotator_units(forward.z, forward_xy);
  float sy = rotator_sin(st, yaw);
  float cy = rotator_sin(st, yaw + k_rotator_quarter_turn);
  float z_dot = (up.y * cy + up.x * -sy) + up.z * -0.0f;
  float y_dot = (right.y * cy + right.x * -sy) + right.z * -0.0f;
  return {pitch, yaw, rotator_units(z_dot, y_dot)};
}

RS_HD QuatAxes quat_axes(Quat q) {
  float z2 = q.z + q.z;
  float y2 = q.y + q.y;
  float x2x = (q.x + q.x) * q.x;
  float x2w = (q.x + q.x) * q.w;
  return {
      {1.0f - (z2 * q.z + y2 * q.y), z2 * q.w + y2 * q.x, z2 * q.x - y2 * q.w},
      {y2 * q.x - z2 * q.w, 1.0f - (z2 * q.z + x2x), x2w + z2 * q.y},
      {y2 * q.w + z2 * q.x, z2 * q.y - x2w, 1.0f - (y2 * q.y + x2x)}};
}

RS_HD Vec3A quat_rotate(Quat q, Vec3A v) {
  float qq = (q.y * q.y + q.x * q.x) + q.z * q.z;
  float d = q.w * q.w - qq;
  float w2 = q.w + q.w;
  float dot = (v.x * q.x + v.y * q.y) + v.z * q.z;
  float t = dot + dot;
  float cx = v.z * q.y - v.y * q.z;
  float cy = v.x * q.z - v.z * q.x;
  float cz = v.y * q.x - v.x * q.y;
  return {(cx * w2 + d * v.x) + t * q.x, (cy * w2 + d * v.y) + t * q.y,
          (cz * w2 + d * v.z) + t * q.z};
}

RS_HD Vec3A auto_flip_up(const StaticWorldData &st, Quat q) {
  Mat3A m = mat3a_from_quat(q);
  return rotator_up_axis(st,
                         rotator_from_axes(st, m.x_axis, m.y_axis, m.z_axis));
}

#else

RS_HD Vec3A rotator_forward(const StaticWorldData &, Vec3A forward) {
  return forward.normalize_or_zero();
}

RS_HD Vec3A dodge_direction(const StaticWorldData &, Vec3A forward) {
  Vec3A horizontal = Vec3A{forward.x, forward.y, 0.0f}.normalize_or_zero();
  if (horizontal == Vec3A::zero()) {
    float yaw = atan2f_glibc(forward.y, forward.x);
    return {cosf_glibc(yaw), sinf_glibc(yaw), 0.0f};
  }
  return horizontal;
}

RS_HD QuatAxes quat_axes(Quat q) {
  Mat3A m = mat3a_from_quat(q);
  return {m.x_axis, m.y_axis, m.z_axis};
}

RS_HD Vec3A quat_rotate(Quat q, Vec3A v) { return quat_mul_vec3a(q, v); }

RS_HD Vec3A auto_flip_up(const StaticWorldData &, Quat q) {
  return quat_mul_vec3a(q, {0.0f, 0.0f, 1.0f});
}

#endif

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
  Quat q = game_quat_normalize_scalar(q_in, 1e-8f);
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
  return game_quat_normalize_scalar(
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

#ifdef RSC_HOST_LANES

inline FloatLanes quantize_component_lanes(FloatLanes c, LaneMask &exact) {
  constexpr int32_t max = (1 << 18) - 1;
  FloatLanes max_f = FloatLanes::splat(static_cast<float>(max));
  FloatLanes half = FloatLanes::splat(0.5f);
  FloatLanes v = ((c * FloatLanes::splat(k_sqrt_2)) * half + half) * max_f;
  v = (v + v) + half;
  exact = exact & (abs(v) < FloatLanes::splat(1073741824.0f));
  FloatLanes r =
      to_float(clamp(round_ties_even_to_int(v) >> 1, 0, max)) / max_f - half;
  return (r + r) * FloatLanes::splat(0.70710677f);
}

inline QuatLanes quantize_quat_lanes(const QuatLanes &q_in, LaneMask &exact) {
  QuatLanes q = game_quat_normalize_lanes(q_in, 1e-8f, exact);
  FloatLanes abs_x = abs(q.x), abs_y = abs(q.y), abs_z = abs(q.z),
             abs_w = abs(q.w);
  LaneMask y_over_x = abs_x < abs_y;
  LaneMask w_over_z = abs_z < abs_w;
  LaneMask big_is_xy =
      select(y_over_x, abs_y, abs_x) >= select(w_over_z, abs_w, abs_z);
  LaneMask big_is_x = and_not(y_over_x, big_is_xy);
  LaneMask big_is_y = big_is_xy & y_over_x;
  LaneMask big_is_w = and_not(big_is_xy, w_over_z);
  FloatLanes big =
      select(big_is_xy, select(y_over_x, q.y, q.x), select(w_over_z, q.w, q.z));
  FloatLanes sign = select(big < FloatLanes::splat(0.0f),
                           FloatLanes::splat(-1.0f), FloatLanes::splat(1.0f));
  FloatLanes d0 =
      quantize_component_lanes(select(big_is_x, q.y, q.x) * sign, exact);
  FloatLanes d1 =
      quantize_component_lanes(select(big_is_xy, q.z, q.y) * sign, exact);
  FloatLanes d2 =
      quantize_component_lanes(select(big_is_w, q.z, q.w) * sign, exact);
  FloatLanes dropped =
      sqrt(((FloatLanes::splat(1.0f) - d0 * d0) - d1 * d1) - d2 * d2);
  return game_quat_normalize_lanes(
      {select(big_is_x, dropped, d0),
       select(big_is_y, dropped, select(big_is_x, d0, d1)),
       select(big_is_xy, d1, select(w_over_z, d2, dropped)),
       select(big_is_w, dropped, d2)},
      1e-8f, exact);
}

#else

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

#endif

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

RS_HD Quat body_rotation_from_basis(const RigidBody &body) {
  return game_basis_rotation(body.world_trans.matrix3);
}

RS_HD void quantize_body_rotation(RigidBody &body, Quat rot) {
  Affine3A trans = body.world_trans;
  trans.matrix3 = game_set_rotation_scalar_columns(rot);
  body.set_world_trans_and_rot(trans, rot);
}

#ifdef RSC_HOST_LANES

inline void quantize_body_rotations(RigidBody *const *bodies, uint32_t n) {
  for (uint32_t first = 0; first < n; first += k_lane_count) {
    uint32_t count = n - first < k_lane_count ? n - first : k_lane_count;
    const Mat3A *bases[k_lane_count];
    for (uint32_t lane = 0; lane < k_lane_count; ++lane) {
      uint32_t source = lane < count ? lane : count - 1;
      bases[lane] = &bodies[first + source]->world_trans.matrix3;
    }
    LaneMask exact = LaneMask::all_set();
    QuatLanes rot_lanes = quant::quantize_quat_lanes(
        game_basis_rotation_lanes(
            BasisLanes::gather(*bases[0], *bases[1], *bases[2], *bases[3]),
            exact),
        exact);
    Quat rots[k_lane_count];
    Mat3A columns[k_lane_count];
    rot_lanes.scatter(rots);
    game_rotation_columns_lanes(rot_lanes).scatter(columns);
    int exact_bits = exact.bits();
    for (uint32_t lane = 0; lane < count; ++lane) {
      RigidBody &body = *bodies[first + lane];
      if ((exact_bits >> lane) & 1) {
        Affine3A trans = body.world_trans;
        trans.matrix3 = columns[lane];
        body.set_world_trans_and_rot(trans, rots[lane]);
      } else {
        quantize_body_rotation(
            body, quant::quantize_quat(body_rotation_from_basis(body)));
      }
    }
  }
}

#else

RS_HD void quantize_body_rotations(RigidBody *const *bodies, uint32_t n) {
  Quat rots[k_max_dyn_bodies];
  for (uint32_t i = 0; i < n; ++i) {
    rots[i] = body_rotation_from_basis(*bodies[i]);
  }
  quant::quantize_quats(rots, n);
  for (uint32_t i = 0; i < n; ++i) {
    quantize_body_rotation(*bodies[i], rots[i]);
  }
}

#endif

RS_HD void quantize_body(RigidBody &body) {
  quantize_body_motion(body);
  if (body.collision_flags & k_flag_no_angular_motion) {
    return;
  }
  quantize_body_rotation(body,
                         quant::quantize_quat(body_rotation_from_basis(body)));
}

}
