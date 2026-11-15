#pragma once

#include "rsc/math/lanes.hpp"
#include "rsc/math/libm.hpp"
#include "rsc/math/shared.hpp"
#include "rsc/math/vec.hpp"
#ifdef RSC_NATIVE_MATH
#include "rsc/math/ucrt_math.hpp"
#endif

namespace rsc {

RS_HD Vec3A quat_mul_vec3a(Quat q, Vec3A rhs) {
  Vec3A b{q.x, q.y, q.z};
  float w = q.w;
  float b2 = b.dot(b);
  float rb = rhs.dot(b);
  Vec3A c = b.cross(rhs);
  float ww_b2 = w * w - b2;
  float rb2 = rb * 2.0f;
  float w2 = w * 2.0f;
  return {(rhs.x * ww_b2 + b.x * rb2) + c.x * w2,
          (rhs.y * ww_b2 + b.y * rb2) + c.y * w2,
          (rhs.z * ww_b2 + b.z * rb2) + c.z * w2};
}

#ifdef RSC_NATIVE_MATH

RS_HD float game_sinf(float x) { return sinf_ucrt(x); }

RS_HD float game_cosf(float x) { return cosf_ucrt(x); }

RS_HD float game_atan2f(float y, float x) { return ucrt_atan2f(y, x); }

RS_HD float game_acosf(float x) { return ucrt_acosf(x); }

RS_HD Quat bullet_quat_from_basis_columns(Vec3A fwd, Vec3A right, Vec3A up) {
  float m00 = fwd.x, m01 = right.x, m02 = up.x;
  float m10 = fwd.y, m11 = right.y, m12 = up.y;
  float m20 = fwd.z, m21 = right.z, m22 = up.z;
  float trace = m00 + m11 + m22;
  float raw0, raw1, raw2, raw3, x;
  if (trace > 0.0f) {
    x = trace + 1.0f;
    raw0 = m21 - m12;
    raw1 = m02 - m20;
    raw2 = m10 - m01;
    raw3 = x;
  } else if (m00 < m11 ? m11 < m22 : m00 < m22) {
    x = m22 - m00 - m11 + 1.0f;
    raw0 = m02 + m20;
    raw1 = m12 + m21;
    raw2 = x;
    raw3 = m10 - m01;
  } else if (m00 < m11) {
    x = m11 - m22 - m00 + 1.0f;
    raw0 = m01 + m10;
    raw1 = x;
    raw2 = m21 + m12;
    raw3 = m02 - m20;
  } else {
    x = m00 - m11 - m22 + 1.0f;
    raw0 = x;
    raw1 = m10 + m01;
    raw2 = m20 + m02;
    raw3 = m21 - m12;
  }
  float scale = 0.5f / sqrt_f32(x);
  return {raw0 * scale, raw1 * scale, raw2 * scale, raw3 * scale};
}

RS_HD Quat game_get_rotation_scalar(Vec3A fwd, Vec3A right, Vec3A up) {
  float m00 = fwd.x, m01 = right.x, m02 = up.x;
  float m10 = fwd.y, m11 = right.y, m12 = up.y;
  float m20 = fwd.z, m21 = right.z, m22 = up.z;
  float trace = (m00 + m11) + m22;
  bool w_largest = trace > 0.0f;
  bool y_over_x = m00 < m11;
  bool z_largest = (y_over_x ? m11 : m00) < m22;
  float root = sqrt_f32(w_largest   ? trace + 1.0f
                        : z_largest ? ((m22 - m00) - m11) + 1.0f
                        : y_over_x  ? ((m11 - m22) - m00) + 1.0f
                                    : ((m00 - m11) - m22) + 1.0f);
  float diagonal = root * 0.5f;
  float s = 0.5f / root;
  float x_sum = (m01 + m10) * s;
  float y_sum = (m12 + m21) * s;
  float z_sum = (m02 + m20) * s;
  float x_diff = (m21 - m12) * s;
  float y_diff = (m02 - m20) * s;
  float z_diff = (m10 - m01) * s;
  return {w_largest   ? x_diff
          : z_largest ? z_sum
          : y_over_x  ? x_sum
                      : diagonal,
          w_largest   ? y_diff
          : z_largest ? y_sum
          : y_over_x  ? diagonal
                      : x_sum,
          w_largest   ? z_diff
          : z_largest ? diagonal
          : y_over_x  ? y_sum
                      : z_sum,
          w_largest   ? diagonal
          : z_largest ? z_diff
          : y_over_x  ? y_diff
                      : x_diff};
}

RS_HD Mat3A game_set_rotation_scalar_columns(Quat q) {
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

RS_HD Quat game_quat_normalize_scalar(Quat q, float min_len_sq) {
  float len_sq = ((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w;
  if (!(len_sq > min_len_sq)) {
    return Quat::identity();
  }
  float s = 1.0f / sqrt_f32(len_sq);
  return {q.x * s, q.y * s, q.z * s, q.w * s};
}

RS_HD Quat game_rotation_step(Vec3A ang_vel, float time_step) {
  float angle_sq =
      (ang_vel.x * ang_vel.x + ang_vel.y * ang_vel.y) + ang_vel.z * ang_vel.z;
  float angle = sqrt_f32(angle_sq);
  if (angle * time_step > k_frac_pi_4) {
    angle = k_frac_pi_4 / time_step;
  }
  float f;
  if (angle < 0.001f) {
    f = time_step * 0.5f -
        (((time_step * time_step) * time_step) * 0.020833334f * angle) * angle;
  } else {
    f = game_sinf((angle * 0.5f) * time_step) / angle;
  }
  return {ang_vel.x * f, ang_vel.y * f, ang_vel.z * f,
          game_cosf((angle * time_step) * 0.5f)};
}

RS_HD Quat game_apply_rotation_step(Quat step, Quat source) {
  const Quat &o = source;
  Quat p{((step.x * o.w + step.w * o.x) + step.y * o.z) - step.z * o.y,
         ((step.w * o.y + step.y * o.w) + step.z * o.x) - step.x * o.z,
         ((step.w * o.z + step.z * o.w) + step.x * o.y) - step.y * o.x,
         ((step.w * o.w - step.x * o.x) - step.y * o.y) - step.z * o.z};
  return game_quat_normalize_scalar(p, -1.0f);
}

RS_HD Vec3A steered_right_column(Vec3A car_up, Vec3A car_right,
                                 float steer_angle) {
  float half_angle = steer_angle * 0.5f;
  float sin_scale = half_angle == 0.0f && car_up.length_squared() > 0.0f
                        ? half_angle
                        : game_sinf(half_angle) / car_up.length();
  float cos_half = half_angle == 0.0f ? 1.0f : game_cosf(half_angle);
  Quat q{car_up.x * sin_scale, car_up.y * sin_scale, car_up.z * sin_scale,
         cos_half};
  Mat3A m = game_set_rotation_scalar_columns(q);
  return {m.x_axis.x * car_right.x + m.y_axis.x * car_right.y +
              m.z_axis.x * car_right.z,
          m.x_axis.y * car_right.x + m.y_axis.y * car_right.y +
              m.z_axis.y * car_right.z,
          m.x_axis.z * car_right.x + m.y_axis.z * car_right.y +
              m.z_axis.z * car_right.z};
}

#else

RS_HD float game_sinf(float x) { return sinf_glibc(x); }

RS_HD float game_cosf(float x) { return cosf_glibc(x); }

RS_HD float game_atan2f(float y, float x) { return atan2f_glibc(y, x); }

RS_HD float game_acosf(float x) { return acosf_glibc(x); }

RS_HD bool vec3a_is_normalized(Vec3A v) {
  return abs_f32(v.length_squared() - 1.0f) <= 2e-4f;
}

RS_HD bool quat_is_normalized(Quat q) {
  return abs_f32(q.dot4_self() - 1.0f) <= 2e-4f;
}

RS_HD bool quat_is_finite(Quat q) {
  return is_finite_f32(q.x) && is_finite_f32(q.y) && is_finite_f32(q.z) &&
         is_finite_f32(q.w);
}

RS_HD Quat game_quat_normalize_scalar(Quat q, float min_len_sq) {
  float len_sq = q.dot4_self();
  if (len_sq == 0.0f || !(len_sq > min_len_sq)) {
    return Quat::identity();
  }
  return q.normalize();
}

RS_HD Vec3A basis_column_or_axis(Vec3A column, Vec3A axis) {
  if (!column.is_finite()) {
    return axis;
  }
  Vec3A unit = column.normalize_or_zero();
  return unit == Vec3A::zero() ? axis : unit;
}

RS_HD Quat game_get_rotation_scalar(Vec3A fwd, Vec3A right, Vec3A up) {
  if (vec3a_is_normalized(fwd) && vec3a_is_normalized(right) &&
      vec3a_is_normalized(up)) {
    Quat quat = quat_from_mat3a(Mat3A::from_cols(fwd, right, up));
    if (quat_is_normalized(quat)) {
      return quat;
    }
    return game_quat_normalize_scalar(quat, 0.0f);
  }
  Quat quat = quat_from_mat3a(
      Mat3A::from_cols(basis_column_or_axis(fwd, {1.0f, 0.0f, 0.0f}),
                       basis_column_or_axis(right, {0.0f, 1.0f, 0.0f}),
                       basis_column_or_axis(up, {0.0f, 0.0f, 1.0f})));
  if (!quat_is_finite(quat)) {
    return Quat::identity();
  }
  return game_quat_normalize_scalar(quat, -1.0f);
}

RS_HD Quat bullet_quat_from_basis_columns(Vec3A fwd, Vec3A right, Vec3A up) {
  return game_get_rotation_scalar(fwd, right, up);
}

RS_HD Mat3A game_set_rotation_scalar_columns(Quat q) {
  return mat3a_from_quat(q);
}

RS_HD Quat game_rotation_step(Vec3A ang_vel, float time_step) {
  float angle = ang_vel.length();
  if (angle * time_step > k_frac_pi_4) {
    angle = k_frac_pi_4 / time_step;
  }
  float half_angle = (angle * time_step) * 0.5f;
  float f;
  if (angle < 0.001f) {
    f = time_step * 0.5f -
        (((time_step * time_step) * time_step) * 0.020833334f * angle) * angle;
  } else {
    f = game_sinf(half_angle) / angle;
  }
  return {ang_vel.x * f, ang_vel.y * f, ang_vel.z * f, game_cosf(half_angle)};
}

RS_HD Quat game_apply_rotation_step(Quat step, Quat source) {
  return game_quat_normalize_scalar(step * source, -1.0f);
}

RS_HD Vec3A steered_right_column(Vec3A car_up, Vec3A car_right,
                                 float steer_angle) {
  float half_angle = steer_angle * 0.5f;
  Vec3A v = car_up.normalize() * game_sinf(half_angle);
  return quat_mul_vec3a({v.x, v.y, v.z, game_cosf(half_angle)}, car_right);
}

#endif

RS_HD Quat game_basis_rotation(const Mat3A &basis) {
  return game_get_rotation_scalar(basis.x_axis, basis.y_axis, basis.z_axis);
}

RS_HD void game_integrate_orientation(Mat3A &basis, Quat &q_mirror, Quat source,
                                      Vec3A ang_vel, float time_step) {
  q_mirror =
      game_apply_rotation_step(game_rotation_step(ang_vel, time_step), source);
  basis = game_set_rotation_scalar_columns(q_mirror);
}

RS_HD void game_integrate_orientation_basis_columns(Mat3A &basis,
                                                    Quat &q_mirror,
                                                    Vec3A ang_vel,
                                                    float time_step) {
  game_integrate_orientation(basis, q_mirror, game_basis_rotation(basis),
                             ang_vel, time_step);
}

#ifdef RSC_HOST_LANES

#ifdef RSC_NATIVE_MATH

inline QuatLanes game_basis_rotation_lanes(const BasisLanes &basis,
                                           LaneMask &exact) {
  FloatLanes m00 = basis.x_axis.x, m01 = basis.y_axis.x, m02 = basis.z_axis.x;
  FloatLanes m10 = basis.x_axis.y, m11 = basis.y_axis.y, m12 = basis.z_axis.y;
  FloatLanes m20 = basis.x_axis.z, m21 = basis.y_axis.z, m22 = basis.z_axis.z;
  FloatLanes zero = FloatLanes::splat(0.0f);
  FloatLanes one = FloatLanes::splat(1.0f);
  FloatLanes half = FloatLanes::splat(0.5f);
  FloatLanes trace = (m00 + m11) + m22;
  LaneMask w_largest = trace > zero;
  LaneMask y_over_x = m00 < m11;
  LaneMask z_largest = select(y_over_x, m11, m00) < m22;
  FloatLanes root =
      sqrt(select(w_largest, trace + one,
                  select(z_largest, ((m22 - m00) - m11) + one,
                         select(y_over_x, ((m11 - m22) - m00) + one,
                                ((m00 - m11) - m22) + one))));
  FloatLanes diagonal = root * half;
  FloatLanes s = half / root;
  FloatLanes x_sum = m01 + m10;
  FloatLanes y_sum = m12 + m21;
  FloatLanes z_sum = m02 + m20;
  FloatLanes x_diff = m21 - m12;
  FloatLanes y_diff = m02 - m20;
  FloatLanes z_diff = m10 - m01;
  LaneMask x_largest =
      and_not(w_largest | z_largest | y_over_x, LaneMask::all_set());
  LaneMask y_largest = and_not(w_largest | z_largest, y_over_x);
  LaneMask z_selected = and_not(w_largest, z_largest);
  QuatLanes quat{
      select(x_largest, diagonal,
             select(w_largest, x_diff, select(z_largest, z_sum, x_sum)) * s),
      select(y_largest, diagonal,
             select(w_largest, y_diff, select(z_largest, y_sum, x_sum)) * s),
      select(z_selected, diagonal,
             select(w_largest, z_diff, select(y_over_x, y_sum, z_sum)) * s),
      select(w_largest, diagonal,
             select(z_largest, z_diff, select(y_over_x, y_diff, x_diff)) * s)};
  exact = exact & (((quat.x * quat.x + quat.y * quat.y) + quat.z * quat.z) +
                       quat.w * quat.w <=
                   FloatLanes::splat(k_f32_max));
  return quat;
}

inline QuatLanes game_quat_normalize_lanes(const QuatLanes &q, float min_len_sq,
                                           LaneMask &exact) {
  FloatLanes len_sq = ((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w;
  exact = exact & (len_sq > FloatLanes::splat(min_len_sq)) &
          (len_sq <= FloatLanes::splat(k_f32_max));
  FloatLanes s = FloatLanes::splat(1.0f) / sqrt(len_sq);
  return {q.x * s, q.y * s, q.z * s, q.w * s};
}

inline BasisLanes game_rotation_columns_lanes(const QuatLanes &q) {
  FloatLanes one = FloatLanes::splat(1.0f);
  FloatLanes d = ((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w;
  FloatLanes s = FloatLanes::splat(2.0f) / d;
  FloatLanes xs = q.x * s, ys = q.y * s, zs = q.z * s;
  FloatLanes wx = q.w * xs, wy = q.w * ys, wz = q.w * zs;
  FloatLanes xx = q.x * xs, xy = q.x * ys, xz = q.x * zs;
  FloatLanes yy = q.y * ys, yz = q.y * zs, zz = q.z * zs;
  return {{one - (yy + zz), xy + wz, xz - wy},
          {xy - wz, one - (xx + zz), yz + wx},
          {xz + wy, yz - wx, one - (xx + yy)}};
}

inline QuatLanes game_apply_rotation_step_lanes(const QuatLanes &step,
                                                const QuatLanes &o,
                                                LaneMask &exact) {
  return game_quat_normalize_lanes(
      {((step.x * o.w + step.w * o.x) + step.y * o.z) - step.z * o.y,
       ((step.w * o.y + step.y * o.w) + step.z * o.x) - step.x * o.z,
       ((step.w * o.z + step.z * o.w) + step.x * o.y) - step.y * o.x,
       ((step.w * o.w - step.x * o.x) - step.y * o.y) - step.z * o.z},
      -1.0f, exact);
}

#else

inline LaneMask is_normalized_lanes(FloatLanes len_sq) {
  return abs(len_sq - FloatLanes::splat(1.0f)) <= FloatLanes::splat(2e-4f);
}

inline FloatLanes length_squared_lanes(const Vec3Lanes &v) {
  return (v.x * v.x + v.y * v.y) + v.z * v.z;
}

inline FloatLanes dot4_lanes(const QuatLanes &q) {
  return (q.x * q.x + q.z * q.z) + (q.y * q.y + q.w * q.w);
}

inline QuatLanes game_basis_rotation_lanes(const BasisLanes &basis,
                                           LaneMask &exact) {
  FloatLanes m00 = basis.x_axis.x, m01 = basis.x_axis.y, m02 = basis.x_axis.z;
  FloatLanes m10 = basis.y_axis.x, m11 = basis.y_axis.y, m12 = basis.y_axis.z;
  FloatLanes m20 = basis.z_axis.x, m21 = basis.z_axis.y, m22 = basis.z_axis.z;
  FloatLanes zero = FloatLanes::splat(0.0f);
  FloatLanes one = FloatLanes::splat(1.0f);
  LaneMask xy_largest = m22 <= zero;
  FloatLanes diagonal = select(xy_largest, m11 - m00, m11 + m00);
  FloatLanes base = select(xy_largest, one - m22, one + m22);
  LaneMask first_largest = diagonal <= zero;
  FloatLanes four_sq = select(first_largest, base - diagonal, base + diagonal);
  FloatLanes inv = FloatLanes::splat(0.5f) / sqrt(four_sq);
  FloatLanes xy_sum = m01 + m10;
  FloatLanes xz_sum = m02 + m20;
  FloatLanes yz_sum = m12 + m21;
  FloatLanes yz_diff = m12 - m21;
  FloatLanes zx_diff = m20 - m02;
  FloatLanes xy_diff = m01 - m10;
  QuatLanes quat{select(xy_largest, select(first_largest, four_sq, xy_sum),
                        select(first_largest, xz_sum, yz_diff)) *
                     inv,
                 select(xy_largest, select(first_largest, xy_sum, four_sq),
                        select(first_largest, yz_sum, zx_diff)) *
                     inv,
                 select(xy_largest, select(first_largest, xz_sum, yz_sum),
                        select(first_largest, four_sq, xy_diff)) *
                     inv,
                 select(xy_largest, select(first_largest, yz_diff, zx_diff),
                        select(first_largest, xy_diff, four_sq)) *
                     inv};
  exact = exact & is_normalized_lanes(length_squared_lanes(basis.x_axis)) &
          is_normalized_lanes(length_squared_lanes(basis.y_axis)) &
          is_normalized_lanes(length_squared_lanes(basis.z_axis)) &
          is_normalized_lanes(dot4_lanes(quat));
  return quat;
}

inline QuatLanes game_quat_normalize_lanes(const QuatLanes &q, float min_len_sq,
                                           LaneMask &exact) {
  FloatLanes len_sq = dot4_lanes(q);
  exact = exact &
          and_not(len_sq == FloatLanes::splat(0.0f),
                  len_sq > FloatLanes::splat(min_len_sq)) &
          (len_sq <= FloatLanes::splat(k_f32_max));
  FloatLanes len = sqrt(len_sq);
  return {q.x / len, q.y / len, q.z / len, q.w / len};
}

inline BasisLanes game_rotation_columns_lanes(const QuatLanes &q) {
  FloatLanes one = FloatLanes::splat(1.0f);
  FloatLanes x2 = q.x + q.x;
  FloatLanes y2 = q.y + q.y;
  FloatLanes z2 = q.z + q.z;
  FloatLanes xx = q.x * x2;
  FloatLanes xy = q.x * y2;
  FloatLanes xz = q.x * z2;
  FloatLanes yy = q.y * y2;
  FloatLanes yz = q.y * z2;
  FloatLanes zz = q.z * z2;
  FloatLanes wx = q.w * x2;
  FloatLanes wy = q.w * y2;
  FloatLanes wz = q.w * z2;
  return {{one - (yy + zz), xy + wz, xz - wy},
          {xy - wz, one - (xx + zz), yz + wx},
          {xz + wy, yz - wx, one - (xx + yy)}};
}

inline QuatLanes game_apply_rotation_step_lanes(const QuatLanes &a,
                                                const QuatLanes &b,
                                                LaneMask &exact) {
  FloatLanes r0x = a.x * b.w + a.w * b.x;
  FloatLanes r0y = -(a.x * b.z) + a.w * b.y;
  FloatLanes r0z = a.x * b.y + a.w * b.z;
  FloatLanes r0w = -(a.x * b.x) + a.w * b.w;
  FloatLanes r1x = -(a.z * b.y) + a.y * b.z;
  FloatLanes r1y = a.z * b.x + a.y * b.w;
  FloatLanes r1z = a.z * b.w + -(a.y * b.x);
  FloatLanes r1w = -(a.z * b.z) + -(a.y * b.y);
  return game_quat_normalize_lanes({r0x + r1x, r0y + r1y, r0z + r1z, r0w + r1w},
                                   -1.0f, exact);
}

#endif

inline uint32_t lane_source(uint32_t lane, uint32_t count) {
  return lane < count ? lane : count - 1;
}

inline void game_basis_rotations(const Mat3A *const *bases, uint32_t n,
                                 Quat *rotations) {
  for (uint32_t first = 0; first < n; first += k_lane_count) {
    uint32_t count = n - first < k_lane_count ? n - first : k_lane_count;
    const Mat3A *const *chunk = bases + first;
    LaneMask exact = LaneMask::all_set();
    Quat lanes[k_lane_count];
    game_basis_rotation_lanes(BasisLanes::gather(*chunk[lane_source(0, count)],
                                                 *chunk[lane_source(1, count)],
                                                 *chunk[lane_source(2, count)],
                                                 *chunk[lane_source(3, count)]),
                              exact)
        .scatter(lanes);
    int exact_bits = exact.bits();
    for (uint32_t lane = 0; lane < count; ++lane) {
      rotations[first + lane] = ((exact_bits >> lane) & 1)
                                    ? lanes[lane]
                                    : game_basis_rotation(*chunk[lane]);
    }
  }
}

inline void game_apply_rotation_steps(const Quat *steps, const Quat *sources,
                                      uint32_t n, Quat *rotations,
                                      Mat3A *columns) {
  for (uint32_t first = 0; first < n; first += k_lane_count) {
    uint32_t count = n - first < k_lane_count ? n - first : k_lane_count;
    const Quat *step = steps + first;
    const Quat *source = sources + first;
    LaneMask exact = LaneMask::all_set();
    QuatLanes rotation_lanes = game_apply_rotation_step_lanes(
        QuatLanes::gather(
            step[lane_source(0, count)], step[lane_source(1, count)],
            step[lane_source(2, count)], step[lane_source(3, count)]),
        QuatLanes::gather(
            source[lane_source(0, count)], source[lane_source(1, count)],
            source[lane_source(2, count)], source[lane_source(3, count)]),
        exact);
    Quat lane_rotations[k_lane_count];
    Mat3A lane_columns[k_lane_count];
    rotation_lanes.scatter(lane_rotations);
    game_rotation_columns_lanes(rotation_lanes).scatter(lane_columns);
    int exact_bits = exact.bits();
    for (uint32_t lane = 0; lane < count; ++lane) {
      if ((exact_bits >> lane) & 1) {
        rotations[first + lane] = lane_rotations[lane];
        columns[first + lane] = lane_columns[lane];
      } else {
        rotations[first + lane] =
            game_apply_rotation_step(step[lane], source[lane]);
        columns[first + lane] =
            game_set_rotation_scalar_columns(rotations[first + lane]);
      }
    }
  }
}

#else

RS_HD void game_basis_rotations(const Mat3A *const *bases, uint32_t n,
                                Quat *rotations) {
  for (uint32_t i = 0; i < n; ++i) {
    rotations[i] = game_basis_rotation(*bases[i]);
  }
}

RS_HD void game_apply_rotation_steps(const Quat *steps, const Quat *sources,
                                     uint32_t n, Quat *rotations,
                                     Mat3A *columns) {
  for (uint32_t i = 0; i < n; ++i) {
    rotations[i] = game_apply_rotation_step(steps[i], sources[i]);
    columns[i] = game_set_rotation_scalar_columns(rotations[i]);
  }
}

#endif

}
