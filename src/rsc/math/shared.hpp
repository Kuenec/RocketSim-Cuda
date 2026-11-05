#pragma once

#include <cstdint>

#include "rsc/math/hd.hpp"
#include "rsc/math/libm.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

struct Aabb {
  Vec3A min, max;

  RS_HDI static Aabb zero() { return {Vec3A::zero(), Vec3A::zero()}; }
  RS_HDI Vec3A center() const { return (min + max) * 0.5f; }
  RS_HDI float area() const {
    Vec3A e = max - min;
    return 2.0f * (e.x * e.y + e.x * e.z + e.y * e.z);
  }
  RS_HDI bool intersects(const Aabb &rhs) const {
    return min.cmple(rhs.max).all() && max.cmpge(rhs.min).all();
  }
  RS_HDI Aabb combine(const Aabb &rhs) const {
    return {min.min(rhs.min), max.max(rhs.max)};
  }
  RS_HDI Aabb &operator+=(const Aabb &rhs) {
    min = min.min(rhs.min);
    max = max.max(rhs.max);
    return *this;
  }
  RS_HDI Aabb operator+(const Aabb &rhs) const { return combine(rhs); }
  RS_HDI Aabb operator-(Vec3A v) const { return {min - v, max - v}; }

  RS_HDI static Aabb from_half_extents_transform(Vec3A half_extents,
                                                 float margin,
                                                 const Affine3A &t) {
    Vec3A half_with_margin = half_extents + margin;
    Mat3A abs_b = t.matrix3.abs();
    Vec3A center = t.translation;
    Vec3A extent = abs_b * half_with_margin;
    return {center - extent, center + extent};
  }

  RS_HDI Aabb transform(const Affine3A &trans, float margin) const {
    Vec3A local_half_extents = 0.5f * (max - min) + margin;
    Vec3A local_center = center();
    Mat3A abs_b = trans.matrix3.abs();
    Vec3A c = trans.transform_point3a(local_center);
    Vec3A extent = abs_b * local_half_extents;
    return {c - extent, c + extent};
  }
};

RS_HD float round_ties_even(float x) {
  constexpr float magic_value = 8388608.0f;
  if (abs_f32(x) < magic_value) {
    float magic = copysign_f32(magic_value, x);
    float rounded = (x + magic) - magic;
    if (rounded == 0.0f) {
      return copysign_f32(0.0f, x);
    }
    return rounded;
  }
  return x;
}

RS_HD Vec3A round_ties_even_vec3a(Vec3A v) {
  Vec3A limit = Vec3A::splat(8388608.0f);
  if (v.abs().cmplt(limit).all()) {
    Vec3A magic = limit.copysign(v);
    Vec3A rounded = (v + magic) - magic;
    return Vec3A::select(rounded.cmpeq(Vec3A::zero()), rounded.copysign(v),
                         rounded);
  }
  return {round_ties_even(v.x), round_ties_even(v.y), round_ties_even(v.z)};
}

struct LinearPiece {
  float base_x, base_y, max_x, max_y, x_diff, y_diff;
};

template <int N> struct LinearPieceCurve {
  LinearPiece curve[N];

  RS_HD static constexpr LinearPieceCurve make(const float (&points)[N][2]) {
    LinearPieceCurve result{};
    result.curve[0] = {0.0f, 0.0f, points[0][0], points[0][1], 0.0f, 0.0f};
    for (int i = 1; i < N; ++i) {
      result.curve[i] = {points[i - 1][0],
                         points[i - 1][1],
                         points[i][0],
                         points[i][1],
                         points[i][0] - points[i - 1][0],
                         points[i][1] - points[i - 1][1]};
    }
    return result;
  }

  RS_HD float get_output(float input) const {
    if (N == 1) {
      return curve[0].max_y;
    }
    if (input <= curve[0].max_x) {
      return curve[0].max_y;
    }
    int i = 1;
    while (i < N) {
      if (curve[i].max_x > input) {
        break;
      }
      ++i;
    }
    if (i == N) {
      return curve[N - 1].max_y;
    }
    const LinearPiece &pair = curve[i];
    float interp_frac = (input - pair.base_x) / pair.x_diff;
    return pair.y_diff * interp_frac + pair.base_y;
  }
};

struct Angle {
  float yaw, pitch, roll;

  RS_HD static Angle from_forward(Vec3A forward) {
    float yaw, pitch;
    if (abs_f32(forward.y) > k_f32_epsilon ||
        abs_f32(forward.x) > k_f32_epsilon) {
      yaw = atan2f_glibc(forward.y, forward.x);
      float dist_2d = forward.with_z(0.0f).length();
      pitch = -atan2f_glibc(-forward.z, dist_2d);
    } else {
      yaw = 0.0f;
      if (forward.z > k_f32_epsilon) {
        pitch = k_frac_pi_2;
      } else if (forward.z < -k_f32_epsilon) {
        pitch = -k_frac_pi_2;
      } else {
        pitch = 0.0f;
      }
    }
    return {yaw, pitch, 0.0f};
  }

  RS_HDI Angle operator-(const Angle &rhs) const {
    return {yaw - rhs.yaw, pitch - rhs.pitch, roll - rhs.roll};
  }

  RS_HD void normalize_fix() {
    yaw = rem_euclid_f32(yaw + k_pi, k_tau) - k_pi;
    pitch = rem_euclid_f32(pitch + k_frac_pi_2, k_pi) - k_frac_pi_2;
    roll = rem_euclid_f32(roll + k_pi, k_tau) - k_pi;
  }

  RS_HD Vec3A get_forward_vec() const {
    float neg_pitch = -pitch;
    float sp = sinf_glibc(neg_pitch);
    float cp = cosf_glibc(neg_pitch);
    float sy = sinf_glibc(yaw);
    float cy = cosf_glibc(yaw);
    return {cp * cy, cp * sy, -sp};
  }

  RS_HD Angle round_ue3() const {
    constexpr float to_ints = static_cast<float>(1 << 15) / k_pi;
    constexpr float back_to_radians = (1.0f / to_ints) * 4.0f;
    constexpr int32_t rounding_mask = 0x4000 - 1;
    int32_t r_yaw = (sat_f32_to_i32(yaw * to_ints) >> 2) & rounding_mask;
    int32_t r_pitch = (sat_f32_to_i32(pitch * to_ints) >> 2) & rounding_mask;
    return {static_cast<float>(r_yaw) * back_to_radians,
            static_cast<float>(r_pitch) * back_to_radians, 0.0f};
  }
};

struct QuadRayPack {
  Vec4 origins[3];
  Vec4 inv_dirs[3];
};

struct QuadRayInfo {
  const Vec3A *ray_sources;
  const Vec3A *ray_targets;
  Vec4 lambda_max;
  Aabb aabb;

  RS_HD static QuadRayInfo make(const Vec3A *sources, const Vec3A *targets) {
    Aabb aabb{sources[0].min(targets[0]), sources[0].max(targets[0])};
    for (int i = 1; i < 4; ++i) {
      aabb += Aabb{sources[i].min(targets[i]), sources[i].max(targets[i])};
    }
    return {sources, targets, Vec4::splat(1.0f), aabb};
  }

  RS_HD QuadRayPack calc_pos_dir() const {
    QuadRayPack pack;
    for (int axis = 0; axis < 3; ++axis) {
      Vec4 src{ray_sources[0][axis], ray_sources[1][axis], ray_sources[2][axis],
               ray_sources[3][axis]};
      Vec4 dst{ray_targets[0][axis], ray_targets[1][axis], ray_targets[2][axis],
               ray_targets[3][axis]};
      Vec4 inv = Vec4::splat(1.0f) / (dst - src);
      pack.origins[axis] = src;
      pack.inv_dirs[axis] =
          Vec4::select(inv.is_finite_mask(), inv, Vec4::splat(k_f32_max));
    }
    return pack;
  }

#if defined(__SSE2__) && !defined(__CUDACC__)
  struct SseSlab {
    __m128 enter;
    __m128 exit;
  };

  static SseSlab sse_slab(Vec4 origin, Vec4 inv_dir, float lo, float hi) {
    __m128 o = to_sse(origin);
    __m128 inv = to_sse(inv_dir);
    __m128 t0 = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(lo), o), inv);
    __m128 t1 = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(hi), o), inv);
    return {_mm_min_ps(t0, t1), _mm_max_ps(t0, t1)};
  }
#endif

  RS_HD static uint32_t intersect_quad_ray_aabb(const Vec4 *origins,
                                                const Vec4 *inv_dir,
                                                const Aabb &bounds,
                                                Vec4 lambda_max) {
#if defined(__SSE2__) && !defined(__CUDACC__)
    SseSlab x = sse_slab(origins[0], inv_dir[0], bounds.min.x, bounds.max.x);
    SseSlab y = sse_slab(origins[1], inv_dir[1], bounds.min.y, bounds.max.y);
    SseSlab z = sse_slab(origins[2], inv_dir[2], bounds.min.z, bounds.max.z);
    __m128 t_enter = _mm_max_ps(_mm_max_ps(x.enter, y.enter), z.enter);
    __m128 t_exit = _mm_min_ps(_mm_min_ps(x.exit, y.exit), z.exit);
    __m128 hit = _mm_and_ps(_mm_and_ps(_mm_cmple_ps(t_enter, t_exit),
                                       _mm_cmpge_ps(t_exit, _mm_set1_ps(0.0f))),
                            _mm_cmple_ps(t_enter, to_sse(lambda_max)));
    return static_cast<uint32_t>(_mm_movemask_ps(hit));
#else
    Vec4 t0x = (Vec4::splat(bounds.min.x) - origins[0]) * inv_dir[0];
    Vec4 t1x = (Vec4::splat(bounds.max.x) - origins[0]) * inv_dir[0];
    Vec4 tminx = t0x.min(t1x);
    Vec4 tmaxx = t0x.max(t1x);
    Vec4 t0y = (Vec4::splat(bounds.min.y) - origins[1]) * inv_dir[1];
    Vec4 t1y = (Vec4::splat(bounds.max.y) - origins[1]) * inv_dir[1];
    Vec4 tminy = t0y.min(t1y);
    Vec4 tmaxy = t0y.max(t1y);
    Vec4 t0z = (Vec4::splat(bounds.min.z) - origins[2]) * inv_dir[2];
    Vec4 t1z = (Vec4::splat(bounds.max.z) - origins[2]) * inv_dir[2];
    Vec4 tminz = t0z.min(t1z);
    Vec4 tmaxz = t0z.max(t1z);
    Vec4 t_enter = tminx.max(tminy).max(tminz);
    Vec4 t_exit = tmaxx.min(tmaxy).min(tmaxz);
    BVec4 cond1 = t_enter.cmple(t_exit);
    BVec4 cond2 = t_exit.cmpge(Vec4::splat(0.0f));
    BVec4 cond3 = t_enter.cmple(lambda_max);
    return (cond1 & cond2 & cond3).bitmask();
#endif
  }
};

}
