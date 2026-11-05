#pragma once

#include <cstdint>

#include "rsc/math/hd.hpp"
#include "rsc/math/libm.hpp"

#if defined(__SSE2__) && !defined(__CUDACC__)
#include <xmmintrin.h>
#endif

namespace rsc {

RS_HDI float sse_min(float a, float b) { return a < b ? a : b; }

RS_HDI float sse_max(float a, float b) { return a > b ? a : b; }

RS_HDI float max_f32_c(float self, float c) { return self > c ? self : c; }

RS_HDI float min_f32_c(float self, float c) { return self < c ? self : c; }

RS_HDI float max_f32(float self, float other) {
  if (is_nan_f32(self)) {
    return other;
  }
  return other > self ? other : self;
}

RS_HDI float min_f32(float self, float other) {
  if (is_nan_f32(self)) {
    return other;
  }
  return other < self ? other : self;
}

RS_HD float glam_floor(float v) {
  uint32_t abs_bits = f32_to_u32(v) & 0x7fffffffu;
  if (static_cast<int32_t>(abs_bits) < 0x4b000000) {
    float result = static_cast<float>(static_cast<int32_t>(v));
    float larger = result > v ? -1.0f : 0.0f;
    return result + larger;
  }
  return v;
}

RS_HD float glam_trunc(float v) {
  uint32_t abs_bits = f32_to_u32(v) & 0x7fffffffu;
  if (static_cast<int32_t>(abs_bits) < 0x4b000000) {
    return static_cast<float>(static_cast<int32_t>(v));
  }
  return v;
}

RS_HD float glam_round_ties_even(float v) {
  uint32_t sign = f32_to_u32(v) & 0x80000000u;
  float s_magic = u32_to_f32(f32_to_u32(8388608.0f) | sign);
  float r1 = (v + s_magic) - s_magic;
  float r2 = u32_to_f32(f32_to_u32(v) & 0x7fffffffu);
  bool mask = r2 <= 8388608.0f;
  uint32_t r2_bits = mask ? 0u : f32_to_u32(v);
  uint32_t r1_bits = mask ? f32_to_u32(r1) : 0u;
  return u32_to_f32(r1_bits ^ r2_bits);
}

RS_HD float glam_round(float v) {
  float rounded = glam_round_ties_even(v);
  uint32_t sign = f32_to_u32(v) & 0x80000000u;
  float half = u32_to_f32(f32_to_u32(0.5f) | sign);
  bool ties_down = (v - rounded) == half;
  float one = u32_to_f32(f32_to_u32(1.0f) | sign);
  return rounded + (ties_down ? one : 0.0f);
}

struct BVec3 {
  bool x, y, z;
  RS_HDI bool all() const { return x && y && z; }
  RS_HDI bool any() const { return x || y || z; }
  RS_HDI uint32_t bitmask() const {
    return static_cast<uint32_t>(x) | (static_cast<uint32_t>(y) << 1) |
           (static_cast<uint32_t>(z) << 2);
  }
};

struct BVec4 {
  bool x, y, z, w;
  RS_HDI bool all() const { return x && y && z && w; }
  RS_HDI bool any() const { return x || y || z || w; }
  RS_HDI uint32_t bitmask() const {
    return static_cast<uint32_t>(x) | (static_cast<uint32_t>(y) << 1) |
           (static_cast<uint32_t>(z) << 2) | (static_cast<uint32_t>(w) << 3);
  }
  RS_HDI BVec4 operator&(BVec4 o) const {
    return {x && o.x, y && o.y, z && o.z, w && o.w};
  }
};

struct Vec3A {
  float x, y, z;

  RS_HDI static Vec3A splat(float v) { return {v, v, v}; }
  RS_HDI static Vec3A zero() { return {0.0f, 0.0f, 0.0f}; }
  RS_HDI static Vec3A one() { return {1.0f, 1.0f, 1.0f}; }

  RS_HDI float &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
  RS_HDI float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

  RS_HDI Vec3A operator+(Vec3A o) const { return {x + o.x, y + o.y, z + o.z}; }
  RS_HDI Vec3A operator-(Vec3A o) const { return {x - o.x, y - o.y, z - o.z}; }
  RS_HDI Vec3A operator*(Vec3A o) const { return {x * o.x, y * o.y, z * o.z}; }
  RS_HDI Vec3A operator/(Vec3A o) const { return {x / o.x, y / o.y, z / o.z}; }
  RS_HDI Vec3A operator+(float s) const { return {x + s, y + s, z + s}; }
  RS_HDI Vec3A operator-(float s) const { return {x - s, y - s, z - s}; }
  RS_HDI Vec3A operator*(float s) const { return {x * s, y * s, z * s}; }
  RS_HDI Vec3A operator/(float s) const { return {x / s, y / s, z / s}; }
  RS_HDI Vec3A operator-() const { return {-x, -y, -z}; }
  RS_HDI Vec3A &operator+=(Vec3A o) {
    *this = *this + o;
    return *this;
  }
  RS_HDI Vec3A &operator-=(Vec3A o) {
    *this = *this - o;
    return *this;
  }
  RS_HDI Vec3A &operator*=(Vec3A o) {
    *this = *this * o;
    return *this;
  }
  RS_HDI Vec3A &operator*=(float s) {
    *this = *this * s;
    return *this;
  }
  RS_HDI Vec3A &operator/=(float s) {
    *this = *this / s;
    return *this;
  }
  RS_HDI bool operator==(Vec3A o) const {
    return x == o.x && y == o.y && z == o.z;
  }
  RS_HDI bool operator!=(Vec3A o) const { return !(*this == o); }

  RS_HDI float dot(Vec3A o) const { return (x * o.x + y * o.y) + z * o.z; }
  RS_HDI Vec3A cross(Vec3A o) const {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
  RS_HDI float length_squared() const { return dot(*this); }
  RS_HDI float length() const { return sqrt_f32(length_squared()); }
  RS_HDI float length_recip() const { return 1.0f / length(); }
  RS_HDI float distance_squared(Vec3A o) const {
    return (*this - o).length_squared();
  }
  RS_HDI Vec3A normalize() const { return *this / length(); }
  RS_HDI Vec3A normalize_or(Vec3A fallback) const {
    float rcp = length_recip();
    if (is_finite_f32(rcp) && rcp > 0.0f) {
      return *this * rcp;
    }
    return fallback;
  }
  RS_HDI Vec3A normalize_or_zero() const { return normalize_or(zero()); }
  RS_HDI bool try_normalize(Vec3A &out) const {
    float rcp = length_recip();
    if (is_finite_f32(rcp) && rcp > 0.0f) {
      out = *this * rcp;
      return true;
    }
    return false;
  }
  RS_HDI Vec3A lerp(Vec3A o, float s) const {
    return *this * (1.0f - s) + o * s;
  }
  RS_HDI Vec3A min(Vec3A o) const {
    return {sse_min(x, o.x), sse_min(y, o.y), sse_min(z, o.z)};
  }
  RS_HDI Vec3A max(Vec3A o) const {
    return {sse_max(x, o.x), sse_max(y, o.y), sse_max(z, o.z)};
  }
  RS_HDI Vec3A clamp(Vec3A lo, Vec3A hi) const { return max(lo).min(hi); }
  RS_HDI Vec3A abs() const { return {abs_f32(x), abs_f32(y), abs_f32(z)}; }
  RS_HDI Vec3A signum() const {
    return {is_nan_f32(x) ? x : copysign_f32(1.0f, x),
            is_nan_f32(y) ? y : copysign_f32(1.0f, y),
            is_nan_f32(z) ? z : copysign_f32(1.0f, z)};
  }
  RS_HDI Vec3A copysign(Vec3A s) const {
    return {copysign_f32(x, s.x), copysign_f32(y, s.y), copysign_f32(z, s.z)};
  }
  RS_HDI Vec3A floor() const {
    return {glam_floor(x), glam_floor(y), glam_floor(z)};
  }
  RS_HDI Vec3A trunc() const {
    return {glam_trunc(x), glam_trunc(y), glam_trunc(z)};
  }
  RS_HDI Vec3A round() const {
    return {glam_round(x), glam_round(y), glam_round(z)};
  }
  RS_HDI Vec3A recip() const { return {1.0f / x, 1.0f / y, 1.0f / z}; }
  RS_HDI float max_element() const {
    float a = sse_max(x, z);
    float b = sse_max(y, z);
    return sse_max(a, b);
  }
  RS_HDI float min_element() const {
    float a = sse_min(x, z);
    float b = sse_min(y, z);
    return sse_min(a, b);
  }
  RS_HDI int max_position() const {
    float m = x;
    int index = 0;
    if (y > m) {
      m = y;
      index = 1;
    }
    if (z > m) {
      index = 2;
    }
    return index;
  }
  RS_HDI Vec3A with_x(float v) const { return {v, y, z}; }
  RS_HDI Vec3A with_y(float v) const { return {x, v, z}; }
  RS_HDI Vec3A with_z(float v) const { return {x, y, v}; }
  RS_HDI BVec3 cmpeq(Vec3A o) const { return {x == o.x, y == o.y, z == o.z}; }
  RS_HDI BVec3 cmplt(Vec3A o) const { return {x < o.x, y < o.y, z < o.z}; }
  RS_HDI BVec3 cmple(Vec3A o) const { return {x <= o.x, y <= o.y, z <= o.z}; }
  RS_HDI BVec3 cmpgt(Vec3A o) const { return {x > o.x, y > o.y, z > o.z}; }
  RS_HDI BVec3 cmpge(Vec3A o) const { return {x >= o.x, y >= o.y, z >= o.z}; }
  RS_HDI BVec3 is_finite_mask() const {
    return {is_finite_f32(x), is_finite_f32(y), is_finite_f32(z)};
  }
  RS_HDI bool is_finite() const { return is_finite_mask().all(); }
  RS_HDI bool is_nan() const {
    return is_nan_f32(x) || is_nan_f32(y) || is_nan_f32(z);
  }
  RS_HDI uint32_t is_negative_bitmask() const {
    return (f32_to_u32(x) >> 31) | ((f32_to_u32(y) >> 31) << 1) |
           ((f32_to_u32(z) >> 31) << 2);
  }
  RS_HDI static Vec3A select(BVec3 m, Vec3A t, Vec3A f) {
    return {m.x ? t.x : f.x, m.y ? t.y : f.y, m.z ? t.z : f.z};
  }
};

RS_HDI Vec3A operator*(float s, Vec3A v) { return Vec3A::splat(s) * v; }

RS_HDI Vec3A operator+(float s, Vec3A v) { return Vec3A::splat(s) + v; }

RS_HDI Vec3A operator-(float s, Vec3A v) { return Vec3A::splat(s) - v; }

RS_HDI Vec3A operator/(float s, Vec3A v) { return Vec3A::splat(s) / v; }

struct Vec4 {
  float x, y, z, w;

  RS_HDI static Vec4 splat(float v) { return {v, v, v, v}; }
  RS_HDI float &operator[](int i) {
    return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
  }
  RS_HDI float operator[](int i) const {
    return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
  }
  RS_HDI Vec4 operator+(Vec4 o) const {
    return {x + o.x, y + o.y, z + o.z, w + o.w};
  }
  RS_HDI Vec4 operator-(Vec4 o) const {
    return {x - o.x, y - o.y, z - o.z, w - o.w};
  }
  RS_HDI Vec4 operator*(Vec4 o) const {
    return {x * o.x, y * o.y, z * o.z, w * o.w};
  }
  RS_HDI Vec4 operator/(Vec4 o) const {
    return {x / o.x, y / o.y, z / o.z, w / o.w};
  }
  RS_HDI Vec4 operator*(float s) const { return *this * splat(s); }
  RS_HDI Vec4 operator-() const { return {-x, -y, -z, -w}; }
  RS_HDI Vec4 min(Vec4 o) const {
    return {sse_min(x, o.x), sse_min(y, o.y), sse_min(z, o.z), sse_min(w, o.w)};
  }
  RS_HDI Vec4 max(Vec4 o) const {
    return {sse_max(x, o.x), sse_max(y, o.y), sse_max(z, o.z), sse_max(w, o.w)};
  }
  RS_HDI Vec4 clamp(Vec4 lo, Vec4 hi) const { return max(lo).min(hi); }
  RS_HDI Vec4 abs() const {
    return {abs_f32(x), abs_f32(y), abs_f32(z), abs_f32(w)};
  }
  RS_HDI Vec4 sqrt() const {
    return {sqrt_f32(x), sqrt_f32(y), sqrt_f32(z), sqrt_f32(w)};
  }
  RS_HDI float max_element() const {
    float a = sse_max(x, z);
    float b = sse_max(y, w);
    return sse_max(a, b);
  }
  RS_HDI int max_position() const {
    float m = x;
    int index = 0;
    if (y > m) {
      m = y;
      index = 1;
    }
    if (z > m) {
      m = z;
      index = 2;
    }
    if (w > m) {
      index = 3;
    }
    return index;
  }
  RS_HDI BVec4 cmple(Vec4 o) const {
    return {x <= o.x, y <= o.y, z <= o.z, w <= o.w};
  }
  RS_HDI BVec4 cmpge(Vec4 o) const {
    return {x >= o.x, y >= o.y, z >= o.z, w >= o.w};
  }
  RS_HDI BVec4 cmplt(Vec4 o) const {
    return {x < o.x, y < o.y, z < o.z, w < o.w};
  }
  RS_HDI BVec4 is_finite_mask() const {
    return {is_finite_f32(x), is_finite_f32(y), is_finite_f32(z),
            is_finite_f32(w)};
  }
  RS_HDI static Vec4 select(BVec4 m, Vec4 t, Vec4 f) {
    return {m.x ? t.x : f.x, m.y ? t.y : f.y, m.z ? t.z : f.z, m.w ? t.w : f.w};
  }
};

#if defined(__SSE2__) && !defined(__CUDACC__)
inline __m128 to_sse(Vec4 v) { return _mm_setr_ps(v.x, v.y, v.z, v.w); }
#endif

RS_HDI Vec4 operator-(float s, Vec4 v) { return Vec4::splat(s) - v; }

struct Mat3A {
  Vec3A x_axis, y_axis, z_axis;

  RS_HDI static Mat3A identity() {
    return {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
  }
  RS_HDI static Mat3A from_cols(Vec3A a, Vec3A b, Vec3A c) { return {a, b, c}; }
  RS_HDI Vec3A &col(int i) {
    return i == 0 ? x_axis : (i == 1 ? y_axis : z_axis);
  }
  RS_HDI Vec3A col(int i) const {
    return i == 0 ? x_axis : (i == 1 ? y_axis : z_axis);
  }
  RS_HDI Vec3A operator*(Vec3A v) const {
    Vec3A res = x_axis * v.x;
    res = res + y_axis * v.y;
    res = res + z_axis * v.z;
    return res;
  }
  RS_HDI Mat3A operator*(const Mat3A &o) const {
    return {*this * o.x_axis, *this * o.y_axis, *this * o.z_axis};
  }
  RS_HDI Mat3A operator*(float s) const {
    return {x_axis * s, y_axis * s, z_axis * s};
  }
  RS_HDI bool operator==(const Mat3A &o) const {
    return x_axis == o.x_axis && y_axis == o.y_axis && z_axis == o.z_axis;
  }
  RS_HDI bool operator!=(const Mat3A &o) const { return !(*this == o); }
  RS_HDI Mat3A transpose() const {
    return {{x_axis.x, y_axis.x, z_axis.x},
            {x_axis.y, y_axis.y, z_axis.y},
            {x_axis.z, y_axis.z, z_axis.z}};
  }
  RS_HDI Vec3A transpose_mul_vec3a(Vec3A v) const {
    return {x_axis.dot(v), y_axis.dot(v), z_axis.dot(v)};
  }
  RS_HDI Mat3A abs() const {
    return {x_axis.abs(), y_axis.abs(), z_axis.abs()};
  }
};

struct Quat {
  float x, y, z, w;

  RS_HDI static Quat identity() { return {0.0f, 0.0f, 0.0f, 1.0f}; }
  RS_HDI static Quat from_xyzw(float x, float y, float z, float w) {
    return {x, y, z, w};
  }
  RS_HDI float dot4_self() const { return (x * x + z * z) + (y * y + w * w); }
  RS_HDI Quat normalize() const {
    float len = sqrt_f32(dot4_self());
    return {x / len, y / len, z / len, w / len};
  }
  RS_HDI Quat operator*(Quat b) const {
    const Quat &a = *this;
    float r0x = a.x * b.w + a.w * b.x;
    float r0y = -(a.x * b.z) + a.w * b.y;
    float r0z = a.x * b.y + a.w * b.z;
    float r0w = -(a.x * b.x) + a.w * b.w;
    float r1x = -(a.z * b.y) + a.y * b.z;
    float r1y = a.z * b.x + a.y * b.w;
    float r1z = a.z * b.w + -(a.y * b.x);
    float r1w = -(a.z * b.z) + -(a.y * b.y);
    return {r0x + r1x, r0y + r1y, r0z + r1z, r0w + r1w};
  }
};

RS_HD Mat3A mat3a_from_quat(Quat q) {
  float x2 = q.x + q.x;
  float y2 = q.y + q.y;
  float z2 = q.z + q.z;
  float xx = q.x * x2;
  float xy = q.x * y2;
  float xz = q.x * z2;
  float yy = q.y * y2;
  float yz = q.y * z2;
  float zz = q.z * z2;
  float wx = q.w * x2;
  float wy = q.w * y2;
  float wz = q.w * z2;
  return Mat3A::from_cols({1.0f - (yy + zz), xy + wz, xz - wy},
                          {xy - wz, 1.0f - (xx + zz), yz + wx},
                          {xz + wy, yz - wx, 1.0f - (xx + yy)});
}

RS_HD Quat quat_from_mat3a(const Mat3A &m) {
  float m00 = m.x_axis.x, m01 = m.x_axis.y, m02 = m.x_axis.z;
  float m10 = m.y_axis.x, m11 = m.y_axis.y, m12 = m.y_axis.z;
  float m20 = m.z_axis.x, m21 = m.z_axis.y, m22 = m.z_axis.z;
  if (m22 <= 0.0f) {
    float dif10 = m11 - m00;
    float omm22 = 1.0f - m22;
    if (dif10 <= 0.0f) {
      float four_xsq = omm22 - dif10;
      float inv4x = 0.5f / sqrt_f32(four_xsq);
      return {four_xsq * inv4x, (m01 + m10) * inv4x, (m02 + m20) * inv4x,
              (m12 - m21) * inv4x};
    }
    float four_ysq = omm22 + dif10;
    float inv4y = 0.5f / sqrt_f32(four_ysq);
    return {(m01 + m10) * inv4y, four_ysq * inv4y, (m12 + m21) * inv4y,
            (m20 - m02) * inv4y};
  }
  float sum10 = m11 + m00;
  float opm22 = 1.0f + m22;
  if (sum10 <= 0.0f) {
    float four_zsq = opm22 - sum10;
    float inv4z = 0.5f / sqrt_f32(four_zsq);
    return {(m02 + m20) * inv4z, (m12 + m21) * inv4z, four_zsq * inv4z,
            (m01 - m10) * inv4z};
  }
  float four_wsq = opm22 + sum10;
  float inv4w = 0.5f / sqrt_f32(four_wsq);
  return {(m12 - m21) * inv4w, (m20 - m02) * inv4w, (m01 - m10) * inv4w,
          four_wsq * inv4w};
}

RS_HD Mat3A mat3a_from_euler_zyx(float a, float b, float c) {
  float si = sinf_glibc(a), ci = cosf_glibc(a);
  float sj = sinf_glibc(b), cj = cosf_glibc(b);
  float sh = sinf_glibc(c), ch = cosf_glibc(c);
  float cc = ci * ch;
  float cs = ci * sh;
  float sc = si * ch;
  float ss = si * sh;
  float m[3][3];
  m[2][2] = cj * ch;
  m[2][1] = sj * sc - cs;
  m[2][0] = sj * cc + ss;
  m[1][2] = cj * sh;
  m[1][1] = sj * ss + cc;
  m[1][0] = sj * cs - sc;
  m[0][2] = -sj;
  m[0][1] = cj * si;
  m[0][0] = cj * ci;
  return Mat3A::from_cols({m[0][0], m[0][1], m[0][2]},
                          {m[1][0], m[1][1], m[1][2]},
                          {m[2][0], m[2][1], m[2][2]});
}

struct Affine3A {
  Mat3A matrix3;
  Vec3A translation;

  RS_HDI static Affine3A identity() {
    return {Mat3A::identity(), Vec3A::zero()};
  }
  RS_HDI Vec3A transform_point3a(Vec3A p) const {
    return matrix3 * p + translation;
  }
  RS_HDI Affine3A operator*(const Affine3A &o) const {
    return {matrix3 * o.matrix3, matrix3 * o.translation + translation};
  }
  RS_HDI bool operator==(const Affine3A &o) const {
    return matrix3 == o.matrix3 && translation == o.translation;
  }
  RS_HDI Affine3A ext_transpose() const {
    Mat3A m = matrix3.transpose();
    return {m, m * -translation};
  }
  RS_HDI Vec3A inv_x_form(Vec3A v) const {
    return matrix3.transpose_mul_vec3a(v - translation);
  }
};

}
