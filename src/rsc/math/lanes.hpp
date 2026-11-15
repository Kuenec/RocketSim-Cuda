#pragma once

#include <cstdint>

#include "rsc/math/hd.hpp"
#include "rsc/math/vec.hpp"

#if defined(__SSE2__) && !defined(__CUDACC__)
#include <emmintrin.h>

#define RSC_HOST_LANES

namespace rsc {

inline constexpr uint32_t k_lane_count = 4;

struct LaneMask {
  __m128 v;

  static LaneMask all_set() { return {_mm_castsi128_ps(_mm_set1_epi32(-1))}; }
  int bits() const { return _mm_movemask_ps(v); }
};

inline LaneMask operator&(LaneMask a, LaneMask b) {
  return {_mm_and_ps(a.v, b.v)};
}

inline LaneMask operator|(LaneMask a, LaneMask b) {
  return {_mm_or_ps(a.v, b.v)};
}

inline LaneMask and_not(LaneMask excluded, LaneMask m) {
  return {_mm_andnot_ps(excluded.v, m.v)};
}

struct FloatLanes {
  __m128 v;

  static FloatLanes splat(float s) { return {_mm_set1_ps(s)}; }
  static FloatLanes from_array(const float *lanes) {
    return {_mm_loadu_ps(lanes)};
  }
  void to_array(float *lanes) const { _mm_storeu_ps(lanes, v); }
};

inline FloatLanes operator+(FloatLanes a, FloatLanes b) {
  return {_mm_add_ps(a.v, b.v)};
}

inline FloatLanes operator-(FloatLanes a, FloatLanes b) {
  return {_mm_sub_ps(a.v, b.v)};
}

inline FloatLanes operator*(FloatLanes a, FloatLanes b) {
  return {_mm_mul_ps(a.v, b.v)};
}

inline FloatLanes operator/(FloatLanes a, FloatLanes b) {
  return {_mm_div_ps(a.v, b.v)};
}

inline FloatLanes operator-(FloatLanes a) {
  return {_mm_xor_ps(a.v, _mm_castsi128_ps(_mm_set1_epi32(INT32_MIN)))};
}

inline FloatLanes sqrt(FloatLanes a) { return {_mm_sqrt_ps(a.v)}; }

inline FloatLanes abs(FloatLanes a) {
  return {_mm_and_ps(a.v, _mm_castsi128_ps(_mm_set1_epi32(INT32_MAX)))};
}

inline LaneMask operator<(FloatLanes a, FloatLanes b) {
  return {_mm_cmplt_ps(a.v, b.v)};
}

inline LaneMask operator<=(FloatLanes a, FloatLanes b) {
  return {_mm_cmple_ps(a.v, b.v)};
}

inline LaneMask operator>(FloatLanes a, FloatLanes b) {
  return {_mm_cmpgt_ps(a.v, b.v)};
}

inline LaneMask operator>=(FloatLanes a, FloatLanes b) {
  return {_mm_cmpge_ps(a.v, b.v)};
}

inline LaneMask operator==(FloatLanes a, FloatLanes b) {
  return {_mm_cmpeq_ps(a.v, b.v)};
}

inline FloatLanes select(LaneMask m, FloatLanes if_set, FloatLanes if_clear) {
  return {_mm_or_ps(_mm_and_ps(m.v, if_set.v), _mm_andnot_ps(m.v, if_clear.v))};
}

struct IntLanes {
  __m128i v;
};

inline IntLanes round_ties_even_to_int(FloatLanes a) {
  return {_mm_cvtps_epi32(a.v)};
}

inline IntLanes operator>>(IntLanes a, int count) {
  return {_mm_srai_epi32(a.v, count)};
}

inline IntLanes clamp(IntLanes a, int32_t lo, int32_t hi) {
  __m128i lower = _mm_set1_epi32(lo);
  __m128i upper = _mm_set1_epi32(hi);
  __m128i below = _mm_cmplt_epi32(a.v, lower);
  __m128i raised =
      _mm_or_si128(_mm_and_si128(below, lower), _mm_andnot_si128(below, a.v));
  __m128i above = _mm_cmpgt_epi32(raised, upper);
  return {_mm_or_si128(_mm_and_si128(above, upper),
                       _mm_andnot_si128(above, raised))};
}

inline FloatLanes to_float(IntLanes a) { return {_mm_cvtepi32_ps(a.v)}; }

struct Vec3Lanes {
  FloatLanes x, y, z;
};

struct QuatLanes {
  FloatLanes x, y, z, w;

  static QuatLanes gather(const Quat &a, const Quat &b, const Quat &c,
                          const Quat &d) {
    __m128 r0 = _mm_loadu_ps(&a.x);
    __m128 r1 = _mm_loadu_ps(&b.x);
    __m128 r2 = _mm_loadu_ps(&c.x);
    __m128 r3 = _mm_loadu_ps(&d.x);
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    return {{r0}, {r1}, {r2}, {r3}};
  }
  void scatter(Quat *lanes) const {
    __m128 r0 = x.v, r1 = y.v, r2 = z.v, r3 = w.v;
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    _mm_storeu_ps(&lanes[0].x, r0);
    _mm_storeu_ps(&lanes[1].x, r1);
    _mm_storeu_ps(&lanes[2].x, r2);
    _mm_storeu_ps(&lanes[3].x, r3);
  }
};

struct BasisLanes {
  Vec3Lanes x_axis, y_axis, z_axis;

  static BasisLanes gather(const Mat3A &a, const Mat3A &b, const Mat3A &c,
                           const Mat3A &d) {
    __m128 h0 = _mm_loadu_ps(&a.x_axis.x);
    __m128 h1 = _mm_loadu_ps(&b.x_axis.x);
    __m128 h2 = _mm_loadu_ps(&c.x_axis.x);
    __m128 h3 = _mm_loadu_ps(&d.x_axis.x);
    _MM_TRANSPOSE4_PS(h0, h1, h2, h3);
    __m128 m0 = _mm_loadu_ps(&a.y_axis.y);
    __m128 m1 = _mm_loadu_ps(&b.y_axis.y);
    __m128 m2 = _mm_loadu_ps(&c.y_axis.y);
    __m128 m3 = _mm_loadu_ps(&d.y_axis.y);
    _MM_TRANSPOSE4_PS(m0, m1, m2, m3);
    return {{{h0}, {h1}, {h2}},
            {{h3}, {m0}, {m1}},
            {{m2},
             {m3},
             {_mm_setr_ps(a.z_axis.z, b.z_axis.z, c.z_axis.z, d.z_axis.z)}}};
  }
  void scatter(Mat3A *lanes) const {
    __m128 h0 = x_axis.x.v, h1 = x_axis.y.v, h2 = x_axis.z.v, h3 = y_axis.x.v;
    _MM_TRANSPOSE4_PS(h0, h1, h2, h3);
    __m128 m0 = y_axis.y.v, m1 = y_axis.z.v, m2 = z_axis.x.v, m3 = z_axis.y.v;
    _MM_TRANSPOSE4_PS(m0, m1, m2, m3);
    float tails[k_lane_count];
    z_axis.z.to_array(tails);
    _mm_storeu_ps(&lanes[0].x_axis.x, h0);
    _mm_storeu_ps(&lanes[1].x_axis.x, h1);
    _mm_storeu_ps(&lanes[2].x_axis.x, h2);
    _mm_storeu_ps(&lanes[3].x_axis.x, h3);
    _mm_storeu_ps(&lanes[0].y_axis.y, m0);
    _mm_storeu_ps(&lanes[1].y_axis.y, m1);
    _mm_storeu_ps(&lanes[2].y_axis.y, m2);
    _mm_storeu_ps(&lanes[3].y_axis.y, m3);
    for (uint32_t i = 0; i < k_lane_count; ++i) {
      lanes[i].z_axis.z = tails[i];
    }
  }
};

static_assert(sizeof(Quat) == 4 * sizeof(float));
static_assert(sizeof(Mat3A) == 9 * sizeof(float));

}

#endif
