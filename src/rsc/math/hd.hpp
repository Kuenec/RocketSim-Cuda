#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define RS_HD __host__ __device__ inline
#define RS_HDI __host__ __device__ __forceinline__
#define RS_HDN __host__ __device__ __noinline__ inline
#else
#define RS_HD inline
#define RS_HDI inline
#define RS_HDN inline
#endif

#ifdef __CUDA_ARCH__
#define RS_NO_UNROLL _Pragma("unroll 1")
#define RS_HOST_NOINLINE
#else
#define RS_NO_UNROLL
#define RS_HOST_NOINLINE __attribute__((noinline))
#endif

namespace rsc {

RS_HDI uint32_t f32_to_u32(float f) {
#ifdef __CUDA_ARCH__
  return __float_as_uint(f);
#else
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
#endif
}

RS_HDI float u32_to_f32(uint32_t u) {
#ifdef __CUDA_ARCH__
  return __uint_as_float(u);
#else
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
#endif
}

RS_HDI uint64_t f64_to_u64(double d) {
#ifdef __CUDA_ARCH__
  return static_cast<uint64_t>(__double_as_longlong(d));
#else
  uint64_t u;
  std::memcpy(&u, &d, sizeof(u));
  return u;
#endif
}

RS_HDI double u64_to_f64(uint64_t u) {
#ifdef __CUDA_ARCH__
  return __longlong_as_double(static_cast<long long>(u));
#else
  double d;
  std::memcpy(&d, &u, sizeof(d));
  return d;
#endif
}

RS_HDI float round_half_away_f32(float x) {
#ifdef __CUDA_ARCH__
  return ::roundf(x);
#else
  return std::round(x);
#endif
}

RS_HDI double fma_f64(double a, double b, double c) { return ::fma(a, b, c); }

RS_HDI float sqrt_f32(float x) { return ::sqrtf(x); }

RS_HDI double sqrt_f64(double x) { return ::sqrt(x); }

RS_HDI float abs_f32(float x) {
  return u32_to_f32(f32_to_u32(x) & 0x7fffffffu);
}

RS_HDI double abs_f64(double x) {
  return u64_to_f64(f64_to_u64(x) & 0x7fffffffffffffffULL);
}

RS_HDI float copysign_f32(float mag, float sign) {
  return u32_to_f32((f32_to_u32(mag) & 0x7fffffffu) |
                    (f32_to_u32(sign) & 0x80000000u));
}

RS_HDI double copysign_f64(double mag, double sign) {
  return u64_to_f64((f64_to_u64(mag) & 0x7fffffffffffffffULL) |
                    (f64_to_u64(sign) & 0x8000000000000000ULL));
}

RS_HDI bool is_nan_f32(float x) {
  return (f32_to_u32(x) & 0x7fffffffu) > 0x7f800000u;
}

RS_HDI float div_f32(float a, float b) {
#ifdef __CUDA_ARCH__
  if (a == 0.0f && b != 0.0f && !is_nan_f32(b)) {
    return u32_to_f32((f32_to_u32(a) ^ f32_to_u32(b)) & 0x80000000u);
  }
#endif
  return a / b;
}

RS_HDI uint32_t lowest_set_bit(uint64_t x) {
#ifdef __CUDA_ARCH__
  return static_cast<uint32_t>(__ffsll(static_cast<long long>(x)) - 1);
#else
  return static_cast<uint32_t>(std::countr_zero(x));
#endif
}

RS_HDI void prefetch_l1(const void *p) {
#ifdef __CUDA_ARCH__
  asm volatile("prefetch.global.L1 [%0];" : : "l"(p) : "memory");
#else
  static_cast<void>(p);
#endif
}

RS_HDI uint32_t popcount32(uint32_t x) {
#ifdef __CUDA_ARCH__
  return static_cast<uint32_t>(__popc(x));
#else
  return static_cast<uint32_t>(std::popcount(x));
#endif
}

RS_HDI bool is_finite_f32(float x) {
  return (f32_to_u32(x) & 0x7fffffffu) < 0x7f800000u;
}

RS_HDI bool is_sign_negative_f32(float x) {
  return (f32_to_u32(x) & 0x80000000u) != 0;
}

RS_HDI int32_t sat_f32_to_i32(float x) {
  if (is_nan_f32(x)) {
    return 0;
  }
  if (x >= 2147483648.0f) {
    return INT32_MAX;
  }
  if (x < -2147483648.0f) {
    return INT32_MIN;
  }
  return static_cast<int32_t>(x);
}

RS_HDI int64_t sat_f32_to_i64(float x) {
  if (is_nan_f32(x)) {
    return 0;
  }
  if (x >= 9223372036854775808.0f) {
    return INT64_MAX;
  }
  if (x < -9223372036854775808.0f) {
    return INT64_MIN;
  }
  return static_cast<int64_t>(x);
}

RS_HDI float clamp_f32(float x, float lo, float hi) {
  if (x < lo) {
    x = lo;
  }
  if (x > hi) {
    x = hi;
  }
  return x;
}

RS_HDI float signum_f32(float x) {
  if (is_nan_f32(x)) {
    return x;
  }
  return copysign_f32(1.0f, x);
}

RS_HDI float fmod_f32(float x, float y) { return ::fmodf(x, y); }

RS_HDI float rem_euclid_f32(float x, float rhs) {
  float r = fmod_f32(x, rhs);
  if (r < 0.0f) {
    return r + abs_f32(rhs);
  }
  return r;
}

inline constexpr float k_f32_epsilon = 1.1920929e-7f;
inline constexpr float k_f32_max = 3.40282347e38f;
inline constexpr float k_f32_min = -3.40282347e38f;
inline constexpr float k_pi = 3.14159265358979323846f;
inline constexpr float k_frac_pi_2 = 1.57079632679489661923f;
inline constexpr float k_frac_pi_4 = 0.785398163397448309616f;
inline constexpr float k_tau = 6.28318530717958647692f;
inline constexpr float k_sqrt_2 = 1.41421356237309504880f;
inline constexpr float k_frac_1_sqrt_2 = 0.707106781186547524401f;

}
