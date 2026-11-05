#pragma once

#include <cstdint>

#include "rsc/math/hd.hpp"

namespace rsc {

RS_HDI uint64_t mul_high_u64(uint64_t a, uint64_t b) {
#ifdef __CUDA_ARCH__
  return __umul64hi(a, b);
#else
  uint64_t a_lo = a & 0xffffffffULL;
  uint64_t a_hi = a >> 32;
  uint64_t b_lo = b & 0xffffffffULL;
  uint64_t b_hi = b >> 32;
  uint64_t lo_lo = a_lo * b_lo;
  uint64_t hi_lo = a_hi * b_lo;
  uint64_t lo_hi = a_lo * b_hi;
  uint64_t hi_hi = a_hi * b_hi;
  uint64_t cross = (lo_lo >> 32) + (hi_lo & 0xffffffffULL) + lo_hi;
  return hi_hi + (hi_lo >> 32) + (cross >> 32);
#endif
}

struct Rng {
  uint64_t state;

  RS_HD uint64_t gen_u64() {
    constexpr uint64_t wy_const_0 = 0x2d358dccaa6c78a5ULL;
    constexpr uint64_t wy_const_1 = 0x8bb84b93962eacc9ULL;
    uint64_t s = state + wy_const_0;
    state = s;
    uint64_t other = s ^ wy_const_1;
    return (s * other) ^ mul_high_u64(s, other);
  }

  RS_HD uint64_t gen_mod_u64(uint64_t n) {
    uint64_t r = gen_u64();
    uint64_t hi = mul_high_u64(r, n);
    uint64_t lo = r * n;
    if (lo < n) {
      uint64_t t = (0 - n) % n;
      while (lo < t) {
        r = gen_u64();
        hi = mul_high_u64(r, n);
        lo = r * n;
      }
    }
    return hi;
  }

  RS_HD uint64_t usize_below(uint64_t end) { return gen_mod_u64(end); }

  RS_HD uint64_t usize_inclusive(uint64_t high) {
    if (high == UINT64_MAX) {
      return gen_u64();
    }
    return gen_mod_u64(high + 1);
  }

  RS_HD bool gen_bool() {
    uint8_t v = static_cast<uint8_t>(static_cast<uint32_t>(gen_u64()));
    return v % 2 == 0;
  }

  template <typename T> RS_HD void shuffle(T *slice, uint64_t len) {
    for (uint64_t i = 1; i < len; ++i) {
      uint64_t j = usize_inclusive(i);
      T tmp = slice[i];
      slice[i] = slice[j];
      slice[j] = tmp;
    }
  }
};

}
