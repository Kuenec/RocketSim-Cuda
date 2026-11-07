#pragma once

#include <cstdint>

#include "rsc/math/glibc_atan_table.hpp"
#include "rsc/math/hd.hpp"

namespace rsc {

namespace detail {

RS_HDI double glibc_c(uint64_t bits) { return u64_to_f64(bits); }

RS_HD double atan2_small_poly(double v) {
  using namespace glibc_atan;
  double p = fma_f64(v, glibc_c(k_d13), glibc_c(k_d11));
  p = fma_f64(v, p, glibc_c(k_d9));
  p = fma_f64(v, p, glibc_c(k_d7));
  p = fma_f64(v, p, glibc_c(k_d5));
  return fma_f64(v, p, glibc_c(k_d3));
}

RS_HD double atan2_table_poly(double v, int i, int lowest) {
  double p = fma_f64(v, glibc_atan::cij(i, 6), glibc_atan::cij(i, 5));
  p = fma_f64(v, p, glibc_atan::cij(i, 4));
  p = fma_f64(v, p, glibc_atan::cij(i, 3));
  if (lowest == 2) {
    p = fma_f64(v, p, glibc_atan::cij(i, 2));
  }
  return p;
}

RS_HDI double atan2_sign(double y, double z) { return copysign_f64(z, y); }

RS_HD double hypot_kernel(double ax, double ay) {
  double t1 = ay + ay;
  double t2 = ax - ay;
  if (t1 >= ax) {
    return sqrt_f64(fma_f64(t1, ax, t2 * t2));
  }
  return sqrt_f64(fma_f64(ax, ax, ay * ay));
}

}

RS_HD RS_FMA_DISPATCH double atan2_glibc(double y, double x) {
  using namespace glibc_atan;
  using detail::atan2_sign;
  using detail::glibc_c;
  const double hpi = glibc_c(k_hpi);
  const double opi = glibc_c(k_opi);
  uint64_t xb = f64_to_u64(x);
  uint64_t yb = f64_to_u64(y);
  int32_t ux = static_cast<int32_t>(xb >> 32);
  int32_t dx = static_cast<int32_t>(xb & 0xffffffffu);
  int32_t uy = static_cast<int32_t>(yb >> 32);
  int32_t dy = static_cast<int32_t>(yb & 0xffffffffu);
  if ((ux & 0x7ff00000) == 0x7ff00000) {
    if (((ux & 0x000fffff) | dx) != 0) {
      return x + y;
    }
  }
  if ((uy & 0x7ff00000) == 0x7ff00000) {
    if (((uy & 0x000fffff) | dy) != 0) {
      return y + y;
    }
  }
  if (uy == 0x00000000) {
    if (dy == 0) {
      if ((ux & static_cast<int32_t>(0x80000000u)) == 0) {
        return 0;
      }
      return opi;
    }
  } else if (uy == static_cast<int32_t>(0x80000000u)) {
    if (dy == 0) {
      if ((ux & static_cast<int32_t>(0x80000000u)) == 0) {
        return -0.0;
      }
      return glibc_c(k_mopi);
    }
  }
  if (x == 0) {
    if ((uy & static_cast<int32_t>(0x80000000u)) == 0) {
      return hpi;
    }
    return glibc_c(k_mhpi);
  }
  if (ux == 0x7ff00000) {
    if (dx == 0) {
      if (uy == 0x7ff00000) {
        if (dy == 0) {
          return glibc_c(k_qpi);
        }
      } else if (uy == static_cast<int32_t>(0xfff00000u)) {
        if (dy == 0) {
          return glibc_c(k_mqpi);
        }
      } else {
        if ((uy & static_cast<int32_t>(0x80000000u)) == 0) {
          return 0;
        }
        return -0.0;
      }
    }
  } else if (ux == static_cast<int32_t>(0xfff00000u)) {
    if (dx == 0) {
      if (uy == 0x7ff00000) {
        if (dy == 0) {
          return glibc_c(k_tqpi);
        }
      } else if (uy == static_cast<int32_t>(0xfff00000u)) {
        if (dy == 0) {
          return glibc_c(k_mtqpi);
        }
      } else {
        if ((uy & static_cast<int32_t>(0x80000000u)) == 0) {
          return opi;
        }
        return glibc_c(k_mopi);
      }
    }
  }
  if (uy == 0x7ff00000) {
    if (dy == 0) {
      return hpi;
    }
  } else if (uy == static_cast<int32_t>(0xfff00000u)) {
    if (dy == 0) {
      return glibc_c(k_mhpi);
    }
  }
  constexpr int ep = 59768832;
  constexpr int em = -59768832;
  double ax = (x < 0) ? -x : x;
  double ay = (y < 0) ? -y : y;
  int de = (uy & 0x7ff00000) - (ux & 0x7ff00000);
  if (de >= ep) {
    return (y > 0) ? hpi : glibc_c(k_mhpi);
  }
  if (de <= em) {
    if (x > 0) {
      return atan2_sign(y, ay / ax);
    }
    return (y > 0) ? opi : glibc_c(k_mopi);
  }
  const double two500 = glibc_c(k_two500);
  const double twom500 = glibc_c(k_twom500);
  if (ax < twom500 || ay < twom500) {
    ax *= two500;
    ay *= two500;
  }
  if (ax > two500 || ay > two500) {
    ax *= twom500;
    ay *= twom500;
  }
  double u, du, v, vv;
  if (ay < ax) {
    u = ay / ax;
    v = ax * u;
    vv = fma_f64(ax, u, -v);
    du = ((ay - v) - vv) / ax;
  } else {
    u = ax / ay;
    v = ay * u;
    vv = fma_f64(ay, u, -v);
    du = ((ax - v) - vv) / ay;
  }
  const double inv16 = glibc_c(k_inv16);
  const double hpi1 = glibc_c(k_hpi1);
  const double opi1 = glibc_c(k_opi1);
  constexpr double two52 = 0x1.0p52;
  if (x > 0) {
    if (ay < ax) {
      if (u < inv16) {
        v = u * u;
        double zz = fma_f64(u * v, detail::atan2_small_poly(v), du);
        double z = u + zz;
        return atan2_sign(y, z);
      }
      int i = static_cast<int>(fma_f64(256.0, u, two52) - two52);
      i -= 16;
      double t3 = u - glibc_atan::cij(i, 0);
      v = t3 + du;
      double dv =
          (abs_f64(t3) > abs_f64(du)) ? ((t3 - v) + du) : ((du - v) + t3);
      double t1 = glibc_atan::cij(i, 1);
      double t2 = glibc_atan::cij(i, 2);
      double p = detail::atan2_table_poly(v, i, 3);
      double zz = fma_f64(v, t2, fma_f64(dv, t2, (v * v) * p));
      double z = t1 + zz;
      return atan2_sign(y, z);
    }
    if (u < inv16) {
      v = u * u;
      double t1 = u * v;
      double p = detail::atan2_small_poly(v);
      double t2 = hpi - u;
      double cor =
          (abs_f64(hpi) > abs_f64(u)) ? ((hpi - t2) - u) : (hpi - (u + t2));
      double zz = t1 * p;
      double t3 = ((hpi1 + cor) - du) - zz;
      double z = t2 + t3;
      return atan2_sign(y, z);
    }
    int i = static_cast<int>(fma_f64(256.0, u, two52) - two52);
    i -= 16;
    v = (u - glibc_atan::cij(i, 0)) + du;
    double zz = fma_f64(-v, detail::atan2_table_poly(v, i, 2), hpi1);
    double t1 = hpi - glibc_atan::cij(i, 1);
    double z = t1 + zz;
    return atan2_sign(y, z);
  }
  if (ax < ay) {
    if (u < inv16) {
      v = u * u;
      double t1 = u * v;
      double p = detail::atan2_small_poly(v);
      double t2 = hpi + u;
      double cor =
          (abs_f64(hpi) > abs_f64(u)) ? ((hpi - t2) + u) : ((u - t2) + hpi);
      double zz = t1 * p;
      double t3 = ((hpi1 + cor) + du) + zz;
      double z = t2 + t3;
      return atan2_sign(y, z);
    }
    int i = static_cast<int>(fma_f64(256.0, u, two52) - two52);
    i -= 16;
    v = (u - glibc_atan::cij(i, 0)) + du;
    double zz = fma_f64(v, detail::atan2_table_poly(v, i, 2), hpi1);
    double t1 = hpi + glibc_atan::cij(i, 1);
    double z = t1 + zz;
    return atan2_sign(y, z);
  }
  if (u < inv16) {
    v = u * u;
    double t1 = u * v;
    double p = detail::atan2_small_poly(v);
    double t2 = opi - u;
    double cor =
        (abs_f64(opi) > abs_f64(u)) ? ((opi - t2) - u) : (opi - (u + t2));
    double zz = t1 * p;
    double t3 = ((opi1 + cor) - du) - zz;
    double z = t2 + t3;
    return atan2_sign(y, z);
  }
  int i = static_cast<int>(fma_f64(256.0, u, two52) - two52);
  i -= 16;
  v = (u - glibc_atan::cij(i, 0)) + du;
  double zz = fma_f64(-v, detail::atan2_table_poly(v, i, 2), opi1);
  double t1 = opi - glibc_atan::cij(i, 1);
  double z = t1 + zz;
  return atan2_sign(y, z);
}

RS_HD RS_FMA_DISPATCH double hypot_glibc(double x, double y) {
  constexpr double scale = 0x1p-600;
  constexpr double large_val = 0x1p+511;
  constexpr double tiny_val = 0x1p-459;
  constexpr double eps = 0x1p-54;
  uint64_t xb = f64_to_u64(x) & 0x7fffffffffffffffULL;
  uint64_t yb = f64_to_u64(y) & 0x7fffffffffffffffULL;
  constexpr uint64_t inf_bits = 0x7ff0000000000000ULL;
  if (xb >= inf_bits || yb >= inf_bits) {
    constexpr uint64_t quiet_bit = 0x0008000000000000ULL;
    bool x_signaling = xb > inf_bits && (xb & quiet_bit) == 0;
    bool y_signaling = yb > inf_bits && (yb & quiet_bit) == 0;
    if ((xb == inf_bits || yb == inf_bits) && !x_signaling && !y_signaling) {
      return u64_to_f64(inf_bits);
    }
    return x + y;
  }
  x = abs_f64(x);
  y = abs_f64(y);
  double ax = x < y ? y : x;
  double ay = x < y ? x : y;
  if (ax > large_val) {
    if (ay <= ax * eps) {
      return ax + ay;
    }
    return detail::hypot_kernel(ax * scale, ay * scale) / scale;
  }
  if (ay < tiny_val) {
    if (ax >= ay / eps) {
      return ax + ay;
    }
    return detail::hypot_kernel(ax / scale, ay / scale) * scale;
  }
  if (ay <= ax * eps) {
    return ax + ay;
  }
  return detail::hypot_kernel(ax, ay);
}

}
