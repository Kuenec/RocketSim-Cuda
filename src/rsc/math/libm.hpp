#pragma once

#include <cstdint>

#include "rsc/math/hd.hpp"

namespace rsc {

namespace detail {

struct SincosTable {
  double sign[4];
  double hpi_inv;
  double hpi;
  double c0, c1, c2, c3, c4;
  double s1, s2, s3;
};

RS_HD const SincosTable &sincosf_table(int idx) {
  static constexpr SincosTable tables[2] = {{{1.0, -1.0, -1.0, 1.0},
                                             0x1.45F306DC9C883p+23,
                                             0x1.921FB54442D18p0,
                                             0x1p0,
                                             -0x1.ffffffd0c621cp-2,
                                             0x1.55553e1068f19p-5,
                                             -0x1.6c087e89a359dp-10,
                                             0x1.99343027bf8c3p-16,
                                             -0x1.555545995a603p-3,
                                             0x1.1107605230bc4p-7,
                                             -0x1.994eb3774cf24p-13},
                                            {{1.0, -1.0, -1.0, 1.0},
                                             0x1.45F306DC9C883p+23,
                                             0x1.921FB54442D18p0,
                                             -0x1p0,
                                             0x1.ffffffd0c621cp-2,
                                             -0x1.55553e1068f19p-5,
                                             0x1.6c087e89a359dp-10,
                                             -0x1.99343027bf8c3p-16,
                                             -0x1.555545995a603p-3,
                                             0x1.1107605230bc4p-7,
                                             -0x1.994eb3774cf24p-13}};
  return tables[idx];
}

RS_HD uint32_t inv_pio4(int idx) {
  static constexpr uint32_t table[24] = {
      0xa2,       0xa2f9,     0xa2f983,   0xa2f9836e, 0xf9836e4e, 0x836e4e44,
      0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1,
      0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62,
      0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041};
  return table[idx];
}

RS_HDI uint32_t abstop12(float x) { return (f32_to_u32(x) >> 20) & 0x7ff; }

RS_HD double sincosf_reduce_fast(double x, const SincosTable &p, int &n) {
  double r = x * p.hpi_inv;
  n = (static_cast<int32_t>(r) + 0x800000) >> 24;
  return fma_f64(static_cast<double>(-n), p.hpi, x);
}

RS_HD double sincosf_reduce_large(uint32_t xi, int &np) {
  int base = (xi >> 26) & 15;
  int shift = (xi >> 23) & 7;
  uint64_t n, res0, res1, res2;
  xi = (xi & 0xffffff) | 0x800000;
  xi <<= shift;
  res0 = xi * inv_pio4(base);
  res1 = static_cast<uint64_t>(xi) * inv_pio4(base + 4);
  res2 = static_cast<uint64_t>(xi) * inv_pio4(base + 8);
  res0 = (res2 >> 32) | (res0 << 32);
  res0 += res1;
  n = (res0 + (1ULL << 61)) >> 62;
  res0 -= n << 62;
  double x = static_cast<double>(static_cast<int64_t>(res0));
  np = static_cast<int>(n);
  return x * 0x1.921FB54442D18p-62;
}

RS_HD float sinf_poly(double x, double x2, const SincosTable &p, int n) {
  if ((n & 1) == 0) {
    double x3 = x * x2;
    double s1 = fma_f64(x2, p.s3, p.s2);
    double x7 = x3 * x2;
    double s = fma_f64(x3, p.s1, x);
    return static_cast<float>(fma_f64(x7, s1, s));
  }
  double x4 = x2 * x2;
  double c2 = fma_f64(x2, p.c4, p.c3);
  double c1 = fma_f64(x2, p.c1, p.c0);
  double x6 = x4 * x2;
  double c = fma_f64(x4, p.c2, c1);
  return static_cast<float>(fma_f64(x6, c2, c));
}

RS_HD float sincosf_eval(float y, int cos_shift) {
  double x = y;
  int n;
  const SincosTable *p = &sincosf_table(0);
  if (abstop12(y) < abstop12(0x1.921FB6p-1f)) {
    double x2 = x * x;
    if (abstop12(y) < abstop12(0x1p-12f)) {
      return cos_shift ? 1.0f : y;
    }
    return sinf_poly(x, x2, *p, cos_shift);
  }
  if (abstop12(y) < abstop12(120.0f)) {
    x = sincosf_reduce_fast(x, *p, n);
    double s = p->sign[n & 3];
    if (n & 2) {
      p = &sincosf_table(1);
    }
    return sinf_poly(x * s, x * x, *p, n ^ cos_shift);
  }
  if (abstop12(y) < 0x7f8) {
    uint32_t xi = f32_to_u32(y);
    int sign = static_cast<int>(xi >> 31);
    x = sincosf_reduce_large(xi, n);
    double s = p->sign[(n + sign) & 3];
    if ((n + sign) & 2) {
      p = &sincosf_table(1);
    }
    return sinf_poly(x * s, x * x, *p, n ^ cos_shift);
  }
  return y - y;
}

RS_HD double atan2f_muldd(double xh, double xl, double ch, double cl,
                          double &l) {
  double ahlh = ch * xl;
  double alhh = cl * xh;
  double ahhh = ch * xh;
  double ahhl = fma_f64(ch, xh, -ahhh);
  ahhl += alhh + ahlh;
  ch = ahhh + ahhl;
  l = (ahhh - ch) + ahhl;
  return ch;
}

RS_HD float atan2f_tiny(float y, float x) {
  double dy = y;
  double dx = x;
  double z = dy / dx;
  double e = fma_f64(-z, x, y);
  constexpr double c = -0x1.5555555555555p-2;
  double zz = z * z;
  double cz = c * z;
  e = e / dx + cz * zz;
  uint64_t t = f64_to_u64(z);
  if ((t & 0xfffffffULL) == 0) {
    if (z * e > 0) {
      t += 1;
    } else {
      t -= 1;
    }
  }
  double r = u64_to_f64(t);
  float rf = static_cast<float>(r);
  if (rf == 0.0f) {
    return ((t >> 63) ? -0x1p-95f : 0x1p-95f) * 0x1p-95f;
  }
  return rf;
}

RS_HD float atan2f_accurate(float y, float x, uint32_t ax, uint32_t ay,
                            uint32_t gt, uint32_t i, double zx, double zy) {
  static constexpr double sgn[2] = {1, -1};
  static constexpr double off[8] = {0.0f,
                                    0x1.921fb54442d18p+0,
                                    0x1.921fb54442d18p+1,
                                    0x1.921fb54442d18p+0,
                                    -0.0f,
                                    -0x1.921fb54442d18p+0,
                                    -0x1.921fb54442d18p+1,
                                    -0x1.921fb54442d18p+0};
  static constexpr double offl[8] = {0.0f,
                                     0x1.1a62633145c07p-54,
                                     2 * 0x1.1a62633145c07p-54,
                                     0x1.1a62633145c07p-54,
                                     -0.0f,
                                     -0x1.1a62633145c07p-54,
                                     -2 * 0x1.1a62633145c07p-54,
                                     -0x1.1a62633145c07p-54};
  static constexpr double c[32][2] = {
      {0x1p+0, -0x1.8c1dac5492248p-87},
      {-0x1.5555555555555p-2, -0x1.55553bf3a2abep-56},
      {0x1.999999999999ap-3, -0x1.99deed1ec9071p-57},
      {-0x1.2492492492492p-3, -0x1.fd99c8d18269ap-58},
      {0x1.c71c71c71c717p-4, -0x1.651eee4c4d9dp-61},
      {-0x1.745d1745d1649p-4, -0x1.632683d6c44a6p-58},
      {0x1.3b13b13b11c63p-4, 0x1.bf69c1f8af41dp-58},
      {-0x1.11111110e6338p-4, 0x1.3c3e431e8bb68p-61},
      {0x1.e1e1e1dc45c4ap-5, -0x1.be2db05c77bbfp-59},
      {-0x1.af286b8164b4fp-5, 0x1.a4673491f0942p-61},
      {0x1.86185e9ad4846p-5, 0x1.e12e32d79fceep-59},
      {-0x1.642c6d5161faep-5, 0x1.3ce76c1ca03fp-59},
      {0x1.47ad6f277e5bfp-5, -0x1.abd8d85bdb714p-60},
      {-0x1.2f64a2ee8896dp-5, 0x1.ef87d4b615323p-61},
      {0x1.1a6a2b31741b5p-5, 0x1.a5d9d973547eep-62},
      {-0x1.07fbdad65e0a6p-5, -0x1.65ac07f5d35f4p-61},
      {0x1.ee9932a9a5f8bp-6, 0x1.f8b9623f6f55ap-61},
      {-0x1.ce8b5b9584dc6p-6, 0x1.fe5af96e8ea2dp-61},
      {0x1.ac9cb288087b7p-6, -0x1.450cdfceaf5cap-60},
      {-0x1.84b025351f3e6p-6, 0x1.579561b0d73dap-61},
      {0x1.52f5b8ecdd52bp-6, 0x1.036bd2c6fba47p-60},
      {-0x1.163a8c44909dcp-6, 0x1.18f735ffb9f16p-60},
      {0x1.a400dce3eea6fp-7, -0x1.c90569c0c1b5cp-61},
      {-0x1.1caa78ae6db3ap-7, -0x1.4c60f8161ea09p-61},
      {0x1.52672453c0731p-8, 0x1.834efb598c338p-62},
      {-0x1.5850c5be137cfp-9, -0x1.445fc150ca7f5p-63},
      {0x1.23eb98d22e1cap-10, -0x1.388fbaf1d783p-64},
      {-0x1.8f4e974a40741p-12, 0x1.271198a97da34p-66},
      {0x1.a5cf2e9cf76e5p-14, -0x1.887eb4a63b665p-68},
      {-0x1.420c270719e32p-16, 0x1.efd595b27888bp-71},
      {0x1.3ba2d69b51677p-19, -0x1.4fb06829cdfc7p-73},
      {-0x1.29b7e6f676385p-23, -0x1.a783b6de718fbp-77}};
  if (ay < ax && ((ax - ay) >> 23 >= 25)) {
    return atan2f_tiny(y, x);
  }
  double zh;
  double zl;
  if (gt == 0) {
    zh = zy / zx;
    zl = fma_f64(zh, -zx, zy) / zx;
  } else {
    zh = zx / zy;
    zl = fma_f64(zh, -zy, zx) / zy;
  }
  double z2l;
  double z2h = atan2f_muldd(zh, zl, zh, zl, z2l);
  int ci = 31;
  double ph = c[ci][0];
  double pl = c[ci][1];
  while (--ci >= 0) {
    ph = atan2f_muldd(z2h, z2l, ph, pl, pl);
    double th = ph + c[ci][0];
    double tl = (c[ci][0] - th) + ph;
    ph = th;
    pl += tl + c[ci][1];
  }
  zh *= sgn[gt];
  zl *= sgn[gt];
  ph = atan2f_muldd(zh, zl, ph, pl, pl);
  double sh = ph + off[i];
  double sl = ((off[i] - sh) + ph) + pl + offl[i];
  float rf = static_cast<float>(sh);
  double th = rf;
  double dh = sh - th;
  double tm = dh + sl;
  uint64_t tth = f64_to_u64(th);
  if (th + th * 0x1p-60 == th - th * 0x1p-60) {
    tth &= 0x7ffULL << 52;
    tth -= 24ULL << 52;
    if (abs_f64(tm) > u64_to_f64(tth)) {
      tm *= 1.25;
    } else {
      tm *= 0.75;
    }
  }
  return static_cast<float>(th + tm);
}

RS_HD double acosf_poly12(double z, const double *c) {
  double z2 = z * z, z4 = z2 * z2;
  double c0 = c[0] + z * c[1];
  double c2 = c[2] + z * c[3];
  double c4 = c[4] + z * c[5];
  double c6 = c[6] + z * c[7];
  double c8 = c[8] + z * c[9];
  double c10 = c[10] + z * c[11];
  c0 += c2 * z2;
  c4 += c6 * z2;
  c8 += z2 * c10;
  c0 += z4 * (c4 + z4 * c8);
  return c0;
}

}

RS_HD float sinf_glibc(float x) { return detail::sincosf_eval(x, 0); }

RS_HD float cosf_glibc(float x) { return detail::sincosf_eval(x, 1); }

RS_HD float atan2f_glibc(float y, float x) {
  static constexpr double cn[7] = {0x1p+0,
                                   0x1.40e0698f94c35p+1,
                                   0x1.248c5da347f0dp+1,
                                   0x1.d873386572976p-1,
                                   0x1.46fa40b20f1dp-3,
                                   0x1.33f5e041eed0fp-7,
                                   0x1.546bbf28667c5p-14};
  static constexpr double cd[7] = {0x1p+0,
                                   0x1.6b8b143a3f6dap+1,
                                   0x1.8421201d18ed5p+1,
                                   0x1.8221d086914ebp+0,
                                   0x1.670657e3a07bap-2,
                                   0x1.0f4951fd1e72dp-5,
                                   0x1.b3874b8798286p-11};
  static constexpr double m[2] = {0, 1};
  constexpr double pi = 0x1.921fb54442d18p+1;
  constexpr double pi2 = 0x1.921fb54442d18p+0;
  static constexpr double off[8] = {0.0f, pi2, pi, pi2, -0.0f, -pi2, -pi, -pi2};
  static constexpr double offl[8] = {0.0f,
                                     0x1.1a62633145c07p-54,
                                     2 * 0x1.1a62633145c07p-54,
                                     0x1.1a62633145c07p-54,
                                     -0.0f,
                                     -0x1.1a62633145c07p-54,
                                     -2 * 0x1.1a62633145c07p-54,
                                     -0x1.1a62633145c07p-54};
  static constexpr double sgn[2] = {1, -1};
  uint32_t ux = f32_to_u32(x);
  uint32_t uy = f32_to_u32(y);
  uint32_t ax = ux & (~0u >> 1);
  uint32_t ay = uy & (~0u >> 1);
  if (ay >= (0xffu << 23) || ax >= (0xffu << 23)) {
    if (ay > (0xffu << 23)) {
      return x + y;
    }
    if (ax > (0xffu << 23)) {
      return x + y;
    }
    bool yinf = ay == (0xffu << 23);
    bool xinf = ax == (0xffu << 23);
    if (yinf && xinf) {
      if (ux >> 31) {
        return static_cast<float>(0x1.2d97c7f3321d2p+1 * sgn[uy >> 31]);
      }
      return static_cast<float>(0x1.921fb54442d18p-1 * sgn[uy >> 31]);
    }
    if (xinf) {
      if (ux >> 31) {
        return static_cast<float>(pi * sgn[uy >> 31]);
      }
      return static_cast<float>(0.0 * sgn[uy >> 31]);
    }
    if (yinf) {
      return static_cast<float>(pi2 * sgn[uy >> 31]);
    }
  }
  if (ay == 0) {
    if (!ax) {
      uint32_t i = (uy >> 31) * 4 + (ux >> 31) * 2;
      if (ux >> 31) {
        return static_cast<float>(off[i] + offl[i]);
      }
      return static_cast<float>(off[i]);
    }
    if (!(ux >> 31)) {
      return static_cast<float>(0.0 * sgn[uy >> 31]);
    }
  }
  uint32_t gt = ay > ax;
  uint32_t i = (uy >> 31) * 4 + (ux >> 31) * 2 + gt;
  double zx = x;
  double zy = y;
  double z = (m[gt] * zx + m[1 - gt] * zy) / (m[gt] * zy + m[1 - gt] * zx);
  double r;
  int d = static_cast<int>(ax) - static_cast<int>(ay);
  if (d < (27 << 23) && d > (-(27 << 23))) {
    double z2 = z * z, z4 = z2 * z2, z8 = z4 * z4;
    double cn0 = cn[0] + z2 * cn[1];
    double cn2 = cn[2] + z2 * cn[3];
    double cn4 = cn[4] + z2 * cn[5];
    double cn6 = cn[6];
    cn0 += z4 * cn2;
    cn4 += z4 * cn6;
    cn0 += z8 * cn4;
    double cd0 = cd[0] + z2 * cd[1];
    double cd2 = cd[2] + z2 * cd[3];
    double cd4 = cd[4] + z2 * cd[5];
    double cd6 = cd[6];
    cd0 += z4 * cd2;
    cd4 += z4 * cd6;
    cd0 += z8 * cd4;
    r = cn0 / cd0;
  } else {
    r = 1;
  }
  z *= sgn[gt];
  r = z * r + off[i];
  if (((f64_to_u64(r) + 8) & 0xfffffffULL) <= 16) {
    r = detail::atan2f_accurate(y, x, ax, ay, gt, i, zx, zy);
  }
  float rf = static_cast<float>(r);
  if (rf == 0.0f) {
    return ((f32_to_u32(rf) >> 1) ? -0x1p-95f : 0x1p-95f) * 0x1p-95f;
  }
  return rf;
}

RS_HD float acosf_glibc(float x) {
  static constexpr double c0[12] = {
      0x1.555555555529cp-3, 0x1.333333337e0ddp-4,  0x1.6db6db3b4465ep-5,
      0x1.f1c72e13ac306p-6, 0x1.6e89cebe06bc4p-6,  0x1.1c6dcf5289094p-6,
      0x1.c6dbbcc7c6315p-7, 0x1.8f8dc2615e996p-7,  0x1.a5833b7bf15e8p-8,
      0x1.43f44ace1665cp-6, -0x1.0fb17df881c73p-6, 0x1.07520c026b2d6p-5};
  static constexpr double c1[12] = {
      0x1.6a09e667f3bcbp+0,  0x1.e2b7dddff2db9p-4,   0x1.b27247ab42dbcp-6,
      0x1.02995cc4e0744p-7,  0x1.5ffb0276ec8eap-9,   0x1.033885a928decp-10,
      0x1.911f2be23f8c7p-12, 0x1.4c3c55d2437fdp-13,  0x1.af477e1d7b461p-15,
      0x1.abd6bdff67dcbp-15, -0x1.1717e86d0fa28p-16, 0x1.6ff526de46023p-16};
  static constexpr double b[16] = {
      0x1.fffffffd9ccb8p-1,  0x1.5555c94838007p-3,  0x1.32ded4b7c20fap-4,
      0x1.8566df703309ep-5,  -0x1.980c959bec9a3p-6, 0x1.56fbb04998344p-1,
      -0x1.403d8e4c49f52p+2, 0x1.b06c3e9f311eap+4,  -0x1.9ea97c4e2c21fp+6,
      0x1.200b8261cc61bp+8,  -0x1.2274c2799a5c7p+9, 0x1.a558a59cc19d3p+9,
      -0x1.aca4b6a529ffp+9,  0x1.228744703f813p+9,  -0x1.d7dbb0b322228p+7,
      0x1.5c2018c0c0105p+5};
  static constexpr double o[2] = {0, 0x1.921fb54442d18p+1};
  constexpr double pi2 = 0x1.921fb54442d18p+0;
  double xs = x;
  double r;
  uint32_t t = f32_to_u32(x);
  uint32_t ax = t << 1;
  if (ax >= 0x7fu << 24) {
    if (t == (0x7fu << 23)) {
      return 0.0f;
    }
    if (t == (0x17fu << 23)) {
      return 0x1.921fb6p+1f + -0x1p-24f;
    }
    if (ax > (0xffu << 24)) {
      return x + x;
    }
    float zero = 0.0f;
    return (zero - zero) / (zero - zero);
  }
  if (ax < 0x7ec2a1dcu) {
    if (ax <= 0x40000000u) {
      return static_cast<float>(pi2);
    }
    double z = xs;
    double z2 = z * z;
    double z4 = z2 * z2;
    double z8 = z4 * z4;
    double z16 = z8 * z8;
    r = z * ((((b[0] + z2 * b[1]) + z4 * (b[2] + z2 * b[3])) +
              z8 * ((b[4] + z2 * b[5]) + z4 * (b[6] + z2 * b[7]))) +
             z16 * (((b[8] + z2 * b[9]) + z4 * (b[10] + z2 * b[11])) +
                    z8 * ((b[12] + z2 * b[13]) + z4 * (b[14] + z2 * b[15]))));
    float ub = static_cast<float>(0x1.921fb54574191p+0 - r);
    float lb = static_cast<float>(0x1.921fb543118ap+0 - r);
    if (ub == lb) {
      return ub;
    }
  }
  if (ax < (0x7eu << 24)) {
    if (t == 0x328885a3u) {
      return 0x1.921fb6p+0f + 0x1p-25f;
    }
    if (t == 0x39826222u) {
      return 0x1.920f6ap+0f + 0x1p-25f;
    }
    double x2 = xs * xs;
    r = (pi2 - xs) - (xs * x2) * detail::acosf_poly12(x2, c0);
  } else {
    double bx = abs_f64(xs);
    double z = 1.0 - bx;
    double s = copysign_f64(sqrt_f64(z), xs);
    r = o[t >> 31] + s * detail::acosf_poly12(z, c1);
  }
  return static_cast<float>(r);
}

}
