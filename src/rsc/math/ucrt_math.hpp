#pragma once

#include <cstdint>

#include "rsc/math/hd.hpp"
#include "rsc/math/libm.hpp"

namespace rsc {

namespace ucrt {

inline constexpr uint32_t k_f32_abs_mask = 0x7fffffffu;
inline constexpr uint32_t k_f32_one_bits = 0x3f800000u;
inline constexpr uint32_t k_f32_inf_bits = 0x7f800000u;
inline constexpr uint32_t k_f32_quiet_nan_bits = 0x7fc00000u;
inline constexpr uint32_t k_f32_canonical_nan_bits = 0xffc00000u;
inline constexpr uint32_t k_quiet_bit = 0x00400000u;
inline constexpr uint32_t k_sign_bit = 0x80000000u;
inline constexpr uint32_t k_exp_mask = 0x7f800000u;
inline constexpr uint64_t k_f64_abs_mask = 0x7fffffffffffffffULL;
inline constexpr uint64_t k_f64_inf_bits = 0x7ff0000000000000ULL;

inline constexpr uint64_t k_one_sixth = 0x3fc5555555555555ULL;
inline constexpr uint64_t k_two_pow_neg_4 = 0x3fb0000000000000ULL;
inline constexpr uint64_t k_pio2 = 0x3ff921fb54442d18ULL;
inline constexpr uint64_t k_pio = 0x400921fb54442d18ULL;
inline constexpr uint32_t k_f32_pi_over_2 = 0x3fc90fdbu;
inline constexpr uint32_t k_f32_pi = 0x40490fdbu;

inline constexpr uint64_t k_acos_offset = 0x3c91a62633145c07ULL;
inline constexpr uint32_t k_acos_c0 = 0x3b81ce6bu;
inline constexpr uint32_t k_acos_c1 = 0xbc5b3fe1u;
inline constexpr uint32_t k_acos_c2 = 0x3d678bddu;
inline constexpr uint32_t k_acos_c3 = 0x3e3c94dcu;
inline constexpr uint32_t k_acos_d0 = 0x3f561f0du;
inline constexpr uint32_t k_acos_d1 = 0x3f8d6fa5u;
inline constexpr uint32_t k_acos_right_angle_exponent = 0x65u;
inline constexpr uint32_t k_acos_sqrt_branch_exponent = 0x7eu;
inline constexpr uint32_t k_acos_unit_exponent = 0x7fu;
inline constexpr uint32_t k_acos_sqrt_head_mask = 0xffff0000u;

inline constexpr uint64_t k_atan_poly_a = 0xbfc2492482bd6be1ULL;
inline constexpr uint64_t k_atan_poly_b = 0x3fc99999999643a3ULL;
inline constexpr uint64_t k_atan_poly_c = 0x3fd5555555555538ULL;
inline constexpr uint64_t k_atan_cubic = 0x3fd5555555550877ULL;
inline constexpr uint64_t k_atan_small = 0x3f1a36e2eb1c432dULL;
inline constexpr uint32_t k_f32_pi_over_4 = 0x3f490fdbu;
inline constexpr uint32_t k_f32_neg_pi_over_4 = 0xbf490fdbu;
inline constexpr uint32_t k_f32_neg_pi_over_2 = 0xbfc90fdbu;
inline constexpr uint32_t k_f32_neg_pi = 0xc0490fdbu;
inline constexpr uint32_t k_f32_three_pi_over_4 = 0x4016cbe4u;
inline constexpr uint32_t k_f32_neg_three_pi_over_4 = 0xc016cbe4u;
inline constexpr int32_t k_atan_table_first_step = 16;

inline constexpr uint64_t k_exp_overflow = 0x40562e4300000000ULL;
inline constexpr uint64_t k_exp_underflow = 0xc059d1da00000000ULL;
inline constexpr uint64_t k_exp_scale = 0x40571547652b82feULL;
inline constexpr uint64_t k_ln2_over_64 = 0x3f862e42fefa39efULL;
inline constexpr uint64_t k_ln2 = 0x3fe62e42fefa39efULL;
inline constexpr uint64_t k_log1p_a = 0x3f89999999bac6d4ULL;
inline constexpr uint64_t k_log1p_b = 0x3fb55555555554e6ULL;
inline constexpr uint64_t k_log1p_c = 0x3f3c8034c85dfff0ULL;
inline constexpr uint64_t k_log1p_d = 0x3f62492307f1519fULL;
inline constexpr uint64_t k_one_third = 0x3fd5555555555555ULL;
inline constexpr uint32_t k_pow_fractional_exponent_limit = 0x4b000000u;
inline constexpr int32_t k_pow_near_one_upper_bits = 0x3f880000;

inline constexpr uint32_t k_sinf_limit = 0x4b800456u;
inline constexpr uint32_t k_cosf_limit = 0x4f490fdbu;
inline constexpr uint32_t k_trig_identity_limit = 0x39000000u;
inline constexpr uint32_t k_trig_short_series_limit = 0x3c000000u;
inline constexpr uint64_t k_sin_c1 = 0x3ec71de3a556c734ULL;
inline constexpr uint64_t k_sin_c2 = 0xbf2a01a01a01a01aULL;
inline constexpr uint64_t k_sin_c3 = 0x3f81111111111111ULL;
inline constexpr uint64_t k_sin_c4 = 0xbfc5555555555555ULL;
inline constexpr uint64_t k_cos_c1 = 0xbe927e4fb7789f5cULL;
inline constexpr uint64_t k_cos_c2 = 0x3efa01a01a01a019ULL;
inline constexpr uint64_t k_cos_c3 = 0xbf56c16c16c16c16ULL;
inline constexpr uint64_t k_cos_c4 = 0x3fa5555555555555ULL;
inline constexpr uint64_t k_inv_pio2 = 0x3fe45f306dc9c883ULL;
inline constexpr uint64_t k_pio2_hi = 0x3ff921fb54400000ULL;
inline constexpr uint64_t k_pio2_lo = 0x3dd0b4611a626331ULL;

enum class ExpRemainder { Fused, Unfused };

RS_HDI uint32_t abs_bits(float x) { return f32_to_u32(x) & k_f32_abs_mask; }

RS_HDI float quiet(float f) { return u32_to_f32(f32_to_u32(f) | k_quiet_bit); }

RS_HDI float fma_f32(float a, float b, float c) { return ::fmaf(a, b, c); }

RS_HDI float round_ties_even_f32(float x) { return ::rintf(x); }

RS_HDI double round_ties_even_f64(double x) { return ::rint(x); }

RS_HDI bool is_nan_f64(double x) {
  return (f64_to_u64(x) & k_f64_abs_mask) > k_f64_inf_bits;
}

RS_HDI int32_t sat_f64_to_i32(double x) {
  if (is_nan_f64(x)) {
    return 0;
  }
  if (x >= 2147483648.0) {
    return INT32_MAX;
  }
  if (x < -2147483648.0) {
    return INT32_MIN;
  }
  return static_cast<int32_t>(x);
}

RS_HD double atan_tab(int32_t i) {
  static constexpr uint64_t table[241] = {
      0x3faff55bb72cfde9ULL, 0x3fb0f99ea71d52a6ULL, 0x3fb1f86dbf082d58ULL,
      0x3fb2f719318a4a9aULL, 0x3fb3f59f0e7c559dULL, 0x3fb4f3fd677292fbULL,
      0x3fb5f2324fd2d7b2ULL, 0x3fb6f03bdcea4b0cULL, 0x3fb7ee182602f10eULL,
      0x3fb8ebc54478fb28ULL, 0x3fb9e94153cfdcf1ULL, 0x3fbae68a71c722b8ULL,
      0x3fbbe39ebe6f07c3ULL, 0x3fbce07c5c3cca32ULL, 0x3fbddd21701eba6eULL,
      0x3fbed98c2190043aULL, 0x3fbfd5ba9aac2f6dULL, 0x3fc068d584212b3dULL,
      0x3fc0e6adccf40881ULL, 0x3fc1646541060850ULL, 0x3fc1e1fafb043726ULL,
      0x3fc25f6e171a535cULL, 0x3fc2dcbdb2fba1ffULL, 0x3fc359e8edeb99a3ULL,
      0x3fc3d6eee8c6626cULL, 0x3fc453cec6092a9eULL, 0x3fc4d087a9da4f17ULL,
      0x3fc54d18ba11570aULL, 0x3fc5c9811e3ec269ULL, 0x3fc645bfffb3aa73ULL,
      0x3fc6c1d4898933d8ULL, 0x3fc73dbde8a7d201ULL, 0x3fc7b97b4bce5b02ULL,
      0x3fc8350be398ebc7ULL, 0x3fc8b06ee2879c28ULL, 0x3fc92ba37d050271ULL,
      0x3fc9a6a8e96c8626ULL, 0x3fca217e601081a5ULL, 0x3fca9c231b403279ULL,
      0x3fcb1696574d780bULL, 0x3fcb90d7529260a2ULL, 0x3fcc0ae54d768466ULL,
      0x3fcc84bf8a742e6dULL, 0x3fccfe654e1d5395ULL, 0x3fcd77d5df205736ULL,
      0x3fcdf110864c9d9dULL, 0x3fce6a148e96ec4dULL, 0x3fcee2e1451d980cULL,
      0x3fcf5b75f92c80ddULL, 0x3fcfd3d1fc40dbe4ULL, 0x3fd025fa510665b5ULL,
      0x3fd061eea03d6290ULL, 0x3fd09dc597d86362ULL, 0x3fd0d97ee509acb3ULL,
      0x3fd1151a362431c9ULL, 0x3fd150973a9ce546ULL, 0x3fd18bf5a30bf178ULL,
      0x3fd1c735212dd883ULL, 0x3fd2025567e47c95ULL, 0x3fd23d562b381041ULL,
      0x3fd278372057ef45ULL, 0x3fd2b2f7fd9b5fe2ULL, 0x3fd2ed987a823cfeULL,
      0x3fd328184fb58951ULL, 0x3fd362773707ebcbULL, 0x3fd39cb4eb76157bULL,
      0x3fd3d6d129271134ULL, 0x3fd410cbad6c7d32ULL, 0x3fd44aa436c2af09ULL,
      0x3fd4845a84d0c21bULL, 0x3fd4bdee586890e6ULL, 0x3fd4f75f73869978ULL,
      0x3fd530ad9951cd49ULL, 0x3fd569d88e1b4cd7ULL, 0x3fd5a2e0175e0f4eULL,
      0x3fd5dbc3fbbe768dULL, 0x3fd614840309cfe1ULL, 0x3fd64d1ff635c1c5ULL,
      0x3fd685979f5fa6fdULL, 0x3fd6bdeac9cbd76cULL, 0x3fd6f61941e4def0ULL,
      0x3fd72e22d53aa2a9ULL, 0x3fd7660752817501ULL, 0x3fd79dc6899118d1ULL,
      0x3fd7d5604b63b3f7ULL, 0x3fd80cd46a14b1d0ULL, 0x3fd84422b8df95d7ULL,
      0x3fd87b4b0c1ebedbULL, 0x3fd8b24d394a1b25ULL, 0x3fd8e92916f5cde8ULL,
      0x3fd91fde7cd0c662ULL, 0x3fd9566d43a34907ULL, 0x3fd98cd5454d6b18ULL,
      0x3fd9c3165cc58107ULL, 0x3fd9f93066168001ULL, 0x3fda2f233e5e530bULL,
      0x3fda64eec3cc23fcULL, 0x3fda9a92d59e98cfULL, 0x3fdad00f5422058bULL,
      0x3fdb056420ae9343ULL, 0x3fdb3a911da65c6cULL, 0x3fdb6f962e737efbULL,
      0x3fdba473378624a5ULL, 0x3fdbd9281e528191ULL, 0x3fdc0db4c94ec9efULL,
      0x3fdc42191ff11eb6ULL, 0x3fdc76550aad71f8ULL, 0x3fdcaa6872f3631bULL,
      0x3fdcde53432c1350ULL, 0x3fdd121566b7f2adULL, 0x3fdd45aec9ec862bULL,
      0x3fdd791f5a1226f4ULL, 0x3fddac670561bb4fULL, 0x3fdddf85bb026974ULL,
      0x3fde127b6b0744afULL, 0x3fde4548066cf51aULL, 0x3fde77eb7f175a34ULL,
      0x3fdeaa65c7cf28c4ULL, 0x3fdedcb6d43f8434ULL, 0x3fdf0ede98f393cfULL,
      0x3fdf40dd0b541417ULL, 0x3fdf72b221a4e495ULL, 0x3fdfa45dd3029258ULL,
      0x3fdfd5e0175fdf83ULL, 0x3fe0039c73c1a40bULL, 0x3fe01c341e82422dULL,
      0x3fe034b709250488ULL, 0x3fe04d25314342e5ULL, 0x3fe0657e94db30cfULL,
      0x3fe07dc3324e9b38ULL, 0x3fe095f30861a58fULL, 0x3fe0ae0e1639866cULL,
      0x3fe0c6145b5b43daULL, 0x3fe0de05d7aa6f7cULL, 0x3fe0f5e28b67e295ULL,
      0x3fe10daa77307a0dULL, 0x3fe1255d9bfbd2a8ULL, 0x3fe13cfbfb1b056eULL,
      0x3fe1548596376469ULL, 0x3fe16bfa6f5137e1ULL, 0x3fe1835a88be7c13ULL,
      0x3fe19aa5e5299f99ULL, 0x3fe1b1dc87904284ULL, 0x3fe1c8fe7341f64fULL,
      0x3fe1e00babdefeb3ULL, 0x3fe1f7043557138aULL, 0x3fe20de813e823b1ULL,
      0x3fe224b74c1d192aULL, 0x3fe23b71e2cc9e6aULL, 0x3fe25217dd17e501ULL,
      0x3fe268a940696da6ULL, 0x3fe27f261273d1b3ULL, 0x3fe2958e59308e30ULL,
      0x3fe2abe21aded073ULL, 0x3fe2c2215e024465ULL, 0x3fe2d84c2961e48bULL,
      0x3fe2ee628406cbcaULL, 0x3fe30464753b090aULL, 0x3fe31a52048874beULL,
      0x3fe3302b39b78856ULL, 0x3fe345f01cce37bbULL, 0x3fe35ba0b60eccceULL,
      0x3fe3713d0df6c503ULL, 0x3fe386c52d3db11eULL, 0x3fe39c391cd41719ULL,
      0x3fe3b198e5e2564aULL, 0x3fe3c6e491c78dc4ULL, 0x3fe3dc1c2a188504ULL,
      0x3fe3f13fb89e96f4ULL, 0x3fe4064f47569f48ULL, 0x3fe41b4ae06fea41ULL,
      0x3fe430328e4b26d5ULL, 0x3fe445065b795b55ULL, 0x3fe459c652badc7fULL,
      0x3fe46e727efe4715ULL, 0x3fe4830aeb5f7bfdULL, 0x3fe4978fa3269ee1ULL,
      0x3fe4ac00b1c71762ULL, 0x3fe4c05e22de94e4ULL, 0x3fe4d4a8023414e8ULL,
      0x3fe4e8de5bb6ec04ULL, 0x3fe4fd013b7dd17eULL, 0x3fe51110adc5ed81ULL,
      0x3fe5250cbef1e9faULL, 0x3fe538f57b89061eULL, 0x3fe54ccaf0362c8fULL,
      0x3fe5608d29c70c34ULL, 0x3fe5743c352b33b9ULL, 0x3fe587d81f732fbaULL,
      0x3fe59b60f5cfab9dULL, 0x3fe5aed6c5909517ULL, 0x3fe5c2399c244260ULL,
      0x3fe5d58987169b18ULL, 0x3fe5e8c6941043cfULL, 0x3fe5fbf0d0d5cc49ULL,
      0x3fe60f084b46e05eULL, 0x3fe6220d115d7b8dULL, 0x3fe634ff312d1f3bULL,
      0x3fe647deb8e20b8fULL, 0x3fe65aabb6c07b02ULL, 0x3fe66d663923e086ULL,
      0x3fe6800e4e7e2857ULL, 0x3fe692a40556fb6aULL, 0x3fe6a5276c4b0575ULL,
      0x3fe6b798920b3d98ULL, 0x3fe6c9f7855c3198ULL, 0x3fe6dc44551553aeULL,
      0x3fe6ee7f10204aefULL, 0x3fe700a7c5784633ULL, 0x3fe712be84295198ULL,
      0x3fe724c35b4fae7bULL, 0x3fe736b65a172dffULL, 0x3fe748978fba8e0fULL,
      0x3fe75a670b82d8d8ULL, 0x3fe76c24dcc6c6c0ULL, 0x3fe77dd112ea22c7ULL,
      0x3fe78f6bbd5d315eULL, 0x3fe7a0f4eb9c19a2ULL, 0x3fe7b26cad2e50fdULL,
      0x3fe7c3d311a6092bULL, 0x3fe7d528289fa093ULL, 0x3fe7e66c01c114fdULL,
      0x3fe7f79eacb97898ULL, 0x3fe808c03940694aULL, 0x3fe819d0b7158a4cULL,
      0x3fe82ad036000005ULL, 0x3fe83bbec5cdee22ULL, 0x3fe84c9c7653f7eaULL,
      0x3fe85d69576cc2c5ULL, 0x3fe86e2578f87ae5ULL, 0x3fe87ed0eadc5a2aULL,
      0x3fe88f6bbd023118ULL, 0x3fe89ff5ff57f1f7ULL, 0x3fe8b06fc1cf3dfeULL,
      0x3fe8c0d9145cf49dULL, 0x3fe8d13206f8c4caULL, 0x3fe8e17aa99cc05dULL,
      0x3fe8f1b30c44f167ULL, 0x3fe901db3eeef187ULL, 0x3fe911f35199833bULL,
      0x3fe921fb54442d18ULL,
  };
  return u64_to_f64(table[i]);
}

RS_HD double pow_exp2(int32_t i) {
  static constexpr uint64_t table[64] = {
      0x3ff0000000000000ULL, 0x3ff02c9a3e778061ULL, 0x3ff059b0d3158574ULL,
      0x3ff0874518759bc8ULL, 0x3ff0b5586cf9890fULL, 0x3ff0e3ec32d3d1a2ULL,
      0x3ff11301d0125b51ULL, 0x3ff1429aaea92de0ULL, 0x3ff172b83c7d517bULL,
      0x3ff1a35beb6fcb75ULL, 0x3ff1d4873168b9aaULL, 0x3ff2063b88628cd6ULL,
      0x3ff2387a6e756238ULL, 0x3ff26b4565e27cddULL, 0x3ff29e9df51fdee1ULL,
      0x3ff2d285a6e4030bULL, 0x3ff306fe0a31b715ULL, 0x3ff33c08b26416ffULL,
      0x3ff371a7373aa9cbULL, 0x3ff3a7db34e59ff7ULL, 0x3ff3dea64c123422ULL,
      0x3ff4160a21f72e2aULL, 0x3ff44e086061892dULL, 0x3ff486a2b5c13cd0ULL,
      0x3ff4bfdad5362a27ULL, 0x3ff4f9b2769d2ca7ULL, 0x3ff5342b569d4f82ULL,
      0x3ff56f4736b527daULL, 0x3ff5ab07dd485429ULL, 0x3ff5e76f15ad2148ULL,
      0x3ff6247eb03a5585ULL, 0x3ff6623882552225ULL, 0x3ff6a09e667f3bcdULL,
      0x3ff6dfb23c651a2fULL, 0x3ff71f75e8ec5f74ULL, 0x3ff75feb564267c9ULL,
      0x3ff7a11473eb0187ULL, 0x3ff7e2f336cf4e62ULL, 0x3ff82589994cce13ULL,
      0x3ff868d99b4492edULL, 0x3ff8ace5422aa0dbULL, 0x3ff8f1ae99157736ULL,
      0x3ff93737b0cdc5e5ULL, 0x3ff97d829fde4e50ULL, 0x3ff9c49182a3f090ULL,
      0x3ffa0c667b5de565ULL, 0x3ffa5503b23e255dULL, 0x3ffa9e6b5579fdbfULL,
      0x3ffae89f995ad3adULL, 0x3ffb33a2b84f15fbULL, 0x3ffb7f76f2fb5e47ULL,
      0x3ffbcc1e904bc1d2ULL, 0x3ffc199bdd85529cULL, 0x3ffc67f12e57d14bULL,
      0x3ffcb720dcef9069ULL, 0x3ffd072d4a07897cULL, 0x3ffd5818dcfba487ULL,
      0x3ffda9e603db3285ULL, 0x3ffdfc97337b9b5fULL, 0x3ffe502ee78b3ff6ULL,
      0x3ffea4afa2a490daULL, 0x3ffefa1bee615a27ULL, 0x3fff50765b6e4540ULL,
      0x3fffa7c1819e90d8ULL,
  };
  return u64_to_f64(table[i]);
}

RS_HD double pow_inv_c(uint64_t i) {
  static constexpr uint64_t table[257] = {
      0x4000000000000000ULL, 0x3fffe01fe01fe020ULL, 0x3fffc07f01fc07f0ULL,
      0x3fffa11caa01fa12ULL, 0x3fff81f81f81f820ULL, 0x3fff6310aca0dbb5ULL,
      0x3fff44659e4a4271ULL, 0x3fff25f644230ab5ULL, 0x3fff07c1f07c1f08ULL,
      0x3ffee9c7f8458e02ULL, 0x3ffecc07b301ecc0ULL, 0x3ffeae807aba01ebULL,
      0x3ffe9131abf0b767ULL, 0x3ffe741aa59750e4ULL, 0x3ffe573ac901e574ULL,
      0x3ffe3a9179dc1a73ULL, 0x3ffe1e1e1e1e1e1eULL, 0x3ffe01e01e01e01eULL,
      0x3ffde5d6e3f8868aULL, 0x3ffdca01dca01dcaULL, 0x3ffdae6076b981dbULL,
      0x3ffd92f2231e7f8aULL, 0x3ffd77b654b82c34ULL, 0x3ffd5cac807572b2ULL,
      0x3ffd41d41d41d41dULL, 0x3ffd272ca3fc5b1aULL, 0x3ffd0cb58f6ec074ULL,
      0x3ffcf26e5c44bfc6ULL, 0x3ffcd85689039b0bULL, 0x3ffcbe6d9601cbe7ULL,
      0x3ffca4b3055ee191ULL, 0x3ffc8b265afb8a42ULL, 0x3ffc71c71c71c71cULL,
      0x3ffc5894d10d4986ULL, 0x3ffc3f8f01c3f8f0ULL, 0x3ffc26b5392ea01cULL,
      0x3ffc0e070381c0e0ULL, 0x3ffbf583ee868d8bULL, 0x3ffbdd2b899406f7ULL,
      0x3ffbc4fd65883e7bULL, 0x3ffbacf914c1bad0ULL, 0x3ffb951e2b18ff23ULL,
      0x3ffb7d6c3dda338bULL, 0x3ffb65e2e3beee05ULL, 0x3ffb4e81b4e81b4fULL,
      0x3ffb37484ad806ceULL, 0x3ffb2036406c80d9ULL, 0x3ffb094b31d922a4ULL,
      0x3ffaf286bca1af28ULL, 0x3ffadbe87f94905eULL, 0x3ffac5701ac5701bULL,
      0x3ffaaf1d2f87ebfdULL, 0x3ffa98ef606a63beULL, 0x3ffa82e65130e159ULL,
      0x3ffa6d01a6d01a6dULL, 0x3ffa574107688a4aULL, 0x3ffa41a41a41a41aULL,
      0x3ffa2c2a87c51ca0ULL, 0x3ffa16d3f97a4b02ULL, 0x3ffa01a01a01a01aULL,
      0x3ff9ec8e951033d9ULL, 0x3ff9d79f176b682dULL, 0x3ff9c2d14ee4a102ULL,
      0x3ff9ae24ea5510daULL, 0x3ff999999999999aULL, 0x3ff9852f0d8ec0ffULL,
      0x3ff970e4f80cb872ULL, 0x3ff95cbb0be377aeULL, 0x3ff948b0fcd6e9e0ULL,
      0x3ff934c67f9b2ce6ULL, 0x3ff920fb49d0e229ULL, 0x3ff90d4f120190d5ULL,
      0x3ff8f9c18f9c18faULL, 0x3ff8e6527af1373fULL, 0x3ff8d3018d3018d3ULL,
      0x3ff8bfce8062ff3aULL, 0x3ff8acb90f6bf3aaULL, 0x3ff899c0f601899cULL,
      0x3ff886e5f0abb04aULL, 0x3ff87427bcc092b9ULL, 0x3ff8618618618618ULL,
      0x3ff84f00c2780614ULL, 0x3ff83c977ab2beddULL, 0x3ff82a4a0182a4a0ULL,
      0x3ff8181818181818ULL, 0x3ff8060180601806ULL, 0x3ff7f405fd017f40ULL,
      0x3ff7e225515a4f1dULL, 0x3ff7d05f417d05f4ULL, 0x3ff7beb3922e017cULL,
      0x3ff7ad2208e0ecc3ULL, 0x3ff79baa6bb6398bULL, 0x3ff78a4c8178a4c8ULL,
      0x3ff77908119ac60dULL, 0x3ff767dce434a9b1ULL, 0x3ff756cac201756dULL,
      0x3ff745d1745d1746ULL, 0x3ff734f0c541fe8dULL, 0x3ff724287f46debcULL,
      0x3ff713786d9c7c09ULL, 0x3ff702e05c0b8170ULL, 0x3ff6f26016f26017ULL,
      0x3ff6e1f76b4337c7ULL, 0x3ff6d1a62681c861ULL, 0x3ff6c16c16c16c17ULL,
      0x3ff6b1490aa31a3dULL, 0x3ff6a13cd1537290ULL, 0x3ff691473a88d0c0ULL,
      0x3ff6816816816817ULL, 0x3ff6719f3601671aULL, 0x3ff661ec6a5122f9ULL,
      0x3ff6524f853b4aa3ULL, 0x3ff642c8590b2164ULL, 0x3ff63356b88ac0deULL,
      0x3ff623fa77016240ULL, 0x3ff614b36831ae94ULL, 0x3ff6058160581606ULL,
      0x3ff5f66434292dfcULL, 0x3ff5e75bb8d015e7ULL, 0x3ff5d867c3ece2a5ULL,
      0x3ff5c9882b931057ULL, 0x3ff5babcc647fa91ULL, 0x3ff5ac056b015ac0ULL,
      0x3ff59d61f123ccaaULL, 0x3ff58ed2308158edULL, 0x3ff5805601580560ULL,
      0x3ff571ed3c506b3aULL, 0x3ff56397ba7c52e2ULL, 0x3ff5555555555555ULL,
      0x3ff54725e6bb82feULL, 0x3ff5390948f40febULL, 0x3ff52aff56a8054bULL,
      0x3ff51d07eae2f815ULL, 0x3ff50f22e111c4c5ULL, 0x3ff5015015015015ULL,
      0x3ff4f38f62dd4c9bULL, 0x3ff4e5e0a72f0539ULL, 0x3ff4d843bedc2c4cULL,
      0x3ff4cab88725af6eULL, 0x3ff4bd3edda68fe1ULL, 0x3ff4afd6a052bf5bULL,
      0x3ff4a27fad76014aULL, 0x3ff49539e3b2d067ULL, 0x3ff4880522014880ULL,
      0x3ff47ae147ae147bULL, 0x3ff46dce34596066ULL, 0x3ff460cbc7f5cf9aULL,
      0x3ff453d9e2c776caULL, 0x3ff446f86562d9fbULL, 0x3ff43a2730abee4dULL,
      0x3ff42d6625d51f87ULL, 0x3ff420b5265e5951ULL, 0x3ff4141414141414ULL,
      0x3ff40782d10e6566ULL, 0x3ff3fb013fb013fbULL, 0x3ff3ee8f42a5af07ULL,
      0x3ff3e22cbce4a902ULL, 0x3ff3d5d991aa75c6ULL, 0x3ff3c995a47babe7ULL,
      0x3ff3bd60d9232955ULL, 0x3ff3b13b13b13b14ULL, 0x3ff3a524387ac822ULL,
      0x3ff3991c2c187f63ULL, 0x3ff38d22d366088eULL, 0x3ff3813813813814ULL,
      0x3ff3755bd1c945eeULL, 0x3ff3698df3de0748ULL, 0x3ff35dce5f9f2af8ULL,
      0x3ff3521cfb2b78c1ULL, 0x3ff34679ace01346ULL, 0x3ff33ae45b57bcb2ULL,
      0x3ff32f5ced6a1dfaULL, 0x3ff323e34a2b10bfULL, 0x3ff3187758e9ebb6ULL,
      0x3ff30d190130d190ULL, 0x3ff301c82ac40260ULL, 0x3ff2f684bda12f68ULL,
      0x3ff2eb4ea1fed14bULL, 0x3ff2e025c04b8097ULL, 0x3ff2d50a012d50a0ULL,
      0x3ff2c9fb4d812ca0ULL, 0x3ff2bef98e5a3711ULL, 0x3ff2b404ad012b40ULL,
      0x3ff2a91c92f3c105ULL, 0x3ff29e4129e4129eULL, 0x3ff293725bb804a5ULL,
      0x3ff288b01288b013ULL, 0x3ff27dfa38a1ce4dULL, 0x3ff27350b8812735ULL,
      0x3ff268b37cd60127ULL, 0x3ff25e22708092f1ULL, 0x3ff2539d7e9177b2ULL,
      0x3ff2492492492492ULL, 0x3ff23eb79717605bULL, 0x3ff23456789abcdfULL,
      0x3ff22a0122a0122aULL, 0x3ff21fb78121fb78ULL, 0x3ff21579804855e6ULL,
      0x3ff20b470c67c0d9ULL, 0x3ff2012012012012ULL, 0x3ff1f7047dc11f70ULL,
      0x3ff1ecf43c7fb84cULL, 0x3ff1e2ef3b3fb874ULL, 0x3ff1d8f5672e4abdULL,
      0x3ff1cf06ada2811dULL, 0x3ff1c522fc1ce059ULL, 0x3ff1bb4a4046ed29ULL,
      0x3ff1b17c67f2bae3ULL, 0x3ff1a7b9611a7b96ULL, 0x3ff19e0119e0119eULL,
      0x3ff19453808ca29cULL, 0x3ff18ab083902bdbULL, 0x3ff1811811811812ULL,
      0x3ff1778a191bd684ULL, 0x3ff16e0689427379ULL, 0x3ff1648d50fc3201ULL,
      0x3ff15b1e5f75270dULL, 0x3ff151b9a3fdd5c9ULL, 0x3ff1485f0e0acd3bULL,
      0x3ff13f0e8d344724ULL, 0x3ff135c81135c811ULL, 0x3ff12c8b89edc0acULL,
      0x3ff12358e75d3033ULL, 0x3ff11a3019a74826ULL, 0x3ff1111111111111ULL,
      0x3ff107fbbe011080ULL, 0x3ff0fef010fef011ULL, 0x3ff0f5edfab325a2ULL,
      0x3ff0ecf56be69c90ULL, 0x3ff0e40655826011ULL, 0x3ff0db20a88f4696ULL,
      0x3ff0d24456359e3aULL, 0x3ff0c9714fbcda3bULL, 0x3ff0c0a7868b4171ULL,
      0x3ff0b7e6ec259dc8ULL, 0x3ff0af2f722eecb5ULL, 0x3ff0a6810a6810a7ULL,
      0x3ff09ddba6af8360ULL, 0x3ff0953f39010954ULL, 0x3ff08cabb37565e2ULL,
      0x3ff0842108421084ULL, 0x3ff07b9f29b8eae2ULL, 0x3ff073260a47f7c6ULL,
      0x3ff06ab59c7912fbULL, 0x3ff0624dd2f1a9fcULL, 0x3ff059eea0727586ULL,
      0x3ff05197f7d73404ULL, 0x3ff04949cc1664c5ULL, 0x3ff0410410410410ULL,
      0x3ff038c6b78247fcULL, 0x3ff03091b51f5e1aULL, 0x3ff02864fc7729e9ULL,
      0x3ff0204081020408ULL, 0x3ff0182436517a37ULL, 0x3ff0101010101010ULL,
      0x3ff0080402010080ULL, 0x3ff0000000000000ULL,
  };
  return u64_to_f64(table[i]);
}

RS_HD double pow_log_c(uint64_t i) {
  static constexpr uint64_t table[257] = {
      0x0000000000000000ULL, 0x3f6ff00aa2b10bc0ULL, 0x3f7fe02a6b106789ULL,
      0x3f87dc475f810a77ULL, 0x3f8fc0a8b0fc03e4ULL, 0x3f93cea44346a575ULL,
      0x3f97b91b07d5b11bULL, 0x3f9b9fc027af9198ULL, 0x3f9f829b0e783300ULL,
      0x3fa1b0d98923d980ULL, 0x3fa39e87b9febd60ULL, 0x3fa58a5bafc8e4d5ULL,
      0x3fa77458f632dcfcULL, 0x3fa95c830ec8e3ebULL, 0x3fab42dd711971bfULL,
      0x3fad276b8adb0b52ULL, 0x3faf0a30c01162a6ULL, 0x3fb075983598e471ULL,
      0x3fb16536eea37ae1ULL, 0x3fb253f62f0a1417ULL, 0x3fb341d7961bd1d1ULL,
      0x3fb42edcbea646f0ULL, 0x3fb51b073f06183fULL, 0x3fb60658a93750c4ULL,
      0x3fb6f0d28ae56b4cULL, 0x3fb7da766d7b12cdULL, 0x3fb8c345d6319b21ULL,
      0x3fb9ab42462033adULL, 0x3fba926d3a4ad563ULL, 0x3fbb78c82bb0eda1ULL,
      0x3fbc5e548f5bc743ULL, 0x3fbd4313d66cb35dULL, 0x3fbe27076e2af2e6ULL,
      0x3fbf0a30c01162a6ULL, 0x3fbfec9131dbeabbULL, 0x3fc0671512ca596eULL,
      0x3fc0d77e7cd08e59ULL, 0x3fc14785846742acULL, 0x3fc1b72ad52f67a0ULL,
      0x3fc2266f190a5acbULL, 0x3fc29552f81ff523ULL, 0x3fc303d718e47fd3ULL,
      0x3fc371fc201e8f74ULL, 0x3fc3dfc2b0ecc62aULL, 0x3fc44d2b6ccb7d1eULL,
      0x3fc4ba36f39a55e5ULL, 0x3fc526e5e3a1b438ULL, 0x3fc59338d9982086ULL,
      0x3fc5ff3070a793d4ULL, 0x3fc66acd4272ad51ULL, 0x3fc6d60fe719d21dULL,
      0x3fc740f8f54037a5ULL, 0x3fc7ab890210d909ULL, 0x3fc815c0a14357ebULL,
      0x3fc87fa06520c911ULL, 0x3fc8e928de886d41ULL, 0x3fc9525a9cf456b4ULL,
      0x3fc9bb362e7dfb83ULL, 0x3fca23bc1fe2b563ULL, 0x3fca8becfc882f19ULL,
      0x3fcaf3c94e80bff3ULL, 0x3fcb5b519e8fb5a4ULL, 0x3fcbc286742d8cd6ULL,
      0x3fcc2968558c18c1ULL, 0x3fcc8ff7c79a9a22ULL, 0x3fccf6354e09c5dcULL,
      0x3fcd5c216b4fbb91ULL, 0x3fcdc1bca0abec7dULL, 0x3fce27076e2af2e6ULL,
      0x3fce8c0252aa5a60ULL, 0x3fcef0adcbdc5936ULL, 0x3fcf550a564b7b37ULL,
      0x3fcfb9186d5e3e2bULL, 0x3fd00e6c45ad501dULL, 0x3fd0402594b4d041ULL,
      0x3fd071b85fcd590dULL, 0x3fd0a324e27390e3ULL, 0x3fd0d46b579ab74bULL,
      0x3fd1058bf9ae4ad5ULL, 0x3fd136870293a8b0ULL, 0x3fd1675cababa60eULL,
      0x3fd1980d2dd4236fULL, 0x3fd1c898c16999fbULL, 0x3fd1f8ff9e48a2f3ULL,
      0x3fd22941fbcf7966ULL, 0x3fd2596010df763aULL, 0x3fd2895a13de86a3ULL,
      0x3fd2b9303ab89d25ULL, 0x3fd2e8e2bae11d31ULL, 0x3fd31871c9544185ULL,
      0x3fd347dd9a987d55ULL, 0x3fd3772662bfd85bULL, 0x3fd3a64c556945eaULL,
      0x3fd3d54fa5c1f710ULL, 0x3fd404308686a7e4ULL, 0x3fd432ef2a04e814ULL,
      0x3fd4618bc21c5ec2ULL, 0x3fd49006804009d1ULL, 0x3fd4be5f957778a1ULL,
      0x3fd4ec973260026aULL, 0x3fd51aad872df82dULL, 0x3fd548a2c3add263ULL,
      0x3fd5767717455a6cULL, 0x3fd5a42ab0f4cfe2ULL, 0x3fd5d1bdbf5809caULL,
      0x3fd5ff3070a793d4ULL, 0x3fd62c82f2b9c795ULL, 0x3fd659b57303e1f3ULL,
      0x3fd686c81e9b14afULL, 0x3fd6b3bb2235943eULL, 0x3fd6e08eaa2ba1e4ULL,
      0x3fd70d42e2789236ULL, 0x3fd739d7f6bbd007ULL, 0x3fd7664e1239dbcfULL,
      0x3fd792a55fdd47a2ULL, 0x3fd7bede0a37afc0ULL, 0x3fd7eaf83b82afc3ULL,
      0x3fd816f41da0d496ULL, 0x3fd842d1da1e8b17ULL, 0x3fd86e919a330ba0ULL,
      0x3fd89a3386c1425bULL, 0x3fd8c5b7c858b48bULL, 0x3fd8f11e873662c7ULL,
      0x3fd91c67eb45a83eULL, 0x3fd947941c2116fbULL, 0x3fd972a341135158ULL,
      0x3fd99d958117e08bULL, 0x3fd9c86b02dc0863ULL, 0x3fd9f323ecbf984cULL,
      0x3fda1dc064d5b995ULL, 0x3fda484090e5bb0aULL, 0x3fda72a4966bd9eaULL,
      0x3fda9cec9a9a084aULL, 0x3fdac718c258b0e4ULL, 0x3fdaf1293247786bULL,
      0x3fdb1b1e0ebdfc5bULL, 0x3fdb44f77bcc8f63ULL, 0x3fdb6eb59d3cf35eULL,
      0x3fdb9858969310fbULL, 0x3fdbc1e08b0dad0aULL, 0x3fdbeb4d9da71b7cULL,
      0x3fdc149ff115f027ULL, 0x3fdc3dd7a7cdad4dULL, 0x3fdc66f4e3ff6ff8ULL,
      0x3fdc8ff7c79a9a22ULL, 0x3fdcb8e0744d7acaULL, 0x3fdce1af0b85f3ebULL,
      0x3fdd0a63ae721e64ULL, 0x3fdd32fe7e00ebd5ULL, 0x3fdd5b7f9ae2c684ULL,
      0x3fdd83e7258a2f3eULL, 0x3fddac353e2c5954ULL, 0x3fddd46a04c1c4a1ULL,
      0x3fddfc859906d5b5ULL, 0x3fde24881a7c6c26ULL, 0x3fde4c71a8687704ULL,
      0x3fde744261d68788ULL, 0x3fde9bfa659861f5ULL, 0x3fdec399d2468cc0ULL,
      0x3fdeeb20c640ddf4ULL, 0x3fdf128f5faf06edULL, 0x3fdf39e5bc811e5cULL,
      0x3fdf6123fa7028acULL, 0x3fdf884a36fe9ec2ULL, 0x3fdfaf588f78f31fULL,
      0x3fdfd64f20f61572ULL, 0x3fdffd2e0857f498ULL, 0x3fe011fab125ff8aULL,
      0x3fe02552a5a5d0ffULL, 0x3fe0389eefce633bULL, 0x3fe04bdf9da926d2ULL,
      0x3fe05f14bd26459cULL, 0x3fe0723e5c1cdf40ULL, 0x3fe0855c884b450eULL,
      0x3fe0986f4f573521ULL, 0x3fe0ab76bece14d2ULL, 0x3fe0be72e4252a83ULL,
      0x3fe0d163ccb9d6b8ULL, 0x3fe0e44985d1cc8cULL, 0x3fe0f7241c9b497dULL,
      0x3fe109f39e2d4c97ULL, 0x3fe11cb81787ccf8ULL, 0x3fe12f719593efbcULL,
      0x3fe1422025243d45ULL, 0x3fe154c3d2f4d5eaULL, 0x3fe1675cababa60eULL,
      0x3fe179eabbd899a1ULL, 0x3fe18c6e0ff5cf06ULL, 0x3fe19ee6b467c96fULL,
      0x3fe1b154b57da29fULL, 0x3fe1c3b81f713c25ULL, 0x3fe1d610fe677003ULL,
      0x3fe1e85f5e7040d0ULL, 0x3fe1faa34b87094cULL, 0x3fe20cdcd192ab6eULL,
      0x3fe21f0bfc65beecULL, 0x3fe23130d7bebf43ULL, 0x3fe2434b6f483934ULL,
      0x3fe2555bce98f7cbULL, 0x3fe26762013430e0ULL, 0x3fe2795e1289b11bULL,
      0x3fe28b500df60783ULL, 0x3fe29d37fec2b08bULL, 0x3fe2af15f02640adULL,
      0x3fe2c0e9ed448e8cULL, 0x3fe2d2b4012edc9eULL, 0x3fe2e47436e40268ULL,
      0x3fe2f62a99509546ULL, 0x3fe307d7334f10beULL, 0x3fe3197a0fa7fe6aULL,
      0x3fe32b1339121d71ULL, 0x3fe33ca2ba328995ULL, 0x3fe34e289d9ce1d3ULL,
      0x3fe35fa4edd36ea0ULL, 0x3fe37117b54747b6ULL, 0x3fe38280fe58797fULL,
      0x3fe393e0d3562a1aULL, 0x3fe3a5373e7ebdfaULL, 0x3fe3b68449fffc23ULL,
      0x3fe3c7c7fff73206ULL, 0x3fe3d9026a7156fbULL, 0x3fe3ea33936b2f5cULL,
      0x3fe3fb5b84d16f42ULL, 0x3fe40c7a4880dce9ULL, 0x3fe41d8fe84672aeULL,
      0x3fe42e9c6ddf80bfULL, 0x3fe43f9fe2f9ce67ULL, 0x3fe4509a5133bb0aULL,
      0x3fe4618bc21c5ec2ULL, 0x3fe472743f33aaadULL, 0x3fe48353d1ea88dfULL,
      0x3fe4942a83a2fc07ULL, 0x3fe4a4f85db03ebbULL, 0x3fe4b5bd6956e274ULL,
      0x3fe4c679afccee3aULL, 0x3fe4d72d3a39fd00ULL, 0x3fe4e7d811b75bb1ULL,
      0x3fe4f87a3f5026e9ULL, 0x3fe50913cc01686bULL, 0x3fe519a4c0ba3446ULL,
      0x3fe52a2d265bc5abULL, 0x3fe53aad05b99b7dULL, 0x3fe54b2467999498ULL,
      0x3fe55b9354b40bcdULL, 0x3fe56bf9d5b3f399ULL, 0x3fe57c57f336f191ULL,
      0x3fe58cadb5cd7989ULL, 0x3fe59cfb25fae87eULL, 0x3fe5ad404c359f2dULL,
      0x3fe5bd7d30e71c73ULL, 0x3fe5cdb1dc6c1765ULL, 0x3fe5ddde57149923ULL,
      0x3fe5ee02a9241675ULL, 0x3fe5fe1edad18919ULL, 0x3fe60e32f44788d9ULL,
      0x3fe61e3efda46467ULL, 0x3fe62e42fefa39efULL,
  };
  return u64_to_f64(table[i]);
}

RS_HD double sin_poly(double r) {
  double r2 = r * r;
  double p = fma_f64(r2, u64_to_f64(k_sin_c1), u64_to_f64(k_sin_c2));
  p = fma_f64(p, r2, u64_to_f64(k_sin_c3));
  p = fma_f64(p, r2, u64_to_f64(k_sin_c4));
  return fma_f64(p, r * r2, r);
}

RS_HD double cos_poly(double r) {
  double r2 = r * r;
  double base = 1.0 - r2 * 0.5;
  double p = fma_f64(r2, u64_to_f64(k_cos_c1), u64_to_f64(k_cos_c2));
  p = fma_f64(p, r2, u64_to_f64(k_cos_c3));
  p = fma_f64(p, r2, u64_to_f64(k_cos_c4));
  return fma_f64(p, r2 * r2, base);
}

RS_HD double reduce_by_pio2(double a, int32_t &quadrant) {
  int32_t n = sat_f64_to_i32(fma_f64(a, u64_to_f64(k_inv_pio2), 0.5));
  double nd = static_cast<double>(n);
  double r1 = fma_f64(-nd, u64_to_f64(k_pio2_hi), a);
  quadrant = n & 3;
  return r1 - nd * u64_to_f64(k_pio2_lo);
}

RS_HD float powf_exp(double t, uint32_t sign, ExpRemainder remainder) {
  if (t > u64_to_f64(k_exp_overflow)) {
    return u32_to_f32(k_f32_inf_bits | sign);
  }
  if (!(t > u64_to_f64(k_exp_underflow))) {
    return u32_to_f32(sign);
  }
  int32_t n =
      static_cast<int32_t>(round_ties_even_f64(t * u64_to_f64(k_exp_scale)));
  double tab = pow_exp2(n & 63);
  double res;
  if (remainder == ExpRemainder::Fused) {
    double r = fma_f64(-static_cast<double>(n), u64_to_f64(k_ln2_over_64), t);
    double poly = fma_f64(fma_f64(u64_to_f64(k_one_sixth), r, 0.5), r, 1.0);
    res = fma_f64(r * poly, tab, tab);
  } else {
    double r = t - static_cast<double>(n) * u64_to_f64(k_ln2_over_64);
    res = ((r * r) * (u64_to_f64(k_one_sixth) * r + 0.5) + r) * tab + tab;
  }
  uint64_t rb = f64_to_u64(res);
  rb += static_cast<uint64_t>(static_cast<int64_t>(n >> 6)) << 52;
  float f = static_cast<float>(u64_to_f64(rb));
  return u32_to_f32(f32_to_u32(f) | sign);
}

RS_HD bool pow_exponent_is_odd_integer(float y, uint32_t ay) {
  if ((ay & k_exp_mask) > k_pow_fractional_exponent_limit) {
    return false;
  }
  float yr = round_ties_even_f32(y);
  return yr == y && (static_cast<int32_t>(yr) & 1) != 0;
}

}

RS_HD RS_FMA_DISPATCH float ucrt_acosf(float x) {
  using namespace ucrt;
  uint32_t ux = f32_to_u32(x);
  uint32_t ax = ux & k_f32_abs_mask;
  uint32_t ex = (ux >> 23) & 0xff;
  if (ax > k_f32_inf_bits) {
    return u32_to_f32(ux | k_quiet_bit);
  }
  if (ex < k_acos_right_angle_exponent) {
    return u32_to_f32(k_f32_pi_over_2);
  }
  if (ex >= k_acos_unit_exponent) {
    if (x == 1.0f) {
      return 0.0f;
    }
    if (x == -1.0f) {
      return u32_to_f32(k_f32_pi);
    }
    return u32_to_f32(k_f32_canonical_nan_bits);
  }
  float a = abs_f32(x);
  bool big = ex >= k_acos_sqrt_branch_exponent;
  float z = big ? (1.0f - a) * 0.5f : a * a;
  float s = big ? sqrt_f32(z) : 0.0f;
  float t = u32_to_f32(k_acos_c0);
  t = fma_f32(-t, z, u32_to_f32(k_acos_c1));
  t = fma_f32(t, z, -u32_to_f32(k_acos_c2));
  t = fma_f32(t, z, u32_to_f32(k_acos_c3));
  float r =
      div_f32(t * z, fma_f32(-u32_to_f32(k_acos_d0), z, u32_to_f32(k_acos_d1)));
  if (!big) {
    double lo = u64_to_f64(k_acos_offset) - static_cast<double>(r * x);
    return static_cast<float>(u64_to_f64(k_pio2) -
                              (static_cast<double>(x) - lo));
  }
  if (static_cast<int32_t>(ux) < 0) {
    double h = (static_cast<double>(s * r) - u64_to_f64(k_acos_offset)) +
               static_cast<double>(s);
    return static_cast<float>(u64_to_f64(k_pio) - (h + h));
  }
  float hi = u32_to_f32(f32_to_u32(s) & k_acos_sqrt_head_mask);
  float c = div_f32(fma_f32(-hi, hi, z), s + hi);
  return fma_f32(r, s + s, c + c) + (hi + hi);
}

RS_HD RS_FMA_DISPATCH float ucrt_atan2f(float y, float x) {
  using namespace ucrt;
  double yd = static_cast<double>(y);
  double xd = static_cast<double>(x);
  uint64_t yb = f64_to_u64(yd);
  uint64_t xb = f64_to_u64(xd);
  uint64_t ax = xb & k_f64_abs_mask;
  uint64_t ay = yb & k_f64_abs_mask;
  bool yneg = (yb >> 63) != 0;
  bool xneg = (xb >> 63) != 0;
  int32_t diff = static_cast<int32_t>((yb >> 52) & 0x7ff) -
                 static_cast<int32_t>((xb >> 52) & 0x7ff);

  if (ax > k_f64_inf_bits) {
    return quiet(x);
  }
  if (ay > k_f64_inf_bits) {
    return quiet(y);
  }
  if (ay == 0) {
    if (xneg) {
      return u32_to_f32(yneg ? k_f32_neg_pi : k_f32_pi);
    }
    return y;
  }
  if (ax == 0 || diff > 26) {
    return u32_to_f32(yneg ? k_f32_neg_pi_over_2 : k_f32_pi_over_2);
  }
  if (diff < -13 && !xneg) {
    if (diff < -150) {
      return yneg ? -0.0f : 0.0f;
    }
    return static_cast<float>(yd / xd);
  }
  if (diff < -26) {
    return u32_to_f32(yneg ? k_f32_neg_pi : k_f32_pi);
  }
  if (ay == k_f64_inf_bits) {
    if (!xneg) {
      return u32_to_f32(yneg ? k_f32_neg_pi_over_4 : k_f32_pi_over_4);
    }
    return u32_to_f32(yneg ? k_f32_neg_three_pi_over_4 : k_f32_three_pi_over_4);
  }

  double u = abs_f64(yd);
  double v = abs_f64(xd);
  bool swapped = u > v;
  if (swapped) {
    double larger = u;
    u = v;
    v = larger;
  }
  double ratio = u / v;
  double res;
  if (ratio > u64_to_f64(k_two_pow_neg_4)) {
    int32_t i = static_cast<int32_t>(fma_f64(256.0, ratio, 0.5));
    double id = static_cast<double>(i);
    double r = fma_f64(u, 256.0, -(id * v)) / fma_f64(u, id, v * 256.0);
    res = fma_f64(-((r * r) * r), u64_to_f64(k_atan_cubic),
                  r + atan_tab(i - k_atan_table_first_step));
  } else if (ratio < u64_to_f64(k_atan_small)) {
    res = ratio;
  } else {
    double r2 = ratio * ratio;
    double p =
        fma_f64(u64_to_f64(k_atan_poly_a), r2, u64_to_f64(k_atan_poly_b));
    p = fma_f64(-p, r2, u64_to_f64(k_atan_poly_c));
    res = fma_f64(-(r2 * ratio), p, ratio);
  }
  if (swapped) {
    res = u64_to_f64(k_pio2) - res;
  }
  if (xneg) {
    res = u64_to_f64(k_pio) - res;
  }
  if (yneg) {
    return static_cast<float>(-res);
  }
  return static_cast<float>(res);
}

RS_HD RS_FMA_DISPATCH float ucrt_powf(float x, float y) {
  using namespace ucrt;
  uint32_t ux = f32_to_u32(x);
  uint32_t uy = f32_to_u32(y);
  uint32_t ax = ux & k_f32_abs_mask;
  uint32_t ay = uy & k_f32_abs_mask;

  if (ay >= k_f32_inf_bits) {
    if (ay > k_f32_inf_bits) {
      if (ux == k_f32_one_bits) {
        return ay < k_f32_quiet_nan_bits ? 1.5f : 1.0f;
      }
      return u32_to_f32((ax > k_f32_inf_bits ? ux : uy) | k_quiet_bit);
    }
    if (ax > k_f32_inf_bits) {
      return u32_to_f32(ux | k_quiet_bit);
    }
    if (ax == k_f32_one_bits) {
      return 1.0f;
    }
    return u32_to_f32((static_cast<int32_t>(uy) < 0) == (ax < k_f32_one_bits)
                          ? k_f32_inf_bits
                          : 0u);
  }
  if (ay == 0) {
    if (ax > k_f32_inf_bits && ax < k_f32_quiet_nan_bits) {
      return u32_to_f32(ux | k_quiet_bit);
    }
    return 1.0f;
  }
  if (uy == k_f32_one_bits) {
    if (ax > k_f32_inf_bits) {
      return u32_to_f32(ux | k_quiet_bit);
    }
    return x;
  }
  if (ax > k_f32_inf_bits) {
    return u32_to_f32(ux | k_quiet_bit);
  }
  if (ax == k_f32_inf_bits || ax == 0) {
    uint32_t mag =
        (static_cast<int32_t>(uy) < 0) == (ax == 0) ? k_f32_inf_bits : 0u;
    uint32_t sign = pow_exponent_is_odd_integer(y, ay) ? ux & k_sign_bit : 0u;
    return u32_to_f32(mag | sign);
  }

  uint32_t sign = 0;
  if (static_cast<int32_t>(ux) < 0 &&
      (ay & k_exp_mask) <= k_pow_fractional_exponent_limit) {
    if (round_ties_even_f32(y) != y) {
      return u32_to_f32(k_f32_canonical_nan_bits);
    }
    if (pow_exponent_is_odd_integer(y, ay)) {
      sign = k_sign_bit;
    }
  }

  double dx = static_cast<double>(x);
  double dy = static_cast<double>(y);

  if (static_cast<int32_t>(ux) < k_pow_near_one_upper_bits) {
    double f = dx - 1.0;
    if (abs_f64(f) < u64_to_f64(k_two_pow_neg_4)) {
      double u = f / (f + 2.0);
      double w = u + u;
      double v = w * w;
      double w3 = w * v;
      double a = v * u64_to_f64(k_log1p_a) + u64_to_f64(k_log1p_b);
      double b = v * u64_to_f64(k_log1p_c) + u64_to_f64(k_log1p_d);
      double s = (((v * v) * w3) * b + w3 * a) - u * f;
      return powf_exp(dy * (f + s), sign, ExpRemainder::Unfused);
    }
  }

  uint64_t bx = f64_to_u64(dx);
  uint64_t mant = bx & 0x000fffffffffffffULL;
  uint64_t idx = (mant >> 44) + ((mant >> 43) & 1);
  uint64_t cb = (0x3fe00ULL | idx) << 44;
  uint64_t mb = mant | 0x3fe0000000000000ULL;
  double c = u64_to_f64(cb);
  double m = u64_to_f64(mb);
  int32_t e = static_cast<int32_t>((bx >> 52) & 0x7ff) - 0x3ff;
  double r = (c - m) * pow_inv_c(idx);
  double rq = r * fma_f64(fma_f64(u64_to_f64(k_one_third), r, 0.5), r, 1.0);
  double log_x =
      (static_cast<double>(e) * u64_to_f64(k_ln2) + pow_log_c(idx)) - rq;
  return powf_exp(dy * log_x, sign, ExpRemainder::Fused);
}

RS_HD RS_FMA_DISPATCH float ucrt_sinf(float x) {
  using namespace ucrt;
  uint32_t ax = abs_bits(x);
  if (ax < k_trig_identity_limit) {
    return x;
  }
  double d = static_cast<double>(x);
  if (ax <= k_f32_pi_over_4) {
    if (ax < k_trig_short_series_limit) {
      return static_cast<float>(
          fma_f64(-((d * d) * d), u64_to_f64(k_one_sixth), d));
    }
    return static_cast<float>(sin_poly(d));
  }
  int32_t q = 0;
  double r = reduce_by_pio2(abs_f64(d), q);
  double y = (q & 1) != 0 ? cos_poly(r) : sin_poly(r);
  if ((q >= 2) != (x < 0.0f)) {
    return static_cast<float>(-y);
  }
  return static_cast<float>(y);
}

RS_HD RS_FMA_DISPATCH float ucrt_cosf(float x) {
  using namespace ucrt;
  uint32_t ax = abs_bits(x);
  if (ax < k_trig_identity_limit) {
    return 1.0f;
  }
  double d = static_cast<double>(x);
  if (ax <= k_f32_pi_over_4) {
    if (ax < k_trig_short_series_limit) {
      return static_cast<float>(fma_f64(-(d * 0.5), d, 1.0));
    }
    return static_cast<float>(cos_poly(d));
  }
  int32_t q = 0;
  double r = reduce_by_pio2(abs_f64(d), q);
  double y = (q & 1) != 0 ? sin_poly(r) : cos_poly(r);
  if (q == 1 || q == 2) {
    return static_cast<float>(-y);
  }
  return static_cast<float>(y);
}

RS_HD float sinf_ucrt(float x) {
  if (ucrt::abs_bits(x) < ucrt::k_sinf_limit) {
    return ucrt_sinf(x);
  }
  return sinf_glibc(x);
}

RS_HD float cosf_ucrt(float x) {
  if (ucrt::abs_bits(x) < ucrt::k_cosf_limit) {
    return ucrt_cosf(x);
  }
  return cosf_glibc(x);
}

}
