#pragma once
#include <cstdint>

// 2026-06-10: Host-side capture of this CPU's RSQRTSS estimate behavior.
// btVector3::normalize() uses _mm_rsqrt_ss + one Newton-Raphson step; the estimate's
// result table is CPU-vendor-specific, so the GPU must EMULATE this CPU's instruction.
// Hypothesis (verified by RsVerifyRsqrtEmulation + the cuda_math_test normalize op):
// the result mantissa depends only on (input exponent parity, top 12 mantissa bits),
// and the result exponent scales arithmetically with the input exponent.
namespace rscuda {

// table[parity][mantissaTop12] = full result bits of rsqrtss at reference exponents
// 127 (parity 0) and 128 (parity 1)
void RsBuildRsqrtTable(uint32_t table[2][4096]);

// Verify the emulation scheme reproduces _mm_rsqrt_ss for `samples` random normal
// floats across the full exponent range; returns number of mismatches (0 = hypothesis
// holds on this CPU).
uint64_t RsVerifyRsqrtEmulation(const uint32_t table[2][4096], uint64_t samples);

} // namespace rscuda
