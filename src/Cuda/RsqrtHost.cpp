#include "RsqrtHost.h"

#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <xmmintrin.h>

namespace rscuda {

static uint32_t RsqrtBits(uint32_t inBits) {
	float x;
	memcpy(&x, &inBits, 4);
	float r = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
	uint32_t out;
	memcpy(&out, &r, 4);
	return out;
}

void RsBuildRsqrtTable(uint32_t table[2][4096]) {
	for (uint32_t parity = 0; parity < 2; parity++) {
		uint32_t exp = 127 + parity;
		for (uint32_t m = 0; m < 4096; m++)
			table[parity][m] = RsqrtBits((exp << 23) | (m << 11));
	}
}

// Mirror of the device-side emulation in CudaMath.cuh (keep in sync)
static uint32_t EmulateBits(const uint32_t table[2][4096], uint32_t u) {
	uint32_t e = (u >> 23) & 0xFF;
	uint32_t m = (u >> 11) & 0xFFF;
	uint32_t p = (e & 1) ^ 1;
	uint32_t refBits = table[p][m];
	int k = ((int)e - (int)(127 + p)) / 2;
	uint32_t refExp = (refBits >> 23) & 0xFF;
	return (refBits & 0x807FFFFFu) | ((uint32_t)((int)refExp - k) << 23);
}

uint64_t RsVerifyRsqrtEmulation(const uint32_t table[2][4096], uint64_t samples) {
	uint64_t state = 0x12345678ABCDEFull, bad = 0;
	for (uint64_t i = 0; i < samples; i++) {
		// splitmix64
		state += 0x9E3779B97F4A7C15ull;
		uint64_t z = state;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
		z ^= z >> 31;
		// positive normal float, full exponent range minus extremes (results must stay normal)
		uint32_t exp = 31 + (uint32_t)(z % 193); // biased [31, 223]
		uint32_t mant = (uint32_t)(z >> 32) & 0x7FFFFF;
		uint32_t u = (exp << 23) | mant;
		if (RsqrtBits(u) != EmulateBits(table, u))
			bad++;
	}
	return bad;
}

} // namespace rscuda

#else
namespace rscuda {
void RsBuildRsqrtTable(uint32_t table[2][4096]) {
	// Non-x86 host: no RSQRTSS to capture. The gates themselves are defined by an x86
	// reference; building on another arch needs a recorded table file instead.
	for (int p = 0; p < 2; p++)
		for (int m = 0; m < 4096; m++)
			table[p][m] = 0;
}
uint64_t RsVerifyRsqrtEmulation(const uint32_t[2][4096], uint64_t) { return ~0ull; }
}
#endif
