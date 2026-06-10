#pragma once
#include <cstdint>

// 2026-06-10: Shared PODs between the GCC-compiled host reference (Bullet) and the
// nvcc-compiled device side of cuda_math_test. Layout-compatible with rscuda types.

namespace mathtest {

struct V3 { float x, y, z; };
struct Q4 { float x, y, z, w; };
struct M3 { V3 r[3]; };

// One sample's inputs (generous superset; each op uses what it needs).
// ma/mb are derived host-side (orthonormal bases from qa/qb); qb is replaced by its
// host-normalized form so both sides consume identical bits.
struct Input {
	V3 a, b;
	V3 w;        // angular velocity (mixed magnitudes incl. Taylor + cap branches)
	Q4 qa, qb;
	M3 ma, mb;
	float dt;
};

// One sample's outputs for every tested op
struct Output {
	float dot3;
	V3 cross;
	Q4 quatMul;
	float quatDot;
	Q4 quatNorm;
	M3 mat3Mul;     // ma * basis(qa)
	Q4 getRot;      // getRotation(basis(qa) built by host setRotation)
	M3 setRot;      // setRotation(qb normalized)
	M3 inertia;     // updateInertiaTensor(basis(qa), a)
	M3 integBasis;  // integrateTransform basis out
	V3 integPos;    // integrateTransform pos out
};

// splitmix64 - identical generator on both sides
struct Gen {
	uint64_t s;
	explicit Gen(uint64_t seed) : s(seed) {}
	uint64_t Next() {
		s += 0x9E3779B97F4A7C15ull;
		uint64_t z = s;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
		return z ^ (z >> 31);
	}
	// float with bounded exponent, random sign/mantissa; occasionally exact zero
	float NextFloat() {
		uint64_t r = Next();
		if ((r & 0xFF) == 0) return 0.0f;
		uint32_t mant = (uint32_t)(r >> 8) & 0x7FFFFF;
		uint32_t exp = 112 + (uint32_t)((r >> 32) % 24); // ~1e-5 .. ~3e4 magnitude
		uint32_t sign = (uint32_t)(r >> 63);
		uint32_t bits = (sign << 31) | (exp << 23) | mant;
		float f;
		__builtin_memcpy(&f, &bits, 4);
		return f;
	}
	// small float for angular velocities (covers Taylor + normal + cap branches)
	float NextAngVel() {
		uint64_t r = Next();
		switch (r % 4) {
		case 0: return NextFloat() * 1e-6f;      // Taylor branch territory
		case 1: return (float)((int64_t)(Next() % 2049) - 1024) / 170.0f; // ~[-6,6] typical RL
		case 2: return (float)((int64_t)(Next() % 2049) - 1024) / 4.0f;   // up to ~256, cap branch
		default: return NextFloat() * 1e-2f;
		}
	}
};

inline void FillInput(Gen& g, Input& in) {
	in.a = { g.NextFloat(), g.NextFloat(), g.NextFloat() };
	in.b = { g.NextFloat(), g.NextFloat(), g.NextFloat() };
	in.w = { g.NextAngVel(), g.NextAngVel(), g.NextAngVel() };
	in.qa = { g.NextFloat(), g.NextFloat(), g.NextFloat(), g.NextFloat() };
	in.qb = { g.NextFloat(), g.NextFloat(), g.NextFloat(), g.NextFloat() };
	in.ma = {};
	in.mb = {};
	in.dt = 1.0f / 120.0f;
}

} // namespace mathtest
