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
	float ang;   // steering-like angle within trig table domain
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
	float rsqrt;    // raw RSQRTSS estimate of |dot3(a,a)|
	V3 norm3;       // btVector3::normalize (rsqrt-NR path), zero if input near zero
	V3 safeNorm3;   // btVector3::safeNormalized
	Q4 axisAngle;   // btQuaternion(axis=a, angle=ang)
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
	// Exact [-1, 1] in 1/1024 steps
	float NextSigned() { return (float)((int64_t)(Next() % 2049) - 1024) / 1024.0f; }
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
	in.ang = (float)((int64_t)(g.Next() % 2049) - 1024) / 1024.0f * 0.75f;
}

// ---- GJK/EPA fuzz (box vs triangle, the convexConvex path used for car-vs-mesh) ----
struct GjkCase {
	V3 boxHalfExtentsCtor;  // btBoxShape ctor arg (margin handling inside)
	M3 boxBasis;            // box world transform
	V3 boxOrigin;
	V3 triV[3];             // triangle verts (mesh local == world for statics)
	float triMargin;        // mesh collision margin
	// host-derived for device:
	V3 boxImplicitHalfExt;  // implicitShapeDimensions (support basis)
	float boxMargin;
	float maxDistSq;        // (marginA+marginB+manifold breaking threshold)^2
};
struct GjkOut {
	int hasContact;
	V3 normalOnB;
	V3 pointOnB;
	float distance;
};

inline void FillGjkCase(Gen& g, GjkCase& c) {
	// box roughly car-hitbox sized (BT units ~ 2.36 x 1.70 x 0.72 half extents 1.18...)
	c.boxHalfExtentsCtor = {
		0.4f + (float)(g.Next() % 1025) / 1024.0f * 1.5f,
		0.3f + (float)(g.Next() % 1025) / 1024.0f * 1.2f,
		0.15f + (float)(g.Next() % 1025) / 1024.0f * 0.6f };
	// triangle around origin, mesh-scale (1-20 BT edge lengths)
	float s = 0.5f + (float)(g.Next() % 1025) / 1024.0f * 10.0f;
	for (int i = 0; i < 3; i++)
		c.triV[i] = { g.NextSigned() * s, g.NextSigned() * s, g.NextSigned() * s };
	c.triMargin = 0.04f; // CONVEX_DISTANCE_MARGIN (mesh getMargin default)
	// box pose near the triangle: separated -> deeply penetrating
	c.boxOrigin = {
		(c.triV[0].x + c.triV[1].x + c.triV[2].x) / 3 + g.NextSigned() * 2.5f,
		(c.triV[0].y + c.triV[1].y + c.triV[2].y) / 3 + g.NextSigned() * 2.5f,
		(c.triV[0].z + c.triV[1].z + c.triV[2].z) / 3 + g.NextSigned() * 2.5f };
	c.boxBasis = {}; // host fills orthonormal basis from a quat
}

} // namespace mathtest
