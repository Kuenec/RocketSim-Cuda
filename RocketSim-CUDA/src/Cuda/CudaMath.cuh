#pragma once
// 2026-06-10: Device math that reproduces Bullet's LinearMath BIT-EXACTLY as compiled
// on this project's CPU reference (GCC -O3 -ffp-contract=off, BT_USE_SSE_IN_API on
// x86-64). Bullet's SSE kernels fix specific summation ASSOCIATIONS that differ from
// the scalar fallbacks; every function here encodes the SSE tree, with the source
// location of the tree it mirrors. Kernels compile with -fmad=false so no contraction
// can change these trees. Verified op-by-op against the host by tools/cuda_math_test.
//
// Conventions: all types are plain float scalars (no float4 vector ops needed - IEEE
// scalar ops are lane-equivalent to SSE element ops). Matrices are row-major like
// btMatrix3x3 (m_el[row]).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rscuda {

#define RSC_HD __host__ __device__ __forceinline__

constexpr float CU_SIMD_EPSILON = 1.19209290e-07f; // FLT_EPSILON, matches btScalar.h
constexpr float CU_ANGULAR_MOTION_THRESHOLD = 0.7853981852531433f; // 0.5f*SIMD_HALF_PI as float

struct CuVec3 { float x, y, z; };
struct CuQuat { float x, y, z, w; };
struct CuMat3 { CuVec3 r[3]; }; // r[i] = btMatrix3x3::m_el[i] (row)

RSC_HD CuVec3 CuV(float x, float y, float z) { return { x, y, z }; }

// element-wise ops (SSE _mm_add_ps etc. are lane-exact to scalar IEEE ops)
RSC_HD CuVec3 operator+(CuVec3 a, CuVec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
RSC_HD CuVec3 operator-(CuVec3 a, CuVec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
RSC_HD CuVec3 operator*(CuVec3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
RSC_HD CuVec3 operator*(CuVec3 a, CuVec3 b) { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
RSC_HD CuVec3 CuNeg(CuVec3 a) { return { -a.x, -a.y, -a.z }; }

// btVector3::dot SSE tree (btVector3.h:230): (x*x' + y*y') + z*z'
RSC_HD float CuDot3(CuVec3 a, CuVec3 b) {
	return (a.x * b.x + a.y * b.y) + a.z * b.z;
}
RSC_HD float CuLength2(CuVec3 a) { return CuDot3(a, a); }

// btVector3::safeNormalize (btVector3.h:287): plain sqrt + operator/= which is
// "*this *= 1.0/s" (reciprocal then element multiply) - NO rsqrt estimate here
RSC_HD CuVec3 CuSafeNormalize3(CuVec3 v) {
	float l2 = CuLength2(v);
	if (l2 >= CU_SIMD_EPSILON * CU_SIMD_EPSILON) {
		float inv = 1.0f / sqrtf(l2);
		return v * inv;
	}
	return { 1, 0, 0 };
}

// btVector3::cross SSE tree (btVector3.h:384) reduces to the classic element formulas
RSC_HD CuVec3 CuCross(CuVec3 a, CuVec3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

// btQuaternion::dot SSE tree (btQuaternion.h:337): (x*x' + z*z') + (y*y' + w*w')
RSC_HD float CuQuatDot4(CuQuat a, CuQuat b) {
	return (a.x * b.x + a.z * b.z) + (a.y * b.y + a.w * b.w);
}

// btQuaternion::operator*= SSE tree (btQuaternion.h:254): per lane (A0 - B1) + (A1 + A2)
RSC_HD CuQuat CuQuatMul(CuQuat a, CuQuat b) {
	return {
		(a.w * b.x - a.z * b.y) + (a.x * b.w + a.y * b.z),
		(a.w * b.y - a.x * b.z) + (a.y * b.w + a.z * b.x),
		(a.w * b.z - a.y * b.x) + (a.z * b.w + a.x * b.y),
		(a.w * b.w - a.z * b.z) - (a.x * b.x + a.y * b.y)
	};
}

// btQuaternion::normalize SSE tree (btQuaternion.h:385): dot4 then sqrt then 1/len
// (real SQRTSS + DIVSS - correctly rounded, NOT the rsqrt estimate path)
RSC_HD CuQuat CuQuatNormalize(CuQuat q) {
	float d = CuQuatDot4(q, q);
	float len = sqrtf(d);
	float inv = 1.0f / len;
	return { q.x * inv, q.y * inv, q.z * inv, q.w * inv };
}

// btQuaternion::safeNormalize (btQuaternion.h:374)
RSC_HD CuQuat CuQuatSafeNormalize(CuQuat q) {
	float l2 = CuQuatDot4(q, q);
	if (l2 > CU_SIMD_EPSILON)
		return CuQuatNormalize(q);
	return q;
}

// btMatrix3x3 operator* SSE tree (btMatrix3x3.h:1265):
// res[i][j] = (m1[i].x*m2[0][j] + m1[i].y*m2[1][j]) + m1[i].z*m2[2][j]
RSC_HD CuMat3 CuMat3Mul(const CuMat3& m1, const CuMat3& m2) {
	CuMat3 r;
	for (int i = 0; i < 3; i++) {
		r.r[i].x = (m1.r[i].x * m2.r[0].x + m1.r[i].y * m2.r[1].x) + m1.r[i].z * m2.r[2].x;
		r.r[i].y = (m1.r[i].x * m2.r[0].y + m1.r[i].y * m2.r[1].y) + m1.r[i].z * m2.r[2].y;
		r.r[i].z = (m1.r[i].x * m2.r[0].z + m1.r[i].y * m2.r[1].z) + m1.r[i].z * m2.r[2].z;
	}
	return r;
}

RSC_HD CuMat3 CuMat3Transpose(const CuMat3& m) {
	return { { { m.r[0].x, m.r[1].x, m.r[2].x },
	           { m.r[0].y, m.r[1].y, m.r[2].y },
	           { m.r[0].z, m.r[1].z, m.r[2].z } } };
}

// btMatrix3x3::scaled - rows element-multiplied by the scale vector
RSC_HD CuMat3 CuMat3Scaled(const CuMat3& m, CuVec3 s) {
	return { { m.r[0] * s, m.r[1] * s, m.r[2] * s } };
}

// btMatrix3x3::getRotation (btMatrix3x3.h:421) - branchy scalar code, SSE only in the
// final q *= s element multiply
RSC_HD CuQuat CuMat3GetRotation(const CuMat3& m) {
	float trace = m.r[0].x + m.r[1].y + m.r[2].z;
	float temp[4];
	float x;
	if (trace > 0.0f) {
		x = trace + 1.0f;
		temp[0] = m.r[2].y - m.r[1].z;
		temp[1] = m.r[0].z - m.r[2].x;
		temp[2] = m.r[1].x - m.r[0].y;
		temp[3] = x;
	} else {
		const float* el[3] = { &m.r[0].x, &m.r[1].x, &m.r[2].x };
		int i, j, k;
		if (m.r[0].x < m.r[1].y) {
			if (m.r[1].y < m.r[2].z) { i = 2; j = 0; k = 1; }
			else { i = 1; j = 2; k = 0; }
		} else {
			if (m.r[0].x < m.r[2].z) { i = 2; j = 0; k = 1; }
			else { i = 0; j = 1; k = 2; }
		}
		x = el[i][i] - el[j][j] - el[k][k] + 1.0f;
		temp[3] = el[k][j] - el[j][k];
		temp[j] = el[j][i] + el[i][j];
		temp[k] = el[k][i] + el[i][k];
		temp[i] = x;
	}
	float s = sqrtf(x);
	s = 0.5f / s;
	return { temp[0] * s, temp[1] * s, temp[2] * s, temp[3] * s };
}

// btMatrix3x3::setRotation SSE tree (btMatrix3x3.h:216), derived lane by lane:
//   row0 = ( 1 + s*(-(y*y) + -(z*z)),  s*(x*y + -(w*z)),  s*(z*x + y*w) )
//   row1 = ( s*(x*y + w*z),  1 + s*(-(x*x) + -(z*z)),  s*(y*z + -(w*x)) )
//   row2 = ( s*(z*x + -(w*y)),  s*(y*z + w*x),  1 + s*(-(x*x) + -(y*y)) )
// with s = 2/dot4(q,q). The +1 happens AFTER the multiply by s (V1*vs + v1000).
RSC_HD CuMat3 CuMat3SetRotation(CuQuat q) {
	float d = CuQuatDot4(q, q);
	float s = 2.0f / d;
	// The SSE code's final step adds v1000/v0100/v0010 to whole ROWS, so off-diagonal
	// lanes get a literal "+ 0.0f" - which canonicalizes -0 to +0. Keep those adds.
	CuMat3 m;
	m.r[0].x = (-(q.y * q.y) + -(q.z * q.z)) * s + 1.0f;
	m.r[0].y = (q.x * q.y + -(q.w * q.z)) * s + 0.0f;
	m.r[0].z = (q.z * q.x + q.y * q.w) * s + 0.0f;
	m.r[1].x = (q.x * q.y + q.w * q.z) * s + 0.0f;
	m.r[1].y = (-(q.x * q.x) + -(q.z * q.z)) * s + 1.0f;
	m.r[1].z = (q.y * q.z + -(q.w * q.x)) * s + 0.0f;
	m.r[2].x = (q.z * q.x + -(q.w * q.y)) * s + 0.0f;
	m.r[2].y = (q.y * q.z + q.w * q.x) * s + 0.0f;
	m.r[2].z = (-(q.x * q.x) + -(q.y * q.y)) * s + 1.0f;
	return m;
}

// btRigidBody::updateInertiaTensor (btRigidBody.cpp:253):
// invInertiaWorld = basis.scaled(invInertiaLocal) * basis.transpose()
RSC_HD CuMat3 CuUpdateInertiaTensor(const CuMat3& basis, CuVec3 invInertiaLocal) {
	return CuMat3Mul(CuMat3Scaled(basis, invInertiaLocal), CuMat3Transpose(basis));
}

// ---- Device trig matching glibc sinf/cosf bit-exactly --------------------------------
// Base value: sin/cos computed in double, rounded once to float. That equals glibc for
// >99.9% of inputs; the rest (double-rounding cases, ~0.08% for sinf) are fixed by an
// EMPIRICAL CORRECTION TABLE measured on this machine by
//   cuda_math_test --emit-trig-table <path>
// over the entire domain integrateTransform can produce ([0, ANGULAR_MOTION_THRESHOLD]).
// Entries are (inputBits, glibcResultBits), sorted by inputBits; device does a binary
// search and substitutes on hit. Measured from the host libm itself, so it is correct
// regardless of glibc version, build flags, or CPU vendor. cuda_math_test verifies the
// result exhaustively against the host.

struct TrigFixEntry { uint32_t in, out; };
struct TrigFixTab {
	const TrigFixEntry* sinTab; int sinN;
	const TrigFixEntry* cosTab; int cosN;
};

#ifdef __CUDACC__
static __device__ TrigFixTab g_trigFix = {}; // per-TU symbol; each executable binds its own

__device__ __forceinline__ bool CuTrigLookup(const TrigFixEntry* tab, int n, uint32_t key, float& out) {
	int lo = 0, hi = n - 1;
	while (lo <= hi) {
		int mid = (lo + hi) >> 1;
		uint32_t v = tab[mid].in;
		if (v == key) {
			uint32_t ob = tab[mid].out;
			memcpy(&out, &ob, 4);
			return true;
		}
		if (v < key) lo = mid + 1; else hi = mid - 1;
	}
	return false;
}

__device__ __forceinline__ float CuSinf(float x) {
	uint32_t key;
	memcpy(&key, &x, 4);
	float fixv;
	if (g_trigFix.sinTab && CuTrigLookup(g_trigFix.sinTab, g_trigFix.sinN, key, fixv))
		return fixv;
	return (float)sin((double)x);
}
__device__ __forceinline__ float CuCosf(float x) {
	uint32_t key;
	memcpy(&key, &x, 4);
	float fixv;
	if (g_trigFix.cosTab && CuTrigLookup(g_trigFix.cosTab, g_trigFix.cosN, key, fixv))
		return fixv;
	return (float)cos((double)x);
}

// Host-side: load a table file emitted by cuda_math_test and bind it to this TU's
// g_trigFix symbol. Returns false (with no side effects) if the file is absent.
inline bool CuLoadTrigFix(const char* path) {
	FILE* f = fopen(path, "rb");
	if (!f) return false;
	uint32_t magic = 0, sinN = 0, cosN = 0;
	if (fread(&magic, 4, 1, f) != 1 || magic != 0x54464958u /*'TFIX'*/ ||
		fread(&sinN, 4, 1, f) != 1 || fread(&cosN, 4, 1, f) != 1) {
		fclose(f);
		return false;
	}
	size_t bytes = (size_t)(sinN + cosN) * sizeof(TrigFixEntry);
	TrigFixEntry* hostBuf = (TrigFixEntry*)malloc(bytes);
	bool ok = fread(hostBuf, 1, bytes, f) == bytes;
	fclose(f);
	if (!ok) { free(hostBuf); return false; }

	TrigFixEntry* devBuf = nullptr;
	if (cudaMalloc(&devBuf, bytes) != cudaSuccess) { free(hostBuf); return false; }
	cudaMemcpy(devBuf, hostBuf, bytes, cudaMemcpyHostToDevice);
	free(hostBuf);
	TrigFixTab tab = { devBuf, (int)sinN, devBuf + sinN, (int)cosN };
	return cudaMemcpyToSymbol(g_trigFix, &tab, sizeof(tab)) == cudaSuccess;
}
#endif // __CUDACC__

#ifdef __CUDACC__
// ---- RSQRTSS emulation -----------------------------------------------------------
// btVector3::normalize() (btVector3.h:304) uses _mm_rsqrt_ss + one Newton-Raphson
// step. The estimate is CPU-vendor-specific, so the device emulates THIS CPU's
// instruction via a host-captured table (RsqrtHost.cpp): result mantissa depends only
// on (exponent parity, top 12 mantissa bits); exponent scales arithmetically.
// Hypothesis verified at init by RsVerifyRsqrtEmulation and continuously by
// cuda_math_test's rsqrt/normalize ops.
static __device__ __constant__ uint32_t g_rsqrtTab[2][4096];

__device__ __forceinline__ float CuRsqrtSS(float x) {
	uint32_t u;
	memcpy(&u, &x, 4);
	uint32_t e = (u >> 23) & 0xFF;
	uint32_t m = (u >> 11) & 0xFFF;
	uint32_t p = (e & 1) ^ 1;
	uint32_t refBits = g_rsqrtTab[p][m];
	int k = ((int)e - (int)(127 + p)) / 2;
	uint32_t refExp = (refBits >> 23) & 0xFF;
	uint32_t out = (refBits & 0x807FFFFFu) | ((uint32_t)((int)refExp - k) << 23);
	float r;
	memcpy(&r, &out, 4);
	return r;
}

// Host: upload the captured table into this TU's constant symbol
inline bool CuUploadRsqrtTable(const uint32_t table[2][4096]) {
	return cudaMemcpyToSymbol(g_rsqrtTab, table, sizeof(uint32_t) * 2 * 4096) == cudaSuccess;
}

// btVector3::normalize() SSE tree (btVector3.h:304): dot3, rsqrt estimate, one NR
// step (y0*(1.5 - d*0.5*y0*y0)), then multiply all lanes
__device__ __forceinline__ CuVec3 CuNormalize3(CuVec3 v) {
	float d = CuLength2(v);
	float y0 = CuRsqrtSS(d);
	float t = d * 0.5f;
	t = t * y0;
	t = t * y0;
	float z = 1.5f - t;
	float y = y0 * z;
	return v * y;
}

// btQuaternion::setRotation(axis, angle) (btQuaternion.h:108)
__device__ __forceinline__ CuQuat CuQuatAxisAngle(CuVec3 axis, float angle) {
	float d = sqrtf(CuLength2(axis)); // btVector3::length
	float s = CuSinf(angle * 0.5f) / d;
	return { axis.x * s, axis.y * s, axis.z * s, CuCosf(angle * 0.5f) };
}

// btTransformUtil::integrateTransform (btTransformUtil.h:37), exponential map branch.
// Argument trees encoded literally: sin arg is (0.5*fAngle)*dt, cos arg is (fAngle*dt)*0.5.
// Device-only: uses the trig fix table.
__device__ __forceinline__ void CuIntegrateTransform(
	const CuMat3& curBasis, CuVec3 curPos, CuVec3 linvel, CuVec3 angvel, float dt, bool noRot,
	CuMat3& outBasis, CuVec3& outPos) {

	outPos = curPos + linvel * dt;
	if (noRot) { // btTransformUtil::integrateTransformNoRot
		outBasis = curBasis;
		return;
	}

	CuVec3 axis;
	float fAngle2 = CuLength2(angvel);
	float fAngle = 0;
	if (fAngle2 > CU_SIMD_EPSILON)
		fAngle = sqrtf(fAngle2);

	if (fAngle * dt > CU_ANGULAR_MOTION_THRESHOLD)
		fAngle = CU_ANGULAR_MOTION_THRESHOLD / dt;

	if (fAngle < 0.001f) {
		// Taylor: angvel * (0.5*dt - dt^3 * 0.020833333333 * fAngle * fAngle), left-assoc
		axis = angvel * (0.5f * dt - (dt * dt * dt) * 0.020833333333f * fAngle * fAngle);
	} else {
		axis = angvel * (CuSinf(0.5f * fAngle * dt) / fAngle);
	}
	CuQuat dorn = { axis.x, axis.y, axis.z, CuCosf(fAngle * dt * 0.5f) };
	CuQuat orn0 = CuMat3GetRotation(curBasis);
	CuQuat predictedOrn = CuQuatMul(dorn, orn0);
	predictedOrn = CuQuatSafeNormalize(predictedOrn);

	if (CuQuatDot4(predictedOrn, predictedOrn) > CU_SIMD_EPSILON)
		outBasis = CuMat3SetRotation(predictedOrn);
	else
		outBasis = curBasis;
}
#endif // __CUDACC__

// btRigidBody::applyDamping (btRigidBody.cpp:153) with the damping factors
// btPow(1-damping, dt) precomputed on HOST with the host's powf (dt is constant per
// run, so the factor is one constant - no device pow needed for bit-exactness).
// RocketSim bodies never set m_additionalDamping, so that branch is dead.
RSC_HD void CuApplyDamping(CuVec3& linVel, CuVec3& angVel, float linFactor, float angFactor) {
	linVel = linVel * linFactor;
	angVel = angVel * angFactor;
}

} // namespace rscuda
