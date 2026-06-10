// 2026-06-10: Host reference half of cuda_math_test. Compiled by the SAME C++ compiler
// and flags as the work library, so these results ARE what the physics code computes.
// Kept out of the .cu so nvcc never parses Bullet's SSE headers.

#include "test_data.h"

#include "../../RocketSim-CUDA/libsrc/bullet3-3.24/LinearMath/btVector3.h"
#include "../../RocketSim-CUDA/libsrc/bullet3-3.24/LinearMath/btQuaternion.h"
#include "../../RocketSim-CUDA/libsrc/bullet3-3.24/LinearMath/btMatrix3x3.h"
#include "../../RocketSim-CUDA/libsrc/bullet3-3.24/LinearMath/btTransform.h"
#include "../../RocketSim-CUDA/libsrc/bullet3-3.24/LinearMath/btTransformUtil.h"

#include <cmath>
#include <cstring>
#include <xmmintrin.h>

namespace mathtest {

static btVector3 BV(const V3& v) { return btVector3(v.x, v.y, v.z); }
static V3 FV(const btVector3& v) { return { v.x(), v.y(), v.z() }; }
static btQuaternion BQ(const Q4& q) { return btQuaternion(q.x, q.y, q.z, q.w); }
static Q4 FQ(const btQuaternion& q) { return { q.x(), q.y(), q.z(), q.w() }; }
static M3 FM(const btMatrix3x3& m) {
	return { { FV(m[0]), FV(m[1]), FV(m[2]) } };
}
static btMatrix3x3 BM(const M3& m) {
	return btMatrix3x3(
		m.r[0].x, m.r[0].y, m.r[0].z,
		m.r[1].x, m.r[1].y, m.r[1].z,
		m.r[2].x, m.r[2].y, m.r[2].z);
}

// Build derived inputs: ma/mb orthonormal bases, qb normalized (in place)
void HostFillDerived(Input* inputs, size_t n) {
	for (size_t i = 0; i < n; i++) {
		Input& in = inputs[i];
		btQuaternion qa = BQ(in.qa), qb = BQ(in.qb);
		if (qa.length2() <= SIMD_EPSILON) qa = btQuaternion(0, 0, 0, 1);
		if (qb.length2() <= SIMD_EPSILON) qb = btQuaternion(0, 0, 0, 1);
		btQuaternion qan = qa; qan.normalize();
		qb.normalize();
		in.qb = FQ(qb);
		btMatrix3x3 ma, mb;
		ma.setRotation(qan);
		mb.setRotation(qb);
		in.ma = FM(ma);
		in.mb = FM(mb);
	}
}

void HostCompute(const Input* inputs, Output* outputs, size_t n) {
	for (size_t i = 0; i < n; i++) {
		const Input& in = inputs[i];
		Output& o = outputs[i];

		o.dot3 = BV(in.a).dot(BV(in.b));
		o.cross = FV(BV(in.a).cross(BV(in.b)));

		btQuaternion qm = BQ(in.qa);
		qm *= BQ(in.qb);
		o.quatMul = FQ(qm);

		o.quatDot = BQ(in.qa).dot(BQ(in.qb));

		btQuaternion qn = BQ(in.qa);
		qn.safeNormalize();
		o.quatNorm = FQ(qn);

		o.mat3Mul = FM(BM(in.ma) * BM(in.mb));

		btQuaternion gr;
		BM(in.ma).getRotation(gr);
		o.getRot = FQ(gr);

		btMatrix3x3 sr;
		sr.setRotation(BQ(in.qb));
		o.setRot = FM(sr);

		// updateInertiaTensor: basis.scaled(invInertiaLocal) * basis.transpose()
		o.inertia = FM(BM(in.ma).scaled(BV(in.a)) * BM(in.ma).transpose());

		btTransform cur(BM(in.ma), BV(in.a));
		btTransform pred;
		btTransformUtil::integrateTransform(cur, BV(in.b), BV(in.w), in.dt, pred);
		o.integBasis = FM(pred.getBasis());
		o.integPos = FV(pred.getOrigin());

		// raw RSQRTSS of a guaranteed-positive normal value
		float d = BV(in.a).dot(BV(in.a));
		o.rsqrt = (d > SIMD_EPSILON) ? _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(d))) : 0.0f;

		if (BV(in.a).length2() > SIMD_EPSILON) {
			btVector3 n = BV(in.a);
			n.normalize();
			o.norm3 = FV(n);
		} else {
			o.norm3 = { 0, 0, 0 };
		}

		o.safeNorm3 = FV(BV(in.a).safeNormalized());

		if (BV(in.a).length2() > SIMD_EPSILON) {
			o.axisAngle = FQ(btQuaternion(BV(in.a), in.ang));
		} else {
			o.axisAngle = { 0, 0, 0, 0 };
		}
	}
}

void HostTrig(const uint32_t* bits, float* sinOut, float* cosOut, size_t n) {
	for (size_t i = 0; i < n; i++) {
		float x;
		__builtin_memcpy(&x, &bits[i], 4);
		sinOut[i] = sinf(x);
		cosOut[i] = cosf(x);
	}
}

} // namespace mathtest
