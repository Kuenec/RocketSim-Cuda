#pragma once
// 2026-06-10: Device port of btVehicleRL (RocketSim's raycast vehicle) + the
// btContactConstraint helpers it uses. Sources: btVehicleRL.cpp, btContactConstraint.cpp,
// btRigidBody.h (applyImpulse/getVelocityInLocalPoint/computeImpulseDenominator),
// btJacobianEntry.h. All trees mirrored exactly; fuzz/compare validated.

#include "CudaMath.cuh"
#include "CudaRaycast.cuh"

#ifdef __CUDACC__

namespace rscuda {

// Minimal dynamic-body view for impulse math (chassis or ground object)
struct CuBodyDyn {
	CuMat3 basis;            // worldTransform basis
	CuVec3 origin;           // worldTransform origin (center of mass)
	CuVec3 linVel, angVel;
	CuMat3 invInertiaWorld;
	CuVec3 invInertiaDiagLocal;
	float invMass;           // 0 for statics
};

// btRigidBody::getVelocityInLocalPoint: linVel + angVel x rel_pos
__device__ __forceinline__ CuVec3 CuVelocityInLocalPoint(const CuBodyDyn& b, CuVec3 relPos) {
	return b.linVel + CuCross(b.angVel, relPos);
}

// btRigidBody::applyImpulse (factors are (1,1,1) for RocketSim bodies; keep the
// multiplies - x*1 is exact). Torque path: angVel += invInertiaWorld * (rel x (imp*linF))
__device__ __forceinline__ void CuApplyImpulse(CuBodyDyn& b, CuVec3 impulse, CuVec3 relPos) {
	if (b.invMass != 0.0f) {
		// applyCentralImpulse: linVel += impulse * linearFactor * invMass
		CuVec3 linF = { 1, 1, 1 };
		b.linVel = b.linVel + (impulse * linF) * b.invMass;
		// applyTorqueImpulse(rel x (impulse * linearFactor)):
		// angVel += invInertiaWorld * torque * angularFactor
		CuVec3 torque = CuCross(relPos, impulse * linF);
		CuVec3 angF = { 1, 1, 1 };
		b.angVel = b.angVel + CuMat3MulVec(b.invInertiaWorld, torque) * angF;
	}
}

// btRigidBody::computeImpulseDenominator: vec = (r0 x n) * invInertiaWorld) x r0;
// note (c0 * tensor) is VECTOR*MATRIX (tdot columns)
__device__ __forceinline__ float CuComputeImpulseDenominator(const CuBodyDyn& b, CuVec3 pos, CuVec3 normal) {
	CuVec3 r0 = pos - b.origin;
	CuVec3 c0 = CuCross(r0, normal);
	CuVec3 vec = CuCross(CuVecMulMat(c0, b.invInertiaWorld), r0);
	return b.invMass + CuDot3(normal, vec);
}

// resolveSingleCollision (btContactConstraint.cpp:60) with applyImpulses=false as the
// vehicle uses it for extraPushback. body2 is the static/dynamic ground object.
__device__ __forceinline__ float CuResolveSingleCollision(
	const CuBodyDyn& body1, const CuBodyDyn& body2, bool body2IsRigid,
	CuVec3 contactPositionWorld, CuVec3 contactNormalOnB,
	float erp, float timeStep, float distance) {

	CuVec3 normal = contactNormalOnB;
	CuVec3 rel_pos1 = contactPositionWorld - body1.origin;
	CuVec3 rel_pos2 = contactPositionWorld - body2.origin;

	CuVec3 vel1 = CuVelocityInLocalPoint(body1, rel_pos1);
	CuVec3 vel2 = body2IsRigid ? CuVelocityInLocalPoint(body2, rel_pos2) : CuVec3{ 0, 0, 0 };
	CuVec3 vel = vel1 - vel2;
	float rel_vel = CuDot3(normal, vel);

	float combinedRestitution = 0.0f;
	float restitution = combinedRestitution * -rel_vel;

	float positionalError = erp * -distance / timeStep;
	float velocityError = -(1.0f + restitution) * rel_vel;
	float denom0 = CuComputeImpulseDenominator(body1, contactPositionWorld, normal);
	float denom1 = body2IsRigid ? CuComputeImpulseDenominator(body2, contactPositionWorld, normal) : 0.0f;
	float relaxation = 1.0f;
	float jacDiagABInv = relaxation / (denom0 + denom1);

	float penetrationImpulse = positionalError * jacDiagABInv;
	float velocityImpulse = velocityError * jacDiagABInv;

	float normalImpulse = penetrationImpulse + velocityImpulse;
	normalImpulse = 0.0f > normalImpulse ? 0.0f : normalImpulse;
	return normalImpulse;
}

// resolveSingleBilateral (btContactConstraint.cpp:108). pos1==pos2 for the vehicle.
// world2A/B = basis transposes; statics contribute zero inertia/mass/velocity but the
// trees still run with their actual (zero) values.
__device__ __forceinline__ float CuResolveSingleBilateral(
	const CuBodyDyn& body1, CuVec3 pos1,
	const CuBodyDyn& body2, CuVec3 pos2,
	CuVec3 normal) {

	float normalLenSqr = CuLength2(normal);
	if (normalLenSqr > 1.1f)
		return 0.0f;

	CuVec3 rel_pos1 = pos1 - body1.origin;
	CuVec3 rel_pos2 = pos2 - body2.origin;

	CuVec3 vel1 = CuVelocityInLocalPoint(body1, rel_pos1);
	CuVec3 vel2 = CuVelocityInLocalPoint(body2, rel_pos2);
	CuVec3 vel = vel1 - vel2;

	// btJacobianEntry(world2A, world2B, rel1, rel2, axis, invInA, mInvA, invInB, mInvB)
	CuMat3 world2A = CuMat3Transpose(body1.basis);
	CuMat3 world2B = CuMat3Transpose(body2.basis);
	CuVec3 aJ = CuMat3MulVec(world2A, CuCross(rel_pos1, normal));
	CuVec3 bJ = CuMat3MulVec(world2B, CuCross(rel_pos2, CuNeg(normal)));
	CuVec3 MinvJt0 = body1.invInertiaDiagLocal * aJ;
	CuVec3 MinvJt1 = body2.invInertiaDiagLocal * bJ;
	float Adiag = body1.invMass + CuDot3(MinvJt0, aJ) + body2.invMass + CuDot3(MinvJt1, bJ);

	float jacDiagABInv = 1.0f / Adiag;

	float rel_vel = CuDot3(normal, vel);

	float contactDamping = 0.2f;
	float velocityImpulse = -contactDamping * rel_vel * jacDiagABInv;
	return velocityImpulse;
}

// ---------------- LinearPieceCurve (Math.cpp:5) ----------------
__device__ __forceinline__ float CuCurveOutput(const GpuCurve& c, float input, float defaultOutput = 1.0f) {
	if (c.n == 0)
		return defaultOutput;
	if (input <= c.xs[0])
		return c.ys[0];
	for (int i = 1; i < c.n; i++) {
		if (c.xs[i] > input) {
			float rangeBetween = c.xs[i] - c.xs[i - 1];
			float valDiffBetween = c.ys[i] - c.ys[i - 1];
			float linearInterpFactor = (input - c.xs[i - 1]) / rangeBetween;
			return c.ys[i - 1] + valDiffBetween * linearInterpFactor;
		}
	}
	return c.ys[c.n - 1];
}

// btMatrix3x3::getColumn(i)
__device__ __forceinline__ CuVec3 CuGetColumn(const CuMat3& m, int i) {
	const float* r0 = &m.r[0].x;
	const float* r1 = &m.r[1].x;
	const float* r2 = &m.r[2].x;
	return { r0[i], r1[i], r2[i] };
}

// btMatrix3x3::inverse (btMatrix3x3.h:1093): cofac trees + det + scale
__device__ __forceinline__ CuMat3 CuMat3Inverse(const CuMat3& m) {
	const float* e[3] = { &m.r[0].x, &m.r[1].x, &m.r[2].x };
	auto cofac = [&](int r1, int c1, int r2, int c2) {
		return e[r1][c1] * e[r2][c2] - e[r1][c2] * e[r2][c1];
	};
	CuVec3 co = { cofac(1, 1, 2, 2), cofac(1, 2, 2, 0), cofac(1, 0, 2, 1) };
	float det = CuDot3(m.r[0], co);
	float s = 1.0f / det;
	CuMat3 r;
	r.r[0] = { co.x * s, cofac(0, 2, 2, 1) * s, cofac(0, 1, 1, 2) * s };
	r.r[1] = { co.y * s, cofac(0, 0, 2, 2) * s, cofac(0, 2, 1, 0) * s };
	r.r[2] = { co.z * s, cofac(0, 1, 2, 0) * s, cofac(0, 0, 1, 1) * s };
	return r;
}

// ---------------- btVehicleRL ----------------

// Per-wheel intra-tick scratch (mirrors btWheelInfoRL transient fields)
struct CuWheelScratch {
	CuMat3 worldBasis;          // m_worldTransform basis (friction reads columns)
	CuVec3 hardPointWS, wheelDirWS, axleWS;
	uint8_t isInContact, isInContactWithWorld;
	int groundIdx;              // -2 none, -1 static world, >=0 dynamic body index
	CuVec3 contactPointWS, contactNormalWS;
	float suspensionLengthNew;
	float suspensionRelativeVelocity;
	float clippedInvContactDotSuspension;
	float extraPushback;
	CuVec3 impulse;
	float dbgWheelTraceLen, dbgPushThresh, dbgResolveRaw;
};

// updateWheelTransformsWS (btVehicleRL.cpp:110)
__device__ __forceinline__ void CuUpdateWheelTransformsWS(
	const CuBodyDyn& chassis, const GpuWheelConsts& wc, const GpuCarConsts& cc, CuWheelScratch& w) {
	w.isInContact = 0;
	w.isInContactWithWorld = 0;
	CuVec3 conn = { wc.connectionPointCS.x, wc.connectionPointCS.y, wc.connectionPointCS.z };
	w.hardPointWS = CuTransformPoint(chassis.basis, chassis.origin, conn);
	w.wheelDirWS = CuMat3MulVec(chassis.basis, { cc.wheelDirectionCS.x, cc.wheelDirectionCS.y, cc.wheelDirectionCS.z });
	w.axleWS = CuMat3MulVec(chassis.basis, { cc.wheelAxleCS.x, cc.wheelAxleCS.y, cc.wheelAxleCS.z });
}

// updateWheelTransform (btVehicleRL.cpp:66). Car::_BulletSetup calls
// setCoordinateSystem(1, 2, 0): indexRight=1, indexUp=2, indexForward=0
// ("X forward, Y right, Z up")
__device__ __forceinline__ void CuUpdateWheelTransform(
	const CuBodyDyn& chassis, const GpuWheelConsts& wc, const GpuCarConsts& cc,
	float steerAngle, float prevSuspensionLength, CuWheelScratch& w) {
	CuUpdateWheelTransformsWS(chassis, wc, cc, w);
	CuVec3 up = CuNeg(w.wheelDirWS);
	CuVec3 right = w.axleWS;
	CuVec3 fwd = CuCross(up, right);
	fwd = CuNormalize3(fwd); // normalize() - rsqrt path

	CuQuat steeringOrn = CuQuatAxisAngle(up, steerAngle);
	CuMat3 steeringMat = CuMat3SetRotation(steeringOrn);

	CuMat3 basis2;
	// columns: [forward axis 0] = fwd, [right axis 1] = -right, [up axis 2] = up
	basis2.r[0] = { fwd.x, -right.x, up.x };
	basis2.r[1] = { fwd.y, -right.y, up.y };
	basis2.r[2] = { fwd.z, -right.z, up.z };
	w.worldBasis = CuMat3Mul(steeringMat, basis2);
	// origin = hardPoint + wheelDir * prevSuspensionLength (physics-dead, basis matters)
	(void)prevSuspensionLength;
}

// btVehicleRL::rayCast (btVehicleRL.cpp:120). Returns updated wheel scratch; needs the
// solver timestep actually stored in solverInfo at this point (1/60 on the very first
// tick before any solve, 1/120 after).
__device__ __forceinline__ void CuWheelRayCast(
	const CuBodyDyn& chassis, int selfDynIdx,
	const GpuWheelConsts& wc, const GpuCarConsts& cc,
	const GpuWorldGeo& geo, const GpuGridParams& grid,
	const GpuDynBody* dyn, int numDyn,
	const CuBodyDyn* dynBodies, // impulse views matching dyn[] indices
	float solverTimeStep,
	CuWheelScratch& w) {

	CuUpdateWheelTransformsWS(chassis, wc, cc, w);

	float suspensionTravel = wc.maxSuspensionTravelCm / 100.0f;
	float realRayLength = wc.suspensionRestLength + suspensionTravel + wc.wheelRadius - cc.suspensionSubtraction;

	CuVec3 source = w.hardPointWS;
	CuVec3 target = source + (w.wheelDirWS * realRayLength);
	w.contactPointWS = target;
	w.groundIdx = -2;

	// btDefaultVehicleRaycaster::castRay: ClosestRayResultCallback(ignore=chassis),
	// group |= addedFilterMask, then the hasContactResponse acceptance rule
	GpuRayHit hit;
	CuRayTestWorld(geo, grid, dyn, numDyn, selfDynIdx, source, target, cc.rayCastGroup, CU_FILTER_ALL, hit);

	bool accepted = false;
	CuVec3 hitNormal = { 0, 0, 0 };
	CuVec3 hitPoint = { 0, 0, 0 };
	int groundIdx = -2;
	if (hit.hitKind >= 0) {
		// all our hit objects are btRigidBody; contact response check:
		bool contactResponse = hit.hitKind == 0 ? true : (dyn[hit.hitKind - 1].hasContactResponse != 0);
		if (contactResponse) {
			accepted = true;
			// ClosestRayResultCallback: hitPoint = setInterpolate3(from, to, fraction)
			hitPoint = CuInterpolate3(source, target, hit.fraction);
			hitNormal = hit.normalWorld;
			hitNormal = CuNormalize3(hitNormal); // castRay: m_hitNormalInWorld.normalize()
			groundIdx = hit.hitKind == 0 ? -1 : (hit.hitKind - 1);
		}
	}

	if (accepted) {
		w.contactPointWS = hitPoint;
		float depth = realRayLength * hit.fraction;
		(void)depth;
		w.contactNormalWS = hitNormal;
		w.isInContact = 1;
		w.isInContactWithWorld = (groundIdx == -1);
		w.groundIdx = groundIdx;

		// getUpVector() = chassis basis column 2
		CuVec3 upVec = CuGetColumn(chassis.basis, 2);
		float wheelTraceLen = CuDot3(w.hardPointWS - w.contactPointWS, upVec);
		w.suspensionLengthNew = wheelTraceLen - wc.wheelRadius;

		// std::clamp(v, lo, hi)
		float lo = wc.suspensionRestLength - suspensionTravel;
		float hi = wc.suspensionRestLength + suspensionTravel;
		float v = w.suspensionLengthNew;
		v = v < lo ? lo : v;
		v = v > hi ? hi : v;
		w.suspensionLengthNew = v;

		float denominator = CuDot3(w.contactNormalWS, upVec);

		CuVec3 relpos = w.contactPointWS - chassis.origin;
		CuVec3 velAtContactPoint = CuVelocityInLocalPoint(chassis, relpos);
		float projVel = CuDot3(w.contactNormalWS, velAtContactPoint);

		if (denominator > 0.1f) {
			float inv = 1.0f / denominator;
			w.suspensionRelativeVelocity = projVel * inv;
			w.clippedInvContactDotSuspension = inv;
		} else {
			w.suspensionRelativeVelocity = 0.0f;
			w.clippedInvContactDotSuspension = 10.0f;
		}

		// NOTE: m_extraPushback is NOT reset here - it persists from previous ticks
		// unless this branch overwrites it (w.extraPushback was pre-seeded from the
		// persistent wheel state by the caller)
		w.dbgWheelTraceLen = wheelTraceLen;
		w.dbgPushThresh = 0;
		w.dbgResolveRaw = 0;
		if (groundIdx == -1) { // static ground
			float rayPushbackThresh = (wc.suspensionRestLength + wc.wheelRadius) - cc.suspensionSubtraction;
			w.dbgPushThresh = rayPushbackThresh;
			if (wheelTraceLen < rayPushbackThresh) {
				float wheelTraceDistDelta = wheelTraceLen - rayPushbackThresh;
				CuBodyDyn staticBody = {};
				staticBody.basis = CuIdentity3();
				float collisionResult = CuResolveSingleCollision(
					chassis, staticBody, false,
					hitPoint, hitNormal, cc.erp, solverTimeStep, wheelTraceDistDelta);
				w.dbgResolveRaw = collisionResult;
				w.extraPushback = collisionResult / 4.0f; // getNumWheels()
			}
		}
	} else {
		w.suspensionLengthNew = wc.suspensionRestLength + suspensionTravel;
		w.suspensionRelativeVelocity = 0.0f;
		w.contactNormalWS = CuNeg(w.wheelDirWS);
		w.clippedInvContactDotSuspension = 1.0f;
		w.extraPushback = 0.0f;
	}
}

// btVehicleRL::calcFrictionImpulses (btVehicleRL.cpp:306); reads PREVIOUS-tick wheel
// persists (engineForce/brake/lat/longFriction) and THIS tick's raycast scratch
__device__ __forceinline__ void CuCalcFrictionImpulse(
	const CuBodyDyn& chassis, const GpuCarConsts& cc,
	const GpuWheelState& persist, const CuBodyDyn* dynBodies, CuWheelScratch& w) {

	if (w.groundIdx == -2) {
		w.impulse = { 0, 0, 0 };
		return;
	}

	CuBodyDyn ground = {};
	ground.basis = CuIdentity3();
	if (w.groundIdx >= 0)
		ground = dynBodies[w.groundIdx];

	float frictionScale = cc.frictionScale; // chassis mass / 3 (host-evaluated)

	// axle = wheel world basis column 1 (m_indexRightAxis after setCoordinateSystem)
	CuVec3 axleDir = CuGetColumn(w.worldBasis, 1);
	CuVec3 surfNormalWS = w.contactNormalWS;
	float proj = CuDot3(axleDir, surfNormalWS);
	axleDir = axleDir - surfNormalWS * proj;
	axleDir = CuSafeNormalize3(axleDir);

	CuVec3 forwardDir = CuSafeNormalize3(CuCross(surfNormalWS, axleDir));

	float sideImpulse = CuResolveSingleBilateral(
		chassis, w.contactPointWS, ground, w.contactPointWS, axleDir);

	float rollingFriction;
	if (persist.engineForce == 0.0f) {
		if (persist.brake != 0.0f) {
			CuVec3 contactPoint = w.contactPointWS;
			CuVec3 carRelContactPoint = contactPoint - chassis.origin;
			// NOTE: bullet quirk preserved - ground body velocity sampled at the
			// CAR-relative contact point
			CuVec3 v1 = CuVelocityInLocalPoint(chassis, carRelContactPoint);
			CuVec3 v2 = CuVelocityInLocalPoint(ground, carRelContactPoint);
			CuVec3 contactVel = v1 - v2;
			float relVel = CuDot3(contactVel, forwardDir);
			// timeStep > 1/80 branch is dead at 120Hz
			constexpr float ROLLING_FRICTION_SCALE_MAGIC = 113.73963f;
			float rf = -relVel * ROLLING_FRICTION_SCALE_MAGIC;
			rf = rf < -persist.brake ? -persist.brake : rf;
			rf = rf > persist.brake ? persist.brake : rf;
			rollingFriction = rf;
		} else {
			rollingFriction = 0.0f;
		}
	} else {
		rollingFriction = -persist.engineForce / frictionScale;
	}

	// (fwd * rolling * longFric) + (axle * side * latFric), then * frictionScale
	CuVec3 totalFrictionForce =
		((forwardDir * rollingFriction) * persist.longFriction) +
		((axleDir * sideImpulse) * persist.latFriction);
	w.impulse = totalFrictionForce * frictionScale;
}

// ---------------- Car pre-tick game logic (Car.cpp) ----------------
// RocketSim Vec trees differ from btVector3: LengthSq = x*x+y*y+z*z+w*w (left-assoc,
// w=0 adds +0.0 - value-identical), operator/(float) = DIRECT per-element division.
__device__ __forceinline__ CuVec3 CuVecNormalizedRS(CuVec3 v) {
	float lengthSq = v.x * v.x + v.y * v.y + v.z * v.z; // + _w*_w with _w=0
	float length = lengthSq > 0 ? sqrtf(lengthSq) : 0.0f; // Vec::Length
	if (length > CU_SIMD_EPSILON * CU_SIMD_EPSILON)
		return { v.x / length, v.y / length, v.z / length };
	return { 0, 0, 0 };
}

__device__ __forceinline__ float CuSgn(float v) { return (float)((v > 0) - (v < 0)); }

// glibc atan2f/asinf stand-ins for the autoflip path: double-rounded; mismatch rate
// vs host glibc measured by cuda_math_test (replace with exact ports if nonzero).
__device__ __forceinline__ float CuAtan2f(float y, float x) { return (float)atan2((double)y, (double)x); }
__device__ __forceinline__ float CuAsinf(float x) { return (float)asin((double)x); }

struct CuCarTickIO {
	// chassis body (velocities mutated by impulses; forces accumulated)
	CuBodyDyn body;
	CuVec3 totalForce, totalTorque;  // btRigidBody accumulators
	GpuCarGameState st;              // game state (CarState mirror)
	GpuWheelState wheels[4];         // cross-tick wheel persists
	GpuCarControls controls;
	float solverTimeStep;            // solverInfo.m_timeStep at ray time
};

__device__ __forceinline__ void CuApplyCentralForce(CuCarTickIO& io, CuVec3 force) {
	CuVec3 linF = { 1, 1, 1 };
	io.totalForce = io.totalForce + force * linF; // m_totalForce += force*m_linearFactor
}
__device__ __forceinline__ void CuApplyTorqueAccum(CuCarTickIO& io, CuVec3 torque) {
	CuVec3 angF = { 1, 1, 1 };
	io.totalTorque = io.totalTorque + torque * angF;
}
__device__ __forceinline__ void CuApplyCentralImpulseBody(CuCarTickIO& io, CuVec3 impulse) {
	CuVec3 linF = { 1, 1, 1 };
	io.body.linVel = io.body.linVel + (impulse * linF) * io.body.invMass;
}

// Full Car::_PreTickUpdate (Car.cpp:71) minus the demo branch (isDemoed early-outs are
// honored; demo RESPAWN is excluded by DemoMode::DISABLED in the gates).
__device__ __forceinline__ void CuCarPreTick(
	CuCarTickIO& io, const GpuCarConsts& cc,
	const GpuWorldGeo& geo, const GpuGridParams& grid,
	const GpuDynBody* dyn, int numDyn, int selfDynIdx,
	const CuBodyDyn* dynBodies,
	float tickTime,
	CuWheelScratch* W /*[4] out, for capture compare*/,
	float* dbgForwardSpeedUU = nullptr) {

	// controls.ClampFix()
	io.controls.throttle = io.controls.throttle < -1 ? -1 : (io.controls.throttle > 1 ? 1 : io.controls.throttle);
	io.controls.steer = io.controls.steer < -1 ? -1 : (io.controls.steer > 1 ? 1 : io.controls.steer);
	io.controls.pitch = io.controls.pitch < -1 ? -1 : (io.controls.pitch > 1 ? 1 : io.controls.pitch);
	io.controls.yaw = io.controls.yaw < -1 ? -1 : (io.controls.yaw > 1 ? 1 : io.controls.yaw);
	io.controls.roll = io.controls.roll < -1 ? -1 : (io.controls.roll > 1 ? 1 : io.controls.roll);

	if (io.st.isDemoed)
		return; // gates: demo respawn disabled; activation flags handled CPU-side later

	// ---- updateVehicleFirst ----
	for (int i = 0; i < 4; i++)
		CuUpdateWheelTransform(io.body, cc.wheels[i], cc, io.wheels[i].steerAngle,
			io.wheels[i].suspensionLength, W[i]);
	for (int i = 0; i < 4; i++) {
		W[i].extraPushback = io.wheels[i].extraPushback; // persistent unless overwritten
		CuWheelRayCast(io.body, selfDynIdx, cc.wheels[i], cc, geo, grid, dyn, numDyn,
			dynBodies, io.solverTimeStep, W[i]);
	}
	for (int i = 0; i < 4; i++)
		CuCalcFrictionImpulse(io.body, cc, io.wheels[i], dynBodies, W[i]);

	// persist wheel raycast state (m_raycastInfo.m_suspensionLength, m_extraPushback)
	for (int i = 0; i < 4; i++) {
		io.wheels[i].suspensionLength = W[i].suspensionLengthNew;
		io.wheels[i].extraPushback = W[i].extraPushback;
	}

	bool jumpPressed = io.controls.jump && !io.st.lastControls.jump;

	int numWheelsInContact = 0;
	for (int i = 0; i < 4; i++) {
		io.st.wheelsWithContact[i] = W[i].isInContact;
		numWheelsInContact += W[i].isInContact;
	}
	io.st.isOnGround = numWheelsInContact >= 3;

	// forwardSpeed_UU = (linVel dot basis column 0 [m_indexForwardAxis]) * BT_TO_UU
	float forwardSpeed_UU = CuDot3(io.body.linVel, CuGetColumn(io.body.basis, 0)) * cc.btToUU;
	if (dbgForwardSpeedUU)
		*dbgForwardSpeedUU = forwardSpeed_UU;

	// ---- _UpdateWheels ----
	{
		float absForwardSpeed_UU = fabsf(forwardSpeed_UU);

		bool wheelsHaveWorldContact = false;
		for (int i = 0; i < 4; i++)
			wheelsHaveWorldContact |= (W[i].isInContactWithWorld != 0);

		if (io.controls.handbrake) {
			io.st.handbrakeVal += cc.powerslideRiseRate * tickTime;
		} else {
			io.st.handbrakeVal -= cc.powerslideFallRate * tickTime;
		}
		io.st.handbrakeVal = fminf(fmaxf(io.st.handbrakeVal, 0.0f), 1.0f);

		float realThrottle = io.controls.throttle;
		float realBrake = 0;

		if (io.controls.boost && io.st.boost > 0)
			realThrottle = 1;

		{
			float driveSpeedScale = CuCurveOutput(cc.driveSpeedTorqueFactor, absForwardSpeed_UU);
			float engineThrottle = realThrottle;

			if (io.controls.handbrake) {
				// unchanged
			} else {
				float absThrottle = fabsf(realThrottle);
				if (absThrottle >= cc.throttleDeadzone) {
					if (absForwardSpeed_UU > cc.stoppingForwardVel && CuSgn(realThrottle) != CuSgn(forwardSpeed_UU)) {
						realBrake = 1;
						if (absForwardSpeed_UU > cc.brakingNoThrottleSpeedThresh)
							engineThrottle = 0;
					}
				} else {
					engineThrottle = 0;
					bool shouldFullStop = (absForwardSpeed_UU < cc.stoppingForwardVel);
					realBrake = shouldFullStop ? 1.0f : cc.coastingBrakeFactor;
				}
			}

			if (numWheelsInContact < 3)
				driveSpeedScale /= 4;

			float driveEngineForce = engineThrottle * cc.throttleTorqueBT * driveSpeedScale;
			float driveBrakeForce = realBrake * cc.brakeTorqueBT;
			for (int i = 0; i < 4; i++) {
				io.wheels[i].engineForce = driveEngineForce;
				io.wheels[i].brake = driveBrakeForce;
			}
		}

		{ // steering
			float steerAngle = CuCurveOutput(cc.steerAngleFromSpeed, absForwardSpeed_UU);
			if (io.st.handbrakeVal) {
				steerAngle += (CuCurveOutput(cc.powerslideSteerAngleFromSpeed, absForwardSpeed_UU) - steerAngle)
					* io.st.handbrakeVal;
			}
			steerAngle *= io.controls.steer;
			io.wheels[0].steerAngle = steerAngle;
			io.wheels[1].steerAngle = steerAngle;
		}

		{ // friction values
			for (int i = 0; i < 4; i++) {
				if (W[i].groundIdx == -2)
					continue;
				CuVec3 vel = io.body.linVel;
				CuVec3 angularVel = io.body.angVel;

				CuVec3 latDir = CuGetColumn(W[i].worldBasis, 1);
				CuVec3 longDir = CuCross(latDir, W[i].contactNormalWS);

				float frictionCurveInput = 0;

				CuVec3 wheelDelta = W[i].hardPointWS - io.body.origin;
				CuVec3 crossVec = (CuCross(angularVel, wheelDelta) + vel) * cc.btToUU;

				float baseFriction = fabsf(CuDot3(crossVec, latDir));
				if (baseFriction > 5)
					frictionCurveInput = baseFriction / (fabsf(CuDot3(crossVec, longDir)) + baseFriction);

				float latFriction = CuCurveOutput(cc.latFriction, frictionCurveInput);
				float longFriction = CuCurveOutput(cc.longFriction, frictionCurveInput);

				if (io.st.handbrakeVal) {
					float handbrakeAmount = io.st.handbrakeVal;
					latFriction *= (CuCurveOutput(cc.handbrakeLatFrictionFactor, frictionCurveInput) - 1) * handbrakeAmount + 1;
					longFriction *= (CuCurveOutput(cc.handbrakeLongFrictionFactor, frictionCurveInput) - 1) * handbrakeAmount + 1;
				} else {
					longFriction = 1;
				}

				bool isContactSticky = realThrottle != 0;
				if (!isContactSticky) {
					float nonStickyScale = CuCurveOutput(cc.nonStickyFrictionFactor, W[i].contactNormalWS.z);
					latFriction *= nonStickyScale;
					longFriction *= nonStickyScale;
				}

				io.wheels[i].latFriction = latFriction;
				io.wheels[i].longFriction = longFriction;
			}
		}

		// sticky forces
		if (wheelsHaveWorldContact) {
			// getUpwardsDirFromWheelContacts: sum contact normals of in-contact wheels
			CuVec3 sumContactDir = { 0, 0, 0 };
			for (int i = 0; i < 4; i++)
				if (W[i].isInContact)
					sumContactDir = sumContactDir + W[i].contactNormalWS;
			CuVec3 upwardsDir;
			if (sumContactDir.x == 0 && sumContactDir.y == 0 && sumContactDir.z == 0)
				upwardsDir = CuGetColumn(io.body.basis, 2); // getUpVector
			else
				upwardsDir = CuSafeNormalize3(sumContactDir);

			bool fullStick = (realThrottle != 0) || (fabsf(forwardSpeed_UU) > cc.stoppingForwardVel);
			float stickyForceScale = 0.5f; // fourWheel
			if (fullStick)
				stickyForceScale += 1 - fabsf(upwardsDir.z);

			CuApplyCentralForce(io, ((upwardsDir * stickyForceScale) * cc.gravityZBT) * cc.massBT);
		}

	}

	// ---- air torque / flip torque (Car::_UpdateAirTorque) or ground flip clear ----
	if (numWheelsInContact < 3) {
		bool updateAirControl = (numWheelsInContact == 0);
		// GetRightDir/UpDir/ForwardDir = CarState rotMat columns = basis columns 1/2/0
		CuVec3 dirPitch_right = CuNeg(CuGetColumn(io.body.basis, 1));
		CuVec3 dirYaw_up = CuGetColumn(io.body.basis, 2);
		CuVec3 dirRoll_forward = CuNeg(CuGetColumn(io.body.basis, 0));

		bool doAirControl = false;
		if (io.st.isFlipping)
			io.st.isFlipping = io.st.hasFlipped && io.st.flipTime < cc.flipTorqueTime;

		if (io.st.isFlipping) {
			CuVec3 relDodgeTorque = { io.st.flipRelTorque.x, io.st.flipRelTorque.y, io.st.flipRelTorque.z };
			bool torqueZero = relDodgeTorque.x == 0 && relDodgeTorque.y == 0 && relDodgeTorque.z == 0;
			if (!torqueZero) {
				float pitchScale = 1;
				if (relDodgeTorque.y != 0 && io.controls.pitch != 0) {
					if (CuSgn(relDodgeTorque.y) == CuSgn(io.controls.pitch)) {
						float a = fabsf(io.controls.pitch);
						pitchScale = 1 - (a < 1.0f ? a : 1.0f); // 1 - RS_MIN(|pitch|, 1)
						doAirControl = true;
					}
				}
				relDodgeTorque.y *= pitchScale;
				CuVec3 dodgeTorque = relDodgeTorque * CuVec3{ cc.flipTorqueX, cc.flipTorqueY, 0 };
				// applyTorque(invInertiaWorld.inverse() * basis * dodgeTorque):
				// chain is (inverse * (basis * dodgeTorque))? No: left-assoc matrix*matrix? Both are
				// mat*vec: (inv) * (basisMat * vec) evaluates basis*vec FIRST? C++: a * b * c with
				// (mat * mat) * vec vs mat * (mat * vec): operator* left-assoc: (inv * basis) * dodgeTorque
				CuMat3 invTensor = CuMat3Inverse(io.body.invInertiaWorld);
				CuMat3 prod = CuMat3Mul(invTensor, io.body.basis);
				CuApplyTorqueAccum(io, CuMat3MulVec(prod, dodgeTorque));
			} else {
				doAirControl = true;
			}
		} else {
			doAirControl = true;
		}

		doAirControl &= !io.st.isAutoFlipping;
		doAirControl &= updateAirControl;
		if (doAirControl) {
			float pitchTorqueScale = 1;
			CuVec3 torque;
			if (io.controls.pitch || io.controls.yaw || io.controls.roll) {
				if (io.st.isFlipping) {
					pitchTorqueScale = 0;
				} else if (io.st.hasFlipped) {
					if (io.st.flipTime < cc.flipTorqueTime + cc.flipPitchlockExtraTime)
						pitchTorqueScale = 0;
				}
				// (pitch * right * scale * Tx) + (yaw * up * Ty) + (roll * fwd * Tz)
				torque = ((dirPitch_right * io.controls.pitch) * pitchTorqueScale) * cc.carAirControlTorque.x +
					(dirYaw_up * io.controls.yaw) * cc.carAirControlTorque.y +
					(dirRoll_forward * io.controls.roll) * cc.carAirControlTorque.z;
			} else {
				torque = { 0, 0, 0 };
			}

			CuVec3 angVel = io.body.angVel;
			float dampPitch = CuDot3(dirPitch_right, angVel) * cc.carAirControlDamping.x *
				(1 - fabsf(doAirControl ? (io.controls.pitch * pitchTorqueScale) : 0.0f));
			float dampYaw = CuDot3(dirYaw_up, angVel) * cc.carAirControlDamping.y *
				(1 - fabsf(doAirControl ? io.controls.yaw : 0.0f));
			float dampRoll = CuDot3(dirRoll_forward, angVel) * cc.carAirControlDamping.z;

			CuVec3 damping = (dirYaw_up * dampYaw) + (dirPitch_right * dampPitch) + (dirRoll_forward * dampRoll);
			// (inverse() * (torque - damping)) * CAR_TORQUE_SCALE - scale applied AFTER mat*vec
			CuMat3 invTensor = CuMat3Inverse(io.body.invInertiaWorld);
			CuApplyTorqueAccum(io, CuMat3MulVec(invTensor, torque - damping) * cc.carTorqueScale);
		}

		if (io.controls.throttle != 0) {
			// fwd * throttle * AIR_ACCEL * UU_TO_BT * MASS (left-assoc chain)
			CuVec3 f = (((CuGetColumn(io.body.basis, 0) * io.controls.throttle) * cc.throttleAirAccel) * cc.uuToBT) * cc.massBT;
			CuApplyCentralForce(io, f);
		}
	} else {
		io.st.isFlipping = 0;
	}

	// ---- _UpdateJump ----
	{
		if (io.st.isOnGround && !io.st.isJumping) {
			if (io.st.hasJumped && io.st.jumpTime < cc.jumpMinTime + cc.jumpResetTimePad) {
				// keep
			} else {
				io.st.hasJumped = 0;
				io.st.jumpTime = 0;
			}
		}
		if (io.st.isJumping) {
			if (io.st.jumpTime < cc.jumpMinTime || (io.controls.jump && io.st.jumpTime < cc.jumpMaxTime)) {
				io.st.isJumping = 1;
			} else {
				io.st.isJumping = 0;
			}
		} else if (io.st.isOnGround && jumpPressed) {
			io.st.isJumping = 1;
			io.st.jumpTime = 0;
			// GetUpDir * jumpImmediateForce * UU_TO_BT * MASS
			CuVec3 jumpStartForce = ((CuGetColumn(io.body.basis, 2) * cc.jumpImmediateForce) * cc.uuToBT) * cc.massBT;
			CuApplyCentralImpulseBody(io, jumpStartForce);
		}
		if (io.st.isJumping) {
			io.st.hasJumped = 1;
			CuVec3 totalJumpForce = CuGetColumn(io.body.basis, 2) * cc.jumpAccel;
			if (io.st.jumpTime < cc.jumpMinTime)
				totalJumpForce = totalJumpForce * 0.62f; // JUMP_PRE_MIN_ACCEL_SCALE
			CuApplyCentralForce(io, (totalJumpForce * cc.uuToBT) * cc.massBT);
		}
		if (io.st.isJumping || io.st.hasJumped)
			io.st.jumpTime += tickTime;
	}

	// ---- _UpdateAutoFlip ----
	{
		if (jumpPressed && io.st.worldContact_hasContact &&
			io.st.worldContact_normal.z > cc.carAutoflipNormZThresh) {
			// Angle::FromRotMat: bullet getEulerYPR on transposed rotMat; roll negated.
			// roll_pre = atan2f(basis[2][1]?? -> bullet[2][1] = rotMat[1][2] = right.z =
			// basis col1 z = basis.r[2].y; bullet[2][2] = up.z = basis.r[2].z.
			float yaw_pre = CuAtan2f(io.body.basis.r[1].x /*bullet[1][0]=rotMat[0][1]=forward.y=col0.y*/,
				io.body.basis.r[0].x);
			float pitch_pre = CuAsinf(-io.body.basis.r[2].x);
			float roll_pre = CuAtan2f(io.body.basis.r[2].y, io.body.basis.r[2].z);
			constexpr float CU_HALF_PI_F = 1.570796326794896619231321691639751442f;
			constexpr float CU_PI_F = 3.141592653589793238462643383279502884f;
			if (fabsf(pitch_pre) == (float)CU_HALF_PI_F) {
				if (yaw_pre > 0) yaw_pre -= (float)CU_PI_F; else yaw_pre += (float)CU_PI_F;
				if (roll_pre > 0) roll_pre -= (float)CU_PI_F; else roll_pre += (float)CU_PI_F;
			}
			float roll = roll_pre * -1.0f; // FromRotMat: result.roll *= -1

			float absRoll = fabsf(roll);
			if (absRoll > cc.carAutoflipRollThresh) {
				io.st.autoFlipTimer = cc.carAutoflipTime * (absRoll / CU_PI_F);
				io.st.autoFlipTorqueScale = (roll > 0) ? 1.0f : -1.0f;
				io.st.isAutoFlipping = 1;
				CuVec3 imp = ((CuNeg(CuGetColumn(io.body.basis, 2)) * cc.carAutoflipImpulse) * cc.uuToBT) * cc.massBT;
				CuApplyCentralImpulseBody(io, imp);
			}
		}

		if (io.st.isAutoFlipping) {
			if (io.st.autoFlipTimer <= 0) {
				io.st.isAutoFlipping = 0;
				io.st.autoFlipTimer = 0;
			} else {
				// angVel += fwd * TORQUE * scale * dt
				io.body.angVel = io.body.angVel +
					((CuGetColumn(io.body.basis, 0) * cc.carAutoflipTorque) * io.st.autoFlipTorqueScale) * tickTime;
				io.st.autoFlipTimer -= tickTime;
			}
		}
	}

	// ---- _UpdateDoubleJumpOrFlip ----
	{
		float tickTimeScale = tickTime / (1 / 120.0f);

		if (io.st.isOnGround) {
			io.st.hasDoubleJumped = 0;
			io.st.hasFlipped = 0;
			io.st.airTime = 0;
			io.st.airTimeSinceJump = 0;
			io.st.flipTime = 0;
		} else {
			io.st.airTime += tickTime;
			if (io.st.hasJumped && !io.st.isJumping) {
				io.st.airTimeSinceJump += tickTime;
			} else {
				io.st.airTimeSinceJump = 0;
			}

			if (jumpPressed && io.st.airTimeSinceJump < cc.doublejumpMaxDelay) {
				float inputMagnitude = fabsf(io.controls.yaw) + fabsf(io.controls.pitch) + fabsf(io.controls.roll);
				bool isFlipInput = inputMagnitude >= cc.dodgeDeadzone;

				bool canUse;
				if (isFlipInput) {
					canUse = (!io.st.hasDoubleJumped && !io.st.hasFlipped) || cc.unlimitedFlips;
				} else {
					canUse = (!io.st.hasDoubleJumped && !io.st.hasFlipped) || cc.unlimitedDoubleJumps;
				}
				if (io.st.isAutoFlipping)
					canUse = false;

				if (canUse) {
					if (isFlipInput) {
						io.st.flipTime = 0;
						io.st.hasFlipped = 1;
						io.st.isFlipping = 1;

						float forwardSpeedRatio = fabsf(forwardSpeed_UU) / cc.carMaxSpeed;
						CuVec3 dodgeDir = { -io.controls.pitch, io.controls.yaw + io.controls.roll, 0 };
						if (fabsf(io.controls.yaw + io.controls.roll) < 0.1f && fabsf(io.controls.pitch) < 0.1f) {
							dodgeDir = { 0, 0, 0 };
						} else {
							dodgeDir = CuSafeNormalize3(dodgeDir); // btVector3::safeNormalized
						}

						io.st.flipRelTorque = { -dodgeDir.y / tickTimeScale, dodgeDir.x / tickTimeScale, 0 };

						if (fabsf(dodgeDir.x) < 0.1f) dodgeDir.x = 0;
						if (fabsf(dodgeDir.y) < 0.1f) dodgeDir.y = 0;

						// fuzzyZero = length2() < SIMD_EPSILON*SIMD_EPSILON
						if (!(CuLength2(dodgeDir) < CU_SIMD_EPSILON * CU_SIMD_EPSILON)) {
							bool shouldDodgeBackwards;
							if (fabsf(forwardSpeed_UU) < 100.0f) {
								shouldDodgeBackwards = dodgeDir.x < 0.0f;
							} else {
								shouldDodgeBackwards = (dodgeDir.x >= 0.0f) != (forwardSpeed_UU >= 0.0f);
							}

							// Vec initalDodgeVel = dodgeDir * FLIP_INITIAL_VEL_SCALE (Vec elementwise)
							CuVec3 initalDodgeVel = dodgeDir * cc.flipInitialVelScale;
							float maxSpeedScaleX = shouldDodgeBackwards ?
								cc.flipBackwardImpulseMaxSpeedScale : cc.flipForwardImpulseMaxSpeedScale;
							initalDodgeVel.x *= ((maxSpeedScaleX - 1) * forwardSpeedRatio) + 1.0f;
							initalDodgeVel.y *= ((cc.flipSideImpulseMaxSpeedScale - 1) * forwardSpeedRatio) + 1.0f;
							if (shouldDodgeBackwards)
								initalDodgeVel.x *= cc.flipBackwardImpulseScaleX;

							// Vec forwardDir2D = GetForwardDir().To2D().Normalized() [RS Vec trees]
							CuVec3 fwdDir = CuGetColumn(io.body.basis, 0);
							CuVec3 forwardDir2D = CuVecNormalizedRS({ fwdDir.x, fwdDir.y, 0 });
							CuVec3 rightDir2D = { -forwardDir2D.y, forwardDir2D.x, 0 };

							// finalDeltaVel = ix * fwd2D + iy * right2D (Vec trees, elementwise equal)
							CuVec3 finalDeltaVel = (forwardDir2D * initalDodgeVel.x) + (rightDir2D * initalDodgeVel.y);
							CuApplyCentralImpulseBody(io, (finalDeltaVel * cc.uuToBT) * cc.massBT);
						}
					} else {
						// double jump uses RLConst::JUMP_IMMEDIATE_FORCE (not the mutator value)
						CuVec3 jumpStartForce = ((CuGetColumn(io.body.basis, 2) * cc.jumpImmediateForceRL) * cc.uuToBT) * cc.massBT;
						CuApplyCentralImpulseBody(io, jumpStartForce);
						io.st.hasDoubleJumped = 1;
					}
				}
			}
		}

		if (io.st.isFlipping) {
			io.st.flipTime += tickTime;
			if (io.st.flipTime <= cc.flipTorqueTime) {
				if (io.st.flipTime >= cc.flipZDampStart &&
					(io.body.linVel.z < 0 || io.st.flipTime < cc.flipZDampEnd)) {
					io.body.linVel.z *= cc.flipZDampFactor; // powf(1-damp, tickTimeScale=1) host-folded
				}
			}
		} else if (io.st.hasFlipped) {
			io.st.flipTime += tickTime;
		}
	}

	// ---- _UpdateAutoRoll (conditional) ----
	if (io.controls.throttle != 0 &&
		((numWheelsInContact > 0 && numWheelsInContact < 4) || io.st.worldContact_hasContact)) {
		CuVec3 groundUpDir;
		if (numWheelsInContact > 0) {
			CuVec3 sumContactDir = { 0, 0, 0 };
			for (int i = 0; i < 4; i++)
				if (W[i].isInContact)
					sumContactDir = sumContactDir + W[i].contactNormalWS;
			if (sumContactDir.x == 0 && sumContactDir.y == 0 && sumContactDir.z == 0)
				groundUpDir = CuGetColumn(io.body.basis, 2);
			else
				groundUpDir = CuSafeNormalize3(sumContactDir);
		} else {
			groundUpDir = { io.st.worldContact_normal.x, io.st.worldContact_normal.y, io.st.worldContact_normal.z };
		}

		CuVec3 groundDownDir = CuNeg(groundUpDir);
		CuVec3 forwardDir = CuGetColumn(io.body.basis, 0);
		CuVec3 rightDir = CuGetColumn(io.body.basis, 1);

		CuVec3 crossRightDir = CuCross(groundUpDir, forwardDir);
		CuVec3 crossForwardDir = CuCross(groundDownDir, crossRightDir);

		float rt = CuDot3(rightDir, crossRightDir);
		rt = fminf(fmaxf(rt, 0.0f), 1.0f);
		float rightTorqueFactor = 1 - rt;
		float ft = CuDot3(forwardDir, crossForwardDir);
		ft = fminf(fmaxf(ft, 0.0f), 1.0f);
		float forwardTorqueFactor = 1 - ft;

		CuVec3 torqueDirRight = forwardDir * (CuDot3(rightDir, groundUpDir) >= 0 ? -1.0f : 1.0f);
		CuVec3 torqueDirForward = rightDir * (CuDot3(forwardDir, groundUpDir) >= 0 ? 1.0f : -1.0f);

		CuVec3 torqueRight = torqueDirRight * rightTorqueFactor;
		CuVec3 torqueForward = torqueDirForward * forwardTorqueFactor;

		CuApplyCentralForce(io, ((groundDownDir * cc.carAutorollForce) * cc.uuToBT) * cc.massBT);
		// (inverse() * (tF + tR)) * CAR_AUTOROLL_TORQUE - scale AFTER mat*vec
		CuMat3 invTensor = CuMat3Inverse(io.body.invInertiaWorld);
		CuApplyTorqueAccum(io, CuMat3MulVec(invTensor, torqueForward + torqueRight) * cc.carAutorollTorque);
	}

	io.st.worldContact_hasContact = 0;

	// ---- updateVehicleSecond ----
	{
		// updateSuspension (btVehicleRL.cpp:270)
		float suspForce[4];
		for (int i = 0; i < 4; i++) {
			if (W[i].isInContact) {
				float force = (cc.wheels[i].suspensionRestLength - W[i].suspensionLengthNew)
					* cc.wheels[i].suspensionStiffness * W[i].clippedInvContactDotSuspension;
				float dampingVelScale = (W[i].suspensionRelativeVelocity < 0)
					? cc.wheels[i].dampingCompression : cc.wheels[i].dampingRelaxation;
				float f = force - (dampingVelScale * W[i].suspensionRelativeVelocity);
				f *= cc.wheels[i].suspensionForceScale;
				if (f < 0) f = 0;
				suspForce[i] = f;
			} else {
				suspForce[i] = 0;
			}
		}
		for (int i = 0; i < 4; i++) {
			if (suspForce[i] != 0) {
				CuVec3 contactPointOffset = W[i].contactPointWS - io.body.origin;
				float baseForceScale = (suspForce[i] * tickTime) + W[i].extraPushback;
				CuVec3 force = W[i].contactNormalWS * baseForceScale;
				CuApplyImpulse(io.body, force, contactPointOffset);
			}
		}

		// applyFrictionImpulses (btVehicleRL.cpp:383)
		CuVec3 upDir = CuGetColumn(io.body.basis, 2);
		for (int i = 0; i < 4; i++) {
			if (!(W[i].impulse.x == 0 && W[i].impulse.y == 0 && W[i].impulse.z == 0)) {
				CuVec3 wheelContactOffset = W[i].contactPointWS - io.body.origin;
				float contactUpDot = CuDot3(upDir, wheelContactOffset);
				CuVec3 wheelRelPos = wheelContactOffset - upDir * contactUpDot;
				CuApplyImpulse(io.body, W[i].impulse * tickTime, wheelRelPos);
			}
		}
	}

	// ---- _UpdateBoost ----
	{
		bool hasBoost = io.st.boost > 0;
		if (hasBoost) {
			if (io.st.isBoosting) {
				io.st.isBoosting = (io.controls.boost || io.st.boostingTime < cc.boostMinTime) ? 1 : 0;
			} else {
				if (io.controls.boost)
					io.st.isBoosting = io.controls.boost;
			}
		} else {
			io.st.isBoosting = 0;
		}

		if (io.st.isBoosting) {
			io.st.boostingTime += tickTime;
		} else {
			io.st.boostingTime = 0;
		}

		if (io.st.isBoosting) {
			float nb = io.st.boost - cc.boostUsedPerSecond * tickTime;
			io.st.boost = nb > 0 ? nb : 0;
			// (accel * UU_TO_BT) * fwdDir * MASS: CPU chain:
			// (boostAccel * UU_TO_BT * GetForwardDir() * CAR_MASS_BT): ((a*u)*fwd)*mass
			float a = (io.st.isOnGround ? cc.boostAccelGround : cc.boostAccelAir) * cc.uuToBT;
			CuVec3 f = (CuGetColumn(io.body.basis, 0) * a) * cc.massBT;
			CuApplyCentralForce(io, f);
			io.st.timeSinceBoosted = 0;
		} else {
			io.st.timeSinceBoosted += tickTime;
			if (cc.rechargeBoostEnabled)
				if (io.st.timeSinceBoosted >= cc.rechargeBoostDelay)
					io.st.boost += cc.rechargeBoostPerSecond * tickTime;
		}
		io.st.boost = io.st.boost < cc.boostMax ? io.st.boost : cc.boostMax;
	}
}

} // namespace rscuda

#endif // __CUDACC__
