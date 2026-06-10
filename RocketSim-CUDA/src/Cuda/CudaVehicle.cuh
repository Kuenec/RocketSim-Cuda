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

} // namespace rscuda

#endif // __CUDACC__
