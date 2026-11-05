#pragma once

#include "GpuTypes.h"
#include "CudaMath.cuh"
#include "CudaCollision.cuh"
#include "CudaCollideWorld.cuh"

namespace rscuda {

#define CU_SOLVER_ITERATIONS 10
#define CU_ERP2 0.8f
#define CU_SPLIT_TURN_ERP 0.1f
#define CU_RESTITUTION_VEL_THRESH 0.2f
#define CU_WARMSTART_FACTOR 0.85f
#define CU_MAX_SOLVER_BODIES 20
#define CU_MAX_CONTACT_ROWS (GPU_MAX_MANIFOLDS * 4 + CU_MAX_SOLVER_BODIES)
#define CU_MAX_FRICTION_ROWS CU_MAX_CONTACT_ROWS

struct CuSolverBody {
	CuMat3 basis;
	CuVec3 origin;
	CuVec3 invMass;
	CuVec3 linVel, angVel;
	CuVec3 extForceImpulse, extTorqueImpulse;
	CuVec3 deltaLin, deltaAng;
	CuVec3 pushVel, turnVel;
	CuMat3 invInertiaWorld;
	int bodySlot;
};

struct CuSolverRow {
	int bodyIdA, bodyIdB;
	CuVec3 contactNormal1, contactNormal2;
	CuVec3 relpos1CrossNormal, relpos2CrossNormal;
	CuVec3 angularComponentA, angularComponentB;
	float jacDiagABInv;
	float rhs, rhsPenetration, cfm;
	float lowerLimit, upperLimit;
	float appliedImpulse, appliedPushImpulse;
	float friction;
	int manifoldIdx;
	uint8_t pointIdx;
	uint8_t isSpecial;
};

struct CuSpecialInfo {
	int num;
	float friction, restitution;
	CuVec3 totalNormal;
	float totalDist;
};

__device__ __forceinline__ float CuSimdDot3(CuVec3 a, CuVec3 b) {
	return a.x * b.x + (a.y * b.y + a.z * b.z);
}

__device__ inline float CuResolveRowGeneric(CuSolverBody& a, CuSolverBody& b, CuSolverRow& c) {
	float cpApplied = c.appliedImpulse;
	float deltaImpulse = c.rhs - c.appliedImpulse * c.cfm;
	float deltaVel1Dotn = CuSimdDot3(c.contactNormal1, a.deltaLin) + CuSimdDot3(c.relpos1CrossNormal, a.deltaAng);
	float deltaVel2Dotn = CuSimdDot3(c.contactNormal2, b.deltaLin) + CuSimdDot3(c.relpos2CrossNormal, b.deltaAng);
	deltaImpulse = deltaImpulse - deltaVel1Dotn * c.jacDiagABInv;
	deltaImpulse = deltaImpulse - deltaVel2Dotn * c.jacDiagABInv;
	float sum = c.appliedImpulse + deltaImpulse;
	bool lowerLess = sum < c.lowerLimit;
	bool upperLess = sum < c.upperLimit;
	float lowMinApplied = c.lowerLimit - cpApplied;
	deltaImpulse = lowerLess ? lowMinApplied : deltaImpulse;
	c.appliedImpulse = lowerLess ? c.lowerLimit : sum;
	float upperMinApplied = c.upperLimit - cpApplied;
	deltaImpulse = upperLess ? deltaImpulse : upperMinApplied;
	c.appliedImpulse = upperLess ? c.appliedImpulse : c.upperLimit;
	a.deltaLin = a.deltaLin + (c.contactNormal1 * a.invMass) * deltaImpulse;
	a.deltaAng = a.deltaAng + c.angularComponentA * deltaImpulse;
	b.deltaLin = b.deltaLin + (c.contactNormal2 * b.invMass) * deltaImpulse;
	b.deltaAng = b.deltaAng + c.angularComponentB * deltaImpulse;
	return deltaImpulse / c.jacDiagABInv;
}

__device__ inline float CuResolveRowLowerLimit(CuSolverBody& a, CuSolverBody& b, CuSolverRow& c) {
	float deltaImpulse = c.rhs - c.appliedImpulse * c.cfm;
	float deltaVel1Dotn = CuSimdDot3(c.contactNormal1, a.deltaLin) + CuSimdDot3(c.relpos1CrossNormal, a.deltaAng);
	float deltaVel2Dotn = CuSimdDot3(c.contactNormal2, b.deltaLin) + CuSimdDot3(c.relpos2CrossNormal, b.deltaAng);
	deltaImpulse = deltaImpulse - deltaVel1Dotn * c.jacDiagABInv;
	deltaImpulse = deltaImpulse - deltaVel2Dotn * c.jacDiagABInv;
	float sum = c.appliedImpulse + deltaImpulse;
	bool lowerLess = sum < c.lowerLimit;
	float lowMinApplied = c.lowerLimit - c.appliedImpulse;
	deltaImpulse = lowerLess ? lowMinApplied : deltaImpulse;
	c.appliedImpulse = lowerLess ? c.lowerLimit : sum;
	a.deltaLin = a.deltaLin + (c.contactNormal1 * a.invMass) * deltaImpulse;
	a.deltaAng = a.deltaAng + c.angularComponentA * deltaImpulse;
	b.deltaLin = b.deltaLin + (c.contactNormal2 * b.invMass) * deltaImpulse;
	b.deltaAng = b.deltaAng + c.angularComponentB * deltaImpulse;
	return deltaImpulse / c.jacDiagABInv;
}

__device__ inline float CuResolveSplitPenetration(CuSolverBody& a, CuSolverBody& b, CuSolverRow& c) {
	if (!(c.rhsPenetration != 0.0f))
		return 0.0f;
	float deltaImpulse = c.rhsPenetration - c.appliedPushImpulse * c.cfm;
	float deltaVel1Dotn = CuSimdDot3(c.contactNormal1, a.pushVel) + CuSimdDot3(c.relpos1CrossNormal, a.turnVel);
	float deltaVel2Dotn = CuSimdDot3(c.contactNormal2, b.pushVel) + CuSimdDot3(c.relpos2CrossNormal, b.turnVel);
	deltaImpulse = deltaImpulse - deltaVel1Dotn * c.jacDiagABInv;
	deltaImpulse = deltaImpulse - deltaVel2Dotn * c.jacDiagABInv;
	float sum = c.appliedPushImpulse + deltaImpulse;
	bool lowerLess = sum < c.lowerLimit;
	float lowMinApplied = c.lowerLimit - c.appliedPushImpulse;
	deltaImpulse = lowerLess ? lowMinApplied : deltaImpulse;
	c.appliedPushImpulse = lowerLess ? c.lowerLimit : sum;
	a.pushVel = a.pushVel + (c.contactNormal1 * a.invMass) * deltaImpulse;
	a.turnVel = a.turnVel + c.angularComponentA * deltaImpulse;
	b.pushVel = b.pushVel + (c.contactNormal2 * b.invMass) * deltaImpulse;
	b.turnVel = b.turnVel + c.angularComponentB * deltaImpulse;
	return deltaImpulse * (1.0f / c.jacDiagABInv);
}

struct CuRowStateBits {
	unsigned w[15];
};

__device__ __forceinline__ CuRowStateBits CuCaptureRowState(const CuSolverBody& a,
	const CuSolverBody& b, const CuSolverRow& c) {
	return { { __float_as_uint(a.deltaLin.x), __float_as_uint(a.deltaLin.y),
		__float_as_uint(a.deltaLin.z), __float_as_uint(a.deltaAng.x),
		__float_as_uint(a.deltaAng.y), __float_as_uint(a.deltaAng.z),
		__float_as_uint(b.deltaLin.x), __float_as_uint(b.deltaLin.y),
		__float_as_uint(b.deltaLin.z), __float_as_uint(b.deltaAng.x),
		__float_as_uint(b.deltaAng.y), __float_as_uint(b.deltaAng.z),
		__float_as_uint(c.appliedImpulse), __float_as_uint(c.lowerLimit),
		__float_as_uint(c.upperLimit) } };
}

__device__ __forceinline__ bool CuSameRowState(const CuRowStateBits& p, const CuRowStateBits& q) {
	unsigned diff = 0;
	for (int i = 0; i < 15; i++) {
		diff |= p.w[i] ^ q.w[i];
	}
	return diff == 0;
}

struct CuSolveCtx {
	CuSolverBody solverBodies[CU_MAX_SOLVER_BODIES];
	int companion[CU_MAX_SOLVER_BODIES];
	CuSpecialInfo special[CU_MAX_SOLVER_BODIES];
	CuSolverRow contactRows[CU_MAX_CONTACT_ROWS];
	CuSolverRow frictionRows[CU_MAX_FRICTION_ROWS];
};

struct CuSolveView {
	GpuBodyState* bodies;
	const GpuBodyConsts* bodyConsts;
	float dt;
	CuSolverBody* solverBodies;
	int* companion;
	CuSpecialInfo* special;
	CuSolverRow* contactRows;
	CuSolverRow* frictionRows;
	int numSolverBodies;
	int fixedBodyId;
	int numContactRows;
	int numFrictionRows;
};

__host__ __device__ inline int CuAlign16(int bytes) {
	return (bytes + 15) & ~15;
}

__device__ inline void CuPlaceSolveStorage(CuSolveView& s, CuSolveCtx* scratch,
	unsigned char* smem, int smemBytes, int numBodies, int maxRows) {
	const int bodyBytes = CuAlign16((int)sizeof(CuSolverBody) * (numBodies + 1));
	const int companionBytes = CuAlign16((int)sizeof(int) * numBodies);
	const int specialBytes = CuAlign16((int)sizeof(CuSpecialInfo) * numBodies);
	const int rowBytes = CuAlign16((int)sizeof(CuSolverRow) * maxRows);
	if (bodyBytes + companionBytes + specialBytes + 2 * rowBytes > smemBytes) {
		s.solverBodies = scratch->solverBodies;
		s.companion = scratch->companion;
		s.special = scratch->special;
		s.contactRows = scratch->contactRows;
		s.frictionRows = scratch->frictionRows;
		return;
	}
	s.solverBodies = reinterpret_cast<CuSolverBody*>(smem);
	smem += bodyBytes;
	s.companion = reinterpret_cast<int*>(smem);
	smem += companionBytes;
	s.special = reinterpret_cast<CuSpecialInfo*>(smem);
	smem += specialBytes;
	s.contactRows = reinterpret_cast<CuSolverRow*>(smem);
	smem += rowBytes;
	s.frictionRows = reinterpret_cast<CuSolverRow*>(smem);
}

__device__ inline void CuInitSolverBody(CuSolveView& s, CuSolverBody& sb, int slot) {
	sb.deltaLin = { 0, 0, 0 };
	sb.deltaAng = { 0, 0, 0 };
	sb.pushVel = { 0, 0, 0 };
	sb.turnVel = { 0, 0, 0 };
	if (slot >= 0) {
		const GpuBodyState& b = s.bodies[slot];
		const GpuBodyConsts& c = s.bodyConsts[slot];
		sb.basis = CuLoadBasis(b.basis);
		sb.origin = GV(b.pos);
		sb.invMass = { c.invMass, c.invMass, c.invMass };
		sb.bodySlot = slot;
		sb.linVel = GV(b.linVel);
		sb.angVel = GV(b.angVel);
		sb.extForceImpulse = (GV(b.totalForce) * c.invMass) * s.dt;
		sb.invInertiaWorld = CuLoadBasis(b.invInertiaWorld);
		sb.extTorqueImpulse = CuVecMulMat(GV(b.totalTorque), sb.invInertiaWorld) * s.dt;
	} else {
		sb.basis = CuIdentity3();
		sb.origin = { 0, 0, 0 };
		sb.invMass = { 0, 0, 0 };
		sb.bodySlot = -1;
		sb.linVel = { 0, 0, 0 };
		sb.angVel = { 0, 0, 0 };
		sb.extForceImpulse = { 0, 0, 0 };
		sb.extTorqueImpulse = { 0, 0, 0 };
		sb.invInertiaWorld = CuIdentity3();
	}
}

__device__ inline void CuSetupContactRow(CuSolveView& s, CuSolverRow& row,
	int idA, int idB, const CuManifoldPoint& cp, float& relaxation,
	CuVec3 rel_pos1, CuVec3 rel_pos2) {

	const CuSolverBody& bodyA = s.solverBodies[idA];
	const CuSolverBody& bodyB = s.solverBodies[idB];
	bool rb0 = bodyA.bodySlot >= 0;
	bool rb1 = bodyB.bodySlot >= 0;
	relaxation = 1.0f;
	float invTimeStep = 1.0f / s.dt;
	float cfm = 0.0f;
	float erp = CU_ERP2;
	cfm *= invTimeStep;

	CuVec3 n = cp.normalWorldOnB;
	CuVec3 torqueAxis0 = CuCross(rel_pos1, n);
	row.angularComponentA = rb0 ? CuMat3MulVec(bodyA.invInertiaWorld, torqueAxis0) : CuVec3{ 0, 0, 0 };
	CuVec3 torqueAxis1 = CuCross(rel_pos2, n);
	row.angularComponentB = rb1 ? CuMat3MulVec(bodyB.invInertiaWorld, CuNeg(torqueAxis1)) : CuVec3{ 0, 0, 0 };

	float denom0 = 0.0f, denom1 = 0.0f;
	if (rb0) {
		CuVec3 vec = CuCross(row.angularComponentA, rel_pos1);
		denom0 = bodyA.invMass.x + CuDot3(n, vec);
	}
	if (rb1) {
		CuVec3 vec = CuCross(CuNeg(row.angularComponentB), rel_pos2);
		denom1 = bodyB.invMass.x + CuDot3(n, vec);
	}
	row.jacDiagABInv = relaxation / (denom0 + denom1 + cfm);

	if (rb0) {
		row.contactNormal1 = n;
		row.relpos1CrossNormal = torqueAxis0;
	} else {
		row.contactNormal1 = { 0, 0, 0 };
		row.relpos1CrossNormal = { 0, 0, 0 };
	}
	if (rb1) {
		row.contactNormal2 = CuNeg(n);
		row.relpos2CrossNormal = CuNeg(torqueAxis1);
	} else {
		row.contactNormal2 = { 0, 0, 0 };
		row.relpos2CrossNormal = { 0, 0, 0 };
	}

	float restitution = 0.0f;
	float penetration = cp.distance1 + 0.0f;
	{
		CuVec3 vel1 = rb0 ? bodyA.linVel + CuCross(bodyA.angVel, rel_pos1) : CuVec3{ 0, 0, 0 };
		CuVec3 vel2 = rb1 ? bodyB.linVel + CuCross(bodyB.angVel, rel_pos2) : CuVec3{ 0, 0, 0 };
		CuVec3 vel = vel1 - vel2;
		float rel_vel = CuDot3(n, vel);
		row.friction = cp.combinedFriction;
		float rest = fabsf(rel_vel) < CU_RESTITUTION_VEL_THRESH ? 0.0f
			: cp.combinedRestitution * -rel_vel;
		restitution = rest <= 0.0f ? 0.0f : rest;
	}

	row.appliedImpulse = cp.appliedImpulse * CU_WARMSTART_FACTOR;
	row.appliedPushImpulse = 0.0f;

	{
		CuVec3 extFA = rb0 ? bodyA.extForceImpulse : CuVec3{ 0, 0, 0 };
		CuVec3 extTA = rb0 ? bodyA.extTorqueImpulse : CuVec3{ 0, 0, 0 };
		CuVec3 extFB = rb1 ? bodyB.extForceImpulse : CuVec3{ 0, 0, 0 };
		CuVec3 extTB = rb1 ? bodyB.extTorqueImpulse : CuVec3{ 0, 0, 0 };
		float vel1Dotn = CuDot3(row.contactNormal1, bodyA.linVel + extFA) +
			CuDot3(row.relpos1CrossNormal, bodyA.angVel + extTA);
		float vel2Dotn = CuDot3(row.contactNormal2, bodyB.linVel + extFB) +
			CuDot3(row.relpos2CrossNormal, bodyB.angVel + extTB);
		float rel_vel = vel1Dotn + vel2Dotn;
		float velocityError = restitution - rel_vel;
		float positionalError = penetration > 0.0f ? 0.0f : -penetration * erp * invTimeStep;
		float penetrationImpulse = positionalError * row.jacDiagABInv;
		float velocityImpulse = velocityError * row.jacDiagABInv;
		row.rhs = velocityImpulse;
		row.rhsPenetration = penetrationImpulse;
		row.cfm = cfm * row.jacDiagABInv;
		row.lowerLimit = 0.0f;
		row.upperLimit = 1e10f;
	}
	row.isSpecial = cp.isSpecial;
}

__device__ inline void CuWarmstartContactRow(CuSolveView& s, const CuSolverRow& row) {
	CuSolverBody& bodyA = s.solverBodies[row.bodyIdA];
	CuSolverBody& bodyB = s.solverBodies[row.bodyIdB];
	if (bodyA.bodySlot >= 0) {
		bodyA.deltaLin = bodyA.deltaLin + (row.contactNormal1 * bodyA.invMass) * row.appliedImpulse;
		bodyA.deltaAng = bodyA.deltaAng + row.angularComponentA * row.appliedImpulse;
	}
	if (bodyB.bodySlot >= 0) {
		bodyB.deltaLin = bodyB.deltaLin + (CuNeg(row.contactNormal2) * bodyB.invMass) * (-row.appliedImpulse);
		bodyB.deltaAng = bodyB.deltaAng + CuNeg(row.angularComponentB) * (-row.appliedImpulse);
	}
}

__device__ inline void CuSetupFrictionRow(CuSolveView& s, CuSolverRow& row,
	CuVec3 normalAxis, int idA, int idB, const CuManifoldPoint& cp,
	CuVec3 rel_pos1, CuVec3 rel_pos2, float relaxation) {

	const CuSolverBody& bodyA = s.solverBodies[idA];
	const CuSolverBody& bodyB = s.solverBodies[idB];
	bool rb0 = bodyA.bodySlot >= 0;
	bool rb1 = bodyB.bodySlot >= 0;
	row.bodyIdA = idA;
	row.bodyIdB = idB;
	row.friction = cp.combinedFriction;
	row.appliedImpulse = 0.0f;
	row.appliedPushImpulse = 0.0f;
	row.isSpecial = 0;

	if (rb0) {
		row.contactNormal1 = normalAxis;
		CuVec3 ftorqueAxis1 = CuCross(rel_pos1, row.contactNormal1);
		row.relpos1CrossNormal = ftorqueAxis1;
		row.angularComponentA = CuMat3MulVec(bodyA.invInertiaWorld, ftorqueAxis1);
	} else {
		row.contactNormal1 = { 0, 0, 0 };
		row.relpos1CrossNormal = { 0, 0, 0 };
		row.angularComponentA = { 0, 0, 0 };
	}
	if (rb1) {
		row.contactNormal2 = CuNeg(normalAxis);
		CuVec3 ftorqueAxis1 = CuCross(rel_pos2, row.contactNormal2);
		row.relpos2CrossNormal = ftorqueAxis1;
		row.angularComponentB = CuMat3MulVec(bodyB.invInertiaWorld, ftorqueAxis1);
	} else {
		row.contactNormal2 = { 0, 0, 0 };
		row.relpos2CrossNormal = { 0, 0, 0 };
		row.angularComponentB = { 0, 0, 0 };
	}
	{
		float denom0 = 0.0f, denom1 = 0.0f;
		if (rb0) {
			CuVec3 vec = CuCross(row.angularComponentA, rel_pos1);
			denom0 = bodyA.invMass.x + CuDot3(normalAxis, vec);
		}
		if (rb1) {
			CuVec3 vec = CuCross(CuNeg(row.angularComponentB), rel_pos2);
			denom1 = bodyB.invMass.x + CuDot3(normalAxis, vec);
		}
		row.jacDiagABInv = relaxation / (denom0 + denom1);
	}
	{
		float vel1Dotn = CuDot3(row.contactNormal1, rb0 ? bodyA.linVel + bodyA.extForceImpulse : CuVec3{ 0, 0, 0 }) +
			CuDot3(row.relpos1CrossNormal, rb0 ? bodyA.angVel : CuVec3{ 0, 0, 0 });
		float vel2Dotn = CuDot3(row.contactNormal2, rb1 ? bodyB.linVel + bodyB.extForceImpulse : CuVec3{ 0, 0, 0 }) +
			CuDot3(row.relpos2CrossNormal, rb1 ? bodyB.angVel : CuVec3{ 0, 0, 0 });
		float rel_vel = vel1Dotn + vel2Dotn;
		float velocityError = 0.0f - rel_vel;
		float velocityImpulse = velocityError * row.jacDiagABInv;
		row.rhs = 0.0f + velocityImpulse;
		row.rhsPenetration = 0.0f;
		row.cfm = 0.0f;
		row.lowerLimit = -row.friction;
		row.upperLimit = row.friction;
	}
}

__device__ inline void CuSetupContactFriction(CuSolveView& s, const CuManifoldPoint& cp,
	CuSolverRow& fr, int idA, int idB, CuVec3 rel_pos1, CuVec3 rel_pos2, float relaxation) {
	const CuSolverBody& bodyA = s.solverBodies[idA];
	const CuSolverBody& bodyB = s.solverBodies[idB];
	CuVec3 vel1 = bodyA.bodySlot >= 0
		? (bodyA.linVel + bodyA.extForceImpulse) + CuCross(bodyA.angVel + bodyA.extTorqueImpulse, rel_pos1)
		: CuVec3{ 0, 0, 0 };
	CuVec3 vel2 = bodyB.bodySlot >= 0
		? (bodyB.linVel + bodyB.extForceImpulse) + CuCross(bodyB.angVel + bodyB.extTorqueImpulse, rel_pos2)
		: CuVec3{ 0, 0, 0 };
	CuVec3 vel = vel1 - vel2;
	CuVec3 n = cp.normalWorldOnB;
	float rel_vel = CuDot3(n, vel);

	CuVec3 lateralDir1 = vel - n * rel_vel;
	float lat_rel_vel = CuLength2(lateralDir1);
	if (lat_rel_vel > CU_SIMD_EPSILON) {
		lateralDir1 = lateralDir1 * (1.0f / sqrtf(lat_rel_vel));
		CuSetupFrictionRow(s, fr, lateralDir1, idA, idB, cp, rel_pos1, rel_pos2, relaxation);
	} else {
		CuVec3 d1, d2;
		CuPlaneSpace1(n, d1, d2);
		CuSetupFrictionRow(s, fr, d1, idA, idB, cp, rel_pos1, rel_pos2, relaxation);
	}
}

__device__ inline void CuSetupPointRows(CuSolveView& s, const CuManifold* manifolds,
	int manifoldIdx, int pointIdx, int rowIdx, int idA, int idB) {
	const CuManifoldPoint& cp = manifolds[manifoldIdx].pts[pointIdx];
	CuSolverRow& row = s.contactRows[rowIdx];
	row.bodyIdA = idA;
	row.bodyIdB = idB;
	row.manifoldIdx = manifoldIdx;
	row.pointIdx = (uint8_t)pointIdx;
	CuVec3 rel_pos1 = cp.positionWorldOnA - s.solverBodies[idA].origin;
	CuVec3 rel_pos2 = cp.positionWorldOnB - s.solverBodies[idB].origin;
	float relaxation;
	CuSetupContactRow(s, row, idA, idB, cp, relaxation, rel_pos1, rel_pos2);
	CuSetupContactFriction(s, cp, s.frictionRows[rowIdx], idA, idB, rel_pos1, rel_pos2, relaxation);
}

__device__ inline void CuAccumulateSpecial(CuSolveView& s, const CuManifold* manifolds,
	const CuSolverRow& row) {
	const CuManifoldPoint& cp = manifolds[row.manifoldIdx].pts[row.pointIdx];
	const CuSolverBody& bodyA = s.solverBodies[row.bodyIdA];
	const CuSolverBody& bodyB = s.solverBodies[row.bodyIdB];
	for (int i = 0; i < 2; i++) {
		const CuSolverBody& body = i ? bodyB : bodyA;
		if (body.bodySlot < 0) {
			continue;
		}
		CuSpecialInfo& spri = s.special[body.bodySlot];
		spri.num++;
		spri.friction = cp.combinedFriction;
		spri.restitution = cp.combinedRestitution;
		spri.totalNormal = spri.totalNormal + cp.normalWorldOnB;
		CuVec3 rp = i ? cp.positionWorldOnB - bodyB.origin : cp.positionWorldOnA - bodyA.origin;
		spri.totalDist += sqrtf(CuDot3(rp, rp));
	}
}

__device__ inline void CuConvertContactSpecial(CuSolveView& s, int slot) {
	CuSpecialInfo& info = s.special[slot];
	float distance = info.totalDist / (float)info.num;
	CuVec3 normal = info.totalNormal * (1.0f / (float)info.num);
	CuManifoldPoint tmp = {};
	tmp.distance1 = distance;
	tmp.normalWorldOnB = normal;
	tmp.combinedFriction = info.friction;
	tmp.combinedRestitution = info.restitution;
	int idA = s.companion[slot];
	if (s.fixedBodyId < 0) {
		s.fixedBodyId = s.numSolverBodies++;
		CuInitSolverBody(s, s.solverBodies[s.fixedBodyId], -1);
	}
	int idB = s.fixedBodyId;
	const int rowIdx = s.numContactRows++;
	s.numFrictionRows++;
	CuSolverRow& row = s.contactRows[rowIdx];
	row.bodyIdA = idA;
	row.bodyIdB = idB;
	CuVec3 rel_pos1 = normal * -distance;
	CuVec3 rel_pos2 = { 0, 0, 0 };
	float relaxation = 0;
	CuSetupContactRow(s, row, idA, idB, tmp, relaxation, rel_pos1, rel_pos2);
	CuWarmstartContactRow(s, row);
	CuSetupContactFriction(s, tmp, s.frictionRows[rowIdx], idA, idB, rel_pos1, rel_pos2, relaxation);
}

__device__ inline void CuQuickSortPairs(int* keys, int* vals, int lo, int hi) {
	int i = lo, j = hi;
	int x = keys[(lo + hi) / 2];
	do {
		while (keys[i] < x) i++;
		while (x < keys[j]) j--;
		if (i <= j) {
			int tk = keys[i]; keys[i] = keys[j]; keys[j] = tk;
			int tv = vals[i]; vals[i] = vals[j]; vals[j] = tv;
			i++;
			j--;
		}
	} while (i <= j);
	if (lo < j) CuQuickSortPairs(keys, vals, lo, j);
	if (i < hi) CuQuickSortPairs(keys, vals, i, hi);
}

__host__ __device__ inline int CuSolveSharedBytes(int numBodies, int numManifolds) {
	return CuAlign16((int)sizeof(int) * (2 * numBodies + 2 * numManifolds));
}

__device__ inline int CuFindRoot(const int* parent, int x) {
	while (parent[x] != x) {
		x = parent[x];
	}
	return x;
}

__device__ inline bool CuMergesIslands(const GpuCarGameState* cars, int slot) {
	return slot == 0 || !cars[slot - 1].noContactResponse;
}

__device__ inline bool CuHasContactResponse(const GpuCarGameState* cars,
	const GpuStaticObj* statics, const uint8_t* tileNoResponse, int bodyRef) {
	if (bodyRef >= 0) {
		return CuMergesIslands(cars, bodyRef);
	}
	const GpuStaticObj& so = statics[-1 - bodyRef];
	if (so.kind == 2 && tileNoResponse) {
		return !tileNoResponse[so.shapeIdx];
	}
	return true;
}

__device__ inline void CuApplyExternalImpulses(GpuBodyState& b, const GpuBodyConsts& c, float dt) {
	const CuVec3 zero = { 0, 0, 0 };
	const CuVec3 extForce = (GV(b.totalForce) * c.invMass) * dt;
	const CuVec3 extTorque = CuVecMulMat(GV(b.totalTorque), CuLoadBasis(b.invInertiaWorld)) * dt;
	const CuVec3 lv = (GV(b.linVel) + zero) + extForce;
	const CuVec3 av = (GV(b.angVel) + zero) + extTorque;
	b.linVel = { lv.x, lv.y, lv.z };
	b.angVel = { av.x, av.y, av.z };
}

__device__ inline void CuWritebackSolverBody(CuSolverBody& sb, GpuBodyState* bodies,
	const GpuBodyConsts* bodyConsts, float dt) {
	if (sb.bodySlot < 0) {
		return;
	}
	GpuBodyState& b = bodies[sb.bodySlot];
	sb.linVel = sb.linVel + sb.deltaLin;
	sb.angVel = sb.angVel + sb.deltaAng;
	if (sb.pushVel.x != 0.0f || sb.pushVel.y != 0.0f || sb.pushVel.z != 0.0f ||
		sb.turnVel.x != 0.0f || sb.turnVel.y != 0.0f || sb.turnVel.z != 0.0f) {
		CuMat3 nb;
		CuVec3 np;
		CuIntegrateTransform(sb.basis, sb.origin, sb.pushVel,
			sb.turnVel * CU_SPLIT_TURN_ERP, dt,
			bodyConsts[sb.bodySlot].noRot != 0, nb, np);
		sb.basis = nb;
		sb.origin = np;
	}
	CuVec3 lv = sb.linVel + sb.extForceImpulse;
	CuVec3 av = sb.angVel + sb.extTorqueImpulse;
	b.linVel = { lv.x, lv.y, lv.z };
	b.angVel = { av.x, av.y, av.z };
	for (int r = 0; r < 3; r++) {
		b.basis.row[r] = { sb.basis.r[r].x, sb.basis.r[r].y, sb.basis.r[r].z };
	}
	b.pos = { sb.origin.x, sb.origin.y, sb.origin.z };
}

__device__ inline void CuBuildBatch(const int* parent, int numBodies, unsigned batchBits,
	int* keys, int* manifoldIdx, int numIslandManifolds, int* batchBodies) {
	if (numIslandManifolds > 1) {
		CuQuickSortPairs(keys, manifoldIdx, 0, numIslandManifolds - 1);
	}
	int sortedIds[CU_MAX_SOLVER_BODIES], sortedSlots[CU_MAX_SOLVER_BODIES];
	for (int i = 0; i < numBodies; i++) {
		sortedIds[i] = CuFindRoot(parent, i);
		sortedSlots[i] = i;
	}
	if (numBodies > 1) {
		CuQuickSortPairs(sortedIds, sortedSlots, 0, numBodies - 1);
	}
	int numBatchBodies = 0;
	int numBatchManifolds = 0;
	int startManifoldIndex = 0, endManifoldIndex = 1;
	for (int start = 0; start < numBodies;) {
		const int islandId = sortedIds[start];
		int end = start + 1;
		while (end < numBodies && sortedIds[end] == islandId) {
			end++;
		}
		int numCur = 0, startMf = -1;
		if (startManifoldIndex < numIslandManifolds && keys[startManifoldIndex] == islandId) {
			startMf = startManifoldIndex;
			for (endManifoldIndex = startManifoldIndex + 1;
				endManifoldIndex < numIslandManifolds && islandId == keys[endManifoldIndex];
				endManifoldIndex++) {}
			numCur = endManifoldIndex - startManifoldIndex;
		}
		if ((batchBits >> sortedSlots[start]) & 1u) {
			for (int k = start; k < end; k++) {
				batchBodies[numBatchBodies++] = sortedSlots[k];
			}
			for (int k = 0; k < numCur; k++) {
				manifoldIdx[numBatchManifolds++] = manifoldIdx[startMf + k];
			}
		}
		if (numCur) {
			startManifoldIndex = endManifoldIndex;
		}
		start = end;
	}
}

__device__ inline void CuSolveIterations(CuSolveView& s) {
	const int numContactRows = s.numContactRows;
	const int numFrictionRows = s.numFrictionRows;
	for (int iteration = 0; iteration < CU_SOLVER_ITERATIONS; iteration++) {
		float leastSquaresResidual = 0.0f;
		for (int j = 0; j < numContactRows; j++) {
			CuSolverRow& row = s.contactRows[j];
			float residual = CuResolveSplitPenetration(
				s.solverBodies[row.bodyIdA], s.solverBodies[row.bodyIdB], row);
			float r2 = residual * residual;
			leastSquaresResidual = leastSquaresResidual > r2 ? leastSquaresResidual : r2;
		}
		if (leastSquaresResidual <= 0.0f || iteration >= (CU_SOLVER_ITERATIONS - 1)) {
			break;
		}
	}
	for (int iteration = 0; iteration < CU_SOLVER_ITERATIONS; iteration++) {
		bool changed = false;
		for (int j = 0; j < numContactRows; j++) {
			CuSolverRow& row = s.contactRows[j];
			if (row.isSpecial) {
				continue;
			}
			CuSolverBody& a = s.solverBodies[row.bodyIdA];
			CuSolverBody& b = s.solverBodies[row.bodyIdB];
			const CuRowStateBits before = CuCaptureRowState(a, b, row);
			CuResolveRowLowerLimit(a, b, row);
			changed = changed || !CuSameRowState(before, CuCaptureRowState(a, b, row));
		}
		for (int j = 0; j < numFrictionRows; j++) {
			CuSolverRow& fr = s.frictionRows[j];
			float totalImpulse = s.contactRows[j].appliedImpulse;
			if (totalImpulse > 0.0f) {
				CuSolverBody& a = s.solverBodies[fr.bodyIdA];
				CuSolverBody& b = s.solverBodies[fr.bodyIdB];
				const CuRowStateBits before = CuCaptureRowState(a, b, fr);
				fr.lowerLimit = -(fr.friction * totalImpulse);
				fr.upperLimit = fr.friction * totalImpulse;
				CuResolveRowGeneric(a, b, fr);
				changed = changed || !CuSameRowState(before, CuCaptureRowState(a, b, fr));
			}
		}
		if (!changed) {
			break;
		}
	}
}

__device__ inline void CuSolveTick(GpuBodyState* bodies, const GpuBodyConsts* bodyConsts,
	const GpuCarGameState* cars, int numBodies, int numStatics,
	CuManifold* manifolds, int numManifolds, const int2* pairs, int numPairs,
	float dt, CuSolveCtx* scratch,
	const GpuStaticObj* statics, const uint8_t* tileNoResponse,
	unsigned char* smem, int smemBytes, int lane, int lanes, int groupBase) {
	const unsigned laneBits = lanes >= 32 ? 0xffffffffu : ((1u << lanes) - 1u);
	const unsigned groupMask = laneBits << groupBase;
	const unsigned lanesBelow = (1u << lane) - 1u;
	const int sharedBytes = CuSolveSharedBytes(numBodies, numManifolds);
	int* sParent = reinterpret_cast<int*>(smem + smemBytes - sharedBytes);
	int* sBatchBodies = sParent + numBodies;
	int* sKey = sBatchBodies + numBodies;
	int* sIdx = sKey + numManifolds;

	const bool isBody = lane < numBodies;
	if (isBody) {
		sParent[lane] = lane;
	}
	__syncwarp(groupMask);
	if (lane == 0) {
		for (int p = 0; p < numPairs; p++) {
			const int2 pr = pairs[p];
			if (pr.x < numStatics || pr.y < numStatics) {
				continue;
			}
			const int a = pr.x - numStatics;
			const int b = pr.y - numStatics;
			if (CuMergesIslands(cars, a) && CuMergesIslands(cars, b)) {
				sParent[CuFindRoot(sParent, a)] = CuFindRoot(sParent, b);
			}
		}
	}
	__syncwarp(groupMask);

	int root = -1;
	int newState = CU_ISLAND_SLEEPING;
	bool inBatch = false;
	if (isBody) {
		root = CuFindRoot(sParent, lane);
		const unsigned bodyMask = ((1u << numBodies) - 1u) << groupBase;
		const unsigned island = __match_any_sync(bodyMask, root);
		GpuBodyState& b = bodies[lane];
		const int state = b.activationState;
		const unsigned awake = __ballot_sync(bodyMask,
			state == CU_ACTIVE_TAG || state == CU_DISABLE_DEACTIVATION);
		newState = state;
		if ((awake & island) == 0) {
			CuSetActivationState(newState, CU_ISLAND_SLEEPING);
		} else if (state == CU_ISLAND_SLEEPING) {
			CuSetActivationState(newState, CU_WANTS_DEACTIVATION);
			b.deactivationTime = 0.0f;
		}
		if (newState != state) {
			b.activationState = newState;
		}
		inBatch = (__ballot_sync(bodyMask, CuIsActive(newState)) & island) != 0;
	}
	const unsigned batchBits = (__ballot_sync(groupMask, inBatch) >> groupBase) & laneBits;
	if (batchBits == 0) {
		return;
	}

	int numIslandManifolds = 0;
	int numBatchManifolds = 0;
	int batchPoints = 0;
	bool hasStatic = false;
	for (int base = 0; base < numManifolds; base += lanes) {
		const int i = base + lane;
		const bool valid = i < numManifolds;
		const int bodyA = valid ? manifolds[i].bodyA : 0;
		const int bodyB = valid ? manifolds[i].bodyB : 0;
		const int srcA = groupBase + (bodyA >= 0 ? bodyA : 0);
		const int srcB = groupBase + (bodyB >= 0 ? bodyB : 0);
		const int stateA = __shfl_sync(groupMask, newState, srcA);
		const int stateB = __shfl_sync(groupMask, newState, srcB);
		const int rootA = __shfl_sync(groupMask, root, srcA);
		const int rootB = __shfl_sync(groupMask, root, srcB);
		const bool awakeA = bodyA >= 0 && stateA != CU_ISLAND_SLEEPING;
		const bool awakeB = bodyB >= 0 && stateB != CU_ISLAND_SLEEPING;
		const bool included = valid && (awakeA || awakeB) &&
			CuHasContactResponse(cars, statics, tileNoResponse, bodyA) &&
			CuHasContactResponse(cars, statics, tileNoResponse, bodyB);
		const int key = bodyA >= 0 ? rootA : rootB;
		const bool batched = included && ((batchBits >> key) & 1u);
		if (batched && ((bodyA >= 0 && !((batchBits >> bodyA) & 1u)) ||
			(bodyB >= 0 && !((batchBits >> bodyB) & 1u)))) {
			__trap();
		}
		const int points = batched ? manifolds[i].numContacts : 0;
		const unsigned bits = (__ballot_sync(groupMask, included) >> groupBase) & laneBits;
		if (included) {
			const int pos = numIslandManifolds + __popc(bits & lanesBelow);
			sKey[pos] = key;
			sIdx[pos] = i;
		}
		numIslandManifolds += __popc(bits);
		numBatchManifolds += __popc(__ballot_sync(groupMask, batched));
		batchPoints += __popc(__ballot_sync(groupMask, points & 1)) +
			2 * __popc(__ballot_sync(groupMask, points & 2)) +
			4 * __popc(__ballot_sync(groupMask, points & 4));
		hasStatic = hasStatic || __any_sync(groupMask, batched && (bodyA < 0 || bodyB < 0));
	}

	if (batchPoints == 0) {
		if (inBatch) {
			CuApplyExternalImpulses(bodies[lane], bodyConsts[lane], dt);
		}
		return;
	}

	const int numBatchBodies = __popc(batchBits);
	CuSolveView s;
	CuPlaceSolveStorage(s, scratch, smem, smemBytes - sharedBytes, numBodies,
		batchPoints + (numBatchBodies < batchPoints ? numBatchBodies : batchPoints));
	s.bodies = bodies;
	s.bodyConsts = bodyConsts;
	s.dt = dt;
	s.numSolverBodies = numBatchBodies + (hasStatic ? 1 : 0);
	s.fixedBodyId = hasStatic ? numBatchBodies : -1;
	__syncwarp(groupMask);
	if (lane == 0) {
		CuBuildBatch(sParent, numBodies, batchBits, sKey, sIdx, numIslandManifolds, sBatchBodies);
		for (int i = 0; i < numBodies; i++) {
			s.companion[i] = -1;
			s.special[i] = {};
		}
		for (int k = 0; k < numBatchBodies; k++) {
			s.companion[sBatchBodies[k]] = k;
		}
	}
	__syncwarp(groupMask);
	for (int k = lane; k < s.numSolverBodies; k += lanes) {
		CuInitSolverBody(s, s.solverBodies[k], k < numBatchBodies ? sBatchBodies[k] : -1);
	}
	__syncwarp(groupMask);

	int numRows = 0;
	for (int base = 0; base < 4 * numBatchManifolds; base += lanes) {
		const int f = base + lane;
		const int pointIdx = f & 3;
		int manifoldIdx = 0, idA = 0, idB = 0;
		bool qualifies = false;
		if (f < 4 * numBatchManifolds) {
			manifoldIdx = sIdx[f >> 2];
			const CuManifold& m = manifolds[manifoldIdx];
			if (pointIdx < m.numContacts) {
				idA = m.bodyA >= 0 ? s.companion[m.bodyA] : s.fixedBodyId;
				idB = m.bodyB >= 0 ? s.companion[m.bodyB] : s.fixedBodyId;
				const CuVec3 invA = s.solverBodies[idA].invMass;
				const CuVec3 invB = s.solverBodies[idB].invMass;
				const bool invA0 = invA.x == 0.0f && invA.y == 0.0f && invA.z == 0.0f;
				const bool invB0 = invB.x == 0.0f && invB.y == 0.0f && invB.z == 0.0f;
				qualifies = !(invA0 && invB0) &&
					m.pts[pointIdx].distance1 <= m.contactProcessingThreshold;
			}
		}
		const unsigned bits = (__ballot_sync(groupMask, qualifies) >> groupBase) & laneBits;
		if (qualifies) {
			CuSetupPointRows(s, manifolds, manifoldIdx, pointIdx,
				numRows + __popc(bits & lanesBelow), idA, idB);
		}
		numRows += __popc(bits);
	}
	__syncwarp(groupMask);

	if (lane == 0) {
		s.numContactRows = numRows;
		s.numFrictionRows = numRows;
		for (int r = 0; r < numRows; r++) {
			CuWarmstartContactRow(s, s.contactRows[r]);
		}
		for (int r = 0; r < numRows; r++) {
			if (s.contactRows[r].isSpecial) {
				CuAccumulateSpecial(s, manifolds, s.contactRows[r]);
			}
		}
		for (int i = 0; i < numBatchBodies; i++) {
			const int slot = sBatchBodies[i];
			if (s.special[slot].num > 0) {
				CuConvertContactSpecial(s, slot);
			}
		}
		CuSolveIterations(s);
	}
	__syncwarp(groupMask);
	for (int k = lane; k < numBatchBodies; k += lanes) {
		CuWritebackSolverBody(s.solverBodies[k], bodies, bodyConsts, dt);
	}
}

}
