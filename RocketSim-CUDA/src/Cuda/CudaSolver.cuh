#pragma once
// 2026-06-10: Device port of the RS solve phase: calculateSimulationIslands +
// solveConstraints (btSimulationIslandManager + InplaceSolverIslandCallback +
// btSequentialImpulseConstraintSolver, RS-modified). Bit-exact replication of the
// Linux GCC build: the _sse2 resolve variants (BT_ALLOW_SSE4 is MSVC-only) with
// their x+(y+z) dot trees; solverInfo: numIterations 10, sor 1, erp2 0.8 (Arena),
// splitImpulsePenetrationThreshold 1e30 (Arena -> rhs/rhsPenetration always split),
// warmstarting on (no-op: manifolds are rebuilt each tick), 1 friction row/contact.
// Spec: Persistent_Memory.md solver notes. Consumes the manifolds + pairs produced
// by CuCollideTick in the same kernel run.

#include "GpuTypes.h"
#include "CudaMath.cuh"
#include "CudaCollision.cuh"
#include "CudaCollideWorld.cuh" // CuIsActive, activation constants

namespace rscuda {

#define CU_SOLVER_ITERATIONS 10
#define CU_ERP2 0.8f
#define CU_SPLIT_TURN_ERP 0.1f
#define CU_RESTITUTION_VEL_THRESH 0.2f
#define CU_WARMSTART_FACTOR 0.85f
#define CU_MAX_SOLVER_BODIES 20   // dyn bodies + fixed
#define CU_MAX_CONTACT_ROWS (GPU_MAX_MANIFOLDS * 4 + CU_MAX_SOLVER_BODIES)
#define CU_MAX_FRICTION_ROWS CU_MAX_CONTACT_ROWS

struct CuSolverBody {
	CuMat3 basis;            // m_worldTransform
	CuVec3 origin;
	CuVec3 invMass;          // invMass * linearFactor(1)
	CuVec3 linVel, angVel;
	CuVec3 extForceImpulse, extTorqueImpulse;
	CuVec3 deltaLin, deltaAng;
	CuVec3 pushVel, turnVel;
	int bodySlot;            // dyn slot, or -1 for the fixed body
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
	int frictionIndex;       // contact row <-> friction row linkage
	uint8_t isSpecial;
};

// RS btCollisionObject::btSpecialResolveInfo accumulator (per dyn body)
struct CuSpecialInfo {
	int num;
	float friction, restitution;
	CuVec3 totalNormal;
	float totalDist;
};

// btSimdDot3: m = a*b; m0 + (m1 + m2)  [x + (y + z) - SSE splat-add order]
__device__ __forceinline__ float CuSimdDot3(CuVec3 a, CuVec3 b) {
	return a.x * b.x + (a.y * b.y + a.z * b.z);
}

// gResolveSingleConstraintRowGeneric_sse2 (scalarized; masks -> selects).
// cpApplied is CACHED AT ENTRY: both clamp deltas subtract the ORIGINAL applied
// impulse (the SSE code keeps cpAppliedImp in a register).
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

// gResolveSingleConstraintRowLowerLimit_sse2
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

// gResolveSplitPenetrationImpulse_sse2 (lower clamp only; exact != 0 early-out)
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

// ---- per-arena solve state ----
struct CuSolveCtx {
	GpuBodyState* bodies;
	const GpuBodyConsts* bodyConsts;
	GpuCarGameState* cars;     // hasContactResponse: demoed cars say no
	int numBodies;
	float dt;
	// solver pools
	CuSolverBody solverBodies[CU_MAX_SOLVER_BODIES];
	int numSolverBodies;
	int fixedBodyId;
	int companion[20];         // per dyn slot -> solver body id (-1 none)
	CuSolverRow contactRows[CU_MAX_CONTACT_ROWS];
	int numContactRows;
	CuSolverRow frictionRows[CU_MAX_FRICTION_ROWS];
	int numFrictionRows;
	CuSpecialInfo special[20]; // per dyn slot
};

__device__ inline void CuInitSolverBody(CuSolveCtx& s, CuSolverBody& sb, int slot) {
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
		// m_externalForceImpulse = totalForce * invMass * dt;
		// m_externalTorqueImpulse = totalTorque * invInertiaWorld * dt - that is
		// btVector3 * btMatrix3x3 = COLUMN dots (invInertiaWorld is not bitwise
		// symmetric, so the order matters)
		sb.extForceImpulse = (GV(b.totalForce) * c.invMass) * s.dt;
		CuMat3 invIW = CuLoadBasis(b.invInertiaWorld);
		sb.extTorqueImpulse = CuVecMulMat(GV(b.totalTorque), invIW) * s.dt;
	} else {
		sb.basis = CuIdentity3();
		sb.origin = { 0, 0, 0 };
		sb.invMass = { 0, 0, 0 };
		sb.bodySlot = -1;
		sb.linVel = { 0, 0, 0 };
		sb.angVel = { 0, 0, 0 };
		sb.extForceImpulse = { 0, 0, 0 };
		sb.extTorqueImpulse = { 0, 0, 0 };
	}
}

// getOrInitSolverBody: encoded body ref (>= 0 dyn slot, < 0 static -> fixed body)
__device__ inline int CuGetOrInitSolverBody(CuSolveCtx& s, int bodyRef) {
	if (bodyRef >= 0) {
		if (s.companion[bodyRef] >= 0)
			return s.companion[bodyRef];
		// rb && invMass != 0 -> own solver body (all our dyns have invMass > 0)
		int id = s.numSolverBodies++;
		CuInitSolverBody(s, s.solverBodies[id], bodyRef);
		s.companion[bodyRef] = id;
		return id;
	}
	if (s.fixedBodyId < 0) {
		s.fixedBodyId = s.numSolverBodies++;
		CuInitSolverBody(s, s.solverBodies[s.fixedBodyId], -1);
	}
	return s.fixedBodyId;
}

// btSequentialImpulseConstraintSolver::setupContactConstraint (RS solverInfo values)
__device__ inline void CuSetupContactRow(CuSolveCtx& s, CuSolverRow& row,
	int idA, int idB, const CuManifoldPoint& cp, float& relaxation,
	CuVec3 rel_pos1, CuVec3 rel_pos2) {

	CuSolverBody& bodyA = s.solverBodies[idA];
	CuSolverBody& bodyB = s.solverBodies[idB];
	bool rb0 = bodyA.bodySlot >= 0;
	bool rb1 = bodyB.bodySlot >= 0;
	relaxation = 1.0f;                     // infoGlobal.m_sor
	float invTimeStep = 1.0f / s.dt;
	float cfm = 0.0f;                      // m_globalCfm
	float erp = CU_ERP2;
	cfm *= invTimeStep;

	CuVec3 n = cp.normalWorldOnB;
	CuVec3 torqueAxis0 = CuCross(rel_pos1, n);
	CuMat3 invIWA = rb0 ? CuLoadBasis(s.bodies[bodyA.bodySlot].invInertiaWorld) : CuIdentity3();
	CuMat3 invIWB = rb1 ? CuLoadBasis(s.bodies[bodyB.bodySlot].invInertiaWorld) : CuIdentity3();
	row.angularComponentA = rb0 ? CuMat3MulVec(invIWA, torqueAxis0) : CuVec3{ 0, 0, 0 };
	CuVec3 torqueAxis1 = CuCross(rel_pos2, n);
	row.angularComponentB = rb1 ? CuMat3MulVec(invIWB, CuNeg(torqueAxis1)) : CuVec3{ 0, 0, 0 };

	float denom0 = 0.0f, denom1 = 0.0f;
	if (rb0) {
		CuVec3 vec = CuCross(row.angularComponentA, rel_pos1);
		denom0 = s.bodyConsts[bodyA.bodySlot].invMass + CuDot3(n, vec);
	}
	if (rb1) {
		CuVec3 vec = CuCross(CuNeg(row.angularComponentB), rel_pos2);
		denom1 = s.bodyConsts[bodyB.bodySlot].invMass + CuDot3(n, vec);
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
	float penetration = cp.distance1 + 0.0f; // linearSlop = 0
	{
		// rb->getVelocityInLocalPoint: REAL body velocities
		CuVec3 vel1 = rb0 ? bodyA.linVel + CuCross(bodyA.angVel, rel_pos1) : CuVec3{ 0, 0, 0 };
		CuVec3 vel2 = rb1 ? bodyB.linVel + CuCross(bodyB.angVel, rel_pos2) : CuVec3{ 0, 0, 0 };
		CuVec3 vel = vel1 - vel2;
		float rel_vel = CuDot3(n, vel);
		row.friction = cp.combinedFriction;
		// restitutionCurve
		float rest = fabsf(rel_vel) < CU_RESTITUTION_VEL_THRESH ? 0.0f
			: cp.combinedRestitution * -rel_vel;
		restitution = rest <= 0.0f ? 0.0f : rest;
	}

	// SOLVER_USE_WARMSTARTING: appliedImpulse = cp.appliedImpulse (0, fresh
	// manifolds) * 0.85 -> internalApplyImpulse with magnitude 0 (replicated)
	row.appliedImpulse = cp.appliedImpulse * CU_WARMSTART_FACTOR;
	if (rb0) {
		bodyA.deltaLin = bodyA.deltaLin + (row.contactNormal1 * bodyA.invMass) * row.appliedImpulse;
		bodyA.deltaAng = bodyA.deltaAng + row.angularComponentA * row.appliedImpulse;
	}
	if (rb1) {
		bodyB.deltaLin = bodyB.deltaLin + (CuNeg(row.contactNormal2) * bodyB.invMass) * (-row.appliedImpulse);
		bodyB.deltaAng = bodyB.deltaAng + CuNeg(row.angularComponentB) * (-row.appliedImpulse);
	}
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
		// Arena: splitImpulsePenetrationThreshold = 1e30 -> always the split branch
		row.rhs = velocityImpulse;
		row.rhsPenetration = penetrationImpulse;
		row.cfm = cfm * row.jacDiagABInv;
		row.lowerLimit = 0.0f;
		row.upperLimit = 1e10f;
	}
	row.isSpecial = cp.isSpecial;
}

// setupFrictionConstraint (desiredVelocity = 0, cfmSlip = 0)
__device__ inline void CuSetupFrictionRow(CuSolveCtx& s, CuSolverRow& row,
	CuVec3 normalAxis, int idA, int idB, const CuManifoldPoint& cp,
	CuVec3 rel_pos1, CuVec3 rel_pos2, float relaxation) {

	CuSolverBody& bodyA = s.solverBodies[idA];
	CuSolverBody& bodyB = s.solverBodies[idB];
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
		CuMat3 invIWA = CuLoadBasis(s.bodies[bodyA.bodySlot].invInertiaWorld);
		row.angularComponentA = CuMat3MulVec(invIWA, ftorqueAxis1);
	} else {
		row.contactNormal1 = { 0, 0, 0 };
		row.relpos1CrossNormal = { 0, 0, 0 };
		row.angularComponentA = { 0, 0, 0 };
	}
	if (rb1) {
		row.contactNormal2 = CuNeg(normalAxis);
		CuVec3 ftorqueAxis1 = CuCross(rel_pos2, row.contactNormal2);
		row.relpos2CrossNormal = ftorqueAxis1;
		CuMat3 invIWB = CuLoadBasis(s.bodies[bodyB.bodySlot].invInertiaWorld);
		row.angularComponentB = CuMat3MulVec(invIWB, ftorqueAxis1);
	} else {
		row.contactNormal2 = { 0, 0, 0 };
		row.relpos2CrossNormal = { 0, 0, 0 };
		row.angularComponentB = { 0, 0, 0 };
	}
	{
		float denom0 = 0.0f, denom1 = 0.0f;
		if (rb0) {
			CuVec3 vec = CuCross(row.angularComponentA, rel_pos1);
			denom0 = s.bodyConsts[bodyA.bodySlot].invMass + CuDot3(normalAxis, vec);
		}
		if (rb1) {
			CuVec3 vec = CuCross(CuNeg(row.angularComponentB), rel_pos2);
			denom1 = s.bodyConsts[bodyB.bodySlot].invMass + CuDot3(normalAxis, vec);
		}
		row.jacDiagABInv = relaxation / (denom0 + denom1);
	}
	{
		float vel1Dotn = CuDot3(row.contactNormal1, rb0 ? bodyA.linVel + bodyA.extForceImpulse : CuVec3{ 0, 0, 0 }) +
			CuDot3(row.relpos1CrossNormal, rb0 ? bodyA.angVel : CuVec3{ 0, 0, 0 });
		float vel2Dotn = CuDot3(row.contactNormal2, rb1 ? bodyB.linVel + bodyB.extForceImpulse : CuVec3{ 0, 0, 0 }) +
			CuDot3(row.relpos2CrossNormal, rb1 ? bodyB.angVel : CuVec3{ 0, 0, 0 });
		float rel_vel = vel1Dotn + vel2Dotn;
		float velocityError = 0.0f - rel_vel; // desiredVelocity = 0
		float velocityImpulse = velocityError * row.jacDiagABInv;
		row.rhs = 0.0f + velocityImpulse;     // no friction anchor
		row.rhsPenetration = 0.0f;
		row.cfm = 0.0f;                       // cfmSlip
		row.lowerLimit = -row.friction;
		row.upperLimit = row.friction;
	}
}

// convertContactInner: friction direction from relative velocity + 1 friction row
__device__ inline void CuConvertContactInner(CuSolveCtx& s, const CuManifoldPoint& cpIn,
	CuSolverRow& contactRow, int idA, int idB, CuVec3 rel_pos1, CuVec3 rel_pos2,
	float relaxation) {

	// no rolling friction (combinedRollingFriction == 0)
	CuSolverBody& bodyA = s.solverBodies[idA];
	CuSolverBody& bodyB = s.solverBodies[idB];
	// getVelocityInLocalPointNoDelta: (linVel + extForce) + (angVel + extTorque) x rel
	CuVec3 vel1 = bodyA.bodySlot >= 0
		? (bodyA.linVel + bodyA.extForceImpulse) + CuCross(bodyA.angVel + bodyA.extTorqueImpulse, rel_pos1)
		: CuVec3{ 0, 0, 0 };
	CuVec3 vel2 = bodyB.bodySlot >= 0
		? (bodyB.linVel + bodyB.extForceImpulse) + CuCross(bodyB.angVel + bodyB.extTorqueImpulse, rel_pos2)
		: CuVec3{ 0, 0, 0 };
	CuVec3 vel = vel1 - vel2;
	CuVec3 n = cpIn.normalWorldOnB;
	float rel_vel = CuDot3(n, vel);

	// (no friction-direction caching flags set -> always recompute)
	CuVec3 lateralDir1 = vel - n * rel_vel;
	float lat_rel_vel = CuLength2(lateralDir1);
	CuSolverRow& fr = s.frictionRows[s.numFrictionRows];
	int frictionIndex = (int)(&contactRow - s.contactRows); // contact pool index
	if (lat_rel_vel > CU_SIMD_EPSILON) {
		lateralDir1 = lateralDir1 * (1.0f / sqrtf(lat_rel_vel)); // btSqrt
		CuSetupFrictionRow(s, fr, lateralDir1, idA, idB, cpIn, rel_pos1, rel_pos2, relaxation);
	} else {
		CuVec3 d1, d2;
		CuPlaneSpace1(n, d1, d2);
		CuSetupFrictionRow(s, fr, d1, idA, idB, cpIn, rel_pos1, rel_pos2, relaxation);
	}
	fr.frictionIndex = frictionIndex;
	s.numFrictionRows++;
	// setFrictionConstraintImpulse: frictionRows[contact.frictionIndex].applied = 0
	// (already 0; the special-contact stale-index write is value-benign - skipped)
}

// convertContact for one manifold (bodies encoded as in CuManifold)
__device__ inline void CuConvertContact(CuSolveCtx& s, CuManifold& m) {
	int idA = CuGetOrInitSolverBody(s, m.bodyA >= 0 ? m.bodyA : -1);
	int idB = CuGetOrInitSolverBody(s, m.bodyB >= 0 ? m.bodyB : -1);
	CuSolverBody& bodyA = s.solverBodies[idA];
	CuSolverBody& bodyB = s.solverBodies[idB];
	bool invA0 = bodyA.invMass.x == 0.0f && bodyA.invMass.y == 0.0f && bodyA.invMass.z == 0.0f;
	bool invB0 = bodyB.invMass.x == 0.0f && bodyB.invMass.y == 0.0f && bodyB.invMass.z == 0.0f;
	if (invA0 && invB0)
		return; // fuzzyZero both
	for (int j = 0; j < m.numContacts; j++) {
		CuManifoldPoint& cp = m.pts[j];
		if (cp.distance1 <= m.contactProcessingThreshold) {
			CuSolverRow& row = s.contactRows[s.numContactRows++];
			row.bodyIdA = idA;
			row.bodyIdB = idB;
			row.frictionIndex = s.numFrictionRows;
			CuVec3 rel_pos1 = cp.positionWorldOnA - bodyA.origin;
			CuVec3 rel_pos2 = cp.positionWorldOnB - bodyB.origin;
			float relaxation;
			CuSetupContactRow(s, row, idA, idB, cp, relaxation, rel_pos1, rel_pos2);

			if (cp.isSpecial) {
				// accumulate into NON-static bodies' specialResolveInfo
				for (int i = 0; i < 2; i++) {
					int slot = i ? m.bodyB : m.bodyA;
					if (slot >= 0) {
						CuSpecialInfo& spri = s.special[slot];
						spri.num++;
						spri.friction = cp.combinedFriction;
						spri.restitution = cp.combinedRestitution;
						spri.totalNormal = spri.totalNormal + cp.normalWorldOnB;
						// (i ? rel_pos2 : rel_pos1).length() - btVector3::length =
						// sqrt(dot3) (plain sqrtf on SSE dot tree)
						CuVec3 rp = i ? rel_pos2 : rel_pos1;
						spri.totalDist += sqrtf(CuDot3(rp, rp));
					}
				}
			}
			CuConvertContactInner(s, cp, row, idA, idB, rel_pos1, rel_pos2, relaxation);
		}
	}
}

// convertContactSpecial: synthetic averaged contact vs the fixed body
__device__ inline void CuConvertContactSpecial(CuSolveCtx& s, int slot) {
	CuSpecialInfo& info = s.special[slot];
	float distance = info.totalDist / (float)info.num;
	// btVector3 operator/(scalar) under SSE = v * (1/s)
	CuVec3 normal = info.totalNormal * (1.0f / (float)info.num);
	CuManifoldPoint tmp = {};
	tmp.distance1 = distance;
	tmp.normalWorldOnB = normal;
	tmp.combinedFriction = info.friction;
	tmp.combinedRestitution = info.restitution;
	int idA = CuGetOrInitSolverBody(s, slot);
	if (s.fixedBodyId < 0) {
		s.fixedBodyId = s.numSolverBodies++;
		CuInitSolverBody(s, s.solverBodies[s.fixedBodyId], -1);
	}
	int idB = s.fixedBodyId;
	CuSolverRow& row = s.contactRows[s.numContactRows];
	int frictionIndex = s.numContactRows;
	s.numContactRows++;
	row.bodyIdA = idA;
	row.bodyIdB = idB;
	CuVec3 rel_pos1 = normal * -distance;
	CuVec3 rel_pos2 = { 0, 0, 0 };
	float relaxation = 0;
	CuSetupContactRow(s, row, idA, idB, tmp, relaxation, rel_pos1, rel_pos2);
	row.frictionIndex = s.numFrictionRows; // NOTE: CPU leaves this STALE; see below
	(void)frictionIndex;
	CuConvertContactInner(s, tmp, row, idA, idB, rel_pos1, rel_pos2, relaxation);
}

// ---- islands (btUnionFind path-compressed + STATIC_SIMULATION_ISLAND_OPT) ----
struct CuUnionFind {
	int id[20], sz[20];
	int n;
	__device__ void reset(int N) {
		n = N;
		for (int i = 0; i < N; i++) { id[i] = i; sz[i] = 1; }
	}
	__device__ int find(int x) {
		while (x != id[x]) {
			id[x] = id[id[x]]; // path compression (USE_PATH_COMPRESSION)
			x = id[x];
		}
		return x;
	}
	__device__ void unite(int p, int q) {
		int i = find(p), j = find(q);
		if (i == j) return;
		id[i] = j;
		sz[j] += sz[i];
	}
};

// btAlignedObjectArray::quickSortInternal ((lo+hi)/2 pivot COPY, Hoare-ish):
// keys[] sorted; vals[] permuted alongside (the payload).
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

// The full solve phase for one arena (single thread). Mirrors
// calculateSimulationIslands + solveConstraints with the batching callback.
// scratch: per-arena CuSolveCtx in GLOBAL memory (too large for the device stack).
__device__ inline void CuSolveTick(GpuBodyState* bodies, const GpuBodyConsts* bodyConsts,
	GpuCarGameState* cars, int numBodies, int numStatics,
	CuManifold* manifolds, int numManifolds, const int2* pairs, int numPairs,
	float dt, CuSolveCtx* scratch) {

	// --- calculateSimulationIslands: tags + unions over the PAIR array ---
	CuUnionFind uf;
	uf.reset(numBodies); // island tag i for dyn slot i (statics: -1, skipped)
	for (int p = 0; p < numPairs; p++) {
		// mergesSimulationIslands: both non-static
		if (pairs[p].x >= numStatics && pairs[p].y >= numStatics)
			uf.unite(pairs[p].x - numStatics, pairs[p].y - numStatics);
	}
	// storeIslandActivationState: tag = find(i) (sets path compression state)
	int tag[20];
	for (int i = 0; i < numBodies; i++)
		tag[i] = uf.find(i);

	// --- buildIslands ---
	// sortIslands: id = find(i) again (no further compression effect), then
	// quickSort elements by id; m_sz (payload) = collisionObjects index, but for
	// processing we keep the dyn slot directly.
	int sortedIds[20], sortedSlots[20];
	for (int i = 0; i < numBodies; i++) {
		sortedIds[i] = uf.find(i);
		sortedSlots[i] = i;
	}
	if (numBodies > 1)
		CuQuickSortPairs(sortedIds, sortedSlots, 0, numBodies - 1);

	// per island: sleeping management
	for (int start = 0; start < numBodies;) {
		int islandId = sortedIds[start];
		int end = start + 1;
		while (end < numBodies && sortedIds[end] == islandId)
			end++;
		bool allSleeping = true;
		for (int k = start; k < end; k++) {
			int st = bodies[sortedSlots[k]].activationState;
			if (st == CU_ACTIVE_TAG || st == CU_DISABLE_DEACTIVATION) {
				allSleeping = false;
				break;
			}
		}
		if (allSleeping) {
			for (int k = start; k < end; k++)
				CuSetActivationState(bodies[sortedSlots[k]].activationState, CU_ISLAND_SLEEPING);
		} else {
			for (int k = start; k < end; k++) {
				GpuBodyState& b = bodies[sortedSlots[k]];
				if (b.activationState == CU_ISLAND_SLEEPING) {
					CuSetActivationState(b.activationState, CU_WANTS_DEACTIVATION);
					b.deactivationTime = 0.0f;
				}
			}
		}
		start = end;
	}

	// m_islandmanifold: dispatcher order, either body awake, needsResponse
	auto hasResponse = [&](int bodyRef) -> bool {
		if (bodyRef < 0) return true;               // statics
		if (bodyRef >= 1 && cars[bodyRef - 1].isDemoed) return false;
		return true;
	};
	auto bodyActivation = [&](int bodyRef) -> int {
		// statics are ISLAND_SLEEPING
		return bodyRef >= 0 ? bodies[bodyRef].activationState : CU_ISLAND_SLEEPING;
	};
	auto manifoldIslandId = [&](const CuManifold& m) -> int {
		int t0 = m.bodyA >= 0 ? tag[m.bodyA] : -1;
		return t0 >= 0 ? t0 : (m.bodyB >= 0 ? tag[m.bodyB] : -1);
	};
	int mfIdx[GPU_MAX_MANIFOLDS], mfKey[GPU_MAX_MANIFOLDS];
	int numIslandManifolds = 0;
	for (int i = 0; i < numManifolds; i++) {
		const CuManifold& m = manifolds[i];
		if (bodyActivation(m.bodyA) != CU_ISLAND_SLEEPING ||
			bodyActivation(m.bodyB) != CU_ISLAND_SLEEPING) {
			// (kinematic-activate block: no kinematics in RS)
			if (hasResponse(m.bodyA) && hasResponse(m.bodyB)) {
				mfKey[numIslandManifolds] = manifoldIslandId(m);
				mfIdx[numIslandManifolds] = i;
				numIslandManifolds++;
			}
		}
	}
	// processIslands: quickSort manifolds by islandId
	if (numIslandManifolds > 1)
		CuQuickSortPairs(mfKey, mfIdx, 0, numIslandManifolds - 1);

	// batch accumulation (minimumSolverBatchSize 128 never reached at our scale ->
	// exactly one solveGroup at the end, in island iteration order)
	int batchBodies[20];
	int numBatchBodies = 0;
	int batchManifolds[GPU_MAX_MANIFOLDS];
	int numBatchManifolds = 0;
	{
		int startManifoldIndex = 0, endManifoldIndex = 1;
		for (int start = 0; start < numBodies;) {
			int islandId = sortedIds[start];
			int end = start;
			bool islandSleeping = true;
			for (; end < numBodies && sortedIds[end] == islandId; end++) {
				if (CuIsActive(bodies[sortedSlots[end]].activationState))
					islandSleeping = false;
			}
			int numCur = 0, startMf = -1;
			if (startManifoldIndex < numIslandManifolds) {
				if (mfKey[startManifoldIndex] == islandId) {
					startMf = startManifoldIndex;
					for (endManifoldIndex = startManifoldIndex + 1;
						endManifoldIndex < numIslandManifolds && islandId == mfKey[endManifoldIndex];
						endManifoldIndex++) {}
					numCur = endManifoldIndex - startManifoldIndex;
				}
			}
			if (!islandSleeping) {
				for (int k = start; k < end; k++)
					batchBodies[numBatchBodies++] = sortedSlots[k];
				for (int k = 0; k < numCur; k++)
					batchManifolds[numBatchManifolds++] = mfIdx[startMf + k];
			}
			if (numCur)
				startManifoldIndex = endManifoldIndex;
			start = end;
		}
	}
	if (numBatchBodies == 0)
		return; // no processConstraints call with an empty batch (solveGroup guards)

	// --- solveGroup(batchBodies, batchManifolds) ---
	CuSolveCtx& s = *scratch;
	s.bodies = bodies;
	s.bodyConsts = bodyConsts;
	s.cars = cars;
	s.numBodies = numBodies;
	s.dt = dt;
	s.numSolverBodies = 0;
	s.fixedBodyId = -1;
	s.numContactRows = 0;
	s.numFrictionRows = 0;
	for (int i = 0; i < numBodies; i++) {
		s.companion[i] = -1;
		s.special[i] = {};
	}

	// Setup: convertBodies (batch order; no gyro flags in RS)
	for (int i = 0; i < numBatchBodies; i++)
		CuGetOrInitSolverBody(s, batchBodies[i]);
	// convertContacts (batched manifold order)
	for (int i = 0; i < numBatchManifolds; i++)
		CuConvertContact(s, manifolds[batchManifolds[i]]);
	// convertContactSpecial per batch body with accumulated specials
	for (int i = 0; i < numBatchBodies; i++) {
		int slot = batchBodies[i];
		if (s.special[slot].num > 0) {
			CuConvertContactSpecial(s, slot);
			s.special[slot] = {};
		}
	}

	// Iterations: split impulse first (early-break on residual^2max <= 0)
	for (int iteration = 0; iteration < CU_SOLVER_ITERATIONS; iteration++) {
		float leastSquaresResidual = 0.0f;
		for (int j = 0; j < s.numContactRows; j++) {
			CuSolverRow& row = s.contactRows[j];
			float residual = CuResolveSplitPenetration(
				s.solverBodies[row.bodyIdA], s.solverBodies[row.bodyIdB], row);
			float r2 = residual * residual;
			leastSquaresResidual = leastSquaresResidual > r2 ? leastSquaresResidual : r2;
		}
		if (leastSquaresResidual <= 0.0f || iteration >= (CU_SOLVER_ITERATIONS - 1))
			break;
	}
	// main iterations (no early exit)
	for (int iteration = 0; iteration < CU_SOLVER_ITERATIONS; iteration++) {
		// contacts in order, skipping RS special rows
		for (int j = 0; j < s.numContactRows; j++) {
			CuSolverRow& row = s.contactRows[j];
			if (row.isSpecial)
				continue;
			CuResolveRowLowerLimit(s.solverBodies[row.bodyIdA], s.solverBodies[row.bodyIdB], row);
		}
		// friction rows in order (limits from the owning contact's applied impulse)
		for (int j = 0; j < s.numFrictionRows; j++) {
			CuSolverRow& fr = s.frictionRows[j];
			float totalImpulse = s.contactRows[fr.frictionIndex].appliedImpulse;
			if (totalImpulse > 0.0f) {
				fr.lowerLimit = -(fr.friction * totalImpulse);
				fr.upperLimit = fr.friction * totalImpulse;
				CuResolveRowGeneric(s.solverBodies[fr.bodyIdA], s.solverBodies[fr.bodyIdB], fr);
			}
		}
	}

	// Finish: writeBackBodies (warmstart contact writeback touches manifold points
	// only - they are rebuilt next tick, no sim effect -> skipped)
	for (int i = 0; i < s.numSolverBodies; i++) {
		CuSolverBody& sb = s.solverBodies[i];
		if (sb.bodySlot < 0)
			continue;
		GpuBodyState& b = bodies[sb.bodySlot];
		// writebackVelocityAndTransform(dt, splitImpulseTurnErp)
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
		// splitImpulse: setWorldTransform (NO inertia update - integrate phase does)
		for (int r = 0; r < 3; r++)
			b.basis.row[r] = { sb.basis.r[r].x, sb.basis.r[r].y, sb.basis.r[r].z };
		b.pos = { sb.origin.x, sb.origin.y, sb.origin.z };
	}
}

} // namespace rscuda
