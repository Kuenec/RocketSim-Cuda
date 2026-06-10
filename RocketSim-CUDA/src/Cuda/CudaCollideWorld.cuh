#pragma once
// 2026-06-10: Device collision-phase orchestrator: bit-exact replication of
// performDiscreteCollisionDetection for the RocketSim soccar world.
//   updateAabbs (RS quirks: forceUpdateAllAabbs=true but statics no-op via the
//   bitwise-change guard; dyn aabb = shape aabb at worldTransform UNION at the
//   freshly-recomputed interpolation transform, each +-gContactBreakingThreshold)
//   -> btRSBroadphase::calculateOverlappingPairs (pairs fully rebuilt every tick;
//   per-cell dyn list order = ascending re-registration sequence)
//   -> btCollisionDispatcher::dispatchAllCollisionPairs with the btDefault
//   CollisionConfiguration algorithm matrix (manifold creation/release order
//   replicated exactly, including ctor-created empty manifolds and the compound
//   post-loop releases) + Arena::_BulletContactAddedCallback game effects.
// Spec: Persistent_Memory.md "COLLISION ORCHESTRATION SPEC". Statics are
// ISLAND_SLEEPING; demolitions are not supported on device (gates: DemoMode
// DISABLED; a demolition would remove a broadphase proxy, which this port does
// not model).

#include "GpuTypes.h"
#include "CudaMath.cuh"
#include "CudaCollision.cuh"
#include "CudaRaycast.cuh"   // GpuWorldGeo, CuGetCellIndices
#include "CudaVehicle.cuh"   // CuCurveOutput, CuVecNormalizedRS, CuGetColumn

namespace rscuda {

// Bullet activation state constants (btCollisionObject.h)
#define CU_ACTIVE_TAG 1
#define CU_ISLAND_SLEEPING 2
#define CU_WANTS_DEACTIVATION 3
#define CU_DISABLE_DEACTIVATION 4
#define CU_DISABLE_SIMULATION 5
#define CU_DEACTIVATION_TIME 2.0f // gDeactivationTime global default

__device__ __forceinline__ bool CuIsActive(int state) {
	// btCollisionObject::isActive(): != FIXED_BASE_MULTI_BODY(6) && != ISLAND_SLEEPING && != DISABLE_SIMULATION
	return state != 6 && state != CU_ISLAND_SLEEPING && state != CU_DISABLE_SIMULATION;
}

#define CU_BT_LARGE_FLOAT 1e18f
#define CU_CONTACT_BREAKING 0.02f  // gContactBreakingThreshold
#define CU_UU_TO_BT (1.f / 50.f)
#define CU_BT_TO_UU 50.f
#define CU_USERIDX_CAR 1
#define CU_USERIDX_BALL 2
#define CU_GAMEMODE_SOCCAR 0
#define CU_DEMOMODE_NORMAL 0      // DemoMode::NORMAL
#define CU_DEMOMODE_ON_CONTACT 1  // DemoMode::ON_CONTACT
#define CU_DEMOMODE_DISABLED 2    // DemoMode::DISABLED

// All device inputs/outputs of one arena's collide phase, bundled.
struct CuCollideIn {
	GpuBodyState* bodies;            // [1 + numCars], post-predict
	const GpuBodyConsts* bodyConsts;
	int numBodies;
	GpuCarGameState* cars;           // [numCars]
	GpuBallGameState* ball;
	GpuDynProxy* proxies;            // [numBodies] persistent
	int* regCounter;                 // persistent per arena
	const GpuStaticObj* statics;
	int numStatics;
	const int* cellStaticStart;      // CSR [numCells+1]
	const int* cellStaticList;
	GpuGridParams grid;
	GpuWorldGeo geo;
	const GpuMeshBvh* bvhs;
	const GpuBvhNode* bvhNodes;
	const GpuBvhSubtree* bvhSubtrees;
	const GpuTriInfo* triInfos;      // [total tris], indexed by mesh triStart + tri
	const float* meshMargins;
	GpuCollideConsts cc;
	uint64_t tickCount;
	float dt;
	// outputs (per arena)
	CuManifold* manifolds;           // capacity GPU_MAX_MANIFOLDS
	int* numManifolds;
	// diagnostic channel: [0]=numPairs, then (h0,h1) per pair (h < numStatics: static)
	int* debug;                      // capacity 2*CU_MAX_PAIRS+1
};

// btAabbUtil2 TestAabbAgainstAabb2 (x, then z, then y)
__device__ __forceinline__ bool CuTestAabb2(CuVec3 min1, CuVec3 max1, CuVec3 min2, CuVec3 max2) {
	bool overlap = true;
	overlap = (min1.x > max2.x || max1.x < min2.x) ? false : overlap;
	overlap = (min1.z > max2.z || max1.z < min2.z) ? false : overlap;
	overlap = (min1.y > max2.y || max1.y < min2.y) ? false : overlap;
	return overlap;
}

// btHashedOverlappingPairCache::needsBroadphaseCollision (no filter callback in RS)
__device__ __forceinline__ bool CuNeedsBroadphase(int g0, int m0, int g1, int m1) {
	return ((g0 & m1) != 0) && ((g1 & m0) != 0);
}

__device__ __forceinline__ CuMat3 CuLoadBasis(const GpuMat3& m) {
	return { { { m.row[0].x, m.row[0].y, m.row[0].z },
	           { m.row[1].x, m.row[1].y, m.row[1].z },
	           { m.row[2].x, m.row[2].y, m.row[2].z } } };
}

// btSphereShape::getAabb: origin +- margin where margin = radius + 0.08f
// ("ROCKETSIM CHANGE: Add 0.08 to sphere margin" - aabb only, GJK keeps radius)
__device__ __forceinline__ void CuSphereAabbAt(CuVec3 origin, float radius,
	CuVec3& mn, CuVec3& mx) {
	float margin = radius + 0.08f;
	mn = { origin.x - margin, origin.y - margin, origin.z - margin };
	mx = { origin.x + margin, origin.y + margin, origin.z + margin };
}

// btCompoundShape::getAabb: localHalfExtents = 0.5*(localMax-localMin) + margin,
// localCenter = 0.5*(localMax+localMin); center = T*localCenter;
// extent = halfExt dot3 |basis rows|.
__device__ __forceinline__ void CuCompoundAabbAt(const GpuCollideConsts& cc,
	const CuMat3& basis, CuVec3 origin, CuVec3& mn, CuVec3& mx) {
	CuVec3 lmin = GV(cc.carLocalAabbMin), lmax = GV(cc.carLocalAabbMax);
	CuVec3 halfExt = (lmax - lmin) * 0.5f;
	CuVec3 center = (lmax + lmin) * 0.5f;
	halfExt = { halfExt.x + cc.carCompoundMargin, halfExt.y + cc.carCompoundMargin,
		halfExt.z + cc.carCompoundMargin };
	CuVec3 c = CuTransformPoint(basis, origin, center);
	CuVec3 r0 = { fabsf(basis.r[0].x), fabsf(basis.r[0].y), fabsf(basis.r[0].z) };
	CuVec3 r1 = { fabsf(basis.r[1].x), fabsf(basis.r[1].y), fabsf(basis.r[1].z) };
	CuVec3 r2 = { fabsf(basis.r[2].x), fabsf(basis.r[2].y), fabsf(basis.r[2].z) };
	CuVec3 extent = { CuDot3(halfExt, r0), CuDot3(halfExt, r1), CuDot3(halfExt, r2) };
	mn = c - extent;
	mx = c + extent;
}

// Body shape aabb at an arbitrary transform (slot 0 = ball sphere, others = compound)
__device__ __forceinline__ void CuBodyAabbAt(const CuCollideIn& in, int slot,
	const CuMat3& basis, CuVec3 origin, CuVec3& mn, CuVec3& mx) {
	if (slot == 0)
		CuSphereAabbAt(origin, in.cc.ballRadius, mn, mx);
	else
		CuCompoundAabbAt(in.cc, basis, origin, mn, mx);
}

// btBoxShape::getAabb = btTransformAabb(halfExtentsWithoutMargin, margin, t)
__device__ __forceinline__ void CuBoxAabbAt(CuVec3 heNoMargin, float margin,
	const CuMat3& basis, CuVec3 origin, CuVec3& mn, CuVec3& mx) {
	CuVec3 he = { heNoMargin.x + margin, heNoMargin.y + margin, heNoMargin.z + margin };
	CuVec3 r0 = { fabsf(basis.r[0].x), fabsf(basis.r[0].y), fabsf(basis.r[0].z) };
	CuVec3 r1 = { fabsf(basis.r[1].x), fabsf(basis.r[1].y), fabsf(basis.r[1].z) };
	CuVec3 r2 = { fabsf(basis.r[2].x), fabsf(basis.r[2].y), fabsf(basis.r[2].z) };
	CuVec3 extent = { CuDot3(he, r0), CuDot3(he, r1), CuDot3(he, r2) };
	mn = origin - extent;
	mx = origin + extent;
}

// ---- updateAabbs + btRSBroadphase::setAabb (dyn bodies only; statics no-op) ----
__device__ inline void CuCollideUpdateAabbs(const CuCollideIn& in) {
	for (int d = 0; d < in.numBodies; d++) {
		const GpuBodyState& s = in.bodies[d];
		CuMat3 basis = CuLoadBasis(s.basis);
		CuVec3 pos = GV(s.pos);
		CuVec3 mn, mx;
		CuBodyAabbAt(in, d, basis, pos, mn, mx);
		CuVec3 thr = { CU_CONTACT_BREAKING, CU_CONTACT_BREAKING, CU_CONTACT_BREAKING };
		mn = mn - thr;
		mx = mx + thr;

		// m_useContinuous: union with the interpolation-transform aabb. RS predict
		// recomputes interp unconditionally each tick right before this, so it is
		// predictIntegratedTransform(world, current damped vels, dt).
		{
			CuMat3 ib;
			CuVec3 ip;
			CuIntegrateTransform(basis, pos, GV(s.linVel), GV(s.angVel), in.dt,
				in.bodyConsts[d].noRot != 0, ib, ip);
			CuVec3 mn2, mx2;
			CuBodyAabbAt(in, d, ib, ip, mn2, mx2);
			mn2 = mn2 - thr;
			mx2 = mx2 + thr;
			mn = { mn.x < mn2.x ? mn.x : mn2.x, mn.y < mn2.y ? mn.y : mn2.y, mn.z < mn2.z ? mn.z : mn2.z };
			mx = { mx.x > mx2.x ? mx.x : mx2.x, mx.y > mx2.y ? mx.y : mx2.y, mx.z > mx2.z ? mx.z : mx2.z };
		}

		GpuDynProxy& p = in.proxies[d];
		CuVec3 omn = GV(p.aabbMin), omx = GV(p.aabbMax);
		// btVector3 operator!= is float compare (so -0 == +0 stays "unchanged")
		bool changed = omn.x != mn.x || omn.y != mn.y || omn.z != mn.z ||
		               omx.x != mx.x || omx.y != mx.y || omx.z != mx.z;
		if (!changed)
			continue;
		p.aabbMin = { mn.x, mn.y, mn.z };
		p.aabbMax = { mx.x, mx.y, mx.z };
		int oldCell = p.cellIdx;
		int ni, nj, nk;
		CuGetCellIndices(in.grid, mn, ni, nj, nk);
		int newCell = ni * in.grid.cellsY * in.grid.cellsZ + nj * in.grid.cellsZ + nk;
		p.cellIdx = newCell;
		if (oldCell != newCell && in.numBodies > 1) { // numDynProxies > 1
			p.i = ni;
			p.j = nj;
			p.k = nk;
			p.regSeq = ++(*in.regCounter);
		}
	}
}

// ---- btRSBroadphase::calculateOverlappingPairs ----
// Pair = (h0, h1) with h < numStatics static handle, else dyn slot h-numStatics.
// internalAddPair orders by uid = handle order, so h0 < h1 always.
#define CU_MAX_PAIRS 96

__device__ __forceinline__ void CuDynFilter(const CuCollideIn& in, int slot, int& g, int& m) {
	if (slot == 0) { g = in.cc.ballFilterGroup; m = in.cc.ballFilterMask; }
	else { g = in.cc.carFilterGroup; m = in.cc.carFilterMask; }
}

__device__ inline int CuCollideGenPairs(const CuCollideIn& in, int2* pairs) {
	int numPairs = 0;
	const int cellsYZ = in.grid.cellsY * in.grid.cellsZ;
	for (int d = 0; d < in.numBodies; d++) {
		const GpuDynProxy& p = in.proxies[d];
		CuVec3 pmin = GV(p.aabbMin), pmax = GV(p.aabbMax);
		int dg, dm;
		CuDynFilter(in, d, dg, dm);

		// TEMP diagnostic (debug[100+])
		if (in.debug && d == 0) {
			in.debug[100] = p.cellIdx;
			in.debug[101] = in.cellStaticStart[p.cellIdx];
			in.debug[102] = in.cellStaticStart[p.cellIdx + 1];
			in.debug[103] = in.cellStaticStart[p.cellIdx + 1] > in.cellStaticStart[p.cellIdx]
				? in.cellStaticList[in.cellStaticStart[p.cellIdx]] : -1;
			in.debug[104] = dg;
			in.debug[105] = dm;
			in.debug[106] = __float_as_int(pmin.z);
			in.debug[107] = in.numBodies;
		}

		// statics of the aabbMin cell (CSR list order = CPU smear insertion order)
		for (int it = in.cellStaticStart[p.cellIdx]; it < in.cellStaticStart[p.cellIdx + 1]; it++) {
			int s = in.cellStaticList[it];
			const GpuStaticObj& so = in.statics[s];
			if (!CuTestAabb2(pmin, pmax, GV(so.aabbMin), GV(so.aabbMax)))
				continue;
			if (!CuNeedsBroadphase(dg, dm, so.group, so.mask))
				continue;
			// findPair dedup: a static appears at most once per cell, each dyn visits
			// one cell -> no duplicate possible
			if (numPairs < CU_MAX_PAIRS)
				pairs[numPairs++] = { s, in.numStatics + d };
		}

		// dyns of the same cell, in cell-list order = ascending regSeq
		if (in.numBodies > 1) {
			int ci = p.cellIdx / cellsYZ;
			int rem = p.cellIdx % cellsYZ;
			int cj = rem / in.grid.cellsZ;
			int ck = rem % in.grid.cellsZ;
			// gather members (cell.dynHandles): q registered over [q.i-1,q.i+1] clamped
			int members[32];
			int numMembers = 0;
			for (int q = 0; q < in.numBodies && numMembers < 32; q++) {
				const GpuDynProxy& qp = in.proxies[q];
				int mni = qp.i - 1 < 0 ? 0 : qp.i - 1;
				int mxi = qp.i + 1 > in.grid.cellsX - 1 ? in.grid.cellsX - 1 : qp.i + 1;
				int mnj = qp.j - 1 < 0 ? 0 : qp.j - 1;
				int mxj = qp.j + 1 > in.grid.cellsY - 1 ? in.grid.cellsY - 1 : qp.j + 1;
				int mnk = qp.k - 1 < 0 ? 0 : qp.k - 1;
				int mxk = qp.k + 1 > in.grid.cellsZ - 1 ? in.grid.cellsZ - 1 : qp.k + 1;
				if (ci >= mni && ci <= mxi && cj >= mnj && cj <= mxj && ck >= mnk && ck <= mxk)
					members[numMembers++] = q;
			}
			// order by regSeq ascending (stable: regSeqs are unique)
			for (int a = 1; a < numMembers; a++) {
				int v = members[a];
				int b = a - 1;
				while (b >= 0 && in.proxies[members[b]].regSeq > in.proxies[v].regSeq) {
					members[b + 1] = members[b];
					b--;
				}
				members[b + 1] = v;
			}
			if (numMembers > 1) {
				for (int mi = 0; mi < numMembers; mi++) {
					int q = members[mi];
					if (q == d)
						continue;
					const GpuDynProxy& qp = in.proxies[q];
					if (!CuTestAabb2(pmin, pmax, GV(qp.aabbMin), GV(qp.aabbMax)))
						continue;
					int qg, qm;
					CuDynFilter(in, q, qg, qm);
					if (!CuNeedsBroadphase(dg, dm, qg, qm))
						continue;
					int h0 = in.numStatics + (d < q ? d : q);
					int h1 = in.numStatics + (d < q ? q : d);
					bool exists = false;
					for (int e = 0; e < numPairs; e++)
						if (pairs[e].x == h0 && pairs[e].y == h1) { exists = true; break; }
					if (!exists && numPairs < CU_MAX_PAIRS)
						pairs[numPairs++] = { h0, h1 };
				}
			}
		}
	}
	return numPairs;
}

// ---- manifold list with btCollisionDispatcher creation/release order ----
struct CuManifoldList {
	CuManifold* arr;
	int n;
	__device__ int create(int bodyA, int bodyB, float breaking) {
		int idx = n;
		if (n < GPU_MAX_MANIFOLDS) {
			CuManifold& m = arr[n++];
			m.bodyA = bodyA;
			m.bodyB = bodyB;
			m.numContacts = 0;
			m.contactBreakingThreshold = breaking;
			m.contactProcessingThreshold = CU_BT_LARGE_FLOAT;
		}
		return idx;
	}
	// btCollisionDispatcher::releaseManifold: swap-with-last + pop
	__device__ void release(int idx) {
		n--;
		if (idx != n)
			arr[idx] = arr[n];
	}
};

// ---- Arena::_BulletContactAddedCallback (game effects; manifold-order bodies) ----
// objA/objB = manifold body0/body1. Implements the soccar-relevant paths; the
// trailing btAdjustInternalEdgeContacts is applied by the caller (mesh pairs only).
__device__ inline void CuContactAddedCallback(const CuCollideIn& in,
	CuManifoldPoint& cp, int bodyA, int bodyB) {

	// hasContactResponse: demolished cars get CF_NO_CONTACT_RESPONSE
	if (bodyA >= 1 && in.cars[bodyA - 1].isDemoed)
		return;
	if (bodyB >= 1 && in.cars[bodyB - 1].isDemoed)
		return;

	// shouldSwap by userIndex: car(1) < ball(2) < none(-1 world). Manifold body0 is
	// always the car for car-ball/car-world, the ball for ball-world, the higher car
	// for car-car -> shouldSwap is false for every soccar pair shape.
	bool carInvolvedA = bodyA >= 1;
	if (carInvolvedA) {
		int c = bodyA - 1;
		GpuCarGameState& car = in.cars[c];
		if (bodyB == 0) {
			// Car + Ball: friction/restitution overrides + Ball::_OnHit
			cp.combinedFriction = in.cc.carBallFriction;
			cp.combinedRestitution = in.cc.carBallRestitution;

			// Ball::_OnHit(car, relPosOnBall = localPointB * BT_TO_UU, ...)
			const GpuBodyState& carB = in.bodies[1 + c];
			const GpuBodyState& ballB = in.bodies[0];
			// GetState(): live body state in UU (RocketSim Vec math from here on)
			CuVec3 carPos = GV(carB.pos) * CU_BT_TO_UU;
			CuVec3 carVel = GV(carB.linVel) * CU_BT_TO_UU;
			CuVec3 ballPos = GV(ballB.pos) * CU_BT_TO_UU;
			CuVec3 ballVel = GV(ballB.linVel) * CU_BT_TO_UU;

			GpuBallHitInfo& bhi = car.ballHitInfo;
			bhi.isValid = 1;
			CuVec3 relPosOnBall = cp.localPointB * CU_BT_TO_UU;
			bhi.relativePosOnBall = { relPosOnBall.x, relPosOnBall.y, relPosOnBall.z };
			bhi.tickCountWhenHit = in.tickCount;
			bhi.ballPos = { ballPos.x, ballPos.y, ballPos.z };
			bhi.extraHitVel = { 0, 0, 0 };

			if ((in.tickCount > bhi.tickCountWhenExtraImpulseApplied + 1) ||
				(bhi.tickCountWhenExtraImpulseApplied > in.tickCount)) {
				bhi.tickCountWhenExtraImpulseApplied = in.tickCount;
				// carForward = cached rotMat.forward == live basis column 0 (the basis
				// cannot have changed since the last post-tick wrote rotMat)
				CuVec3 carForward = CuGetColumn(CuLoadBasis(carB.basis), 0);
				CuVec3 relPos = ballPos - carPos;
				CuVec3 relVel = ballVel - carVel;

				float relLenSq = relVel.x * relVel.x + relVel.y * relVel.y + relVel.z * relVel.z;
				float relLen = relLenSq > 0 ? sqrtf(relLenSq) : 0.0f; // Vec::Length
				float relSpeed = relLen < in.cc.ballCarExtraMaxDeltaVelUU
					? relLen : in.cc.ballCarExtraMaxDeltaVelUU; // RS_MIN
				if (relSpeed > 0) {
					// soccar: no hoops zScale branch
					float zScale = in.cc.ballCarExtraZScale;
					CuVec3 scaled = { relPos.x * 1.0f, relPos.y * 1.0f, relPos.z * zScale };
					CuVec3 hitDir = CuVecNormalizedRS(scaled);
					float dotF = hitDir.x * carForward.x + hitDir.y * carForward.y + hitDir.z * carForward.z;
					CuVec3 fwdAdj = (carForward * dotF) * (1.0f - in.cc.ballCarExtraForwardScale);
					hitDir = CuVecNormalizedRS(hitDir - fwdAdj);
					float factor = CuCurveOutput(in.cc.ballExtraImpulseFactorCurve, relSpeed);
					CuVec3 addedVel = ((hitDir * relSpeed) * factor) * in.cc.ballHitExtraForceScale;
					bhi.extraHitVel = { addedVel.x, addedVel.y, addedVel.z };
					CuVec3 cache = GV(in.ball->velocityImpulseCache) + addedVel * CU_UU_TO_BT;
					in.ball->velocityImpulseCache = { cache.x, cache.y, cache.z };
				}
			}
			// (heatseeker / dropshot charge paths: non-soccar, not on device)
		} else if (bodyB >= 1) {
			// Car + Car: overrides + bump logic both ways
			cp.combinedFriction = in.cc.carCarFriction;
			cp.combinedRestitution = in.cc.carCarRestitution;

			for (int i = 0; i < 2; i++) {
				bool isSwapped = (i == 1);
				int c1 = isSwapped ? bodyB - 1 : bodyA - 1;
				int c2 = isSwapped ? bodyA - 1 : bodyB - 1;
				GpuCarGameState& car1 = in.cars[c1];
				GpuCarGameState& car2 = in.cars[c2];
				if (car1.isDemoed || car2.isDemoed)
					return;
				if (car1.carContact_otherCarID == car2.carID && car1.carContact_cooldownTimer > 0)
					continue;

				const GpuBodyState& b1 = in.bodies[1 + c1];
				const GpuBodyState& b2 = in.bodies[1 + c2];
				CuVec3 pos1 = GV(b1.pos) * CU_BT_TO_UU, vel1 = GV(b1.linVel) * CU_BT_TO_UU;
				CuVec3 pos2 = GV(b2.pos) * CU_BT_TO_UU, vel2 = GV(b2.linVel) * CU_BT_TO_UU;
				CuVec3 deltaPos = pos2 - pos1;
				float velDotDelta = vel1.x * deltaPos.x + vel1.y * deltaPos.y + vel1.z * deltaPos.z;
				if (velDotDelta > 0) {
					CuVec3 velDir = CuVecNormalizedRS(vel1);
					CuVec3 dirToOtherCar = CuVecNormalizedRS(deltaPos);
					float speedTowards = vel1.x * dirToOtherCar.x + vel1.y * dirToOtherCar.y + vel1.z * dirToOtherCar.z;
					float otherAway = vel2.x * velDir.x + vel2.y * velDir.y + vel2.z * velDir.z;
					if (speedTowards > otherAway) {
						CuVec3 localPoint = isSwapped ? cp.localPointB : cp.localPointA;
						bool hitWithBumper = (localPoint.x * CU_BT_TO_UU) > in.cc.bumpMinForwardDist;
						if (hitWithBumper) {
							bool isDemo;
							switch (in.cc.demoMode) {
							case CU_DEMOMODE_ON_CONTACT: isDemo = true; break;
							case CU_DEMOMODE_DISABLED: isDemo = false; break;
							default: isDemo = car1.isSupersonic != 0; break;
							}
							if (isDemo && !in.cc.enableTeamDemos)
								isDemo = car1.team != car2.team;
							if (isDemo) {
								// Car::Demolish: full demolition (body removal + respawn)
								// is not modeled on device; gates run DemoMode::DISABLED.
								car2.isDemoed = 1;
								car2.demoRespawnTimer = in.cc.respawnDelay;
							} else {
								bool groundHit = car2.isOnGround;
								float baseScale = CuCurveOutput(
									groundHit ? in.cc.bumpVelGroundCurve : in.cc.bumpVelAirCurve,
									speedTowards);
								CuVec3 hitUpDir = car2.isOnGround
									? CuGetColumn(CuLoadBasis(b2.basis), 2)
									: CuVec3{ 0, 0, 1 };
								float upScale = CuCurveOutput(in.cc.bumpUpwardVelCurve, speedTowards);
								CuVec3 bumpImpulse = velDir * baseScale +
									(hitUpDir * upScale) * in.cc.bumpForceScale;
								CuVec3 cache = GV(car2.velocityImpulseCache) + bumpImpulse * CU_UU_TO_BT;
								car2.velocityImpulseCache = { cache.x, cache.y, cache.z };
							}
							car1.carContact_otherCarID = car2.carID;
							car1.carContact_cooldownTimer = in.cc.bumpCooldownTime;
						}
					}
				}
			}
		} else {
			// Car + World
			car.worldContact_hasContact = 1;
			car.worldContact_normal = { cp.normalWorldOnB.x, cp.normalWorldOnB.y, cp.normalWorldOnB.z };
			cp.combinedFriction = in.cc.carWorldFriction;
			cp.combinedRestitution = in.cc.carWorldRestitution;
		}
	} else if (bodyA == 0 && bodyB < 0) {
		// Ball + World: _OnWorldCollision is a no-op in soccar; mark special
		if (in.cc.gameMode != /*SNOWDAY*/ 4)
			cp.isSpecial = 1;
	}
}

// ---- the collide phase for one arena (single thread) ----
__device__ inline void CuCollideTick(const CuCollideIn& in) {
	CuCollideUpdateAabbs(in);

	int2 pairs[CU_MAX_PAIRS];
	int numPairs = CuCollideGenPairs(in, pairs);

	if (in.debug) {
		in.debug[0] = numPairs;
		for (int i = 0; i < numPairs; i++) {
			in.debug[1 + 2 * i] = pairs[i].x;
			in.debug[2 + 2 * i] = pairs[i].y;
		}
	}

	CuManifoldList ml = { in.manifolds, 0 };

	const CuMat3 ident = CuIdentity3();
	CuVec3 zero = { 0, 0, 0 };

	for (int pi = 0; pi < numPairs; pi++) {
		int h0 = pairs[pi].x, h1 = pairs[pi].y;

		if (h0 < in.numStatics) {
			// ---- static (proxy0) vs dyn (proxy1) ----
			int d = h1 - in.numStatics;
			const GpuStaticObj& so = in.statics[h0];
			// needsCollision: statics are ISLAND_SLEEPING -> requires the dyn active
			if (!CuIsActive(in.bodies[d].activationState))
				continue;

			const GpuBodyState& ds = in.bodies[d];
			CuMat3 dynBasis = CuLoadBasis(ds.basis);
			CuVec3 dynPos = GV(ds.pos);

			CuContactBodyInfo staticInfo = { ident, zero, so.friction, so.restitution, true, -1 };
			float dynBreaking = d == 0 ? in.cc.ballBreakingThreshold : in.cc.carBreakingThreshold;
			float breaking = dynBreaking < so.breakingThreshold ? dynBreaking : so.breakingThreshold;

			if (d == 0) {
				// ball: swappedConvexConcave / planeConvex; manifold (ball, static) at ctor
				CuContactBodyInfo ballInfo = { dynBasis, dynPos,
					in.cc.ballFriction, in.cc.ballRestitution, false, CU_USERIDX_BALL };
				int mIdx = ml.create(0, -1 - h0, breaking);
				if (mIdx >= GPU_MAX_MANIFOLDS)
					continue;
				CuManifold& m = ml.arr[mIdx];
				if (so.kind == 0) {
					// sphere vs mesh (refresh happens inside CuConvexVsMesh)
					CuGjkShape sphere;
					sphere.kind = CU_SHAPE_SPHERE;
					sphere.margin = in.cc.ballRadius;
					const GpuMeshBvh& mb = in.bvhs[so.shapeIdx];
					CuBvhView bvh = { in.bvhNodes + mb.nodeStart, in.bvhSubtrees + mb.subtreeStart,
						mb.numSubtrees, GV(mb.bvhAabbMin), GV(mb.bvhAabbMax), GV(mb.bvhQuantization) };
					int triStart = in.geo.meshes[so.shapeIdx].triStart;
					CuConvexVsMesh(sphere, dynBasis, dynPos, in.geo, so.shapeIdx, bvh,
						in.meshMargins[so.shapeIdx], ballInfo, staticInfo, m, breaking,
						-1, -1,
						[&](int idx, int triIdx) {
							CuContactAddedCallback(in, m.pts[idx], 0, -1 - h0);
							const GpuVec3* tv = in.geo.meshTris + (size_t)triStart * 3;
							CuAdjustInternalEdgeContacts(m.pts[idx],
								GV(tv[triIdx * 3]), GV(tv[triIdx * 3 + 1]), GV(tv[triIdx * 3 + 2]),
								in.triInfos[triStart + triIdx]);
						});
				} else {
					// sphere vs plane (result order: (plane, ball) -> manifold swapped)
					const GpuPlane& pl = in.geo.planes[so.shapeIdx];
					CuConvexShape sphere = { CU_SHAPE_SPHERE, in.cc.ballRadius, { 0, 0, 0 } };
					CuVec3 nrm, pt;
					float dist;
					if (CuConvexPlaneCollide(sphere, dynBasis, dynPos, GV(pl.normal),
							pl.planeConst, ident, GV(pl.origin), breaking, nrm, pt, dist)) {
						int idx = CuAddContactPoint(m, staticInfo, ballInfo, true,
							nrm, pt, dist, -1, -1, -1, -1);
						if (idx >= 0)
							CuContactAddedCallback(in, m.pts[idx], 0, -1 - h0);
					}
					if (m.numContacts)
						CuManifoldRefresh(m, dynBasis, dynPos, ident, GV(pl.origin));
				}
			} else {
				// car: swappedCompound -> child box; manifold (car, static) at ctor
				CuContactBodyInfo carInfo = { dynBasis, dynPos,
					in.cc.carFriction, in.cc.carRestitution, false, CU_USERIDX_CAR };
				int mIdx = ml.create(d, -1 - h0, breaking);
				if (mIdx >= GPU_MAX_MANIFOLDS)
					continue;
				CuManifold& m = ml.arr[mIdx];
				// newChildWorldTrans = carWT * childTrans (basis through the multiply
				// trees with identity: canonicalizes -0)
				CuMat3 childBasis = CuMat3Mul(dynBasis, ident);
				CuVec3 childOrigin = CuTransformPoint(dynBasis, dynPos, GV(in.cc.carChildOffset));
				CuVec3 cmn, cmx;
				CuBoxAabbAt(GV(in.cc.carBoxHalfExtNoMargin), in.cc.carBoxMargin,
					childBasis, childOrigin, cmn, cmx);
				// ProcessChildShape tests against the RAW shape aabb, not the proxy
				bool overlap = CuTestAabb2(cmn, cmx, GV(so.shapeAabbMin), GV(so.shapeAabbMax));
				if (overlap) {
					if (so.kind == 0) {
						CuGjkShape box;
						box.kind = CU_SHAPE_BOX;
						box.boxHalfExtNoMargin = GV(in.cc.carBoxHalfExtNoMargin);
						box.margin = in.cc.carBoxMargin;
						const GpuMeshBvh& mb = in.bvhs[so.shapeIdx];
						CuBvhView bvh = { in.bvhNodes + mb.nodeStart, in.bvhSubtrees + mb.subtreeStart,
							mb.numSubtrees, GV(mb.bvhAabbMin), GV(mb.bvhAabbMax), GV(mb.bvhQuantization) };
						int triStart = in.geo.meshes[so.shapeIdx].triStart;
						if (in.debug) { // TEMP diagnostic: last car-mesh pair walk stats
							in.debug[110] = h0;
							in.debug[111] = 0; // leaves visited
							in.debug[112] = 0; // passed tri-aabb
							in.debug[113] = 0; // gjk/added contacts
						}
						CuConvexVsMesh(box, childBasis, childOrigin, in.geo, so.shapeIdx, bvh,
							in.meshMargins[so.shapeIdx], carInfo, staticInfo, m, breaking,
							-1, 0,
							[&](int idx, int triIdx) {
								if (in.debug) in.debug[113]++;
								CuContactAddedCallback(in, m.pts[idx], d, -1 - h0);
								const GpuVec3* tv = in.geo.meshTris + (size_t)triStart * 3;
								CuAdjustInternalEdgeContacts(m.pts[idx],
									GV(tv[triIdx * 3]), GV(tv[triIdx * 3 + 1]), GV(tv[triIdx * 3 + 2]),
									in.triInfos[triStart + triIdx]);
							},
							in.debug ? in.debug + 111 : nullptr);
					} else {
						const GpuPlane& pl = in.geo.planes[so.shapeIdx];
						CuConvexShape box = { CU_SHAPE_BOX, 0.0f,
							{ in.cc.carBoxHalfExtNoMargin.x + in.cc.carBoxMargin,
							  in.cc.carBoxHalfExtNoMargin.y + in.cc.carBoxMargin,
							  in.cc.carBoxHalfExtNoMargin.z + in.cc.carBoxMargin } };
						CuVec3 nrm, pt;
						float dist;
						if (CuConvexPlaneCollide(box, childBasis, childOrigin, GV(pl.normal),
								pl.planeConst, ident, GV(pl.origin), breaking, nrm, pt, dist)) {
							// result A ids unset (-1,-1); B ids set by ProcessChildShape (-1, 0)
							int idx = CuAddContactPoint(m, staticInfo, carInfo, true,
								nrm, pt, dist, -1, -1, -1, 0);
							if (idx >= 0)
								CuContactAddedCallback(in, m.pts[idx], d, -1 - h0);
						}
						if (m.numContacts)
							CuManifoldRefresh(m, dynBasis, dynPos, ident, GV(pl.origin));
					}
				}
				// compound post-loop: same aabb test without extension; releases the
				// ctor-created manifold when the child never overlapped
				if (!overlap)
					ml.release(mIdx);
			}
		} else if (h0 == in.numStatics) {
			// ---- ball (proxy0) vs car (proxy1): swappedCompound, lazy convex-convex ----
			int c = h1 - in.numStatics - 1;
			const GpuBodyState& ballS = in.bodies[0];
			const GpuBodyState& carS = in.bodies[1 + c];
			if (!CuIsActive(ballS.activationState) && !CuIsActive(carS.activationState))
				continue;
			CuMat3 ballBasis = CuLoadBasis(ballS.basis);
			CuVec3 ballPos = GV(ballS.pos);
			CuMat3 carBasis = CuLoadBasis(carS.basis);
			CuVec3 carPos = GV(carS.pos);

			CuMat3 childBasis = CuMat3Mul(carBasis, ident);
			CuVec3 childOrigin = CuTransformPoint(carBasis, carPos, GV(in.cc.carChildOffset));
			CuVec3 cmn, cmx, bmn, bmx;
			CuBoxAabbAt(GV(in.cc.carBoxHalfExtNoMargin), in.cc.carBoxMargin,
				childBasis, childOrigin, cmn, cmx);
			CuSphereAabbAt(ballPos, in.cc.ballRadius, bmn, bmx);
			if (!CuTestAabb2(cmn, cmx, bmn, bmx))
				continue; // no manifold was ever created (lazy)

			float breaking = in.cc.ballBreakingThreshold < in.cc.carBreakingThreshold
				? in.cc.ballBreakingThreshold : in.cc.carBreakingThreshold;
			int mIdx = ml.create(1 + c, 0, breaking);
			if (mIdx >= GPU_MAX_MANIFOLDS)
				continue;
			CuManifold& m = ml.arr[mIdx];

			CuGjkShape box;
			box.kind = CU_SHAPE_BOX;
			box.boxHalfExtNoMargin = GV(in.cc.carBoxHalfExtNoMargin);
			box.margin = in.cc.carBoxMargin;
			CuGjkShape sphere;
			sphere.kind = CU_SHAPE_SPHERE;
			sphere.margin = in.cc.ballRadius;
			float maxd = in.cc.carBoxMargin + in.cc.ballRadius + breaking;
			CuGjkResult r;
			CuGjkClosestPoints(box, childBasis, childOrigin, sphere, ballBasis, ballPos,
				maxd * maxd, true, r);
			CuContactBodyInfo ballInfo = { ballBasis, ballPos,
				in.cc.ballFriction, in.cc.ballRestitution, false, CU_USERIDX_BALL };
			CuContactBodyInfo carInfo = { carBasis, carPos,
				in.cc.carFriction, in.cc.carRestitution, false, CU_USERIDX_CAR };
			if (r.hasContact) {
				// result order (ball, car-child); manifold (car, ball) -> swapped
				int idx = CuAddContactPoint(m, ballInfo, carInfo, true,
					r.normalOnB, r.pointOnBWorld, r.distance, -1, -1, -1, 0);
				if (idx >= 0)
					CuContactAddedCallback(in, m.pts[idx], 1 + c, 0);
			}
			// convexConvex tail: refresh (own manifold), unconditional
			CuManifoldRefresh(m, carBasis, carPos, ballBasis, ballPos);
		} else {
			// ---- car (lower) vs car (higher): compoundCompound -> treeless fallback ----
			int cLO = h0 - in.numStatics - 1;
			int cHI = h1 - in.numStatics - 1;
			const GpuBodyState& sLO = in.bodies[1 + cLO];
			const GpuBodyState& sHI = in.bodies[1 + cHI];
			if (!CuIsActive(sLO.activationState) && !CuIsActive(sHI.activationState))
				continue;
			CuMat3 basisLO = CuLoadBasis(sLO.basis);
			CuVec3 posLO = GV(sLO.pos);
			CuMat3 basisHI = CuLoadBasis(sHI.basis);
			CuVec3 posHI = GV(sHI.pos);

			// boxbox ctor at pair-algo creation: manifold (carHI, carLO)
			int mIdx = ml.create(1 + cHI, 1 + cLO, in.cc.carBreakingThreshold);
			if (mIdx >= GPU_MAX_MANIFOLDS)
				continue;
			CuManifold& m = ml.arr[mIdx];

			CuMat3 childBasisLO = CuMat3Mul(basisLO, ident);
			CuVec3 childOriginLO = CuTransformPoint(basisLO, posLO, GV(in.cc.carChildOffset));
			CuVec3 lmn, lmx;
			CuBoxAabbAt(GV(in.cc.carBoxHalfExtNoMargin), in.cc.carBoxMargin,
				childBasisLO, childOriginLO, lmn, lmx);
			// outer ProcessChildShape: carLO child vs carHI COMPOUND shape aabb
			CuVec3 hmn, hmx;
			CuCompoundAabbAt(in.cc, basisHI, posHI, hmn, hmx);
			bool overlapOuter = CuTestAabb2(lmn, lmx, hmn, hmx);
			if (overlapOuter) {
				CuMat3 childBasisHI = CuMat3Mul(basisHI, ident);
				CuVec3 childOriginHI = CuTransformPoint(basisHI, posHI, GV(in.cc.carChildOffset));
				CuVec3 hcmn, hcmx;
				CuBoxAabbAt(GV(in.cc.carBoxHalfExtNoMargin), in.cc.carBoxMargin,
					childBasisHI, childOriginHI, hcmn, hcmx);
				bool overlapNested = CuTestAabb2(hcmn, hcmx, lmn, lmx);
				if (overlapNested) {
					CuVec3 heWithMargin = {
						in.cc.carBoxHalfExtNoMargin.x + in.cc.carBoxMargin,
						in.cc.carBoxHalfExtNoMargin.y + in.cc.carBoxMargin,
						in.cc.carBoxHalfExtNoMargin.z + in.cc.carBoxMargin };
					CuBoxBoxOut bb;
					CuBoxBoxClosestPoints(childBasisHI, childOriginHI, heWithMargin,
						childBasisLO, childOriginLO, heWithMargin, bb);
					CuContactBodyInfo loInfo = { basisLO, posLO,
						in.cc.carFriction, in.cc.carRestitution, false, CU_USERIDX_CAR };
					CuContactBodyInfo hiInfo = { basisHI, posHI,
						in.cc.carFriction, in.cc.carRestitution, false, CU_USERIDX_CAR };
					for (int bi = 0; bi < bb.n; bi++) {
						// result order (carLO, carHI); manifold (carHI, carLO) -> swapped
						int idx = CuAddContactPoint(m, loInfo, hiInfo, true,
							bb.c[bi].normal, bb.c[bi].point, bb.c[bi].depth, -1, 0, -1, 0);
						if (idx >= 0)
							CuContactAddedCallback(in, m.pts[idx], 1 + cHI, 1 + cLO);
					}
					// boxbox tail: refresh (own manifold), unconditional
					CuManifoldRefresh(m, basisHI, posHI, basisLO, posLO);
				} else {
					ml.release(mIdx); // nested post-loop destroys boxbox -> release
				}
			} else {
				ml.release(mIdx); // outer post-loop destroys nested compound -> release
			}
		}
	}

	*in.numManifolds = ml.n;
}

} // namespace rscuda
