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
// ISLAND_SLEEPING. Demolitions: the demoed body STAYS in the broadphase (CPU
// keeps the proxy); it goes DISABLE_SIMULATION + noContactResponse at the next
// pre-tick, respawn handled in the kernel GAME_PRE demo block.

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

// btCollisionObject::setActivationState (respects DISABLE_* lock-in)
__device__ __forceinline__ void CuSetActivationState(int& cur, int newState) {
	if (cur != CU_DISABLE_DEACTIVATION && cur != CU_DISABLE_SIMULATION)
		cur = newState;
}

#define CU_BT_LARGE_FLOAT 1e18f
#define CU_CONTACT_BREAKING 0.02f  // gContactBreakingThreshold
#define CU_UU_TO_BT (1.f / 50.f)
#define CU_BT_TO_UU 50.f
#define CU_USERIDX_CAR 1
#define CU_USERIDX_BALL 2
#define CU_USERIDX_DROPSHOT_TILE 3
// GameMode enum (src/Sim/GameMode.h)
#define CU_GAMEMODE_SOCCAR 0
#define CU_GAMEMODE_HOOPS 1
#define CU_GAMEMODE_HEATSEEKER 2
#define CU_GAMEMODE_SNOWDAY 3
#define CU_GAMEMODE_DROPSHOT 4
#define CU_DEMOMODE_NORMAL 0      // DemoMode::NORMAL
#define CU_DEMOMODE_ON_CONTACT 1  // DemoMode::ON_CONTACT
#define CU_DEMOMODE_DISABLED 2    // DemoMode::DISABLED

struct __align__(16) CuCellStatic {
	GpuVec3 aabbMin;
	int group;
	GpuVec3 aabbMax;
	int mask;
	int staticIdx;
};

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
	const int* cellStaticCount;
	const CuCellStatic* cellStatics;
	int cellStaticStride;
	GpuGridParams grid;
	GpuWorldGeo geo;
	const GpuMeshBvh* bvhs;
	const GpuBvhNode* bvhNodes;
	const uint4* bvhSubtreeBoxes;
	const GpuTriInfo* triInfos;      // [total tris], indexed by mesh triStart + tri
	const CuTriCull* triCull;
	const float* meshMargins;
	const GpuCollideConsts* cc;
	uint64_t tickCount;
	float dt;
	// dropshot per-arena tile state (null outside dropshot). Damage mutates during
	// the tick (neighbor breaks); response flags only re-sync at GAME_POST.
	uint8_t* tileDamage;
	const uint8_t* tileNoResponse;
	// outputs (per arena)
	CuManifold* manifolds;           // capacity GPU_MAX_MANIFOLDS
	int* numManifolds;
	int2* pairsOut;                  // capacity CU_MAX_PAIRS (solver islands input)
	int* numPairsOut;
#ifdef RS_GPU_PHASE_PROF
	unsigned long long* prof; // [5]: aabbs, pairs, narrowphase cycles, pair count, manifold count
#endif
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

// btPolyhedralConvexAabbCachingShape::getAabb (snowday puck hull): btTransformAabb
// over the cached local aabb + collision margin
__device__ __forceinline__ void CuHullAabbAt(const GpuCollideConsts& cc,
	const CuMat3& basis, CuVec3 origin, CuVec3& mn, CuVec3& mx) {
	CuVec3 lmin = GV(cc.ballHullLocalAabbMin), lmax = GV(cc.ballHullLocalAabbMax);
	CuVec3 halfExt = (lmax - lmin) * 0.5f;
	halfExt = { halfExt.x + cc.ballHullMargin, halfExt.y + cc.ballHullMargin,
		halfExt.z + cc.ballHullMargin };
	CuVec3 center = (lmax + lmin) * 0.5f;
	CuVec3 c = CuTransformPoint(basis, origin, center);
	CuVec3 r0 = { fabsf(basis.r[0].x), fabsf(basis.r[0].y), fabsf(basis.r[0].z) };
	CuVec3 r1 = { fabsf(basis.r[1].x), fabsf(basis.r[1].y), fabsf(basis.r[1].z) };
	CuVec3 r2 = { fabsf(basis.r[2].x), fabsf(basis.r[2].y), fabsf(basis.r[2].z) };
	CuVec3 extent = { CuDot3(halfExt, r0), CuDot3(halfExt, r1), CuDot3(halfExt, r2) };
	mn = c - extent;
	mx = c + extent;
}

// Body shape aabb at an arbitrary transform (slot 0 = ball, others = compound)
__device__ __forceinline__ void CuBodyAabbAt(const CuCollideIn& in, int slot,
	const CuMat3& basis, CuVec3 origin, CuVec3& mn, CuVec3& mx) {
	if (slot == 0) {
		if (in.cc->ballIsHull)
			CuHullAabbAt(*in.cc, basis, origin, mn, mx);
		else
			CuSphereAabbAt(origin, in.cc->ballRadius, mn, mx);
	} else {
		CuCompoundAabbAt(*in.cc, basis, origin, mn, mx);
	}
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

__device__ inline bool CuCollideUpdateAabbBody(const CuCollideIn& in, int d, GpuDynProxy& p) {
	const GpuBodyState& s = in.bodies[d];
	CuMat3 basis = CuLoadBasis(s.basis);
	CuVec3 pos = GV(s.pos);
	CuVec3 mn, mx;
	CuBodyAabbAt(in, d, basis, pos, mn, mx);
	CuVec3 thr = { CU_CONTACT_BREAKING, CU_CONTACT_BREAKING, CU_CONTACT_BREAKING };
	mn = mn - thr;
	mx = mx + thr;

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

	CuVec3 omn = GV(p.aabbMin), omx = GV(p.aabbMax);
	bool changed = omn.x != mn.x || omn.y != mn.y || omn.z != mn.z ||
		omx.x != mx.x || omx.y != mx.y || omx.z != mx.z;
	if (!changed)
		return false;
	p.aabbMin = { mn.x, mn.y, mn.z };
	p.aabbMax = { mx.x, mx.y, mx.z };
	int oldCell = p.cellIdx;
	int ni, nj, nk;
	CuGetCellIndices(in.grid, mn, ni, nj, nk);
	int newCell = ni * in.grid.cellsY * in.grid.cellsZ + nj * in.grid.cellsZ + nk;
	p.cellIdx = newCell;
	if (oldCell != newCell && in.numBodies > 1) {
		p.i = ni;
		p.j = nj;
		p.k = nk;
		return true;
	}
	return false;
}

// ---- btRSBroadphase::calculateOverlappingPairs ----
// Pair = (h0, h1) with h < numStatics static handle, else dyn slot h-numStatics.
// internalAddPair orders by uid = handle order, so h0 < h1 always.
#define CU_MAX_PAIRS 96

__device__ __forceinline__ void CuDynFilter(const CuCollideIn& in, int slot, int& g, int& m) {
	if (slot == 0) { g = in.cc->ballFilterGroup; m = in.cc->ballFilterMask; }
	else { g = in.cc->carFilterGroup; m = in.cc->carFilterMask; }
}

__device__ __forceinline__ const CuCellStatic* CuCellStaticsOf(const CuCollideIn& in, int cellIdx) {
	return in.cellStatics + (size_t)cellIdx * in.cellStaticStride;
}

__device__ __forceinline__ bool CuCellStaticOverlaps(const CuCellStatic& cs, CuVec3 pmin,
	CuVec3 pmax, int dg, int dm) {
	return CuTestAabb2(pmin, pmax, GV(cs.aabbMin), GV(cs.aabbMax)) &&
		CuNeedsBroadphase(dg, dm, cs.group, cs.mask);
}

#define CU_STATIC_CAND_MAX 7
__device__ inline void CuCollideStaticCandidates(const CuCollideIn& in, const GpuDynProxy& p,
	int d, int* out) {
	CuVec3 pmin = GV(p.aabbMin), pmax = GV(p.aabbMax);
	int dg, dm;
	CuDynFilter(in, d, dg, dm);
	const int count = in.cellStaticCount[p.cellIdx];
	const CuCellStatic* cell = CuCellStaticsOf(in, p.cellIdx);
	int n = 0;
#pragma unroll 4
	for (int i = 0; i < count; i++) {
		const CuCellStatic cs = cell[i];
		if (!CuCellStaticOverlaps(cs, pmin, pmax, dg, dm))
			continue;
		if (n == CU_STATIC_CAND_MAX) {
			n = -1;
			break;
		}
		out[1 + n++] = cs.staticIdx;
	}
	out[0] = n;
}

__device__ inline int CuCollideGenPairs(const CuCollideIn& in, const GpuDynProxy* proxies,
	int2* pairs, const int* staticCand) {
	int numPairs = 0;
	const int cellsYZ = in.grid.cellsY * in.grid.cellsZ;
	if (staticCand)
		for (int d = 0; d < in.numBodies; d++)
			if (staticCand[d * (1 + CU_STATIC_CAND_MAX)] < 0)
				staticCand = nullptr;
	for (int d = 0; d < in.numBodies; d++) {
		const GpuDynProxy& p = proxies[d];
		CuVec3 pmin = GV(p.aabbMin), pmax = GV(p.aabbMax);
		int dg, dm;
		CuDynFilter(in, d, dg, dm);

		if (staticCand) {
			const int* c = staticCand + d * (1 + CU_STATIC_CAND_MAX);
			for (int i = 0; i < c[0]; i++)
				if (numPairs < CU_MAX_PAIRS)
					pairs[numPairs++] = { c[1 + i], in.numStatics + d };
		} else {
			const int count = in.cellStaticCount[p.cellIdx];
			const CuCellStatic* cell = CuCellStaticsOf(in, p.cellIdx);
			for (int i = 0; i < count; i++)
				if (CuCellStaticOverlaps(cell[i], pmin, pmax, dg, dm) && numPairs < CU_MAX_PAIRS)
					pairs[numPairs++] = { cell[i].staticIdx, in.numStatics + d };
		}

		if (in.numBodies > 1) {
			int ci = p.cellIdx / cellsYZ;
			int rem = p.cellIdx % cellsYZ;
			int cj = rem / in.grid.cellsZ;
			int ck = rem % in.grid.cellsZ;
			int members[32];
			int numMembers = 0;
			for (int q = 0; q < in.numBodies && numMembers < 32; q++) {
				const GpuDynProxy& qp = proxies[q];
				int mni = qp.i - 1 < 0 ? 0 : qp.i - 1;
				int mxi = qp.i + 1 > in.grid.cellsX - 1 ? in.grid.cellsX - 1 : qp.i + 1;
				int mnj = qp.j - 1 < 0 ? 0 : qp.j - 1;
				int mxj = qp.j + 1 > in.grid.cellsY - 1 ? in.grid.cellsY - 1 : qp.j + 1;
				int mnk = qp.k - 1 < 0 ? 0 : qp.k - 1;
				int mxk = qp.k + 1 > in.grid.cellsZ - 1 ? in.grid.cellsZ - 1 : qp.k + 1;
				if (ci >= mni && ci <= mxi && cj >= mnj && cj <= mxj && ck >= mnk && ck <= mxk)
					members[numMembers++] = q;
			}
			for (int a = 1; a < numMembers; a++) {
				int v = members[a];
				int b = a - 1;
				while (b >= 0 && proxies[members[b]].regSeq > proxies[v].regSeq) {
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
					const GpuDynProxy& qp = proxies[q];
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

static_assert(CU_MAX_PAIRS <= GPU_MAX_MANIFOLDS, "a pair keeps at most one manifold");

__device__ __forceinline__ void CuManifoldBegin(CuManifold& m, int bodyA, int bodyB, float breaking) {
	m.bodyA = bodyA;
	m.bodyB = bodyB;
	m.numContacts = 0;
	m.contactBreakingThreshold = breaking;
	m.contactProcessingThreshold = CU_BT_LARGE_FLOAT;
}

__device__ __forceinline__ void CuManifoldStore(CuManifold& dst, const CuManifold& src) {
	dst.bodyA = src.bodyA;
	dst.bodyB = src.bodyB;
	dst.numContacts = src.numContacts;
	dst.contactBreakingThreshold = src.contactBreakingThreshold;
	dst.contactProcessingThreshold = src.contactProcessingThreshold;
	for (int i = 0; i < src.numContacts; i++)
		dst.pts[i] = src.pts[i];
}

// Body contact response for a callback/solver body ref (>= 0 dyn slot, < 0 static).
// Cars mirror CF_NO_CONTACT_RESPONSE (set/cleared at pre-tick - demo flag timing);
// broken dropshot tiles get the flag at the post-tick sync.
__device__ __forceinline__ bool CuBodyHasResponse(const CuCollideIn& in, int bodyRef) {
	if (bodyRef >= 1)
		return !in.cars[bodyRef - 1].noContactResponse;
	if (bodyRef < 0) {
		const GpuStaticObj& so = in.statics[-1 - bodyRef];
		if (so.kind == 2 && in.tileNoResponse)
			return !in.tileNoResponse[so.shapeIdx];
	}
	return true;
}

__device__ __forceinline__ void CuContactOverrides(const CuCollideIn& in, CuManifoldPoint& cp,
	int bodyA, int bodyB) {
	if (bodyA >= 1) {
		if (bodyB == 0) {
			cp.combinedFriction = in.cc->carBallFriction;
			cp.combinedRestitution = in.cc->carBallRestitution;
		} else if (bodyB >= 1) {
			cp.combinedFriction = in.cc->carCarFriction;
			cp.combinedRestitution = in.cc->carCarRestitution;
		} else {
			cp.combinedFriction = in.cc->carWorldFriction;
			cp.combinedRestitution = in.cc->carWorldRestitution;
		}
	} else if (bodyA == 0 && bodyB < 0 && in.statics[-1 - bodyB].kind != 2 &&
		in.cc->gameMode != CU_GAMEMODE_SNOWDAY) {
		cp.isSpecial = 1;
	}
}

__device__ inline void CuContactEffects(const CuCollideIn& in, int bodyA, int bodyB,
	CuVec3 localPointA, CuVec3 localPointB, CuVec3 normalWorldOnB) {
	bool carInvolvedA = bodyA >= 1;
	if (carInvolvedA) {
		int c = bodyA - 1;
		GpuCarGameState& car = in.cars[c];
		if (bodyB == 0) {
			const GpuBodyState& carB = in.bodies[1 + c];
			const GpuBodyState& ballB = in.bodies[0];
			CuVec3 carPos = GV(carB.pos) * CU_BT_TO_UU;
			CuVec3 carVel = GV(carB.linVel) * CU_BT_TO_UU;
			CuVec3 ballPos = GV(ballB.pos) * CU_BT_TO_UU;
			CuVec3 ballVel = GV(ballB.linVel) * CU_BT_TO_UU;

			GpuBallHitInfo& bhi = car.ballHitInfo;
			bhi.isValid = 1;
			CuVec3 relPosOnBall = localPointB * CU_BT_TO_UU;
			bhi.relativePosOnBall = { relPosOnBall.x, relPosOnBall.y, relPosOnBall.z };
			bhi.tickCountWhenHit = in.tickCount;
			bhi.ballPos = { ballPos.x, ballPos.y, ballPos.z };
			bhi.extraHitVel = { 0, 0, 0 };

			if ((in.tickCount > bhi.tickCountWhenExtraImpulseApplied + 1) ||
				(bhi.tickCountWhenExtraImpulseApplied > in.tickCount)) {
				bhi.tickCountWhenExtraImpulseApplied = in.tickCount;
				CuVec3 carForward = CuGetColumn(CuLoadBasis(carB.basis), 0);
				CuVec3 relPos = ballPos - carPos;
				CuVec3 relVel = ballVel - carVel;

				float relLenSq = relVel.x * relVel.x + relVel.y * relVel.y + relVel.z * relVel.z;
				float relLen = relLenSq > 0 ? sqrtf(relLenSq) : 0.0f;
				float relSpeed = relLen < in.cc->ballCarExtraMaxDeltaVelUU
					? relLen : in.cc->ballCarExtraMaxDeltaVelUU;
				if (relSpeed > 0) {
					bool extraZScale = in.cc->gameMode == CU_GAMEMODE_HOOPS &&
						car.isOnGround &&
						car.rotMat.row[2].z > in.cc->hoopsZScaleThresh;
					float zScale = extraZScale ? in.cc->hoopsZScaleGround
						: in.cc->ballCarExtraZScale;
					CuVec3 scaled = { relPos.x * 1.0f, relPos.y * 1.0f, relPos.z * zScale };
					CuVec3 hitDir = CuVecNormalizedRS(scaled);
					float dotF = hitDir.x * carForward.x + hitDir.y * carForward.y + hitDir.z * carForward.z;
					CuVec3 fwdAdj = (carForward * dotF) * (1.0f - in.cc->ballCarExtraForwardScale);
					hitDir = CuVecNormalizedRS(hitDir - fwdAdj);
					float factor = CuCurveOutput(in.cc->ballExtraImpulseFactorCurve, relSpeed);
					CuVec3 addedVel = ((hitDir * relSpeed) * factor) * in.cc->ballHitExtraForceScale;
					bhi.extraHitVel = { addedVel.x, addedVel.y, addedVel.z };
					CuVec3 cache = GV(in.ball->velocityImpulseCache) + addedVel * CU_UU_TO_BT;
					in.ball->velocityImpulseCache = { cache.x, cache.y, cache.z };
				}
			} else {
				return;
			}
			if (in.cc->gameMode == CU_GAMEMODE_HEATSEEKER) {
				GpuBallGameState& bs = *in.ball;
				bool canIncrease = (bs.hsTimeSinceHit > in.cc->hsMinSpeedupInterval) ||
					(bs.hsYTargetDir == 0.0f);
				float newTargetDir = (car.team == 0) ? 1.0f : -1.0f;
				if (canIncrease && (newTargetDir != bs.hsYTargetDir)) {
					bs.hsTimeSinceHit = 0;
					float ns = bs.hsCurTargetSpeed + in.cc->hsTargetSpeedIncrement;
					bs.hsCurTargetSpeed = ns < in.cc->hsMaxSpeed ? ns : in.cc->hsMaxSpeed;
				}
				bs.hsYTargetDir = newTargetDir;
			} else if (in.cc->gameMode == CU_GAMEMODE_DROPSHOT) {
				GpuBallGameState& bs = *in.ball;
				CuVec3 dirFromCar = CuVecNormalizedRS(ballPos - carPos);
				CuVec3 relVelFromCar = carVel - ballVel;
				float velIntoBall = dirFromCar.x * relVelFromCar.x +
					dirFromCar.y * relVelFromCar.y + dirFromCar.z * relVelFromCar.z;
				if (velIntoBall >= in.cc->dsMinChargeHitSpeed) {
					bs.dsAccumulatedHitForce += velIntoBall;
					if (bs.dsAccumulatedHitForce >= in.cc->dsMinAbsorbedForCharge)
						bs.dsChargeLevel = 2;
					if (bs.dsAccumulatedHitForce >= in.cc->dsMinAbsorbedForSupercharge)
						bs.dsChargeLevel = 3;
				}
				if (bs.dsChargeLevel > 1) {
					float newTargetDir = (car.team == 0) ? 1.0f : -1.0f;
					bs.dsYTargetDir = newTargetDir;
				}
			}
		} else if (bodyB >= 1) {
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
						CuVec3 localPoint = isSwapped ? localPointB : localPointA;
						bool hitWithBumper = (localPoint.x * CU_BT_TO_UU) > in.cc->bumpMinForwardDist;
						if (hitWithBumper) {
							bool isDemo;
							switch (in.cc->demoMode) {
							case CU_DEMOMODE_ON_CONTACT: isDemo = true; break;
							case CU_DEMOMODE_DISABLED: isDemo = false; break;
							default: isDemo = car1.isSupersonic != 0; break;
							}
							if (isDemo && !in.cc->enableTeamDemos)
								isDemo = car1.team != car2.team;
							if (isDemo) {
								car2.isDemoed = 1;
								car2.demoRespawnTimer = in.cc->respawnDelay;
							} else {
								bool groundHit = car2.isOnGround;
								float baseScale = CuCurveOutput(
									groundHit ? in.cc->bumpVelGroundCurve : in.cc->bumpVelAirCurve,
									speedTowards);
								CuVec3 hitUpDir = car2.isOnGround
									? CuGetColumn(CuLoadBasis(b2.basis), 2)
									: CuVec3{ 0, 0, 1 };
								float upScale = CuCurveOutput(in.cc->bumpUpwardVelCurve, speedTowards);
								CuVec3 bumpImpulse = velDir * baseScale +
									(hitUpDir * upScale) * in.cc->bumpForceScale;
								CuVec3 cache = GV(car2.velocityImpulseCache) + bumpImpulse * CU_UU_TO_BT;
								car2.velocityImpulseCache = { cache.x, cache.y, cache.z };
							}
							car1.carContact_otherCarID = car2.carID;
							car1.carContact_cooldownTimer = in.cc->bumpCooldownTime;
						}
					}
				}
			}
		} else {
			car.worldContact_hasContact = 1;
			car.worldContact_normal = { normalWorldOnB.x, normalWorldOnB.y, normalWorldOnB.z };
		}
	} else if (bodyA == 0 && bodyB < 0 && in.statics[-1 - bodyB].kind == 2) {
		int tileTotalIndex = in.statics[-1 - bodyB].shapeIdx;
		int teamIdx = tileTotalIndex / GPU_DS_TILES_PER_TEAM;
		int tileIdx = tileTotalIndex % GPU_DS_TILES_PER_TEAM;
		GpuBallGameState& bs = *in.ball;
		do {
			if (in.tileDamage[tileTotalIndex] == GPU_DS_TILE_BROKEN)
				break;
			if (bs.dsHasDamaged) {
				float timeSinceDamage =
					(float)(in.tickCount - bs.dsLastDamageTick) * in.dt;
				if (timeSinceDamage <= in.cc->dsMinDamageInterval)
					break;
			}
			CuVec3 velUU = GV(in.bodies[0].linVel) * CU_BT_TO_UU;
			if (velUU.z > -in.cc->dsMinDownwardSpeedToDamage)
				break;
			if (bs.dsChargeLevel > 1 && bs.dsYTargetDir != 0.0f) {
				float tileYSign = (teamIdx == 0) ? -1.0f : 1.0f;
				if (tileYSign != bs.dsYTargetDir)
					break;
			}
			const GpuTileTables& tt = *in.geo.tileTables;
			if (bs.dsChargeLevel == 1) {
				uint8_t& st = in.tileDamage[teamIdx * GPU_DS_TILES_PER_TEAM + tileIdx];
				if (st != GPU_DS_TILE_BROKEN)
					st++;
			} else if (bs.dsChargeLevel == 2) {
				for (int i = 0; i < tt.n1Cnt[tileIdx]; i++) {
					uint8_t& st = in.tileDamage[teamIdx * GPU_DS_TILES_PER_TEAM + tt.n1[tileIdx][i]];
					if (st != GPU_DS_TILE_BROKEN)
						st++;
				}
			} else {
				for (int i = 0; i < tt.n2Cnt[tileIdx]; i++) {
					uint8_t& st = in.tileDamage[teamIdx * GPU_DS_TILES_PER_TEAM + tt.n2[tileIdx][i]];
					if (st != GPU_DS_TILE_BROKEN)
						st++;
				}
			}
			bs.dsHasDamaged = 1;
			bs.dsLastDamageTick = in.tickCount;
			bs.dsAccumulatedHitForce = 0;
			bs.dsChargeLevel = 1;
			bs.dsYTargetDir = 0;
		} while (false);
	} else if (bodyA == 0 && bodyB < 0) {
		if (in.cc->gameMode == CU_GAMEMODE_HEATSEEKER) {
			GpuBallGameState& bs = *in.ball;
			if (bs.hsYTargetDir != 0.0f) {
				const GpuBodyState& ballB = in.bodies[0];
				CuVec3 pos = GV(ballB.pos) * CU_BT_TO_UU;
				float relNormalY = normalWorldOnB.y * bs.hsYTargetDir;
				float relY = pos.y * bs.hsYTargetDir;
				if (relNormalY <= -in.cc->hsWallBounceChangeYNormal &&
					relY >= in.cc->arenaExtentY - in.cc->hsWallBounceChangeYThresh) {
					bs.hsYTargetDir *= -1.0f;
					CuVec3 pos2 = GV(ballB.pos) * CU_BT_TO_UU;
					CuVec3 vel = GV(ballB.linVel) * CU_BT_TO_UU;
					CuVec3 goalTargetPos = { 0, in.cc->hsTargetY * bs.hsYTargetDir, in.cc->hsTargetZ };
					CuVec3 dirToGoal = CuVecNormalizedRS(goalTargetPos - pos2);
					CuVec3 bounceDir = dirToGoal * (1.0f - in.cc->hsWallBounceUpFrac) +
						CuVec3{ 0, 0, 1 } * in.cc->hsWallBounceUpFrac;
					float velLenSq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
					float velLen = velLenSq > 0 ? sqrtf(velLenSq) : 0.0f;
					CuVec3 bounceImpulse = (bounceDir * velLen) * in.cc->hsWallBounceForceScale;
					CuVec3 cache = GV(bs.velocityImpulseCache) + bounceImpulse * CU_UU_TO_BT;
					bs.velocityImpulseCache = { cache.x, cache.y, cache.z };
				}
			}
		} else if (in.cc->gameMode == CU_GAMEMODE_SNOWDAY) {
			if (!in.ball->groundStickApplied) {
				GpuBodyState& ballB = in.bodies[0];
				CuVec3 f = GV(ballB.totalForce) +
					CuNeg(normalWorldOnB) * in.cc->puckGroundStickForce;
				ballB.totalForce = { f.x, f.y, f.z };
				in.ball->groundStickApplied = 1;
			}
		}
	}
}

#define CU_PAIR_EVENT_CAP 8

struct CuContactEvent {
	CuVec3 localPointA, localPointB, normalWorldOnB;
};

enum class CuSinkMode : int { Log, Overflow, Apply };

struct CuContactSink {
	CuSinkMode mode;
	int bodyA, bodyB;
	int numEvents;
	CuContactEvent events[CU_PAIR_EVENT_CAP];
};

__device__ inline void CuContactAdded(const CuCollideIn& in, CuContactSink& sink,
	CuManifoldPoint& cp, int bodyA, int bodyB) {
	if (!CuBodyHasResponse(in, bodyA) || !CuBodyHasResponse(in, bodyB))
		return;
	CuContactOverrides(in, cp, bodyA, bodyB);
	if (sink.mode == CuSinkMode::Apply) {
		CuContactEffects(in, bodyA, bodyB, cp.localPointA, cp.localPointB, cp.normalWorldOnB);
		return;
	}
	if (sink.numEvents == CU_PAIR_EVENT_CAP) {
		sink.mode = CuSinkMode::Overflow;
		return;
	}
	sink.bodyA = bodyA;
	sink.bodyB = bodyB;
	sink.events[sink.numEvents++] = { cp.localPointA, cp.localPointB, cp.normalWorldOnB };
}

__device__ inline int CuCollideBroadphase(const CuCollideIn& in, const GpuDynProxy* proxies,
	const int* staticCand, int2* pairs) {
#ifdef RS_GPU_PHASE_PROF
	long long profT = clock64();
#endif
	const int numPairs = CuCollideGenPairs(in, proxies, pairs, staticCand);
	for (int i = 0; i < numPairs; i++)
		in.pairsOut[i] = pairs[i];
	*in.numPairsOut = numPairs;
#ifdef RS_GPU_PHASE_PROF
	if (in.prof) {
		atomicAdd(&in.prof[1], (unsigned long long)(clock64() - profT));
		atomicAdd(&in.prof[3], (unsigned long long)numPairs);
	}
#endif
	return numPairs;
}

struct CuStaticPairFrame {
	CuContactBodyInfo dynInfo, staticInfo;
	CuMat3 convexBasis;
	CuVec3 convexOrigin;
	float breaking;
};

__device__ inline CuStaticPairFrame CuStaticPairFrameOf(const CuCollideIn& in, int d,
	const GpuStaticObj& so) {
	const CuMat3 ident = CuIdentity3();
	const GpuBodyState& ds = in.bodies[d];
	const CuMat3 dynBasis = CuLoadBasis(ds.basis);
	const CuVec3 dynPos = GV(ds.pos);
	const bool isBall = d == 0;
	const float dynBreaking = isBall ? in.cc->ballBreakingThreshold : in.cc->carBreakingThreshold;
	CuStaticPairFrame f;
	f.staticInfo = { ident,
		so.kind == 1 ? GV(in.geo.planes[so.shapeIdx].origin) : CuVec3{ 0, 0, 0 },
		so.friction, so.restitution, true, -1 };
	f.breaking = dynBreaking < so.breakingThreshold ? dynBreaking : so.breakingThreshold;
	f.dynInfo = isBall
		? CuContactBodyInfo{ dynBasis, dynPos,
			in.cc->ballFriction, in.cc->ballRestitution, false, CU_USERIDX_BALL }
		: CuContactBodyInfo{ dynBasis, dynPos,
			in.cc->carFriction, in.cc->carRestitution, false, CU_USERIDX_CAR };
	f.convexBasis = isBall ? dynBasis : CuMat3Mul(dynBasis, ident);
	f.convexOrigin = isBall ? dynPos
		: CuTransformPoint(dynBasis, dynPos, GV(in.cc->carChildOffset));
	return f;
}

__device__ inline void CuMeshTaskBuild(const CuCollideIn& in, int2 pair, CuMeshTask& t) {
	const int d = pair.y - in.numStatics;
	const GpuStaticObj& so = in.statics[pair.x];
	const CuStaticPairFrame f = CuStaticPairFrameOf(in, d, so);
	CuGjkShape convex = {};
	if (d != 0) {
		convex.kind = CU_SHAPE_BOX;
		convex.boxHalfExtNoMargin = GV(in.cc->carBoxHalfExtNoMargin);
		convex.margin = in.cc->carBoxMargin;
	} else if (in.cc->ballIsHull) {
		convex.kind = CU_SHAPE_HULL;
		convex.margin = in.cc->ballHullMargin;
		convex.hullPts = in.geo.hullPoints;
		convex.hullN = in.geo.numHullPoints;
		convex.hullLocalAabbMin = GV(in.cc->ballHullLocalAabbMin);
		convex.hullLocalAabbMax = GV(in.cc->ballHullLocalAabbMax);
	} else {
		convex.kind = CU_SHAPE_SPHERE;
		convex.margin = in.cc->ballRadius;
	}
	const GpuMeshBvh& mb = in.bvhs[so.shapeIdx];
	const CuBvhView bvh = { reinterpret_cast<const uint4*>(in.bvhNodes + mb.nodeStart),
		in.bvhSubtreeBoxes + mb.subtreeStart,
		mb.numSubtrees, GV(mb.bvhAabbMin), GV(mb.bvhAabbMax), GV(mb.bvhQuantization) };
	const int triStart = in.geo.meshes[so.shapeIdx].triStart;
	const float meshMargin = in.meshMargins[so.shapeIdx];

	const CuMat3 ident = CuIdentity3();
	const CuVec3 meshInvOrigin = CuMat3MulVec(ident, CuNeg({ 0, 0, 0 }));
	const CuMat3 convexInTriBasis = CuMat3Mul(ident, f.convexBasis);
	const CuVec3 convexInTriOrigin = CuTransformPoint(ident, meshInvOrigin, f.convexOrigin);
	CuVec3 aabbMin, aabbMax;
	CuShapeAabb(convex, convexInTriBasis, convexInTriOrigin, aabbMin, aabbMax);
	const CuVec3 extra = { meshMargin, meshMargin, meshMargin };
	aabbMax = aabbMax + extra;
	aabbMin = aabbMin - extra;

	t.convex = convex;
	t.convexBody = f.dynInfo;
	t.meshBody = f.staticInfo;
	t.convexBasis = f.convexBasis;
	t.basisInv = CuMat3Inverse(f.convexBasis);
	t.convexOrigin = f.convexOrigin;
	t.aabbMin = aabbMin;
	t.aabbMax = aabbMax;
	t.q = CuBvhQuantizeAabb(bvh, aabbMin, aabbMax);
	t.nodes = bvh.nodes;
	t.subtreeBoxes = bvh.subtreeBoxes;
	t.cull = in.triCull + triStart;
	t.tris = in.geo.meshTris + (size_t)triStart * 3;
	t.triInfos = in.triInfos + triStart;
	t.numSubtrees = bvh.numSubtrees;
	t.meshMargin = meshMargin;
	t.threshold = f.breaking;
}

__device__ inline void CuMeshPhase(const CuCollideIn& in, const CuWarpMeshSmem& wsm,
	const CuLaneGroup& g, CuManifold& m, CuContactSink& sink, bool meshPending, int2 meshPair) {
	const unsigned full = 0xffffffffu;
	const int warpLane = g.base + g.lane;
	const int numGroups = 32 >> g.shift;
	const int myGroup = g.base >> g.shift;
	const CuMeshSmem sm = wsm.Of(myGroup);
	const CuMeshTask& t = *sm.task;
	const int h0 = meshPair.x;
	const int d = meshPair.y - in.numStatics;
	unsigned meshBits = g.Ballot(meshPending);
	bool hasPair = false;
	bool owner = false;
	bool added = false;
	CuMeshWalk walk;
	walk.Reset();
	while (true) {
		if (!hasPair && meshBits) {
			const int ownerLane = __ffs(meshBits) - 1;
			meshBits &= meshBits - 1;
			owner = g.lane == ownerLane;
			if (owner)
				CuMeshTaskBuild(in, meshPair, *sm.task);
			g.Sync();
			walk.Reset();
			hasPair = true;
		}
		if (!__any_sync(full, hasPair))
			break;
		if (hasPair)
			CuMeshWalkFill(sm, g, walk);
		__syncwarp(full);
		const int pending = hasPair ? walk.numSurvivors : 0;
		int first[4];
		int total = 0, myFirst = 0;
#pragma unroll
		for (int k = 0; k < 4; k++) {
			first[k] = total;
			const int c = __shfl_sync(full, pending, (k * g.lanes) & 31);
			if (k < numGroups) {
				if (k < myGroup)
					myFirst += c;
				total += c;
			}
		}
		for (int base = 0; base < total; base += 32) {
			const int item = base + warpLane;
			CuTriHit hit = {};
			if (item < total) {
				int group = 0, groupFirst = 0;
#pragma unroll
				for (int k = 1; k < 4; k++) {
					if (k < numGroups && item >= first[k]) {
						group = k;
						groupFirst = first[k];
					}
				}
				const CuMeshSmem other = wsm.Of(group);
				hit = CuMeshTriangleTest(*other.task, other.survivors[item - groupFirst]);
			}
			const int lo = myFirst > base ? myFirst : base;
			const int hi = myFirst + pending < base + 32 ? myFirst + pending : base + 32;
			const int count = hi > lo ? hi - lo : 0;
			const int maxCount = (int)__reduce_max_sync(full, (unsigned)count);
			for (int k = 0; k < maxCount; k++) {
				const CuTriHit h = CuShflTriHit(hit, (lo + k - base) & 31);
				if (!owner || k >= count || !h.hasContact)
					continue;
				const int triIdx = sm.survivors[lo + k - myFirst];
				const int idx = CuAddContactPoint(m, t.convexBody, t.meshBody, false,
					h.normal, h.point, h.depth, -1, d == 0 ? -1 : 0, 0, triIdx);
				if (idx < 0)
					continue;
				added = true;
				CuManifoldPoint& cp = m.pts[idx];
				CuContactAdded(in, sink, cp, d, -1 - h0);
				cp.normalWorldOnB = h.edgeNormal;
				cp.positionWorldOnB = h.edgePointB;
				cp.localPointB = h.edgeLocalB;
			}
			__syncwarp(full);
		}
		if (hasPair) {
			walk.numSurvivors = 0;
			if (walk.done) {
				if (owner && added)
					CuManifoldRefresh(m, t.convexBody.basis, t.convexBody.origin,
						t.meshBody.basis, t.meshBody.origin);
				g.Sync();
				hasPair = false;
			}
		}
	}
}

__device__ inline bool CuCollidePair(const CuCollideIn& in, int2 pair, CuManifold& m,
	CuContactSink& sink, bool& meshPending) {
	const CuMat3 ident = CuIdentity3();
	const int h0 = pair.x, h1 = pair.y;

	if (h0 < in.numStatics) {
		const int d = h1 - in.numStatics;
		const GpuStaticObj& so = in.statics[h0];
		if (!CuIsActive(in.bodies[d].activationState))
			return false;

		const CuStaticPairFrame f = CuStaticPairFrameOf(in, d, so);
		const CuMat3 dynBasis = f.dynInfo.basis;
		const CuVec3 dynPos = f.dynInfo.origin;

		if (d == 0 && so.kind == 2) {
			const GpuTileShape& ts = in.geo.tiles[so.shapeIdx];
			CuVec3 tileOrigin = GV(ts.origin);
			CuContactBodyInfo tileInfo = { ident, tileOrigin,
				so.friction, so.restitution, true, CU_USERIDX_DROPSHOT_TILE };
			CuManifoldBegin(m, -1 - h0, 0, f.breaking);
			CuGjkShape tileShape = {};
			tileShape.kind = CU_SHAPE_HULL;
			tileShape.margin = in.cc->tileMargin;
			tileShape.hullPts = ts.pts;
			tileShape.hullN = 6;
			tileShape.hullLocalAabbMin = GV(ts.localAabbMin);
			tileShape.hullLocalAabbMax = GV(ts.localAabbMax);
			CuGjkShape ballShape = {};
			ballShape.kind = CU_SHAPE_SPHERE;
			ballShape.margin = in.cc->ballRadius;
			float maxd = in.cc->tileMargin + in.cc->ballRadius + f.breaking;
			CuGjkResult r;
			CuGjkClosestPoints(tileShape, ident, tileOrigin, ballShape, dynBasis, dynPos,
				maxd * maxd, true, r);
			if (r.hasContact) {
				int idx = CuAddContactPoint(m, tileInfo, f.dynInfo, false,
					r.normalOnB, r.pointOnBWorld, r.distance, -1, -1, -1, -1);
				if (idx >= 0)
					CuContactAdded(in, sink, m.pts[idx], 0, -1 - h0);
			}
			CuManifoldRefresh(m, ident, tileOrigin, dynBasis, dynPos);
			return true;
		}
		const bool isBall = d == 0;
		CuManifoldBegin(m, d, -1 - h0, f.breaking);
		if (!isBall) {
			CuVec3 cmn, cmx;
			CuBoxAabbAt(GV(in.cc->carBoxHalfExtNoMargin), in.cc->carBoxMargin,
				f.convexBasis, f.convexOrigin, cmn, cmx);
			if (!CuTestAabb2(cmn, cmx, GV(so.shapeAabbMin), GV(so.shapeAabbMax)))
				return false;
		}
		if (so.kind == 0) {
			meshPending = true;
			return true;
		}
		const GpuPlane& pl = in.geo.planes[so.shapeIdx];
		CuConvexShape convex = !isBall
			? CuConvexShape{ CU_SHAPE_BOX, 0.0f,
				{ in.cc->carBoxHalfExtNoMargin.x + in.cc->carBoxMargin,
				  in.cc->carBoxHalfExtNoMargin.y + in.cc->carBoxMargin,
				  in.cc->carBoxHalfExtNoMargin.z + in.cc->carBoxMargin } }
			: in.cc->ballIsHull
			? CuConvexShape{ CU_SHAPE_HULL, in.cc->ballHullMargin, { 0, 0, 0 },
				in.geo.hullPoints, in.geo.numHullPoints }
			: CuConvexShape{ CU_SHAPE_SPHERE, in.cc->ballRadius, { 0, 0, 0 } };
		CuVec3 nrm, pt;
		float dist;
		if (CuConvexPlaneCollide(convex, f.convexBasis, f.convexOrigin, GV(pl.normal),
				pl.planeConst, ident, GV(pl.origin), f.breaking, nrm, pt, dist)) {
			int idx = CuAddContactPoint(m, f.staticInfo, f.dynInfo, true,
				nrm, pt, dist, -1, -1, -1, isBall ? -1 : 0);
			if (idx >= 0)
				CuContactAdded(in, sink, m.pts[idx], d, -1 - h0);
		}
		if (m.numContacts)
			CuManifoldRefresh(m, dynBasis, dynPos, ident, GV(pl.origin));
		return true;
	} else if (h0 == in.numStatics) {
		int c = h1 - in.numStatics - 1;
		const GpuBodyState& ballS = in.bodies[0];
		const GpuBodyState& carS = in.bodies[1 + c];
		if (!CuIsActive(ballS.activationState) && !CuIsActive(carS.activationState))
			return false;
		CuMat3 ballBasis = CuLoadBasis(ballS.basis);
		CuVec3 ballPos = GV(ballS.pos);
		CuMat3 carBasis = CuLoadBasis(carS.basis);
		CuVec3 carPos = GV(carS.pos);

		CuMat3 childBasis = CuMat3Mul(carBasis, ident);
		CuVec3 childOrigin = CuTransformPoint(carBasis, carPos, GV(in.cc->carChildOffset));
		CuVec3 cmn, cmx, bmn, bmx;
		CuBoxAabbAt(GV(in.cc->carBoxHalfExtNoMargin), in.cc->carBoxMargin,
			childBasis, childOrigin, cmn, cmx);
		if (in.cc->ballIsHull)
			CuHullAabbAt(*in.cc, ballBasis, ballPos, bmn, bmx);
		else
			CuSphereAabbAt(ballPos, in.cc->ballRadius, bmn, bmx);
		if (!CuTestAabb2(cmn, cmx, bmn, bmx))
			return false;

		float breaking = in.cc->ballBreakingThreshold < in.cc->carBreakingThreshold
			? in.cc->ballBreakingThreshold : in.cc->carBreakingThreshold;
		CuManifoldBegin(m, 1 + c, 0, breaking);

		CuGjkShape box = {};
		box.kind = CU_SHAPE_BOX;
		box.boxHalfExtNoMargin = GV(in.cc->carBoxHalfExtNoMargin);
		box.margin = in.cc->carBoxMargin;
		CuGjkShape ballShape = {};
		if (in.cc->ballIsHull) {
			ballShape.kind = CU_SHAPE_HULL;
			ballShape.margin = in.cc->ballHullMargin;
			ballShape.hullPts = in.geo.hullPoints;
			ballShape.hullN = in.geo.numHullPoints;
			ballShape.hullLocalAabbMin = GV(in.cc->ballHullLocalAabbMin);
			ballShape.hullLocalAabbMax = GV(in.cc->ballHullLocalAabbMax);
		} else {
			ballShape.kind = CU_SHAPE_SPHERE;
			ballShape.margin = in.cc->ballRadius;
		}
		float maxd = in.cc->carBoxMargin + ballShape.margin + breaking;
		CuGjkResult r;
		CuGjkClosestPoints(box, childBasis, childOrigin, ballShape, ballBasis, ballPos,
			maxd * maxd, true, r);
		CuContactBodyInfo ballInfo = { ballBasis, ballPos,
			in.cc->ballFriction, in.cc->ballRestitution, false, CU_USERIDX_BALL };
		CuContactBodyInfo carInfo = { carBasis, carPos,
			in.cc->carFriction, in.cc->carRestitution, false, CU_USERIDX_CAR };
		if (r.hasContact) {
			int idx = CuAddContactPoint(m, ballInfo, carInfo, true,
				r.normalOnB, r.pointOnBWorld, r.distance, -1, -1, -1, 0);
			if (idx >= 0)
				CuContactAdded(in, sink, m.pts[idx], 1 + c, 0);
		}
		CuManifoldRefresh(m, carBasis, carPos, ballBasis, ballPos);
		return true;
	} else {
		int cLO = h0 - in.numStatics - 1;
		int cHI = h1 - in.numStatics - 1;
		const GpuBodyState& sLO = in.bodies[1 + cLO];
		const GpuBodyState& sHI = in.bodies[1 + cHI];
		if (!CuIsActive(sLO.activationState) && !CuIsActive(sHI.activationState))
			return false;
		CuMat3 basisLO = CuLoadBasis(sLO.basis);
		CuVec3 posLO = GV(sLO.pos);
		CuMat3 basisHI = CuLoadBasis(sHI.basis);
		CuVec3 posHI = GV(sHI.pos);

		CuManifoldBegin(m, 1 + cHI, 1 + cLO, in.cc->carBreakingThreshold);

		CuMat3 childBasisLO = CuMat3Mul(basisLO, ident);
		CuVec3 childOriginLO = CuTransformPoint(basisLO, posLO, GV(in.cc->carChildOffset));
		CuVec3 lmn, lmx;
		CuBoxAabbAt(GV(in.cc->carBoxHalfExtNoMargin), in.cc->carBoxMargin,
			childBasisLO, childOriginLO, lmn, lmx);
		CuVec3 hmn, hmx;
		CuCompoundAabbAt(*in.cc, basisHI, posHI, hmn, hmx);
		bool overlapOuter = CuTestAabb2(lmn, lmx, hmn, hmx);
		if (overlapOuter) {
			CuMat3 childBasisHI = CuMat3Mul(basisHI, ident);
			CuVec3 childOriginHI = CuTransformPoint(basisHI, posHI, GV(in.cc->carChildOffset));
			CuVec3 hcmn, hcmx;
			CuBoxAabbAt(GV(in.cc->carBoxHalfExtNoMargin), in.cc->carBoxMargin,
				childBasisHI, childOriginHI, hcmn, hcmx);
			bool overlapNested = CuTestAabb2(hcmn, hcmx, lmn, lmx);
			if (overlapNested) {
				CuVec3 heWithMargin = {
					in.cc->carBoxHalfExtNoMargin.x + in.cc->carBoxMargin,
					in.cc->carBoxHalfExtNoMargin.y + in.cc->carBoxMargin,
					in.cc->carBoxHalfExtNoMargin.z + in.cc->carBoxMargin };
				CuBoxBoxOut bb;
				CuBoxBoxClosestPoints(childBasisHI, childOriginHI, heWithMargin,
					childBasisLO, childOriginLO, heWithMargin, bb);
				CuContactBodyInfo loInfo = { basisLO, posLO,
					in.cc->carFriction, in.cc->carRestitution, false, CU_USERIDX_CAR };
				CuContactBodyInfo hiInfo = { basisHI, posHI,
					in.cc->carFriction, in.cc->carRestitution, false, CU_USERIDX_CAR };
				for (int bi = 0; bi < bb.n; bi++) {
					int idx = CuAddContactPoint(m, loInfo, hiInfo, true,
						bb.c[bi].normal, bb.c[bi].point, bb.c[bi].depth, -1, 0, -1, 0);
					if (idx >= 0)
						CuContactAdded(in, sink, m.pts[idx], 1 + cHI, 1 + cLO);
				}
				CuManifoldRefresh(m, basisHI, posHI, basisLO, posLO);
				return true;
			}
		}
	}
	return false;
}

__device__ inline void CuCollideNarrowphase(const CuCollideIn& in, const int2* pairs, int numPairs,
	bool active, int lane, int lanes, int groupBase, const CuWarpMeshSmem& meshSmem) {
#ifdef RS_GPU_PHASE_PROF
	long long profT = clock64();
#endif
	const unsigned laneBits = lanes >= 32 ? 0xffffffffu : (1u << lanes) - 1u;
	const unsigned groupMask = laneBits << groupBase;
	const CuLaneGroup group = { lane, lanes, __ffs(lanes) - 1, groupBase, groupMask, laneBits };
	int numManifolds = 0;
	int round = 0;
	int rerun = -1;
	CuManifold m;
	CuContactSink sink;
	while (true) {
		const bool more = rerun >= 0 || round < numPairs;
		if (!__any_sync(0xffffffffu, more))
			break;
		const bool serial = rerun >= 0;
		int pi = numPairs;
		bool kept = false;
		bool meshPending = false;
		if (more) {
			pi = round + lane;
			if (serial) {
				pi = lane == 0 ? round + rerun : numPairs;
				if (lane == 0)
					sink.mode = CuSinkMode::Apply;
			} else {
				sink.mode = CuSinkMode::Log;
				sink.bodyA = 0;
				sink.bodyB = 0;
				sink.numEvents = 0;
			}
			kept = pi < numPairs && CuCollidePair(in, pairs[pi], m, sink, meshPending);
		}
		CuMeshPhase(in, meshSmem, group, m, sink, meshPending,
			meshPending ? pairs[pi] : int2{ 0, 0 });
		if (!more)
			continue;
		int k = rerun + 1;
		rerun = -1;
		if (!serial) {
			const unsigned keptBits = (__ballot_sync(groupMask, kept) >> groupBase) & laneBits;
			if (kept)
				CuManifoldStore(in.manifolds[numManifolds + __popc(keptBits & ((1u << lane) - 1u))], m);
			numManifolds += __popc(keptBits);
		}
		const int roundPairs = numPairs - round < lanes ? numPairs - round : lanes;
		const bool overflow = sink.mode == CuSinkMode::Overflow;
		const bool logged = lane >= k && lane < roundPairs && (overflow || sink.numEvents > 0);
		const unsigned overflowBits = (__ballot_sync(groupMask, overflow) >> groupBase) & laneBits;
		unsigned loggedBits = (__ballot_sync(groupMask, logged) >> groupBase) & laneBits;
		for (; loggedBits; loggedBits &= loggedBits - 1) {
			k = __ffs(loggedBits) - 1;
			if ((overflowBits >> k) & 1u) {
				rerun = k;
				break;
			}
			const int numEvents = __shfl_sync(groupMask, sink.numEvents, k, lanes);
			const int bodyA = __shfl_sync(groupMask, sink.bodyA, k, lanes);
			const int bodyB = __shfl_sync(groupMask, sink.bodyB, k, lanes);
			for (int e = 0; e < numEvents; e++) {
				const CuContactEvent ev = e < sink.numEvents ? sink.events[e] : CuContactEvent{};
				CuVec3 localPointA, localPointB, normalWorldOnB;
				localPointA.x = __shfl_sync(groupMask, ev.localPointA.x, k, lanes);
				localPointA.y = __shfl_sync(groupMask, ev.localPointA.y, k, lanes);
				localPointA.z = __shfl_sync(groupMask, ev.localPointA.z, k, lanes);
				localPointB.x = __shfl_sync(groupMask, ev.localPointB.x, k, lanes);
				localPointB.y = __shfl_sync(groupMask, ev.localPointB.y, k, lanes);
				localPointB.z = __shfl_sync(groupMask, ev.localPointB.z, k, lanes);
				normalWorldOnB.x = __shfl_sync(groupMask, ev.normalWorldOnB.x, k, lanes);
				normalWorldOnB.y = __shfl_sync(groupMask, ev.normalWorldOnB.y, k, lanes);
				normalWorldOnB.z = __shfl_sync(groupMask, ev.normalWorldOnB.z, k, lanes);
				if (lane == 0)
					CuContactEffects(in, bodyA, bodyB, localPointA, localPointB, normalWorldOnB);
			}
		}
		if (rerun < 0)
			round += lanes;
	}
	if (active && lane == 0)
		*in.numManifolds = numManifolds;
#ifdef RS_GPU_PHASE_PROF
	if (active && lane == 0 && in.prof) {
		atomicAdd(&in.prof[2], (unsigned long long)(clock64() - profT));
		atomicAdd(&in.prof[4], (unsigned long long)numManifolds);
	}
#endif
}

} // namespace rscuda
