#pragma once
// 2026-06-10: Device collision narrowphase, faithful to the work-tree bullet sources.
// Per-tick manifolds only (RS broadphase rebuilds all pairs every tick; RS
// getCacheEntry always returns -1 so points are only ever ADDED, never replaced).
// Sources: btManifoldResult.cpp, btPersistentManifold.cpp/h, SphereTriangleDetector.cpp,
// btSphereBoxCollisionAlgorithm.cpp, btConvexPlaneCollisionAlgorithm.cpp,
// btGjkPairDetector.cpp. Validated by the manifold-compare harness in rocketsim_bench.

#include "CudaMath.cuh"
#include "CudaConvexCast.cuh" // CuSimplexSolver (GJK reuses it)
#include "GpuTypes.h"

#ifdef __CUDACC__

namespace rscuda {

#define CU_CONTACT_BREAKING_THRESHOLD 0.02f // gContactBreakingThreshold

// One contact point (fields the simulation actually consumes downstream)
struct CuManifoldPoint {
	CuVec3 localPointA, localPointB;
	CuVec3 positionWorldOnA, positionWorldOnB;
	CuVec3 normalWorldOnB;
	float distance1;
	float combinedFriction, combinedRestitution;
	CuVec3 lateralFrictionDir1, lateralFrictionDir2;
	int partId0, partId1, index0, index1;
	float appliedImpulse, appliedImpulseLateral1, appliedImpulseLateral2, prevRHS;
	int lifeTime;
	uint8_t isSpecial; // RS m_isSpecial (ball-world contacts outside snowday)
};

// Per-pair manifold (4-point cache)
struct CuManifold {
	int bodyA, bodyB;        // dynamic body indices; -1-staticIdx for statics (kind enc.)
	int numContacts;
	CuManifoldPoint pts[4];
	float contactBreakingThreshold; // min of both shapes' getAngularMotionDisc()*0.02
	float contactProcessingThreshold; // gContactProcessingThreshold default BT_LARGE_FLOAT
};

// btPersistentManifold::calcArea4Points (btPersistentManifold.cpp:90)
__device__ __forceinline__ float CuCalcArea4Points(CuVec3 p0, CuVec3 p1, CuVec3 p2, CuVec3 p3) {
	CuVec3 a0 = p0 - p1, a1 = p0 - p2, a2 = p0 - p3;
	CuVec3 b0 = p2 - p3, b1 = p1 - p3, b2 = p1 - p2;
	CuVec3 tmp0 = CuCross(a0, b0);
	CuVec3 tmp1 = CuCross(a1, b1);
	CuVec3 tmp2 = CuCross(a2, b2);
	float l0 = CuLength2(tmp0), l1 = CuLength2(tmp1), l2 = CuLength2(tmp2);
	float m = l0 > l1 ? l0 : l1;
	return m > l2 ? m : l2;
}

// btVector4::closestAxis4 over non-negative inputs = index of max, FIRST wins ties
// (sequential > comparisons as in btVector3.h maxAxis4)
__device__ __forceinline__ int CuMaxAxis4(float a, float b, float c, float d) {
	int maxIndex = -1;
	float maxVal = -3.402823466e+38f; // -BT_LARGE_FLOAT
	if (a > maxVal) { maxIndex = 0; maxVal = a; }
	if (b > maxVal) { maxIndex = 1; maxVal = b; }
	if (c > maxVal) { maxIndex = 2; maxVal = c; }
	if (d > maxVal) { maxIndex = 3; maxVal = d; }
	return maxIndex;
}

// btPersistentManifold::sortCachedPoints (btPersistentManifold.cpp:110),
// gContactCalcArea3Points = true path
__device__ __forceinline__ int CuSortCachedPoints(const CuManifold& m, const CuManifoldPoint& pt) {
	int maxPenetrationIndex = -1;
	float maxPenetration = pt.distance1;
	for (int i = 0; i < 4; i++) {
		if (m.pts[i].distance1 < maxPenetration) {
			maxPenetrationIndex = i;
			maxPenetration = m.pts[i].distance1;
		}
	}
	float res0 = 0, res1 = 0, res2 = 0, res3 = 0;
	if (maxPenetrationIndex != 0) {
		CuVec3 a0 = pt.localPointA - m.pts[1].localPointA;
		CuVec3 b0 = m.pts[3].localPointA - m.pts[2].localPointA;
		res0 = CuLength2(CuCross(a0, b0));
	}
	if (maxPenetrationIndex != 1) {
		CuVec3 a1 = pt.localPointA - m.pts[0].localPointA;
		CuVec3 b1 = m.pts[3].localPointA - m.pts[2].localPointA;
		res1 = CuLength2(CuCross(a1, b1));
	}
	if (maxPenetrationIndex != 2) {
		CuVec3 a2 = pt.localPointA - m.pts[0].localPointA;
		CuVec3 b2 = m.pts[3].localPointA - m.pts[1].localPointA;
		res2 = CuLength2(CuCross(a2, b2));
	}
	if (maxPenetrationIndex != 3) {
		CuVec3 a3 = pt.localPointA - m.pts[0].localPointA;
		CuVec3 b3 = m.pts[2].localPointA - m.pts[1].localPointA;
		res3 = CuLength2(CuCross(a3, b3));
	}
	return CuMaxAxis4(res0, res1, res2, res3);
}

// btPersistentManifold::addManifoldPoint with RS getCacheEntry==-1 (always add)
__device__ __forceinline__ int CuManifoldAddPoint(CuManifold& m, const CuManifoldPoint& pt) {
	int insertIndex = m.numContacts;
	if (insertIndex == 4) {
		insertIndex = CuSortCachedPoints(m, pt);
		if (insertIndex < 0)
			insertIndex = 0;
	} else {
		m.numContacts++;
	}
	m.pts[insertIndex] = pt;
	return insertIndex;
}

// btPersistentManifold::removeContactPoint (swap-with-last, order-sensitive)
__device__ __forceinline__ void CuManifoldRemovePoint(CuManifold& m, int index) {
	int lastUsedIndex = m.numContacts - 1;
	if (index != lastUsedIndex)
		m.pts[index] = m.pts[lastUsedIndex];
	m.numContacts--;
}

// btPersistentManifold::refreshContactPoints (btPersistentManifold.cpp:250):
// recompute world positions/distance, prune by distance and tangential drift.
// trA/trB are the bodies' CURRENT world transforms. Game contact-processed callback
// is null in RocketSim.
__device__ __forceinline__ void CuManifoldRefresh(CuManifold& m,
	const CuMat3& basisA, CuVec3 originA, const CuMat3& basisB, CuVec3 originB) {
	for (int i = m.numContacts - 1; i >= 0; i--) {
		CuManifoldPoint& p = m.pts[i];
		p.positionWorldOnA = CuTransformPoint(basisA, originA, p.localPointA);
		p.positionWorldOnB = CuTransformPoint(basisB, originB, p.localPointB);
		p.distance1 = CuDot3(p.positionWorldOnA - p.positionWorldOnB, p.normalWorldOnB);
		p.lifeTime++;
	}
	for (int i = m.numContacts - 1; i >= 0; i--) {
		CuManifoldPoint& p = m.pts[i];
		if (!(p.distance1 <= m.contactBreakingThreshold)) { // validContactDistance
			CuManifoldRemovePoint(m, i);
		} else {
			CuVec3 projectedPoint = p.positionWorldOnA - p.normalWorldOnB * p.distance1;
			CuVec3 projectedDifference = p.positionWorldOnB - projectedPoint;
			float distance2d = CuDot3(projectedDifference, projectedDifference);
			if (distance2d > m.contactBreakingThreshold * m.contactBreakingThreshold)
				CuManifoldRemovePoint(m, i);
		}
	}
}

// ---- btManifoldResult::addContactPoint (btManifoldResult.cpp:109) ----
// The friction/restitution combine rules are RS-MODIFIED (static -> min/max).
// Game-callback effects (Arena::_BulletContactAddedCallback) are applied by the
// CALLER after this returns the inserted index, mirroring CPU order.

struct CuContactBodyInfo {
	CuMat3 basis;       // world transform (for invXform)
	CuVec3 origin;
	float friction, restitution;
	bool isStatic;
	int userIndex;      // BT_USERINFO_TYPE_CAR/BALL or -1 world
};

// btTransform::invXform(point): basisT * (point - origin)
__device__ __forceinline__ CuVec3 CuInvXform(const CuMat3& basis, CuVec3 origin, CuVec3 p) {
	CuVec3 v = p - origin;
	return CuMat3MulVec(CuMat3Transpose(basis), v);
}

// Returns inserted index, or -1 if rejected. partId/index are the swap-normalized
// triangle ids (m_partId0 etc. already arranged by caller as bullet does).
__device__ __forceinline__ int CuAddContactPoint(
	CuManifold& m,
	const CuContactBodyInfo& body0, const CuContactBodyInfo& body1, bool manifoldSwapped,
	CuVec3 normalOnBInWorld, CuVec3 pointInWorld, float depth,
	int partId0, int index0, int partId1, int index1) {

	if (depth > m.contactBreakingThreshold)
		return -1;

	bool isSwapped = manifoldSwapped; // manifold body0 != result body0
	CuVec3 pointA = pointInWorld + normalOnBInWorld * depth;

	CuManifoldPoint pt = {};
	if (isSwapped) {
		pt.localPointA = CuInvXform(body1.basis, body1.origin, pointA);
		pt.localPointB = CuInvXform(body0.basis, body0.origin, pointInWorld);
	} else {
		pt.localPointA = CuInvXform(body0.basis, body0.origin, pointA);
		pt.localPointB = CuInvXform(body1.basis, body1.origin, pointInWorld);
	}
	pt.positionWorldOnA = pointA;
	pt.positionWorldOnB = pointInWorld;
	pt.normalWorldOnB = normalOnBInWorld;
	pt.distance1 = depth;

	// RS-modified combine rules (btManifoldResult.cpp:56/67)
	{
		float a = body0.friction, b = body1.friction;
		pt.combinedFriction = (body0.isStatic || body1.isStatic) ? ((a < b) ? a : b) : a * b;
		float ra = body0.restitution, rb = body1.restitution;
		pt.combinedRestitution = (body0.isStatic || body1.isStatic) ? ((ra > rb) ? ra : rb) : ra * rb;
	}

	CuPlaneSpace1(pt.normalWorldOnB, pt.lateralFrictionDir1, pt.lateralFrictionDir2);

	if (isSwapped) {
		pt.partId0 = partId1; pt.partId1 = partId0;
		pt.index0 = index1; pt.index1 = index0;
	} else {
		pt.partId0 = partId0; pt.partId1 = partId1;
		pt.index0 = index0; pt.index1 = index1;
	}

	return CuManifoldAddPoint(m, pt);
}

// ---- SphereTriangleDetector (RS/VirxEC variant) ----

// closestPointTriangle (SphereTriangleDetector.cpp, embree-style)
__device__ __forceinline__ CuVec3 CuClosestPointTriangle(CuVec3 p, CuVec3 a, CuVec3 b, CuVec3 c) {
	CuVec3 ab = b - a;
	CuVec3 ac = c - a;
	CuVec3 ap = p - a;

	float d1 = CuDot3(ab, ap);
	float d2 = CuDot3(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f) return a;

	CuVec3 bp = p - b;
	float d3 = CuDot3(ab, bp);
	float d4 = CuDot3(ac, bp);
	if (d3 >= 0.0f && d4 <= d3) return b;

	CuVec3 cp = p - c;
	float d5 = CuDot3(ab, cp);
	float d6 = CuDot3(ac, cp);
	if (d6 >= 0.0f && d5 <= d6) return c;

	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		float v = d1 / (d1 - d3);
		return a + ab * v;
	}
	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		float v = d2 / (d2 - d6);
		return a + ac * v;
	}
	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + (c - b) * v;
	}
	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;
	return a + ab * v + ac * w;
}

// pointInTriangle barycentric variant (SphereTriangleDetector.cpp tail)
__device__ __forceinline__ bool CuPointInTriangle(CuVec3 p, CuVec3 p0, CuVec3 p1, CuVec3 p2) {
	CuVec3 u = p1 - p0;
	CuVec3 v = p2 - p0;
	CuVec3 n = CuCross(u, v);
	float nLenSq = CuDot3(n, n);
	CuVec3 w = p - p0;
	float gamma = CuDot3(CuCross(u, w), n) / nLenSq;
	float beta = CuDot3(CuCross(w, v), n) / nLenSq;
	float alpha = 1 - gamma - beta;
	return ((0 <= alpha) && (alpha <= 1) && (0 <= beta) && (beta <= 1) && (0 <= gamma) && (gamma <= 1));
}

// SphereTriangleDetector::collide in TRIANGLE LOCAL space. Returns contact in local
// space (caller transforms to world like getClosestPoints does).
__device__ __forceinline__ bool CuSphereTriangleCollide(
	CuVec3 sphereCenterLocal, float radius,
	CuVec3 v0, CuVec3 v1, CuVec3 v2,
	float contactBreakingThreshold,
	CuVec3& point, CuVec3& resultNormal, float& depth) {

	float radiusWithThreshold = radius + contactBreakingThreshold;
	CuVec3 normal = CuCross(v1 - v0, v2 - v0);
	float l2 = CuLength2(normal);
	bool hasContact = false;
	CuVec3 contactPoint;

	if (l2 >= CU_SIMD_EPSILON * CU_SIMD_EPSILON) {
		// normal /= sqrt(l2): operator/= = *(1/s)
		float inv = 1.0f / sqrtf(l2);
		normal = normal * inv;

		CuVec3 p1ToCentre = sphereCenterLocal - v0;
		float distanceFromPlane = CuDot3(p1ToCentre, normal);
		if (distanceFromPlane < 0.0f) {
			distanceFromPlane *= -1.0f;
			normal = normal * -1.0f;
		}
		bool isInsideContactPlane = distanceFromPlane < radiusWithThreshold;
		if (isInsideContactPlane) {
			if (CuPointInTriangle(sphereCenterLocal, v0, v1, v2)) {
				hasContact = true;
				contactPoint = sphereCenterLocal - normal * distanceFromPlane;
			} else {
				float contactCapsuleRadiusSqr = radiusWithThreshold * radiusWithThreshold;
				float minDistSqr = contactCapsuleRadiusSqr;
				CuVec3 nearestOnEdge = CuClosestPointTriangle(sphereCenterLocal, v0, v1, v2);
				CuVec3 d = nearestOnEdge - sphereCenterLocal; // distance2 tree
				float distanceSqr = CuLength2(d);
				if (distanceSqr < minDistSqr) {
					hasContact = true;
					contactPoint = nearestOnEdge;
				}
			}
		}
	}

	if (hasContact) {
		CuVec3 contactToCentre = sphereCenterLocal - contactPoint;
		float distanceSqr = CuLength2(contactToCentre);
		if (distanceSqr < radiusWithThreshold * radiusWithThreshold) {
			if (distanceSqr > CU_SIMD_EPSILON) {
				float distance = sqrtf(distanceSqr);
				resultNormal = CuNormalize3(contactToCentre); // normalize() rsqrt path
				point = contactPoint;
				depth = -(radius - distance);
			} else {
				resultNormal = normal;
				point = contactPoint;
				depth = -radius;
			}
			return true;
		}
	}
	return false;
}

// ---- btSphereBoxCollisionAlgorithm ----
// Sphere center in box-local space; returns world-space contact via caller transforms.
__device__ __forceinline__ bool CuSphereBoxGetDistance(
	const CuMat3& boxBasis, CuVec3 boxOrigin, CuVec3 boxHalfExtWithoutMargin, float boxMargin,
	CuVec3 sphereCenterWorld, float radius, float maxContactDistance,
	CuVec3& pointOnBoxWorld, CuVec3& normalWorld, float& penetrationDepth) {

	CuVec3 sphereRelPos = CuInvXform(boxBasis, boxOrigin, sphereCenterWorld);

	CuVec3 closestPoint = sphereRelPos;
	closestPoint.x = fminf(boxHalfExtWithoutMargin.x, closestPoint.x);
	closestPoint.x = fmaxf(-boxHalfExtWithoutMargin.x, closestPoint.x);
	closestPoint.y = fminf(boxHalfExtWithoutMargin.y, closestPoint.y);
	closestPoint.y = fmaxf(-boxHalfExtWithoutMargin.y, closestPoint.y);
	closestPoint.z = fminf(boxHalfExtWithoutMargin.z, closestPoint.z);
	closestPoint.z = fmaxf(-boxHalfExtWithoutMargin.z, closestPoint.z);

	float intersectionDist = radius + boxMargin;
	float contactDist = intersectionDist + maxContactDistance;
	CuVec3 normal = sphereRelPos - closestPoint;

	float dist2 = CuLength2(normal);
	if (dist2 > contactDist * contactDist)
		return false;

	float distance;
	if (dist2 <= CU_SIMD_EPSILON) {
		// deep: face fallback (btSphereBoxCollisionAlgorithm::getSpherePenetration)
		float faceDist = boxHalfExtWithoutMargin.x - sphereRelPos.x;
		float minDist = faceDist;
		closestPoint = sphereRelPos;
		closestPoint.x = boxHalfExtWithoutMargin.x;
		normal = { 1, 0, 0 };
		faceDist = boxHalfExtWithoutMargin.x + sphereRelPos.x;
		if (faceDist < minDist) {
			minDist = faceDist;
			closestPoint = sphereRelPos;
			closestPoint.x = -boxHalfExtWithoutMargin.x;
			normal = { -1, 0, 0 };
		}
		faceDist = boxHalfExtWithoutMargin.y - sphereRelPos.y;
		if (faceDist < minDist) {
			minDist = faceDist;
			closestPoint = sphereRelPos;
			closestPoint.y = boxHalfExtWithoutMargin.y;
			normal = { 0, 1, 0 };
		}
		faceDist = boxHalfExtWithoutMargin.y + sphereRelPos.y;
		if (faceDist < minDist) {
			minDist = faceDist;
			closestPoint = sphereRelPos;
			closestPoint.y = -boxHalfExtWithoutMargin.y;
			normal = { 0, -1, 0 };
		}
		faceDist = boxHalfExtWithoutMargin.z - sphereRelPos.z;
		if (faceDist < minDist) {
			minDist = faceDist;
			closestPoint = sphereRelPos;
			closestPoint.z = boxHalfExtWithoutMargin.z;
			normal = { 0, 0, 1 };
		}
		faceDist = boxHalfExtWithoutMargin.z + sphereRelPos.z;
		if (faceDist < minDist) {
			minDist = faceDist;
			closestPoint = sphereRelPos;
			closestPoint.z = -boxHalfExtWithoutMargin.z;
			normal = { 0, 0, -1 };
		}
		distance = -minDist;
	} else {
		distance = sqrtf(dist2); // length()
		float inv = 1.0f / distance; // operator/= scalar
		normal = normal * inv;
	}

	CuVec3 pointOnBox = closestPoint + normal * boxMargin;
	penetrationDepth = distance - intersectionDist;

	pointOnBoxWorld = CuTransformPoint(boxBasis, boxOrigin, pointOnBox);
	normalWorld = CuMat3MulVec(boxBasis, normal);
	return true;
}

// ---- btConvexPlaneCollisionAlgorithm::processCollision (perturbation dead at RS
// defaults). Convex = sphere (support margin*normalize) or box (support fsel WITH
// margin: localGetSupportingVertex adds margin to halfExtentsWithoutMargin). ----
__device__ __forceinline__ bool CuConvexPlaneCollide(
	const CuConvexShape& convex, const CuMat3& convexBasis, CuVec3 convexOrigin,
	CuVec3 planeNormal, float planeConst, const CuMat3& planeBasis, CuVec3 planeOrigin,
	float contactBreakingThreshold,
	CuVec3& normalOnSurfaceB, CuVec3& vtxInPlaneWorld, float& distance) {

	// planeInConvex = convexWT.inverse() * planeWT; convexInPlaneTrans = planeWT.inverse() * convexWT
	// btTransform::inverse: basisT, origin = basisT * -origin; compose trees
	CuMat3 convexBasisT = CuMat3Transpose(convexBasis);
	CuMat3 planeInConvexBasis = CuMat3Mul(convexBasisT, planeBasis);

	CuMat3 planeBasisT = CuMat3Transpose(planeBasis);
	CuVec3 planeInvOrigin = CuMat3MulVec(planeBasisT, CuNeg(planeOrigin));
	CuMat3 convexInPlaneBasis = CuMat3Mul(planeBasisT, convexBasis);
	CuVec3 convexInPlaneOrigin = CuTransformPoint(planeBasisT, planeInvOrigin, convexOrigin);

	CuVec3 supportDir = CuMat3MulVec(planeInConvexBasis, CuNeg(planeNormal));
	CuVec3 vtx;
	if (convex.kind == CU_SHAPE_BOX) {
		// localGetSupportingVertex = fsel on (halfExtentsWithoutMargin + margin);
		// boxHalfExtWithMargin already carries the sum
		vtx = CuSupportBox(convex.boxHalfExtWithMargin, supportDir);
	} else {
		vtx = CuSupportSphere(convex.sphereRadius, supportDir);
	}

	CuVec3 vtxInPlane = CuTransformPoint(convexInPlaneBasis, convexInPlaneOrigin, vtx);
	distance = CuDot3(planeNormal, vtxInPlane) - planeConst;

	if (!(distance < contactBreakingThreshold)) // + m_closestPointDistanceThreshold(0)
		return false;

	CuVec3 vtxInPlaneProjected = vtxInPlane - planeNormal * distance;
	vtxInPlaneWorld = CuTransformPoint(planeBasis, planeOrigin, vtxInPlaneProjected);
	normalOnSurfaceB = CuMat3MulVec(planeBasis, planeNormal);
	return true;
}

// ---- btGjkPairDetector::getClosestPointsNonVirtual (btGjkPairDetector.cpp:688) ----
// Shapes: support WITHOUT margin (box: fsel on implicitShapeDimensions =
// halfExtentsWithoutMargin; sphere: (0,0,0); triangle: max-dot vertex). Margins via
// getMarginNonVirtual (box margin, triangle margin, sphere radius) passed by caller.

#define CU_REL_ERROR2 1.0e-6f

struct CuGjkShape {
	int kind;                 // CU_SHAPE_SPHERE / CU_SHAPE_BOX / 3 = triangle
	CuVec3 boxHalfExtNoMargin; // implicitShapeDimensions
	CuVec3 triV[3];
	float margin;             // getMarginNonVirtual
};
#define CU_SHAPE_TRIANGLE 3

// btVector3::maxAxis comparison tree
__device__ __forceinline__ int CuMaxAxis3(float x, float y, float z) {
	return x < y ? (y < z ? 2 : 1) : (x < z ? 2 : 0);
}

__device__ __forceinline__ CuVec3 CuSupportNoMargin(const CuGjkShape& s, CuVec3 dir) {
	if (s.kind == CU_SHAPE_BOX)
		return CuSupportBox(s.boxHalfExtNoMargin, dir);
	if (s.kind == CU_SHAPE_TRIANGLE) {
		float d0 = CuDot3(dir, s.triV[0]);
		float d1 = CuDot3(dir, s.triV[1]);
		float d2 = CuDot3(dir, s.triV[2]);
		return s.triV[CuMaxAxis3(d0, d1, d2)];
	}
	return { 0, 0, 0 }; // sphere
}

// Shape getAabb for the fix-normal-direction branch (local transforms).
// Box: btTransformAabb - extent = (he+margin) dot3 abs(basis rows). Sphere: RS adds
// +0.08 to margin (btSphereShape.cpp getAabb). Triangle: 6-support slow path
// (btPolyhedralConvexShape::getAabbSlow with margin) - VERIFY tree vs source when the
// compare harness first exercises a triangle fix-normal case.
__device__ __forceinline__ void CuShapeAabb(const CuGjkShape& s,
	const CuMat3& basis, CuVec3 origin, CuVec3& aabbMin, CuVec3& aabbMax) {
	if (s.kind == CU_SHAPE_BOX) {
		// btTransformAabb(halfExtentsWithoutMargin, margin, t):
		CuVec3 halfExtentsWithMargin = {
			s.boxHalfExtNoMargin.x + s.margin,
			s.boxHalfExtNoMargin.y + s.margin,
			s.boxHalfExtNoMargin.z + s.margin };
		// extent = he dot3 |basis rows|
		CuVec3 r0 = { fabsf(basis.r[0].x), fabsf(basis.r[0].y), fabsf(basis.r[0].z) };
		CuVec3 r1 = { fabsf(basis.r[1].x), fabsf(basis.r[1].y), fabsf(basis.r[1].z) };
		CuVec3 r2 = { fabsf(basis.r[2].x), fabsf(basis.r[2].y), fabsf(basis.r[2].z) };
		CuVec3 extent = { CuDot3(halfExtentsWithMargin, r0), CuDot3(halfExtentsWithMargin, r1), CuDot3(halfExtentsWithMargin, r2) };
		aabbMin = origin - extent;
		aabbMax = origin + extent;
	} else if (s.kind == CU_SHAPE_SPHERE) {
		float m = s.margin + 0.08f; // ROCKETSIM CHANGE in btSphereShape::getAabb
		CuVec3 extent = { m, m, m };
		aabbMin = origin - extent;
		aabbMax = origin + extent;
	} else {
		// triangle: support in +-axis dirs WITH margin (getAabbSlow form)
		CuVec3 mn, mx;
		for (int axis = 0; axis < 3; axis++) {
			CuVec3 dirW = { 0, 0, 0 };
			(&dirW.x)[axis] = 1.0f;
			CuVec3 dirL = CuVecMulMat(dirW, basis);
			CuVec3 sup = CuSupportNoMargin(s, dirL);
			CuVec3 supW = CuTransformPoint(basis, origin, sup);
			(&mx.x)[axis] = (&supW.x)[axis] + s.margin;
			(&dirW.x)[axis] = -1.0f;
			dirL = CuVecMulMat(dirW, basis);
			sup = CuSupportNoMargin(s, dirL);
			supW = CuTransformPoint(basis, origin, sup);
			(&mn.x)[axis] = (&supW.x)[axis] - s.margin;
		}
		aabbMin = mn;
		aabbMax = mx;
	}
}

// ================== btGjkEpa2 port (gjkepa2_impl, float-precision build) ==========
// Constants: GJK_MAX_ITERATIONS 128, GJK_ACCURACY/MIN_DISTANCE/DUPLICATED_EPS 1e-4,
// GJK_SIMPLEX*_EPS 0; EPA_MAX_VERTICES 128, FACES 256, ITERATIONS 255,
// EPA_ACCURACY 1e-4, PLANE_EPS 1e-5, INSIDE_EPS 1e-2. Pointers -> indices for device.

#define CU2_GJK_MAX_ITERATIONS 128
#define CU2_GJK_ACCURACY 0.0001f
#define CU2_GJK_MIN_DISTANCE 0.0001f
#define CU2_GJK_DUPLICATED_EPS 0.0001f
#define CU2_GJK_SIMPLEX2_EPS 0.0f
#define CU2_GJK_SIMPLEX3_EPS 0.0f
#define CU2_GJK_SIMPLEX4_EPS 0.0f
#define CU2_EPA_MAX_VERTICES 128
#define CU2_EPA_MAX_FACES 256
#define CU2_EPA_MAX_ITERATIONS 255
#define CU2_EPA_ACCURACY 0.0001f
#define CU2_EPA_PLANE_EPS 0.00001f

// MinkowskiDiff over our shapes; margins toggled like EnableMargin
struct Cu2Shape {
	const CuGjkShape* shapes[2];
	CuMat3 toshape1;       // wtrs1.basis ^T * wtrs0.basis (transposeTimes tree)
	CuMat3 toshape0Basis;  // wtrs0.inverseTimes(wtrs1)
	CuVec3 toshape0Origin;
	bool withMargins;

	__device__ CuVec3 SupportShape(int idx, CuVec3 d) const {
		const CuGjkShape& s = *shapes[idx];
		if (withMargins) {
			// localGetSupportVertexNonVirtual: normalize dir (fallback (-1,-1,-1)),
			// support without margin + margin * dirNorm
			CuVec3 dn = d;
			if (CuLength2(dn) < CU_SIMD_EPSILON * CU_SIMD_EPSILON)
				dn = { -1, -1, -1 };
			dn = CuNormalize3(dn);
			CuVec3 base = CuSupportNoMargin(s, dn);
			return base + dn * s.margin;
		}
		return CuSupportNoMargin(s, d);
	}
	__device__ CuVec3 Support0(CuVec3 d) const { return SupportShape(0, d); }
	__device__ CuVec3 Support1(CuVec3 d) const {
		return CuTransformPoint(toshape0Basis, toshape0Origin, SupportShape(1, CuMat3MulVec(toshape1, d)));
	}
	__device__ CuVec3 Support(CuVec3 d) const { return Support0(d) - Support1(CuNeg(d)); }
	__device__ CuVec3 Support(CuVec3 d, int index) const {
		return index ? Support1(d) : Support0(d);
	}
};

// btMatrix3x3::transposeTimes: out = this^T * m; per-lane left-assoc accumulation
__device__ __forceinline__ CuMat3 CuTransposeTimes(const CuMat3& t, const CuMat3& m) {
	CuMat3 r;
	for (int i = 0; i < 3; i++) {
		const float t0 = (&t.r[0].x)[i], t1 = (&t.r[1].x)[i], t2 = (&t.r[2].x)[i];
		r.r[i].x = (m.r[0].x * t0 + m.r[1].x * t1) + m.r[2].x * t2;
		r.r[i].y = (m.r[0].y * t0 + m.r[1].y * t1) + m.r[2].y * t2;
		r.r[i].z = (m.r[0].z * t0 + m.r[1].z * t1) + m.r[2].z * t2;
	}
	return r;
}

struct Cu2SV { CuVec3 d, w; };
struct Cu2Simplex { int c[4]; float p[4]; int rank; };

struct Cu2GJK {
	Cu2Shape shape;
	CuVec3 ray;
	float distance;
	Cu2Simplex simplices[2];
	Cu2SV store[4];
	int freeList[4];
	int nfree;
	int current;
	int simplexIdx; // index into simplices
	int status;     // 0 Valid, 1 Inside, 2 Failed

	__device__ void getsupport(CuVec3 d, Cu2SV& sv) const {
		float len = sqrtf(CuLength2(d)); // d.length()
		float inv = 1.0f / len;          // operator/ scalar
		sv.d = d * inv;
		sv.w = shape.Support(sv.d);
	}
	__device__ void removevertice(Cu2Simplex& s) {
		freeList[nfree++] = s.c[--s.rank];
	}
	__device__ void appendvertice(Cu2Simplex& s, CuVec3 v) {
		s.p[s.rank] = 0;
		s.c[s.rank] = freeList[--nfree];
		getsupport(v, store[s.c[s.rank++]]);
	}
	__device__ static float det3(CuVec3 a, CuVec3 b, CuVec3 c) {
		return (a.y * b.z * c.x + a.z * b.x * c.y -
			a.x * b.z * c.y - a.y * b.x * c.z +
			a.x * b.y * c.z - a.z * b.y * c.x);
	}
	__device__ static float projectorigin2(CuVec3 a, CuVec3 b, float* w, int& m) {
		CuVec3 d = b - a;
		float l = CuLength2(d);
		if (l > CU2_GJK_SIMPLEX2_EPS) {
			float t = (l > 0 ? -CuDot3(a, d) / l : 0.0f);
			if (t >= 1) {
				w[0] = 0; w[1] = 1; m = 2;
				return CuLength2(b);
			} else if (t <= 0) {
				w[0] = 1; w[1] = 0; m = 1;
				return CuLength2(a);
			} else {
				w[0] = 1 - (w[1] = t);
				m = 3;
				return CuLength2(a + d * t);
			}
		}
		return -1;
	}
	__device__ static float projectorigin3(CuVec3 a, CuVec3 b, CuVec3 c, float* w, int& m) {
		const int imd3[] = { 1, 2, 0 };
		CuVec3 vt[3] = { a, b, c };
		CuVec3 dl[3] = { a - b, b - c, c - a };
		CuVec3 n = CuCross(dl[0], dl[1]);
		float l = CuLength2(n);
		if (l > CU2_GJK_SIMPLEX3_EPS) {
			float mindist = -1;
			float subw[2] = { 0, 0 };
			int subm = 0;
			for (int i = 0; i < 3; i++) {
				if (CuDot3(vt[i], CuCross(dl[i], n)) > 0) {
					int j = imd3[i];
					float subd = projectorigin2(vt[i], vt[j], subw, subm);
					if ((mindist < 0) || (subd < mindist)) {
						mindist = subd;
						m = ((subm & 1) ? 1 << i : 0) + ((subm & 2) ? 1 << j : 0);
						w[i] = subw[0];
						w[j] = subw[1];
						w[imd3[j]] = 0;
					}
				}
			}
			if (mindist < 0) {
				float d = CuDot3(a, n);
				float s = sqrtf(l);
				CuVec3 p = n * (d / l);
				mindist = CuLength2(p);
				m = 7;
				w[0] = sqrtf(CuLength2(CuCross(dl[1], b - p))) / s; // .length()/s
				w[1] = sqrtf(CuLength2(CuCross(dl[2], c - p))) / s;
				w[2] = 1 - (w[0] + w[1]);
			}
			return mindist;
		}
		return -1;
	}
	__device__ static float projectorigin4(CuVec3 a, CuVec3 b, CuVec3 c, CuVec3 d, float* w, int& m) {
		const int imd3[] = { 1, 2, 0 };
		CuVec3 vt[4] = { a, b, c, d };
		CuVec3 dl[3] = { a - d, b - d, c - d };
		float vl = det3(dl[0], dl[1], dl[2]);
		bool ng = (vl * CuDot3(a, CuCross(b - c, a - b))) <= 0;
		if (ng && (fabsf(vl) > CU2_GJK_SIMPLEX4_EPS)) {
			float mindist = -1;
			float subw[3] = { 0, 0, 0 };
			int subm = 0;
			for (int i = 0; i < 3; i++) {
				int j = imd3[i];
				float s = vl * CuDot3(d, CuCross(dl[i], dl[j]));
				if (s > 0) {
					float subd = projectorigin3(vt[i], vt[j], d, subw, subm);
					if ((mindist < 0) || (subd < mindist)) {
						mindist = subd;
						m = (subm & 1 ? 1 << i : 0) + (subm & 2 ? 1 << j : 0) + (subm & 4 ? 8 : 0);
						w[i] = subw[0];
						w[j] = subw[1];
						w[imd3[j]] = 0;
						w[3] = subw[2];
					}
				}
			}
			if (mindist < 0) {
				mindist = 0;
				m = 15;
				w[0] = det3(c, b, d) / vl;
				w[1] = det3(a, c, d) / vl;
				w[2] = det3(b, a, d) / vl;
				w[3] = 1 - (w[0] + w[1] + w[2]);
			}
			return mindist;
		}
		return -1;
	}

	__device__ int Evaluate(const Cu2Shape& shapearg, CuVec3 guess) {
		int iterations = 0;
		float sqdist = 0;
		float alpha = 0;
		CuVec3 lastw[4];
		int clastw = 0;

		freeList[0] = 0; freeList[1] = 1; freeList[2] = 2; freeList[3] = 3;
		nfree = 4;
		current = 0;
		status = 0; // Valid
		shape = shapearg;
		distance = 0;

		simplices[0].rank = 0;
		ray = guess;
		float sqrl = CuLength2(ray);
		appendvertice(simplices[0], sqrl > 0 ? CuNeg(ray) : CuVec3{ 1, 0, 0 });
		simplices[0].p[0] = 1;
		ray = store[simplices[0].c[0]].w;
		sqdist = sqrl;
		lastw[0] = lastw[1] = lastw[2] = lastw[3] = ray;

		do {
			int next = 1 - current;
			Cu2Simplex& cs = simplices[current];
			Cu2Simplex& ns = simplices[next];

			float rl = sqrtf(CuLength2(ray)); // length()
			if (rl < CU2_GJK_MIN_DISTANCE) {
				status = 1; // Inside
				break;
			}
			appendvertice(cs, CuNeg(ray));
			CuVec3 w = store[cs.c[cs.rank - 1]].w;
			bool found = false;
			for (int i = 0; i < 4; i++) {
				CuVec3 diff = w - lastw[i];
				if (CuLength2(diff) < CU2_GJK_DUPLICATED_EPS) {
					found = true;
					break;
				}
			}
			if (found) {
				removevertice(simplices[current]);
				break;
			} else {
				lastw[clastw = (clastw + 1) & 3] = w;
			}

			float omega = CuDot3(ray, w) / rl;
			alpha = omega > alpha ? omega : alpha; // btMax
			if (((rl - alpha) - (CU2_GJK_ACCURACY * rl)) <= 0) {
				removevertice(simplices[current]);
				break;
			}

			float weights[4];
			int mask = 0;
			switch (cs.rank) {
			case 2:
				sqdist = projectorigin2(store[cs.c[0]].w, store[cs.c[1]].w, weights, mask);
				break;
			case 3:
				sqdist = projectorigin3(store[cs.c[0]].w, store[cs.c[1]].w, store[cs.c[2]].w, weights, mask);
				break;
			case 4:
				sqdist = projectorigin4(store[cs.c[0]].w, store[cs.c[1]].w, store[cs.c[2]].w, store[cs.c[3]].w, weights, mask);
				break;
			}
			if (sqdist >= 0) {
				ns.rank = 0;
				ray = { 0, 0, 0 };
				current = next;
				for (int i = 0, ni = cs.rank; i < ni; i++) {
					if (mask & (1 << i)) {
						ns.c[ns.rank] = cs.c[i];
						ns.p[ns.rank++] = weights[i];
						ray = ray + store[cs.c[i]].w * weights[i];
					} else {
						freeList[nfree++] = cs.c[i];
					}
				}
				if (mask == 15)
					status = 1; // Inside
			} else {
				removevertice(simplices[current]);
				break;
			}
			status = ((++iterations) < CU2_GJK_MAX_ITERATIONS) ? status : 2; // Failed
		} while (status == 0);

		simplexIdx = current;
		if (status == 0)
			distance = sqrtf(CuLength2(ray));
		else if (status == 1)
			distance = 0;
		return status;
	}

	__device__ bool EncloseOrigin() {
		Cu2Simplex& s = simplices[simplexIdx];
		switch (s.rank) {
		case 1: {
			for (int i = 0; i < 3; i++) {
				CuVec3 axis = { 0, 0, 0 };
				(&axis.x)[i] = 1;
				appendvertice(s, axis);
				if (EncloseOrigin()) return true;
				removevertice(s);
				appendvertice(s, CuNeg(axis));
				if (EncloseOrigin()) return true;
				removevertice(s);
			}
			break;
		}
		case 2: {
			CuVec3 d = store[s.c[1]].w - store[s.c[0]].w;
			for (int i = 0; i < 3; i++) {
				CuVec3 axis = { 0, 0, 0 };
				(&axis.x)[i] = 1;
				CuVec3 p = CuCross(d, axis);
				if (CuLength2(p) > 0) {
					appendvertice(s, p);
					if (EncloseOrigin()) return true;
					removevertice(s);
					appendvertice(s, CuNeg(p));
					if (EncloseOrigin()) return true;
					removevertice(s);
				}
			}
			break;
		}
		case 3: {
			CuVec3 n = CuCross(store[s.c[1]].w - store[s.c[0]].w,
				store[s.c[2]].w - store[s.c[0]].w);
			if (CuLength2(n) > 0) {
				appendvertice(s, n);
				if (EncloseOrigin()) return true;
				removevertice(s);
				appendvertice(s, CuNeg(n));
				if (EncloseOrigin()) return true;
				removevertice(s);
			}
			break;
		}
		case 4: {
			if (fabsf(det3(store[s.c[0]].w - store[s.c[3]].w,
					store[s.c[1]].w - store[s.c[3]].w,
					store[s.c[2]].w - store[s.c[3]].w)) > 0)
				return true;
			break;
		}
		}
		return false;
	}
};

struct Cu2Face {
	CuVec3 n;
	float d;
	int c[3];     // sSV indices into EPA sv store
	int f[3];     // adjacent face indices (-1 null)
	int l[2];     // list links (-1 null)
	uint8_t e[3];
	uint8_t pass;
};

struct Cu2EPA {
	int status; // matches eStatus order: 0 Valid .. 8 FallBack, 9 Failed
	Cu2Simplex result;     // c[] index: 0..3 = initial hull verts, 4+ = svStore[i-4]
	CuVec3 normal;
	float depth;
	Cu2SV initStore[4];    // copies of the 4 GJK simplex verts (CPU keeps pointers;
	                       // EPA's own 128-slot store is for NEW vertices only)
	Cu2SV svStore[CU2_EPA_MAX_VERTICES];
	Cu2Face fcStore[CU2_EPA_MAX_FACES];
	int nextsv;
	int hullRoot, hullCount;
	int stockRoot, stockCount;

	__device__ Cu2SV& svAt(int i) { return i < 4 ? initStore[i] : svStore[i - 4]; }

	__device__ void bind(int fa, int ea, int fb, int eb) {
		fcStore[fa].e[ea] = (uint8_t)eb;
		fcStore[fa].f[ea] = fb;
		fcStore[fb].e[eb] = (uint8_t)ea;
		fcStore[fb].f[eb] = fa;
	}
	__device__ void appendList(int& root, int& count, int face) {
		fcStore[face].l[0] = -1;
		fcStore[face].l[1] = root;
		if (root >= 0) fcStore[root].l[0] = face;
		root = face;
		++count;
	}
	__device__ void removeList(int& root, int& count, int face) {
		if (fcStore[face].l[1] >= 0) fcStore[fcStore[face].l[1]].l[0] = fcStore[face].l[0];
		if (fcStore[face].l[0] >= 0) fcStore[fcStore[face].l[0]].l[1] = fcStore[face].l[1];
		if (face == root) root = fcStore[face].l[1];
		--count;
	}

	__device__ void Initialize() {
		status = 9;
		normal = { 0, 0, 0 };
		depth = 0;
		nextsv = 0;
		hullRoot = -1; hullCount = 0;
		stockRoot = -1; stockCount = 0;
		for (int i = 0; i < CU2_EPA_MAX_FACES; i++)
			appendList(stockRoot, stockCount, CU2_EPA_MAX_FACES - i - 1);
	}

	__device__ bool getedgedist(Cu2Face& face, Cu2SV& a, Cu2SV& b, float& dist) {
		CuVec3 ba = b.w - a.w;
		CuVec3 n_ab = CuCross(ba, face.n);
		float a_dot_nab = CuDot3(a.w, n_ab);
		if (a_dot_nab < 0) {
			float ba_l2 = CuLength2(ba);
			float a_dot_ba = CuDot3(a.w, ba);
			float b_dot_ba = CuDot3(b.w, ba);
			if (a_dot_ba > 0) {
				dist = sqrtf(CuLength2(a.w)); // length()
			} else if (b_dot_ba < 0) {
				dist = sqrtf(CuLength2(b.w));
			} else {
				float a_dot_b = CuDot3(a.w, b.w);
				float num = (CuLength2(a.w) * CuLength2(b.w) - a_dot_b * a_dot_b) / ba_l2;
				dist = sqrtf(num > 0.0f ? num : 0.0f); // btMax(.., 0)
			}
			return true;
		}
		return false;
	}

	__device__ int newface(int a, int b, int c, bool forced) {
		if (stockRoot >= 0) {
			int face = stockRoot;
			Cu2Face& f = fcStore[face];
			removeList(stockRoot, stockCount, face);
			appendList(hullRoot, hullCount, face);
			f.pass = 0;
			f.c[0] = a; f.c[1] = b; f.c[2] = c;
			f.n = CuCross(svAt(b).w - svAt(a).w, svAt(c).w - svAt(a).w);
			float l = sqrtf(CuLength2(f.n)); // length()
			bool v = l > CU2_EPA_ACCURACY;
			if (v) {
				if (!(getedgedist(f, svAt(a), svAt(b), f.d) ||
					  getedgedist(f, svAt(b), svAt(c), f.d) ||
					  getedgedist(f, svAt(c), svAt(a), f.d))) {
					f.d = CuDot3(svAt(a).w, f.n) / l;
				}
				float inv = 1.0f / l; // n /= l
				f.n = f.n * inv;
				if (forced || (f.d >= -CU2_EPA_PLANE_EPS)) {
					return face;
				} else {
					status = 3; // NonConvex
				}
			} else {
				status = 2; // Degenerated
			}
			removeList(hullRoot, hullCount, face);
			appendList(stockRoot, stockCount, face);
			return -1;
		}
		status = stockRoot >= 0 ? 6 : 5; // OutOfVertices : OutOfFaces
		return -1;
	}

	__device__ int findbest() {
		int minf = hullRoot;
		float mind = fcStore[minf].d * fcStore[minf].d;
		for (int f = fcStore[minf].l[1]; f >= 0; f = fcStore[f].l[1]) {
			float sqd = fcStore[f].d * fcStore[f].d;
			if (sqd < mind) {
				minf = f;
				mind = sqd;
			}
		}
		return minf;
	}

	__device__ bool expand(int pass, int w, int fIdx, int e, int& horizonCf, int& horizonFf, int& horizonNf) {
		const int i1m3[] = { 1, 2, 0 };
		const int i2m3[] = { 2, 0, 1 };
		Cu2Face& f = fcStore[fIdx];
		if (f.pass != pass) {
			int e1 = i1m3[e];
			if ((CuDot3(f.n, svAt(w).w) - f.d) < -CU2_EPA_PLANE_EPS) {
				int nf = newface(f.c[e1], f.c[e], w, false);
				if (nf >= 0) {
					bind(nf, 0, fIdx, e);
					if (horizonCf >= 0)
						bind(horizonCf, 1, nf, 2);
					else
						horizonFf = nf;
					horizonCf = nf;
					++horizonNf;
					return true;
				}
			} else {
				int e2 = i2m3[e];
				f.pass = (uint8_t)pass;
				if (expand(pass, w, f.f[e1], f.e[e1], horizonCf, horizonFf, horizonNf) &&
					expand(pass, w, f.f[e2], f.e[e2], horizonCf, horizonFf, horizonNf)) {
					removeList(hullRoot, hullCount, fIdx);
					appendList(stockRoot, stockCount, fIdx);
					return true;
				}
			}
		}
		return false;
	}

	__device__ int Evaluate(Cu2GJK& gjk, CuVec3 guess) {
		Cu2Simplex& simplex = gjk.simplices[gjk.simplexIdx];
		if ((simplex.rank > 1) && gjk.EncloseOrigin()) {
			while (hullRoot >= 0) {
				int f = hullRoot;
				removeList(hullRoot, hullCount, f);
				appendList(stockRoot, stockCount, f);
			}
			status = 0; // Valid
			nextsv = 0;

			if (Cu2GJK::det3(gjk.store[simplex.c[0]].w - gjk.store[simplex.c[3]].w,
					gjk.store[simplex.c[1]].w - gjk.store[simplex.c[3]].w,
					gjk.store[simplex.c[2]].w - gjk.store[simplex.c[3]].w) < 0) {
				int tc = simplex.c[0]; simplex.c[0] = simplex.c[1]; simplex.c[1] = tc;
				float tp = simplex.p[0]; simplex.p[0] = simplex.p[1]; simplex.p[1] = tp;
			}

			// copy GJK simplex verts (CPU keeps POINTERS; EPA's own store stays
			// reserved for new vertices so capacity semantics match exactly)
			for (int i = 0; i < 4; i++)
				initStore[i] = gjk.store[simplex.c[i]];
			nextsv = 0;

			int tetra[4];
			tetra[0] = newface(0, 1, 2, true);
			tetra[1] = newface(1, 0, 3, true);
			tetra[2] = newface(2, 1, 3, true);
			tetra[3] = newface(0, 2, 3, true);

			if (hullCount == 4) {
				int best = findbest();
				Cu2Face outer = fcStore[best];
				int pass = 0;
				int iterations = 0;
				bind(tetra[0], 0, tetra[1], 0);
				bind(tetra[0], 1, tetra[2], 0);
				bind(tetra[0], 2, tetra[3], 0);
				bind(tetra[1], 1, tetra[3], 2);
				bind(tetra[1], 2, tetra[2], 1);
				bind(tetra[2], 2, tetra[3], 1);
				status = 0;
				for (; iterations < CU2_EPA_MAX_ITERATIONS; ++iterations) {
					if (nextsv < CU2_EPA_MAX_VERTICES) {
						int horizonCf = -1, horizonFf = -1, horizonNf = 0;
						int w = 4 + nextsv++;
						bool valid = true;
						fcStore[best].pass = (uint8_t)(++pass);
						gjk.getsupport(fcStore[best].n, svAt(w));
						float wdist = CuDot3(fcStore[best].n, svAt(w).w) - fcStore[best].d;
						if (wdist > CU2_EPA_ACCURACY) {
							for (int j = 0; (j < 3) && valid; ++j) {
								valid &= expand(pass, w, fcStore[best].f[j], fcStore[best].e[j],
									horizonCf, horizonFf, horizonNf);
							}
							if (valid && (horizonNf >= 3)) {
								bind(horizonCf, 1, horizonFf, 2);
								removeList(hullRoot, hullCount, best);
								appendList(stockRoot, stockCount, best);
								best = findbest();
								outer = fcStore[best];
							} else {
								status = 4; // InvalidHull
								break;
							}
						} else {
							status = 7; // AccuraryReached
							break;
						}
					} else {
						status = 6; // OutOfVertices
						break;
					}
				}

				CuVec3 projection = outer.n * outer.d;
				normal = outer.n;
				depth = outer.d;
				result.rank = 3;
				result.c[0] = outer.c[0];
				result.c[1] = outer.c[1];
				result.c[2] = outer.c[2];
				result.p[0] = sqrtf(CuLength2(CuCross(svAt(outer.c[1]).w - projection,
					svAt(outer.c[2]).w - projection)));
				result.p[1] = sqrtf(CuLength2(CuCross(svAt(outer.c[2]).w - projection,
					svAt(outer.c[0]).w - projection)));
				result.p[2] = sqrtf(CuLength2(CuCross(svAt(outer.c[0]).w - projection,
					svAt(outer.c[1]).w - projection)));
				float sum = result.p[0] + result.p[1] + result.p[2];
				result.p[0] /= sum;
				result.p[1] /= sum;
				result.p[2] /= sum;
				return status;
			}
		}
		// Fallback
		status = 8; // FallBack
		normal = CuNeg(guess);
		float nl = sqrtf(CuLength2(normal));
		if (nl > 0) {
			float inv = 1.0f / nl;
			normal = normal * inv;
		} else {
			normal = { 1, 0, 0 };
		}
		depth = 0;
		result.rank = 1;
		result.c[0] = simplex.c[0]; // NOTE: GJK-store index (rank-1 fallback path)
		result.p[0] = 1;
		return status;
	}
};

struct Cu2Results {
	int status; // 0 Separated, 1 Penetrating, 2 GJK_Failed, 3 EPA_Failed
	CuVec3 witnesses[2];
	CuVec3 normal;
	float distance;
};

__device__ inline void Cu2Initialize(
	const CuGjkShape& shape0, const CuMat3& basis0, CuVec3 origin0,
	const CuGjkShape& shape1, const CuMat3& basis1, CuVec3 origin1,
	Cu2Results& results, Cu2Shape& shape, bool withmargins) {
	results.witnesses[0] = { 0, 0, 0 };
	results.witnesses[1] = { 0, 0, 0 };
	results.status = 0;
	shape.shapes[0] = &shape0;
	shape.shapes[1] = &shape1;
	shape.toshape1 = CuTransposeTimes(basis1, basis0);
	// wtrs0.inverseTimes(wtrs1): basis = basis0^T*basis1 (transposeTimes);
	// origin = (origin1 - origin0) * basis0 (vec*mat)
	shape.toshape0Basis = CuTransposeTimes(basis0, basis1);
	CuVec3 v = origin1 - origin0;
	shape.toshape0Origin = CuVecMulMat(v, basis0);
	shape.withMargins = withmargins;
}

// btGjkEpaSolver2::Penetration (usemargins=true as calcPenDepth calls it)
__device__ inline bool Cu2Penetration(
	const CuGjkShape& shape0, const CuMat3& basis0, CuVec3 origin0,
	const CuGjkShape& shape1, const CuMat3& basis1, CuVec3 origin1,
	CuVec3 guess, Cu2Results& results, Cu2GJK& gjk, Cu2EPA& epa) {

	Cu2Shape shape;
	Cu2Initialize(shape0, basis0, origin0, shape1, basis1, origin1, results, shape, true);
	int gjk_status = gjk.Evaluate(shape, CuNeg(guess));
	switch (gjk_status) {
	case 1: { // Inside
		epa.Initialize();
		int epa_status = epa.Evaluate(gjk, CuNeg(guess));
		if (epa_status != 9) { // != Failed
			CuVec3 w0 = { 0, 0, 0 };
			for (int i = 0; i < epa.result.rank; i++) {
				// fallback path stores a GJK index; main path stores EPA-store indices
				CuVec3 d = (epa.status == 8) ? gjk.store[epa.result.c[i]].d : epa.svAt(epa.result.c[i]).d;
				w0 = w0 + shape.Support(d, 0) * epa.result.p[i];
			}
			results.status = 1; // Penetrating
			results.witnesses[0] = CuTransformPoint(basis0, origin0, w0);
			results.witnesses[1] = CuTransformPoint(basis0, origin0, w0 - epa.normal * epa.depth);
			results.normal = CuNeg(epa.normal);
			results.distance = -epa.depth;
			return true;
		} else {
			results.status = 3; // EPA_Failed
		}
		break;
	}
	case 2: // Failed
		results.status = 2; // GJK_Failed
		break;
	default:
		break;
	}
	return false;
}

// btGjkEpaSolver2::Distance (no margins)
__device__ inline bool Cu2Distance(
	const CuGjkShape& shape0, const CuMat3& basis0, CuVec3 origin0,
	const CuGjkShape& shape1, const CuMat3& basis1, CuVec3 origin1,
	CuVec3 guess, Cu2Results& results, Cu2GJK& gjk) {

	Cu2Shape shape;
	Cu2Initialize(shape0, basis0, origin0, shape1, basis1, origin1, results, shape, false);
	int gjk_status = gjk.Evaluate(shape, guess);
	if (gjk_status == 0) { // Valid
		CuVec3 w0 = { 0, 0, 0 };
		CuVec3 w1 = { 0, 0, 0 };
		Cu2Simplex& s = gjk.simplices[gjk.simplexIdx];
		for (int i = 0; i < s.rank; i++) {
			float p = s.p[i];
			w0 = w0 + shape.Support(gjk.store[s.c[i]].d, 0) * p;
			w1 = w1 + shape.Support(CuNeg(gjk.store[s.c[i]].d), 1) * p;
		}
		results.witnesses[0] = CuTransformPoint(basis0, origin0, w0);
		results.witnesses[1] = CuTransformPoint(basis0, origin0, w1);
		results.normal = w0 - w1;
		results.distance = sqrtf(CuLength2(results.normal)); // length()
		float div = results.distance > CU2_GJK_MIN_DISTANCE ? results.distance : 1.0f;
		float inv = 1.0f / div; // operator/= scalar
		results.normal = results.normal * inv;
		return true;
	} else {
		results.status = gjk_status == 1 ? 1 : 2; // Inside->Penetrating : GJK_Failed
		return false;
	}
}

// btGjkEpaPenetrationDepthSolver::calcPenDepth: 9 guess vectors, Penetration ->
// Distance fallback per guess. simplexSolver param unused on CPU (reset only).
__device__ inline bool CuEpaPenDepth(CuSimplexSolver& simplexSolver,
	const CuGjkShape& shapeA, const CuGjkShape& shapeB,
	const CuMat3& basisA, CuVec3 originA, const CuMat3& basisB, CuVec3 originB,
	CuVec3& sepAxis, CuVec3& pOnA, CuVec3& pOnB) {

	CuVec3 guessVectors[9];
	guessVectors[0] = CuSafeNormalize3(originB - originA); // safeNormalize
	guessVectors[1] = CuSafeNormalize3(originA - originB);
	guessVectors[2] = { 0, 0, 1 };
	guessVectors[3] = { 0, 1, 0 };
	guessVectors[4] = { 1, 0, 0 };
	guessVectors[5] = { 1, 1, 0 };
	guessVectors[6] = { 1, 1, 1 };
	guessVectors[7] = { 0, 1, 1 };
	guessVectors[8] = { 1, 0, 1 };

	for (int i = 0; i < 9; i++) {
		simplexSolver.reset();
		CuVec3 guessVector = guessVectors[i];
		Cu2Results results;
		Cu2GJK gjk;
		Cu2EPA epa;
		if (Cu2Penetration(shapeA, basisA, originA, shapeB, basisB, originB,
				guessVector, results, gjk, epa)) {
			pOnA = results.witnesses[0];
			pOnB = results.witnesses[1];
			sepAxis = results.normal;
			return true;
		} else {
			if (Cu2Distance(shapeA, basisA, originA, shapeB, basisB, originB,
					guessVector, results, gjk)) {
				pOnA = results.witnesses[0];
				pOnB = results.witnesses[1];
				sepAxis = results.normal;
				return false;
			}
		}
	}

	pOnA = { 0, 0, 0 };
	pOnB = { 0, 0, 0 };
	sepAxis = { 0, 0, 0 };
	return false;
}

struct CuGjkResult {
	bool hasContact;
	CuVec3 normalOnB;     // world
	CuVec3 pointOnBWorld; // world (positionOffset re-added)
	float distance;
};

__device__ inline void CuGjkClosestPoints(
	const CuGjkShape& shapeA, const CuMat3& basisA, CuVec3 originAIn,
	const CuGjkShape& shapeB, const CuMat3& basisB, CuVec3 originBIn,
	float maximumDistanceSquared, bool epaAvailable,
	CuGjkResult& out) {

	out.hasContact = false;

	float distance = 0.0f;
	CuVec3 normalInB = { 0, 0, 0 };
	CuVec3 pointOnA = { 0, 0, 0 }, pointOnB = { 0, 0, 0 };

	// positionOffset centering
	CuVec3 positionOffset = (originAIn + originBIn) * 0.5f;
	CuVec3 localOriginA = originAIn - positionOffset;
	CuVec3 localOriginB = originBIn - positionOffset;

	float marginA = shapeA.margin;
	float marginB = shapeB.margin;

	int curIter = 0;
	const int gGjkMaxIter = 1000;
	CuVec3 cachedSeparatingAxis = { 0, 1, 0 };

	bool isValid = false;
	bool checkSimplex = false;
	const bool checkPenetration = true;
	int degenerateSimplex = 0;

	float squaredDistance = 3.402823466e+38f; // BT_LARGE_FLOAT
	float delta = 0.0f;
	float margin = marginA + marginB;

	CuSimplexSolver solver;
	solver.reset();

	while (true) {
		// (-axis) * basisA / axis * basisB: vec*mat tdot trees
		CuVec3 seperatingAxisInA = CuVecMulMat(CuNeg(cachedSeparatingAxis), basisA);
		CuVec3 seperatingAxisInB = CuVecMulMat(cachedSeparatingAxis, basisB);

		CuVec3 pInA = CuSupportNoMargin(shapeA, seperatingAxisInA);
		CuVec3 qInB = CuSupportNoMargin(shapeB, seperatingAxisInB);

		CuVec3 pWorld = CuTransformPoint(basisA, localOriginA, pInA);
		CuVec3 qWorld = CuTransformPoint(basisB, localOriginB, qInB);

		CuVec3 w = pWorld - qWorld;
		delta = CuDot3(cachedSeparatingAxis, w);

		if ((delta > 0.0f) && (delta * delta > squaredDistance * maximumDistanceSquared)) {
			degenerateSimplex = 10;
			checkSimplex = true;
			break;
		}
		if (solver.inSimplex(w)) {
			degenerateSimplex = 1;
			checkSimplex = true;
			break;
		}
		float f0 = squaredDistance - delta;
		float f1 = squaredDistance * CU_REL_ERROR2;
		if (f0 <= f1) {
			degenerateSimplex = (f0 <= 0.0f) ? 2 : 11;
			checkSimplex = true;
			break;
		}

		solver.addVertex(w, pWorld, qWorld);
		CuVec3 newCachedSeparatingAxis;
		if (!solver.closest(newCachedSeparatingAxis)) {
			degenerateSimplex = 3;
			checkSimplex = true;
			break;
		}
		if (CuLength2(newCachedSeparatingAxis) < CU_REL_ERROR2) {
			cachedSeparatingAxis = newCachedSeparatingAxis;
			degenerateSimplex = 6;
			checkSimplex = true;
			break;
		}

		float previousSquaredDistance = squaredDistance;
		squaredDistance = CuLength2(newCachedSeparatingAxis);

		if (previousSquaredDistance - squaredDistance <= CU_SIMD_EPSILON * previousSquaredDistance) {
			checkSimplex = true;
			degenerateSimplex = 12;
			break;
		}
		cachedSeparatingAxis = newCachedSeparatingAxis;

		if (curIter++ > gGjkMaxIter)
			break;

		if (solver.numVertices == 4) { // fullSimplex
			degenerateSimplex = 13;
			break;
		}
	}

	if (checkSimplex) {
		CuVec3 v;
		solver.updateClosestVectorAndPoints(); // compute_points
		pointOnA = solver.cachedP1;
		pointOnB = solver.cachedP2;
		(void)v;
		normalInB = cachedSeparatingAxis;
		float lenSqr = CuLength2(cachedSeparatingAxis);

		if (lenSqr < 0.0001f)
			degenerateSimplex = 5;
		if (lenSqr > CU_SIMD_EPSILON * CU_SIMD_EPSILON) {
			float rlen = 1.0f / sqrtf(lenSqr);
			normalInB = normalInB * rlen;
			float s = sqrtf(squaredDistance);
			pointOnA = pointOnA - cachedSeparatingAxis * (marginA / s);
			pointOnB = pointOnB + cachedSeparatingAxis * (marginB / s);
			distance = (1.0f / rlen) - margin;
			isValid = true;
		}
	}

	// m_catchDegeneracies=1
	bool catchDegeneratePenetrationCase =
		(epaAvailable && degenerateSimplex && ((distance + margin) < 0.01f));

	if (checkPenetration && (!isValid || catchDegeneratePenetrationCase)) {
		if (epaAvailable) {
			CuVec3 tmpPointOnA, tmpPointOnB;
			cachedSeparatingAxis = { 0, 0, 0 };

			bool isValid2 = CuEpaPenDepth(solver, shapeA, shapeB,
				basisA, localOriginA, basisB, localOriginB,
				cachedSeparatingAxis, tmpPointOnA, tmpPointOnB);

			if (isValid2) {
				CuVec3 tmpNormalInB = tmpPointOnB - tmpPointOnA;
				float lenSqr = CuLength2(tmpNormalInB);
				if (lenSqr <= CU_SIMD_EPSILON * CU_SIMD_EPSILON) {
					tmpNormalInB = cachedSeparatingAxis;
					lenSqr = CuLength2(cachedSeparatingAxis);
				}
				if (lenSqr > CU_SIMD_EPSILON * CU_SIMD_EPSILON) {
					float inv = 1.0f / sqrtf(lenSqr); // /= btSqrt
					tmpNormalInB = tmpNormalInB * inv;
					CuVec3 dAB = tmpPointOnA - tmpPointOnB;
					float distance2 = -sqrtf(CuLength2(dAB)); // -(..).length()
					if (!isValid || (distance2 < distance)) {
						distance = distance2;
						pointOnA = tmpPointOnA;
						pointOnB = tmpPointOnB;
						normalInB = tmpNormalInB;
						isValid = true;
					}
				}
			} else {
				if (CuLength2(cachedSeparatingAxis) > 0.0f) {
					CuVec3 dAB = tmpPointOnA - tmpPointOnB;
					float distance2 = sqrtf(CuLength2(dAB)) - margin;
					if (!isValid || (distance2 < distance)) {
						distance = distance2;
						pointOnA = tmpPointOnA;
						pointOnB = tmpPointOnB;
						pointOnA = pointOnA - cachedSeparatingAxis * marginA;
						pointOnB = pointOnB + cachedSeparatingAxis * marginB;
						normalInB = cachedSeparatingAxis;
						normalInB = CuNormalize3(normalInB); // normalize() rsqrt
						isValid = true;
					}
				}
			}
		}
	}

	if (isValid && ((distance < 0) || (distance * distance < maximumDistanceSquared))) {
		// m_fixContactNormalDirection=1: aabb-center check (shape getAabb at LOCAL
		// transforms). Implemented for our shapes:
		CuVec3 posA, posB;
		{
			CuVec3 mn, mx;
			CuShapeAabb(shapeA, basisA, localOriginA, mn, mx);
			posA = (mx + mn) * 0.5f;
			CuShapeAabb(shapeB, basisB, localOriginB, mn, mx);
			posB = (mn + mx) * 0.5f; // note CPU writes (aabbMin+aabbMax)*0.5 here
		}
		CuVec3 diff = posA - posB;
		if (CuDot3(diff, normalInB) < 0.0f)
			normalInB = normalInB * -1.0f;

		out.hasContact = true;
		out.normalOnB = normalInB;
		out.pointOnBWorld = pointOnB + positionOffset;
		out.distance = distance;
	}
}

// ================== btBoxBoxDetector (ODE dBoxBox2) port ==========================
// Plain scalar code; trees port verbatim. cullPoints2 uses atan2f - double-rounded
// stand-in, mismatch risk measured by fuzz/manifold-compare.
// Output contacts collected into CuBoxBoxOut (caller feeds the manifold in order).

struct CuBoxBoxContact { CuVec3 normal; CuVec3 point; float depth; };
struct CuBoxBoxOut { int n; CuBoxBoxContact c[4]; };

__device__ __forceinline__ float CuAtan2fBB(float y, float x) { return (float)atan2((double)y, (double)x); }

__device__ __forceinline__ float CuDDOTpq(const float* a, const float* b, int p, int q) {
	return a[0] * b[0] + a[p] * b[q] + a[2 * p] * b[2 * q];
}
__device__ __forceinline__ float CuDDOT(const float* a, const float* b) { return CuDDOTpq(a, b, 1, 1); }
__device__ __forceinline__ float CuDDOT44(const float* a, const float* b) { return CuDDOTpq(a, b, 4, 4); }
__device__ __forceinline__ float CuDDOT41(const float* a, const float* b) { return CuDDOTpq(a, b, 4, 1); }
__device__ __forceinline__ float CuDDOT14(const float* a, const float* b) { return CuDDOTpq(a, b, 1, 4); }

__device__ __forceinline__ void CuLineClosestApproach(
	const float* pa, const float* ua, const float* pb, const float* ub,
	float* alpha, float* beta) {
	float p[3];
	p[0] = pb[0] - pa[0];
	p[1] = pb[1] - pa[1];
	p[2] = pb[2] - pa[2];
	float uaub = CuDDOT(ua, ub);
	float q1 = CuDDOT(ua, p);
	float q2 = -CuDDOT(ub, p);
	float d = 1 - uaub * uaub;
	if (d <= 0.0001f) {
		*alpha = 0;
		*beta = 0;
	} else {
		d = 1.f / d;
		*alpha = (q1 + uaub * q2) * d;
		*beta = (uaub * q1 + q2) * d;
	}
}

__device__ inline int CuIntersectRectQuad2(float h[2], float p[8], float ret[16]) {
	int nq = 4, nr = 0;
	float buffer[16];
	float* q = p;
	float* r = ret;
	for (int dir = 0; dir <= 1; dir++) {
		for (int sign = -1; sign <= 1; sign += 2) {
			float* pq = q;
			float* pr = r;
			nr = 0;
			for (int i = nq; i > 0; i--) {
				if (sign * pq[dir] < h[dir]) {
					pr[0] = pq[0];
					pr[1] = pq[1];
					pr += 2;
					nr++;
					if (nr & 8) {
						q = r;
						goto done;
					}
				}
				float* nextq = (i > 1) ? pq + 2 : q;
				if ((sign * pq[dir] < h[dir]) ^ (sign * nextq[dir] < h[dir])) {
					pr[1 - dir] = pq[1 - dir] + (nextq[1 - dir] - pq[1 - dir]) /
						(nextq[dir] - pq[dir]) * (sign * h[dir] - pq[dir]);
					pr[dir] = sign * h[dir];
					pr += 2;
					nr++;
					if (nr & 8) {
						q = r;
						goto done;
					}
				}
				pq += 2;
			}
			q = r;
			r = (q == ret) ? buffer : ret;
			nq = nr;
		}
	}
done:
	if (q != ret)
		for (int i = 0; i < nr * 2; i++) ret[i] = q[i];
	return nr;
}

__device__ inline void CuCullPoints2(int n, float p[], int m, int i0, int iret[]) {
	constexpr float M_PI_BB = 3.14159265f;
	int i, j;
	float a, cx, cy, q;
	if (n == 1) {
		cx = p[0];
		cy = p[1];
	} else if (n == 2) {
		cx = 0.5f * (p[0] + p[2]);
		cy = 0.5f * (p[1] + p[3]);
	} else {
		a = 0;
		cx = 0;
		cy = 0;
		for (i = 0; i < (n - 1); i++) {
			q = p[i * 2] * p[i * 2 + 3] - p[i * 2 + 2] * p[i * 2 + 1];
			a += q;
			cx += q * (p[i * 2] + p[i * 2 + 2]);
			cy += q * (p[i * 2 + 1] + p[i * 2 + 3]);
		}
		q = p[n * 2 - 2] * p[1] - p[0] * p[n * 2 - 1];
		if (fabsf(a + q) > CU_SIMD_EPSILON) {
			a = 1.f / (3.0f * (a + q));
		} else {
			a = 3.402823466e+38f; // BT_LARGE_FLOAT
		}
		cx = a * (cx + q * (p[n * 2 - 2] + p[0]));
		cy = a * (cy + q * (p[n * 2 - 1] + p[1]));
	}

	float A[8];
	for (i = 0; i < n; i++) A[i] = CuAtan2fBB(p[i * 2 + 1] - cy, p[i * 2] - cx);

	int avail[8];
	for (i = 0; i < n; i++) avail[i] = 1;
	avail[i0] = 0;
	iret[0] = i0;
	int ir = 1;
	for (j = 1; j < m; j++) {
		a = (float)j * (2 * M_PI_BB / m) + A[i0];
		if (a > M_PI_BB) a -= 2 * M_PI_BB;
		float maxdiff = 1e9f, diff;
		iret[ir] = i0;
		for (i = 0; i < n; i++) {
			if (avail[i]) {
				diff = fabsf(A[i] - a);
				if (diff > M_PI_BB) diff = 2 * M_PI_BB - diff;
				if (diff < maxdiff) {
					maxdiff = diff;
					iret[ir] = i;
				}
			}
		}
		avail[iret[ir]] = 0;
		ir++;
	}
}

// dBoxBox2 (btBoxBoxDetector.cpp:184): sides are FULL extents (2*halfExtWithMargin).
// Emits up to 4 contacts in CPU order.
__device__ inline int CuDBoxBox2(
	const float* p1, const float* R1, const float* side1,
	const float* p2, const float* R2, const float* side2,
	CuBoxBoxOut& output) {

	const float fudge_factor = 1.05f;
	float p[3], pp[3], normalC[3] = { 0, 0, 0 };
	const float* normalR = 0;
	float A[3], B[3], R11, R12, R13, R21, R22, R23, R31, R32, R33,
		Q11, Q12, Q13, Q21, Q22, Q23, Q31, Q32, Q33, s, s2, l;
	int i, j, invert_normal, code;

	p[0] = p2[0] - p1[0];
	p[1] = p2[1] - p1[1];
	p[2] = p2[2] - p1[2];
	pp[0] = CuDDOT41(R1 + 0, p);
	pp[1] = CuDDOT41(R1 + 1, p);
	pp[2] = CuDDOT41(R1 + 2, p);

	A[0] = side1[0] * 0.5f;
	A[1] = side1[1] * 0.5f;
	A[2] = side1[2] * 0.5f;
	B[0] = side2[0] * 0.5f;
	B[1] = side2[1] * 0.5f;
	B[2] = side2[2] * 0.5f;

	R11 = CuDDOT44(R1 + 0, R2 + 0);
	R12 = CuDDOT44(R1 + 0, R2 + 1);
	R13 = CuDDOT44(R1 + 0, R2 + 2);
	R21 = CuDDOT44(R1 + 1, R2 + 0);
	R22 = CuDDOT44(R1 + 1, R2 + 1);
	R23 = CuDDOT44(R1 + 1, R2 + 2);
	R31 = CuDDOT44(R1 + 2, R2 + 0);
	R32 = CuDDOT44(R1 + 2, R2 + 1);
	R33 = CuDDOT44(R1 + 2, R2 + 2);

	Q11 = fabsf(R11); Q12 = fabsf(R12); Q13 = fabsf(R13);
	Q21 = fabsf(R21); Q22 = fabsf(R22); Q23 = fabsf(R23);
	Q31 = fabsf(R31); Q32 = fabsf(R32); Q33 = fabsf(R33);

	s = -3.402823466e+38f;
	invert_normal = 0;
	code = 0;

#define CU_TST(expr1, expr2, norm, cc)  \
	s2 = fabsf(expr1) - (expr2);        \
	if (s2 > 0) return 0;               \
	if (s2 > s) {                       \
		s = s2;                         \
		normalR = norm;                 \
		invert_normal = ((expr1) < 0);  \
		code = (cc);                    \
	}

	CU_TST(pp[0], (A[0] + B[0] * Q11 + B[1] * Q12 + B[2] * Q13), R1 + 0, 1);
	CU_TST(pp[1], (A[1] + B[0] * Q21 + B[1] * Q22 + B[2] * Q23), R1 + 1, 2);
	CU_TST(pp[2], (A[2] + B[0] * Q31 + B[1] * Q32 + B[2] * Q33), R1 + 2, 3);

	CU_TST(CuDDOT41(R2 + 0, p), (A[0] * Q11 + A[1] * Q21 + A[2] * Q31 + B[0]), R2 + 0, 4);
	CU_TST(CuDDOT41(R2 + 1, p), (A[0] * Q12 + A[1] * Q22 + A[2] * Q32 + B[1]), R2 + 1, 5);
	CU_TST(CuDDOT41(R2 + 2, p), (A[0] * Q13 + A[1] * Q23 + A[2] * Q33 + B[2]), R2 + 2, 6);
#undef CU_TST

#define CU_TST(expr1, expr2, n1, n2, n3, cc)               \
	s2 = fabsf(expr1) - (expr2);                           \
	if (s2 > CU_SIMD_EPSILON) return 0;                    \
	l = sqrtf((n1) * (n1) + (n2) * (n2) + (n3) * (n3));    \
	if (l > CU_SIMD_EPSILON) {                             \
		s2 /= l;                                           \
		if (s2 * fudge_factor > s) {                       \
			s = s2;                                        \
			normalR = 0;                                   \
			normalC[0] = (n1) / l;                         \
			normalC[1] = (n2) / l;                         \
			normalC[2] = (n3) / l;                         \
			invert_normal = ((expr1) < 0);                 \
			code = (cc);                                   \
		}                                                  \
	}

	float fudge2 = 1.0e-5f;
	Q11 += fudge2; Q12 += fudge2; Q13 += fudge2;
	Q21 += fudge2; Q22 += fudge2; Q23 += fudge2;
	Q31 += fudge2; Q32 += fudge2; Q33 += fudge2;

	CU_TST(pp[2] * R21 - pp[1] * R31, (A[1] * Q31 + A[2] * Q21 + B[1] * Q13 + B[2] * Q12), 0, -R31, R21, 7);
	CU_TST(pp[2] * R22 - pp[1] * R32, (A[1] * Q32 + A[2] * Q22 + B[0] * Q13 + B[2] * Q11), 0, -R32, R22, 8);
	CU_TST(pp[2] * R23 - pp[1] * R33, (A[1] * Q33 + A[2] * Q23 + B[0] * Q12 + B[1] * Q11), 0, -R33, R23, 9);

	CU_TST(pp[0] * R31 - pp[2] * R11, (A[0] * Q31 + A[2] * Q11 + B[1] * Q23 + B[2] * Q22), R31, 0, -R11, 10);
	CU_TST(pp[0] * R32 - pp[2] * R12, (A[0] * Q32 + A[2] * Q12 + B[0] * Q23 + B[2] * Q21), R32, 0, -R12, 11);
	CU_TST(pp[0] * R33 - pp[2] * R13, (A[0] * Q33 + A[2] * Q13 + B[0] * Q22 + B[1] * Q21), R33, 0, -R13, 12);

	CU_TST(pp[1] * R11 - pp[0] * R21, (A[0] * Q21 + A[1] * Q11 + B[1] * Q33 + B[2] * Q32), -R21, R11, 0, 13);
	CU_TST(pp[1] * R12 - pp[0] * R22, (A[0] * Q22 + A[1] * Q12 + B[0] * Q33 + B[2] * Q31), -R22, R12, 0, 14);
	CU_TST(pp[1] * R13 - pp[0] * R23, (A[0] * Q23 + A[1] * Q13 + B[0] * Q32 + B[1] * Q31), -R23, R13, 0, 15);
#undef CU_TST

	if (!code) return 0;

	float normal[3];
	if (normalR) {
		normal[0] = normalR[0];
		normal[1] = normalR[4];
		normal[2] = normalR[8];
	} else {
		normal[0] = CuDDOT(R1 + 0, normalC);
		normal[1] = CuDDOT(R1 + 4, normalC);
		normal[2] = CuDDOT(R1 + 8, normalC);
	}
	if (invert_normal) {
		normal[0] = -normal[0];
		normal[1] = -normal[1];
		normal[2] = -normal[2];
	}
	float depth = -s;

	if (code > 6) {
		float pa[3], pb[3], sign;
		for (i = 0; i < 3; i++) pa[i] = p1[i];
		for (j = 0; j < 3; j++) {
			sign = (CuDDOT14(normal, R1 + j) > 0) ? 1.0f : -1.0f;
			for (i = 0; i < 3; i++) pa[i] += sign * A[j] * R1[i * 4 + j];
		}
		for (i = 0; i < 3; i++) pb[i] = p2[i];
		for (j = 0; j < 3; j++) {
			sign = (CuDDOT14(normal, R2 + j) > 0) ? -1.0f : 1.0f;
			for (i = 0; i < 3; i++) pb[i] += sign * B[j] * R2[i * 4 + j];
		}

		float alpha, beta;
		float ua[3], ub[3];
		for (i = 0; i < 3; i++) ua[i] = R1[((code) - 7) / 3 + i * 4];
		for (i = 0; i < 3; i++) ub[i] = R2[((code) - 7) % 3 + i * 4];

		CuLineClosestApproach(pa, ua, pb, ub, &alpha, &beta);
		for (i = 0; i < 3; i++) pa[i] += ua[i] * alpha;
		for (i = 0; i < 3; i++) pb[i] += ub[i] * beta;

		output.c[output.n].normal = { -normal[0], -normal[1], -normal[2] };
		output.c[output.n].point = { pb[0], pb[1], pb[2] };
		output.c[output.n].depth = -depth;
		output.n++;
		return 1;
	}

	const float *Ra, *Rb, *paf, *pbf, *Sa, *Sb;
	if (code <= 3) {
		Ra = R1; Rb = R2; paf = p1; pbf = p2; Sa = A; Sb = B;
	} else {
		Ra = R2; Rb = R1; paf = p2; pbf = p1; Sa = B; Sb = A;
	}

	float normal2[3], nr[3], anr[3];
	if (code <= 3) {
		normal2[0] = normal[0]; normal2[1] = normal[1]; normal2[2] = normal[2];
	} else {
		normal2[0] = -normal[0]; normal2[1] = -normal[1]; normal2[2] = -normal[2];
	}
	nr[0] = CuDDOT41(Rb + 0, normal2);
	nr[1] = CuDDOT41(Rb + 1, normal2);
	nr[2] = CuDDOT41(Rb + 2, normal2);
	anr[0] = fabsf(nr[0]);
	anr[1] = fabsf(nr[1]);
	anr[2] = fabsf(nr[2]);

	int lanr, a1, a2;
	if (anr[1] > anr[0]) {
		if (anr[1] > anr[2]) { a1 = 0; lanr = 1; a2 = 2; }
		else { a1 = 0; a2 = 1; lanr = 2; }
	} else {
		if (anr[0] > anr[2]) { lanr = 0; a1 = 1; a2 = 2; }
		else { a1 = 0; a2 = 1; lanr = 2; }
	}

	float center[3];
	if (nr[lanr] < 0) {
		for (i = 0; i < 3; i++) center[i] = pbf[i] - paf[i] + Sb[lanr] * Rb[i * 4 + lanr];
	} else {
		for (i = 0; i < 3; i++) center[i] = pbf[i] - paf[i] - Sb[lanr] * Rb[i * 4 + lanr];
	}

	int codeN, code1, code2;
	if (code <= 3) codeN = code - 1; else codeN = code - 4;
	if (codeN == 0) { code1 = 1; code2 = 2; }
	else if (codeN == 1) { code1 = 0; code2 = 2; }
	else { code1 = 0; code2 = 1; }

	float quad[8];
	float c1, c2, m11, m12, m21, m22;
	c1 = CuDDOT14(center, Ra + code1);
	c2 = CuDDOT14(center, Ra + code2);
	m11 = CuDDOT44(Ra + code1, Rb + a1);
	m12 = CuDDOT44(Ra + code1, Rb + a2);
	m21 = CuDDOT44(Ra + code2, Rb + a1);
	m22 = CuDDOT44(Ra + code2, Rb + a2);
	{
		float k1 = m11 * Sb[a1];
		float k2 = m21 * Sb[a1];
		float k3 = m12 * Sb[a2];
		float k4 = m22 * Sb[a2];
		quad[0] = c1 - k1 - k3;
		quad[1] = c2 - k2 - k4;
		quad[2] = c1 - k1 + k3;
		quad[3] = c2 - k2 + k4;
		quad[4] = c1 + k1 + k3;
		quad[5] = c2 + k2 + k4;
		quad[6] = c1 + k1 - k3;
		quad[7] = c2 + k2 - k4;
	}

	float rect[2];
	rect[0] = Sa[code1];
	rect[1] = Sa[code2];

	float ret[16];
	int n = CuIntersectRectQuad2(rect, quad, ret);
	if (n < 1) return 0;

	float point[3 * 8];
	float dep[8];
	float det1 = 1.f / (m11 * m22 - m12 * m21);
	m11 *= det1;
	m12 *= det1;
	m21 *= det1;
	m22 *= det1;
	int cnum = 0;
	for (j = 0; j < n; j++) {
		float k1 = m22 * (ret[j * 2] - c1) - m12 * (ret[j * 2 + 1] - c2);
		float k2 = -m21 * (ret[j * 2] - c1) + m11 * (ret[j * 2 + 1] - c2);
		for (i = 0; i < 3; i++)
			point[cnum * 3 + i] = center[i] + k1 * Rb[i * 4 + a1] + k2 * Rb[i * 4 + a2];
		dep[cnum] = Sa[codeN] - CuDDOT(normal2, point + cnum * 3);
		if (dep[cnum] >= 0) {
			ret[cnum * 2] = ret[j * 2];
			ret[cnum * 2 + 1] = ret[j * 2 + 1];
			cnum++;
		}
	}
	if (cnum < 1) return 0;

	int maxc = 4;
	if (maxc > cnum) maxc = cnum;
	if (maxc < 1) maxc = 1;

	if (cnum <= maxc) {
		if (code < 4) {
			for (j = 0; j < cnum; j++) {
				CuBoxBoxContact& cc = output.c[output.n++];
				cc.normal = { -normal[0], -normal[1], -normal[2] };
				cc.point = { point[j * 3 + 0] + paf[0], point[j * 3 + 1] + paf[1], point[j * 3 + 2] + paf[2] };
				cc.depth = -dep[j];
			}
		} else {
			for (j = 0; j < cnum; j++) {
				CuBoxBoxContact& cc = output.c[output.n++];
				cc.normal = { -normal[0], -normal[1], -normal[2] };
				cc.point = {
					point[j * 3 + 0] + paf[0] - normal[0] * dep[j],
					point[j * 3 + 1] + paf[1] - normal[1] * dep[j],
					point[j * 3 + 2] + paf[2] - normal[2] * dep[j] };
				cc.depth = -dep[j];
			}
		}
	} else {
		int i1 = 0;
		float maxdepth = dep[0];
		for (i = 1; i < cnum; i++) {
			if (dep[i] > maxdepth) {
				maxdepth = dep[i];
				i1 = i;
			}
		}
		int iret[8];
		CuCullPoints2(cnum, ret, maxc, i1, iret);
		for (j = 0; j < maxc; j++) {
			float posInWorld[3];
			for (i = 0; i < 3; i++)
				posInWorld[i] = point[iret[j] * 3 + i] + paf[i];
			CuBoxBoxContact& cc = output.c[output.n++];
			cc.normal = { -normal[0], -normal[1], -normal[2] };
			if (code < 4) {
				cc.point = { posInWorld[0], posInWorld[1], posInWorld[2] };
			} else {
				cc.point = {
					posInWorld[0] - normal[0] * dep[iret[j]],
					posInWorld[1] - normal[1] * dep[iret[j]],
					posInWorld[2] - normal[2] * dep[iret[j]] };
			}
			cc.depth = -dep[iret[j]];
		}
		cnum = maxc;
	}
	return cnum;
}

// btBoxBoxDetector::getClosestPoints packing: R[c + 4*j] = basis[j][c],
// sides = 2*halfExtentsWithMargin
__device__ inline void CuBoxBoxClosestPoints(
	const CuMat3& basisA, CuVec3 originA, CuVec3 halfExtWithMarginA,
	const CuMat3& basisB, CuVec3 originB, CuVec3 halfExtWithMarginB,
	CuBoxBoxOut& out) {

	float R1[12], R2[12];
	for (int j = 0; j < 3; j++) {
		R1[0 + 4 * j] = (&basisA.r[j].x)[0];
		R2[0 + 4 * j] = (&basisB.r[j].x)[0];
		R1[1 + 4 * j] = (&basisA.r[j].x)[1];
		R2[1 + 4 * j] = (&basisB.r[j].x)[1];
		R1[2 + 4 * j] = (&basisA.r[j].x)[2];
		R2[2 + 4 * j] = (&basisB.r[j].x)[2];
	}
	float p1[3] = { originA.x, originA.y, originA.z };
	float p2[3] = { originB.x, originB.y, originB.z };
	float side1[3] = { 2.f * halfExtWithMarginA.x, 2.f * halfExtWithMarginA.y, 2.f * halfExtWithMarginA.z };
	float side2[3] = { 2.f * halfExtWithMarginB.x, 2.f * halfExtWithMarginB.y, 2.f * halfExtWithMarginB.z };

	out.n = 0;
	CuDBoxBox2(p1, R1, side1, p2, R2, side2, out);
}

// ================== Quantized BVH walk + convex-vs-mesh (btConvexConcave) ==========

struct CuBvhView {
	const GpuBvhNode* nodes;
	const GpuBvhSubtree* subtrees;
	int numSubtrees;
	CuVec3 aabbMin, aabbMax, quantization;
};

// btQuantizedBvh::quantizeWithClamp: clamp (setMax(min)/setMin(max)), v = (p-min)*q,
// max: (ushort)(v+1)|1, min: (ushort)v & 0xfffe (v>=0 after clamp; C trunc cast)
__device__ __forceinline__ void CuBvhQuantizeClamp(const CuBvhView& b, CuVec3 point, int isMax, uint16_t out[3]) {
	CuVec3 c = point;
	c.x = fmaxf(c.x, b.aabbMin.x); c.y = fmaxf(c.y, b.aabbMin.y); c.z = fmaxf(c.z, b.aabbMin.z);
	c.x = fminf(c.x, b.aabbMax.x); c.y = fminf(c.y, b.aabbMax.y); c.z = fminf(c.z, b.aabbMax.z);
	CuVec3 v = (c - b.aabbMin) * b.quantization;
	if (isMax) {
		out[0] = (uint16_t)(((uint16_t)(v.x + 1.0f)) | 1);
		out[1] = (uint16_t)(((uint16_t)(v.y + 1.0f)) | 1);
		out[2] = (uint16_t)(((uint16_t)(v.z + 1.0f)) | 1);
	} else {
		out[0] = (uint16_t)(((uint16_t)(v.x)) & 0xfffe);
		out[1] = (uint16_t)(((uint16_t)(v.y)) & 0xfffe);
		out[2] = (uint16_t)(((uint16_t)(v.z)) & 0xfffe);
	}
}

__device__ __forceinline__ bool CuTestQuantAabb(
	const uint16_t* min1, const uint16_t* max1, const uint16_t* min2, const uint16_t* max2) {
	bool overlap = true;
	overlap = (min1[0] > max2[0] || max1[0] < min2[0]) ? false : overlap;
	overlap = (min1[2] > max2[2] || max1[2] < min2[2]) ? false : overlap;
	overlap = (min1[1] > max2[1] || max1[1] < min2[1]) ? false : overlap;
	return overlap;
}

// walkStacklessQuantizedTreeCacheFriendly + walkStacklessQuantizedTree: triangles
// stream to the functor in EXACT CPU order (subtree-header order, then array order)
template <typename F>
__device__ inline void CuBvhWalk(const CuBvhView& b, const uint16_t qmin[3], const uint16_t qmax[3], F&& processTriangle) {
	for (int s = 0; s < b.numSubtrees; s++) {
		const GpuBvhSubtree& st = b.subtrees[s];
		if (CuTestQuantAabb(qmin, qmax, st.qmin, st.qmax)) {
			int curIndex = st.rootNodeIndex;
			int endIndex = st.rootNodeIndex + st.subtreeSize;
			const GpuBvhNode* node = &b.nodes[curIndex];
			while (curIndex < endIndex) {
				bool overlap = CuTestQuantAabb(qmin, qmax, node->qmin, node->qmax);
				bool isLeaf = node->escapeIndexOrTriangleIndex >= 0;
				if (isLeaf && overlap) {
					// triangleIndex = e & ~(~0u << 21); partId = e >> 21 (0 for our meshes)
					int triIdx = node->escapeIndexOrTriangleIndex & ~((~0u) << (31 - 10));
					processTriangle(triIdx);
				}
				if (overlap || isLeaf) {
					node++;
					curIndex++;
				} else {
					int escape = -node->escapeIndexOrTriangleIndex;
					node += escape;
					curIndex += escape;
				}
			}
		}
	}
}

// TestTriangleAgainstAabb2 (btAabbUtil2.h:54): Z, X, Y order, min/max trees
__device__ __forceinline__ bool CuTestTriangleAabb(
	CuVec3 p1, CuVec3 p2, CuVec3 p3, CuVec3 aabbMin, CuVec3 aabbMax) {
	const float* v1 = &p1.x;
	const float* v2 = &p2.x;
	const float* v3 = &p3.x;
	const float* mn = &aabbMin.x;
	const float* mx = &aabbMax.x;
	const int order[3] = { 2, 0, 1 };
	for (int k = 0; k < 3; k++) {
		int i = order[k];
		float lo = fminf(fminf(v1[i], v2[i]), v3[i]);
		if (lo > mx[i]) return false;
		float hi = fmaxf(fmaxf(v1[i], v2[i]), v3[i]);
		if (hi < mn[i]) return false;
	}
	return true;
}

// Contact sink for the concave path: accumulates into the shared per-pair manifold
// via CuAddContactPoint, recording the triangle index for callback/internal-edge use.
struct CuConcaveContactCtx {
	CuManifold* manifold;
	const CuContactBodyInfo* convexBody; // body0 (dyn)
	const CuContactBodyInfo* meshBody;   // body1 (static)
	int lastInsertIndex;
	int lastTriangleIndex;
};

// btConvexConcaveCollisionAlgorithm::processCollision per pair (convex vs one mesh).
// Mirrors btConvexTriangleCallback: AABB gather via BVH, per-tri early-out, dispatch
// (sphere -> SphereTriangleDetector, box -> convexConvex GJK), accumulate into the
// SHARED manifold, no per-triangle refresh. The Arena contact callback + internal-edge
// adjustment hooks are invoked by the CALLER per inserted point (CPU order preserved:
// callback fires inside addContactPoint per triangle).
template <typename FCallback>
__device__ inline void CuConvexVsMesh(
	const CuGjkShape& convex, // box or sphere (margins set)
	const CuMat3& convexBasis, CuVec3 convexOrigin,
	const GpuWorldGeo& geo, int meshIdx, const CuBvhView& bvh, float meshMargin,
	const CuContactBodyInfo& convexBody, const CuContactBodyInfo& meshBody,
	CuManifold& manifold, float manifoldBreakingThreshold,
	FCallback&& onContactAdded /*(insertIndex, triangleIndex)*/) {

	// convexInTriangleSpace = meshWT.inverse() * convexWT (mesh = identity transform:
	// canon trees). Identity inverse: basisT = I, invOrigin = I * -(0,0,0)
	const CuMat3 ident = CuIdentity3();
	CuVec3 meshInvOrigin = CuMat3MulVec(ident, CuNeg({ 0, 0, 0 }));
	CuMat3 convexInTriBasis = CuMat3Mul(ident, convexBasis);
	CuVec3 convexInTriOrigin = CuTransformPoint(ident, meshInvOrigin, convexOrigin);

	// convexShape->getAabb(convexInTriangleSpace) +- (meshMargin + 0)
	CuVec3 aabbMin, aabbMax;
	CuShapeAabb(convex, convexInTriBasis, convexInTriOrigin, aabbMin, aabbMax);
	CuVec3 extra = { meshMargin, meshMargin, meshMargin };
	aabbMax = aabbMax + extra;
	aabbMin = aabbMin - extra;

	uint16_t qmin[3], qmax[3];
	CuBvhQuantizeClamp(bvh, aabbMin, 0, qmin);
	CuBvhQuantizeClamp(bvh, aabbMax, 1, qmax);

	const GpuMeshInfo& mi = geo.meshes[meshIdx];
	const GpuVec3* tv = geo.meshTris + (size_t)mi.triStart * 3;

	CuBvhWalk(bvh, qmin, qmax, [&](int triIdx) {
		CuVec3 t0 = GV(tv[triIdx * 3]);
		CuVec3 t1 = GV(tv[triIdx * 3 + 1]);
		CuVec3 t2 = GV(tv[triIdx * 3 + 2]);

		// btConvexTriangleCallback::processTriangle
		if (!CuTestTriangleAabb(t0, t1, t2, aabbMin, aabbMax))
			return;

		{ // early-out: support projection on both sides of the world triangle plane
			// (mesh world transform = identity: tri verts already world, but CPU runs
			// the transform trees - identity canon)
			CuVec3 v0 = CuTransformPoint(ident, { 0, 0, 0 }, t0);
			CuVec3 v1 = CuTransformPoint(ident, { 0, 0, 0 }, t1);
			CuVec3 v2 = CuTransformPoint(ident, { 0, 0, 0 }, t2);

			CuVec3 triangle_normal_world = CuCross(v1 - v0, v2 - v0);
			triangle_normal_world = CuNormalize3(triangle_normal_world); // rsqrt path

			// localGetSupportingVertex(basis.inverse() * n) WITH margin
			CuMat3 basisInv = CuMat3Inverse(convexBasis);
			float contact_threshold = manifoldBreakingThreshold; // + closestPoint(0)

			for (int side = 0; side < 2; side++) {
				CuVec3 dirLocal = CuMat3MulVec(basisInv, triangle_normal_world);
				CuVec3 localPt;
				if (convex.kind == CU_SHAPE_BOX) {
					CuVec3 heWithMargin = {
						convex.boxHalfExtNoMargin.x + convex.margin,
						convex.boxHalfExtNoMargin.y + convex.margin,
						convex.boxHalfExtNoMargin.z + convex.margin };
					localPt = CuSupportBox(heWithMargin, dirLocal);
				} else {
					localPt = CuSupportSphere(convex.margin, dirLocal);
				}
				CuVec3 worldPt = CuTransformPoint(convexBasis, convexOrigin, localPt);
				float proj_distPt = CuDot3(triangle_normal_world, worldPt);
				float proj_distTr = CuDot3(triangle_normal_world, v0);
				float dist = proj_distTr - proj_distPt;
				if (dist > contact_threshold)
					return;
				triangle_normal_world = triangle_normal_world * -1.0f;
			}
		}

		// dispatch: triangle shape with margin = meshMargin, at mesh transform (identity)
		if (convex.kind == CU_SHAPE_SPHERE) {
			// btSphereTriangleCollisionAlgorithm: detector threshold = manifold breaking
			// sphereInTr = triWT.inverseTimes(sphereWT): identity tri -> canon trees
			CuVec3 centerLocal = CuVecMulMat(convexOrigin - CuVec3{ 0, 0, 0 }, ident);
			CuGjkShape triShape;
			(void)triShape;
			CuVec3 point, normal;
			float depth;
			if (CuSphereTriangleCollide(centerLocal, convex.margin /*radius*/,
					t0, t1, t2, manifoldBreakingThreshold, point, normal, depth)) {
				CuVec3 nW = CuMat3MulVec(ident, normal);
				CuVec3 pW = CuTransformPoint(ident, { 0, 0, 0 }, point);
				int idx = CuAddContactPoint(manifold, *(&convexBody), *(&meshBody), false,
					nW, pW, depth, -1, -1, 0, triIdx);
				if (idx >= 0)
					onContactAdded(idx, triIdx);
			}
		} else {
			// convexConvex plain GJK (box vs triangle); maxDistSq = (marginA+marginB+
			// manifold breaking threshold)^2
			CuGjkShape tri;
			tri.kind = CU_SHAPE_TRIANGLE;
			tri.triV[0] = t0;
			tri.triV[1] = t1;
			tri.triV[2] = t2;
			tri.margin = meshMargin;
			float maxd = convex.margin + tri.margin + manifoldBreakingThreshold;
			CuGjkResult r;
			CuGjkClosestPoints(convex, convexBasis, convexOrigin, tri, ident, { 0, 0, 0 },
				maxd * maxd, true, r);
			if (r.hasContact) {
				int idx = CuAddContactPoint(manifold, *(&convexBody), *(&meshBody), false,
					r.normalOnB, r.pointOnBWorld, r.distance, -1, -1, 0, triIdx);
				if (idx >= 0)
					onContactAdded(idx, triIdx);
			}
		}
	});

	// resultOut->refreshContactPoints() at the end of the concave pass
	CuManifoldRefresh(manifold, convexBasis, convexOrigin, ident, { 0, 0, 0 });
}

// ================== btInternalEdgeUtility port =====================================
// btAdjustInternalEdgeContacts with flags=0 (Arena passes none): frontFacing=1,
// no double-sided. Triangle info extracted host-side into a flat per-triangle array
// (partId always 0 for our meshes). Thresholds (btTriangleInfoMap defaults):
// convexEpsilon=0, edgeDistanceThreshold=0.1, maxEdgeAngleThreshold=2*SIMD_PI.

#define CU_TRI_INFO_V0V1_CONVEX 1
#define CU_TRI_INFO_V1V2_CONVEX 2
#define CU_TRI_INFO_V2V0_CONVEX 4
#define CU_TRI_INFO_V0V1_SWAP_NORMALB 8
#define CU_TRI_INFO_V1V2_SWAP_NORMALB 16
#define CU_TRI_INFO_V2V0_SWAP_NORMALB 32
#define CU_EDGE_DISTANCE_THRESHOLD 0.1f
#define CU_MAX_EDGE_ANGLE_THRESHOLD 6.283185307179586232f // SIMD_2_PI
#define CU_CONVEX_EPSILON 0.0f

// (GpuTriInfo lives in GpuTypes.h - host extraction needs it)

// btQuaternion operator*(quat, vec) SSE tree:
//  x = (w*vx + y*vz) - z*vy ; y = (w*vy + z*vx) - x*vz
//  z = (w*vz + x*vy) - y*vx ; w = -(x*vx + y*vy) - z*vz
__device__ __forceinline__ CuQuat CuQuatMulVec(CuQuat q, CuVec3 v) {
	return {
		(q.w * v.x + q.y * v.z) - q.z * v.y,
		(q.w * v.y + q.z * v.x) - q.x * v.z,
		(q.w * v.z + q.x * v.y) - q.y * v.x,
		-(q.x * v.x + q.y * v.y) - q.z * v.z
	};
}

// quatRotate(q, v) = (q * v) * q.inverse(); xyz of the result
__device__ __forceinline__ CuVec3 CuQuatRotate(CuQuat rotation, CuVec3 v) {
	CuQuat q = CuQuatMulVec(rotation, v);
	CuQuat inv = { -rotation.x, -rotation.y, -rotation.z, rotation.w };
	q = CuQuatMul(q, inv);
	return { q.x, q.y, q.z };
}

// btNearestPointInLineSegment (btInternalEdgeUtility.cpp:357); fuzzyZero =
// length2 < SIMD_EPSILON*SIMD_EPSILON
__device__ __forceinline__ CuVec3 CuNearestPointInLineSegment(CuVec3 point, CuVec3 line0, CuVec3 line1) {
	CuVec3 lineDelta = line1 - line0;
	if (CuLength2(lineDelta) < CU_SIMD_EPSILON * CU_SIMD_EPSILON)
		return line0;
	float delta = CuDot3(point - line0, lineDelta) / CuDot3(lineDelta, lineDelta);
	if (delta < 0) delta = 0;
	else if (delta > 1) delta = 1;
	return line0 + lineDelta * delta;
}

// btGetAngle: atan2(swing.refAxis0, swing.refAxis1) - double-rounded stand-in
__device__ __forceinline__ float CuGetAngleIE(CuVec3 edgeA, CuVec3 normalA, CuVec3 normalB) {
	return (float)atan2((double)CuDot3(normalB, edgeA), (double)CuDot3(normalB, normalA));
}

// btClampNormal (btInternalEdgeUtility.cpp:380)
__device__ __forceinline__ bool CuClampNormal(CuVec3 edge, CuVec3 tri_normal, CuVec3 localContactNormalOnB,
	float correctedEdgeAngle, CuVec3& clampedLocalNormal) {
	CuVec3 edgeCross = CuNormalize3(CuCross(edge, tri_normal)); // .normalize() rsqrt
	float curAngle = CuGetAngleIE(edgeCross, tri_normal, localContactNormalOnB);

	if (correctedEdgeAngle < 0) {
		if (curAngle < correctedEdgeAngle) {
			float diffAngle = correctedEdgeAngle - curAngle;
			CuQuat rotation = CuQuatAxisAngle(edge, diffAngle);
			clampedLocalNormal = CuMat3MulVec(CuMat3SetRotation(rotation), localContactNormalOnB);
			return true;
		}
	}
	if (correctedEdgeAngle >= 0) {
		if (curAngle > correctedEdgeAngle) {
			float diffAngle = correctedEdgeAngle - curAngle;
			CuQuat rotation = CuQuatAxisAngle(edge, diffAngle);
			clampedLocalNormal = CuMat3MulVec(CuMat3SetRotation(rotation), localContactNormalOnB);
			return true;
		}
	}
	return false;
}

// btAdjustInternalEdgeContacts (mesh transform identity, flags=0). Mutates the
// manifold point exactly like CPU. Mesh body = "colObj0Wrap" (the triangle side).
__device__ inline void CuAdjustInternalEdgeContacts(
	CuManifoldPoint& cp,
	CuVec3 v0, CuVec3 v1, CuVec3 v2,
	const GpuTriInfo& info) {

	if (!info.hasInfo)
		return;

	const CuMat3 ident = CuIdentity3(); // mesh world transform basis
	const float frontFacing = 1.0f;

	// tri_shape->calcNormal: cross + normalize() [rsqrt]
	CuVec3 tri_normal = CuNormalize3(CuCross(v1 - v0, v2 - v0));

	// stale-nearest quirk: computed for edge V0V1 BEFORE the bestedge loop
	CuVec3 nearest = CuNearestPointInLineSegment(cp.localPointB, v0, v1);
	CuVec3 contact = cp.localPointB;

	bool isNearEdge = false;
	int numConcaveEdgeHits = 0;
	int numConvexEdgeHits = 0;

	// basisT * normalWorld (identity: canon trees), then normalize() [rsqrt]
	CuVec3 localContactNormalOnB = CuMat3MulVec(CuMat3Transpose(ident), cp.normalWorldOnB);
	localContactNormalOnB = CuNormalize3(localContactNormalOnB);

	int bestedge = -1;
	float disttobestedge = 3.402823466e+38f;
	if (fabsf(info.edgeV0V1Angle) < CU_MAX_EDGE_ANGLE_THRESHOLD) {
		CuVec3 nr = CuNearestPointInLineSegment(cp.localPointB, v0, v1);
		float len = sqrtf(CuLength2(contact - nr));
		if (len < disttobestedge) {
			bestedge = 0;
			disttobestedge = len;
		}
	}
	if (fabsf(info.edgeV1V2Angle) < CU_MAX_EDGE_ANGLE_THRESHOLD) {
		CuVec3 nr = CuNearestPointInLineSegment(cp.localPointB, v1, v2);
		float len = sqrtf(CuLength2(contact - nr));
		if (len < disttobestedge) {
			bestedge = 1;
			disttobestedge = len;
		}
	}
	if (fabsf(info.edgeV2V0Angle) < CU_MAX_EDGE_ANGLE_THRESHOLD) {
		CuVec3 nr = CuNearestPointInLineSegment(cp.localPointB, v2, v0);
		float len = sqrtf(CuLength2(contact - nr));
		if (len < disttobestedge) {
			bestedge = 2;
			disttobestedge = len;
		}
	}

	// ---- edge V0V1 (uses the STALE pre-loop `nearest`) ----
	if (fabsf(info.edgeV0V1Angle) < CU_MAX_EDGE_ANGLE_THRESHOLD) {
		float len = sqrtf(CuLength2(contact - nearest));
		if (len < CU_EDGE_DISTANCE_THRESHOLD)
			if (bestedge == 0) {
				CuVec3 edge = v0 - v1;
				isNearEdge = true;
				if (info.edgeV0V1Angle == 0.0f) {
					numConcaveEdgeHits++;
				} else {
					bool isEdgeConvex = (info.flags & CU_TRI_INFO_V0V1_CONVEX) != 0;
					float swapFactor = isEdgeConvex ? 1.0f : -1.0f;
					CuVec3 nA = tri_normal * swapFactor;
					CuQuat orn = CuQuatAxisAngle(edge, info.edgeV0V1Angle);
					CuVec3 computedNormalB = CuQuatRotate(orn, tri_normal);
					if (info.flags & CU_TRI_INFO_V0V1_SWAP_NORMALB)
						computedNormalB = computedNormalB * -1.0f;
					CuVec3 nB = computedNormalB * swapFactor;

					float NdotA = CuDot3(localContactNormalOnB, nA);
					float NdotB = CuDot3(localContactNormalOnB, nB);
					bool backFacingNormal = (NdotA < CU_CONVEX_EPSILON) && (NdotB < CU_CONVEX_EPSILON);

					if (backFacingNormal) {
						numConcaveEdgeHits++;
					} else {
						numConvexEdgeHits++;
						CuVec3 clampedLocalNormal;
						bool isClamped = CuClampNormal(edge, tri_normal * swapFactor, localContactNormalOnB, info.edgeV0V1Angle, clampedLocalNormal);
						if (isClamped) {
							if (CuDot3(clampedLocalNormal, tri_normal * frontFacing) > 0) {
								CuVec3 newNormal = CuMat3MulVec(ident, clampedLocalNormal);
								cp.normalWorldOnB = newNormal;
								cp.positionWorldOnB = cp.positionWorldOnA - cp.normalWorldOnB * cp.distance1;
								cp.localPointB = CuInvXform(ident, { 0, 0, 0 }, cp.positionWorldOnB);
							}
						}
					}
				}
			}
	}

	// ---- edge V1V2 ----
	nearest = CuNearestPointInLineSegment(contact, v1, v2);
	if (fabsf(info.edgeV1V2Angle) < CU_MAX_EDGE_ANGLE_THRESHOLD) {
		float len = sqrtf(CuLength2(contact - nearest));
		if (len < CU_EDGE_DISTANCE_THRESHOLD)
			if (bestedge == 1) {
				isNearEdge = true;
				CuVec3 edge = v1 - v2;
				if (info.edgeV1V2Angle == 0.0f) {
					numConcaveEdgeHits++;
				} else {
					bool isEdgeConvex = (info.flags & CU_TRI_INFO_V1V2_CONVEX) != 0;
					float swapFactor = isEdgeConvex ? 1.0f : -1.0f;
					CuVec3 nA = tri_normal * swapFactor;
					CuQuat orn = CuQuatAxisAngle(edge, info.edgeV1V2Angle);
					CuVec3 computedNormalB = CuQuatRotate(orn, tri_normal);
					if (info.flags & CU_TRI_INFO_V1V2_SWAP_NORMALB)
						computedNormalB = computedNormalB * -1.0f;
					CuVec3 nB = computedNormalB * swapFactor;

					float NdotA = CuDot3(localContactNormalOnB, nA);
					float NdotB = CuDot3(localContactNormalOnB, nB);
					bool backFacingNormal = (NdotA < CU_CONVEX_EPSILON) && (NdotB < CU_CONVEX_EPSILON);

					if (backFacingNormal) {
						numConcaveEdgeHits++;
					} else {
						numConvexEdgeHits++;
						// CPU quirk: shadowed local recompute, NOT normalized
						CuVec3 localN2 = CuMat3MulVec(CuMat3Transpose(ident), cp.normalWorldOnB);
						CuVec3 clampedLocalNormal;
						bool isClamped = CuClampNormal(edge, tri_normal * swapFactor, localN2, info.edgeV1V2Angle, clampedLocalNormal);
						if (isClamped) {
							if (CuDot3(clampedLocalNormal, tri_normal * frontFacing) > 0) {
								CuVec3 newNormal = CuMat3MulVec(ident, clampedLocalNormal);
								cp.normalWorldOnB = newNormal;
								cp.positionWorldOnB = cp.positionWorldOnA - cp.normalWorldOnB * cp.distance1;
								cp.localPointB = CuInvXform(ident, { 0, 0, 0 }, cp.positionWorldOnB);
							}
						}
					}
				}
			}
	}

	// ---- edge V2V0 ----
	nearest = CuNearestPointInLineSegment(contact, v2, v0);
	if (fabsf(info.edgeV2V0Angle) < CU_MAX_EDGE_ANGLE_THRESHOLD) {
		float len = sqrtf(CuLength2(contact - nearest));
		if (len < CU_EDGE_DISTANCE_THRESHOLD)
			if (bestedge == 2) {
				isNearEdge = true;
				CuVec3 edge = v2 - v0;
				if (info.edgeV2V0Angle == 0.0f) {
					numConcaveEdgeHits++;
				} else {
					bool isEdgeConvex = (info.flags & CU_TRI_INFO_V2V0_CONVEX) != 0;
					float swapFactor = isEdgeConvex ? 1.0f : -1.0f;
					CuVec3 nA = tri_normal * swapFactor;
					CuQuat orn = CuQuatAxisAngle(edge, info.edgeV2V0Angle);
					CuVec3 computedNormalB = CuQuatRotate(orn, tri_normal);
					if (info.flags & CU_TRI_INFO_V2V0_SWAP_NORMALB)
						computedNormalB = computedNormalB * -1.0f;
					CuVec3 nB = computedNormalB * swapFactor;
					(void)nA; (void)nB;

					float NdotA = CuDot3(localContactNormalOnB, nA);
					float NdotB = CuDot3(localContactNormalOnB, nB);
					bool backFacingNormal = (NdotA < CU_CONVEX_EPSILON) && (NdotB < CU_CONVEX_EPSILON);

					if (backFacingNormal) {
						numConcaveEdgeHits++;
					} else {
						numConvexEdgeHits++;
						// CPU quirk: shadowed local recompute, NOT normalized
						CuVec3 localN2 = CuMat3MulVec(CuMat3Transpose(ident), cp.normalWorldOnB);
						CuVec3 clampedLocalNormal;
						bool isClamped = CuClampNormal(edge, tri_normal * swapFactor, localN2, info.edgeV2V0Angle, clampedLocalNormal);
						if (isClamped) {
							if (CuDot3(clampedLocalNormal, tri_normal * frontFacing) > 0) {
								CuVec3 newNormal = CuMat3MulVec(ident, clampedLocalNormal);
								cp.normalWorldOnB = newNormal;
								cp.positionWorldOnB = cp.positionWorldOnA - cp.normalWorldOnB * cp.distance1;
								cp.localPointB = CuInvXform(ident, { 0, 0, 0 }, cp.positionWorldOnB);
							}
						}
					}
				}
			}
	}

	if (isNearEdge) {
		if (numConcaveEdgeHits > 0) {
			// flags=0: not concave-double-sided
			CuVec3 newNormal = tri_normal * frontFacing;
			float d = CuDot3(newNormal, localContactNormalOnB);
			if (d < 0)
				return;
			cp.normalWorldOnB = CuMat3MulVec(ident, newNormal);
			cp.positionWorldOnB = cp.positionWorldOnA - cp.normalWorldOnB * cp.distance1;
			cp.localPointB = CuInvXform(ident, { 0, 0, 0 }, cp.positionWorldOnB);
		}
	}
}

} // namespace rscuda

#endif // __CUDACC__
