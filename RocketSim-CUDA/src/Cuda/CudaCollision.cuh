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

// EPA penetration solver (btGjkEpaPenetrationDepthSolver -> btGjkEpa2). PENDING PORT:
// this stub returns false, which on CPU only happens when EPA itself fails. The
// manifold-compare harness FAILS LOUDLY the first time a deep-penetration pair needs
// it, so the stub cannot silently ship. Callers pass epaAvailable=true (CPU always
// has the solver).
__device__ __forceinline__ bool CuEpaPenDepth(CuSimplexSolver& simplexSolver,
	const CuGjkShape& shapeA, const CuGjkShape& shapeB,
	const CuMat3& basisA, CuVec3 originA, const CuMat3& basisB, CuVec3 originB,
	CuVec3& sepAxis, CuVec3& pOnA, CuVec3& pOnB) {
	(void)simplexSolver; (void)shapeA; (void)shapeB;
	(void)basisA; (void)originA; (void)basisB; (void)originB;
	sepAxis = { 0, 0, 0 };
	pOnA = { 0, 0, 0 };
	pOnB = { 0, 0, 0 };
	return false; // TODO(EPA port): replace with btGjkEpa2 Penetration
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

} // namespace rscuda

#endif // __CUDACC__
