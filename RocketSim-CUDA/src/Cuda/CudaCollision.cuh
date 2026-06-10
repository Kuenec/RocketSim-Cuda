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

} // namespace rscuda

#endif // __CUDACC__
