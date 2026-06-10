#pragma once
// 2026-06-10: Device raycast replicating Bullet's rayTest pipeline BIT-EXACTLY for the
// shapes RocketSim uses. Static path (btBvhTriangleMeshShape brute-force-equivalent +
// btStaticPlaneShape synthesized quad) is implemented; dynamic convex path (sphere/box
// via btSubsimplexConvexCast) lands with the voronoi solver port.
//
// Semantics mirrored from btCollisionWorld::rayTestSingleInternal,
// btTriangleRaycastCallback::processTriangle (btRaycastCallback.cpp:35) and
// btStaticPlaneShape::processAllTriangles (btStaticPlaneShape.cpp:56).
// Fraction culling is strict (<), so equal-fraction ties resolve by test order; CPU
// BVH/dbvt order may differ from our linear order on exact ties (accepted risk,
// caught by compare harnesses if ever hit).

#include "CudaMath.cuh"
#include "CudaConvexCast.cuh"
#include "GpuTypes.h"

namespace rscuda {

// Collision filter groups (btBroadphaseProxy)
#define CU_FILTER_DEFAULT 1
#define CU_FILTER_STATIC 2
#define CU_FILTER_ALL (-1)

// Uploaded once per batch; identical for all arenas (GpuPlane/GpuMeshInfo in GpuTypes.h)
struct GpuWorldGeo {
	const GpuPlane* planes;
	int numPlanes;
	const GpuMeshInfo* meshes;
	int numMeshes;
	const GpuVec3* meshTris;  // [triStart*3 + 3*t + corner]
};

struct GpuRayHit {
	float fraction;     // m_closestHitFraction (1.0 = no hit)
	CuVec3 normalWorld; // un-normalized at accumulation; caller normalizes like castRay
	int hitKind;        // -1 none, 0 = static world, 1+dynIdx = dynamic body (later)
};

__device__ __forceinline__ CuVec3 GV(const GpuVec3& v) { return { v.x, v.y, v.z }; }

// btTriangleRaycastCallback::processTriangle, flags=0 (no backface filter, flip normal
// on backface hits). Updates hit fraction/normal (normal in object-local space ==
// world space for identity-basis statics). Returns true if hit accepted.
__device__ __forceinline__ bool CuRayTriangle(
	CuVec3 from, CuVec3 to,
	CuVec3 vert0, CuVec3 vert1, CuVec3 vert2,
	float& hitFraction, CuVec3& hitNormal) {

	CuVec3 v10 = vert1 - vert0;
	CuVec3 v20 = vert2 - vert0;
	CuVec3 triangleNormal = CuCross(v10, v20);

	float dist = CuDot3(vert0, triangleNormal);
	float dist_a = CuDot3(triangleNormal, from);
	dist_a -= dist;
	float dist_b = CuDot3(triangleNormal, to);
	dist_b -= dist;

	if (dist_a * dist_b >= 0.0f)
		return false; // same sign

	float proj_length = dist_a - dist_b;
	float distance = dist_a / proj_length;

	if (distance >= hitFraction)
		return false;

	float edge_tolerance = CuLength2(triangleNormal);
	edge_tolerance *= -0.0001f;
	CuVec3 point = CuInterpolate3(from, to, distance);

	CuVec3 v0p = vert0 - point;
	CuVec3 v1p = vert1 - point;
	CuVec3 cp0 = CuCross(v0p, v1p);
	if (CuDot3(cp0, triangleNormal) < edge_tolerance)
		return false;
	CuVec3 v2p = vert2 - point;
	CuVec3 cp1 = CuCross(v1p, v2p);
	if (CuDot3(cp1, triangleNormal) < edge_tolerance)
		return false;
	CuVec3 cp2 = CuCross(v2p, v0p);
	if (CuDot3(cp2, triangleNormal) < edge_tolerance)
		return false;

	CuVec3 n = CuNormalize3(triangleNormal);
	// flags==0: flip normal for backface hits (kF_KeepUnflippedNormal not set)
	hitNormal = (dist_a <= 0.0f) ? CuNeg(n) : n;
	hitFraction = distance;
	return true;
}

// btPlaneSpace1 (btVector3.h:1266); btRecipSqrt is exact 1/sqrt
__device__ __forceinline__ void CuPlaneSpace1(CuVec3 n, CuVec3& p, CuVec3& q) {
	constexpr float CU_SQRT12 = 0.7071067811865475244f;
	if (fabsf(n.z) > CU_SQRT12) {
		float a = n.y * n.y + n.z * n.z;
		float k = 1.0f / sqrtf(a);
		p = { 0, -n.z * k, n.y * k };
		q = { a * k, -n.x * p.z, n.x * p.y };
	} else {
		float a = n.x * n.x + n.y * n.y;
		float k = 1.0f / sqrtf(a);
		p = { -n.y * k, n.x * k, 0 };
		q = { -n.z * p.y, n.z * p.x, a * k };
	}
}

__device__ __forceinline__ CuVec3 CuMin3(CuVec3 a, CuVec3 b) {
	return { fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z) };
}
__device__ __forceinline__ CuVec3 CuMax3(CuVec3 a, CuVec3 b) {
	return { fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z) };
}

// btStaticPlaneShape::processAllTriangles for a ray AABB, then the two triangles
// through CuRayTriangle. fromL/toL are in object-local space (world minus origin).
__device__ __forceinline__ bool CuRayPlane(
	CuVec3 fromL, CuVec3 toL, CuVec3 planeNormal, float planeConst,
	float& hitFraction, CuVec3& hitNormal) {

	CuVec3 aabbMin = CuMin3(fromL, toL);
	CuVec3 aabbMax = CuMax3(fromL, toL);

	CuVec3 halfExtents = (aabbMax - aabbMin) * 0.5f;
	float radius = sqrtf(CuLength2(halfExtents)); // btVector3::length
	CuVec3 center = (aabbMax + aabbMin) * 0.5f;

	CuVec3 t0, t1;
	CuPlaneSpace1(planeNormal, t0, t1);

	CuVec3 projectedCenter = center - planeNormal * (CuDot3(planeNormal, center) - planeConst);

	CuVec3 triA0 = projectedCenter + t0 * radius + t1 * radius;
	CuVec3 triA1 = projectedCenter + t0 * radius - t1 * radius;
	CuVec3 triA2 = (projectedCenter - t0 * radius) - t1 * radius;

	bool hit = CuRayTriangle(fromL, toL, triA0, triA1, triA2, hitFraction, hitNormal);

	CuVec3 triB0 = (projectedCenter - t0 * radius) - t1 * radius;
	CuVec3 triB1 = (projectedCenter - t0 * radius) + t1 * radius;
	CuVec3 triB2 = projectedCenter + t0 * radius + t1 * radius;

	hit |= CuRayTriangle(fromL, toL, triB0, triB1, triB2, hitFraction, hitNormal);
	return hit;
}

// Closest-hit ray test against the static world (meshes then planes, mirroring arena
// setup order). cbGroup/cbMask = callback filters. Returns hit with fraction strictly
// < 1 if anything was hit. Normal is world-space, NOT normalized (castRay normalizes).
__device__ __forceinline__ void CuRayTestStatic(
	const GpuWorldGeo& geo, CuVec3 from, CuVec3 to, int cbGroup, int cbMask,
	GpuRayHit& out) {

	out.fraction = 1.0f;
	out.hitKind = -1;

	const CuMat3 ident = CuIdentity3();

	for (int m = 0; m < geo.numMeshes; m++) {
		const GpuMeshInfo& mi = geo.meshes[m];
		if (!((mi.group & cbMask) && (cbGroup & mi.mask)))
			continue;
		// worldToCollisionObject = inverse(identity, origin=0): basisT=I, invOrigin =
		// I * -(0,0,0) = (-0,-0,-0); CPU runs endpoints through the transform trees
		CuVec3 invOrigin = CuMat3MulVec(ident, CuNeg({ 0, 0, 0 }));
		CuVec3 fromL = CuTransformPoint(ident, invOrigin, from);
		CuVec3 toL = CuTransformPoint(ident, invOrigin, to);
		const GpuVec3* tv = geo.meshTris + (size_t)mi.triStart * 3;
		CuVec3 localNormal;
		bool hit = false;
		for (int t = 0; t < mi.triCount; t++) {
			if (CuRayTriangle(fromL, toL, GV(tv[t * 3]), GV(tv[t * 3 + 1]), GV(tv[t * 3 + 2]),
					out.fraction, localNormal))
				hit = true;
		}
		if (hit) {
			// reportHit: hitNormalWorld = basis * localNormal (canonicalizes -0)
			out.normalWorld = CuMat3MulVec(ident, localNormal);
			out.hitKind = 0;
		}
	}

	for (int p = 0; p < geo.numPlanes; p++) {
		const GpuPlane& pl = geo.planes[p];
		if (!((pl.group & cbMask) && (cbGroup & pl.mask)))
			continue;
		CuVec3 invOrigin = CuMat3MulVec(ident, CuNeg(GV(pl.origin)));
		CuVec3 fromL = CuTransformPoint(ident, invOrigin, from);
		CuVec3 toL = CuTransformPoint(ident, invOrigin, to);
		CuVec3 localNormal;
		if (CuRayPlane(fromL, toL, GV(pl.normal), pl.planeConst, out.fraction, localNormal)) {
			out.normalWorld = CuMat3MulVec(ident, localNormal);
			out.hitKind = 0;
		}
	}
}

__device__ __forceinline__ CuMat3 GM(const GpuMat3& m) {
	return { { GV(m.row[0]), GV(m.row[1]), GV(m.row[2]) } };
}

// btTransform compose: basis = bA*bB (mat-mul tree), origin = (A)(oB)
__device__ __forceinline__ void CuTransformCompose(
	const CuMat3& bA, CuVec3 oA, const CuMat3& bB, CuVec3 oB,
	CuMat3& bOut, CuVec3& oOut) {
	bOut = CuMat3Mul(bA, bB);
	oOut = CuTransformPoint(bA, oA, oB);
}

// btRSBroadphase::GetCellIndices (btRSBroadphase.h:90): (pos-min)/cellSize via
// reciprocal multiply (btVector3 operator/), int truncation, clamp
__device__ __forceinline__ void CuGetCellIndices(const GpuGridParams& g, CuVec3 pos,
	int& i, int& j, int& k) {
	CuVec3 rel = pos - GV(g.minPos);
	float inv = 1.0f / g.cellSize;
	CuVec3 f = rel * inv;
	i = (int)f.x;
	j = (int)f.y;
	k = (int)f.z;
	i = i < 0 ? 0 : (i > g.cellsX - 1 ? g.cellsX - 1 : i);
	j = j < 0 ? 0 : (j > g.cellsY - 1 ? g.cellsY - 1 : j);
	k = k < 0 ? 0 : (k > g.cellsZ - 1 ? g.cellsZ - 1 : k);
}

// Full closest-hit ray test replicating btRSBroadphase::rayTest candidate semantics:
// statics brute force (provably reference-equal: +-1-cell smear >= max short-ray
// length), dynamics gated by the voxel cell rule for short rays. ignoreIdx excludes
// a dynamic body (the vehicle's own chassis).
__device__ __forceinline__ void CuRayTestWorld(
	const GpuWorldGeo& geo, const GpuGridParams& grid,
	const GpuDynBody* dyn, int numDyn, int ignoreIdx,
	CuVec3 from, CuVec3 to, int cbGroup, int cbMask, GpuRayHit& out) {

	CuRayTestStatic(geo, from, to, cbGroup, cbMask, out);

	// btVector3::distance2(rayTo): (rayTo - rayFrom).length2()
	CuVec3 rayD = to - from;
	float rayLenSq = CuLength2(rayD);
	bool shortRay = rayLenSq < grid.cellSizeSq;
	int ri = 0, rj = 0, rk = 0;
	if (shortRay)
		CuGetCellIndices(grid, from, ri, rj, rk);

	const CuMat3 ident = CuIdentity3();
	for (int i = 0; i < numDyn; i++) {
		if (i == ignoreIdx)
			continue;
		const GpuDynBody& d = dyn[i];
		if (!((d.group & cbMask) && (cbGroup & d.mask)))
			continue;
		if (shortRay) {
			// dyn proxy is registered in the 3x3x3 cells around its aabbMin's cell;
			// the ray only tests rayFrom's cell
			int di, dj, dk;
			CuGetCellIndices(grid, GV(d.aabbMin), di, dj, dk);
			int ddi = ri - di, ddj = rj - dj, ddk = rk - dk;
			if (ddi < -1 || ddi > 1 || ddj < -1 || ddj > 1 || ddk < -1 || ddk > 1)
				continue;
		}

		CuConvexShape shape;
		CuMat3 shapeBasis;
		CuVec3 shapeOrigin;
		if (d.kind == CU_SHAPE_BOX) {
			// compound branch: child world transform = bodyTrans * childTrans
			// (child basis identity, origin childOffset) - compose trees matter
			shape = { CU_SHAPE_BOX, 0.0f, GV(d.boxHalfExtWithMargin) };
			CuMat3 childBasis = CuIdentity3();
			CuTransformCompose(GM(d.basis), GV(d.origin), childBasis, GV(d.childOffset),
				shapeBasis, shapeOrigin);
		} else {
			shape = { CU_SHAPE_SPHERE, d.sphereRadius, { 0, 0, 0 } };
			shapeBasis = GM(d.basis);
			shapeOrigin = GV(d.origin);
		}

		CuCastResult castResult;
		castResult.fraction = out.fraction;
		if (CuSubsimplexCastRay(from, to, shape, shapeBasis, shapeOrigin, castResult)) {
			if (CuLength2(castResult.normal) > 0.0001f) {
				if (castResult.fraction < out.fraction) {
					// rayTestSingleInternal: normal.normalize() (second normalize)
					out.normalWorld = CuNormalize3(castResult.normal);
					out.fraction = castResult.fraction;
					out.hitKind = 1 + i;
				}
			}
		}
	}
}

} // namespace rscuda
