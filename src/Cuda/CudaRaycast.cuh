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
	// per-mesh quantized BVHs: ray tests must visit triangles in the CPU's
	// btQuantizedBvh ray-walk order (equal-fraction ties keep the FIRST visit -
	// proven by a wheel ray crossing a triangle edge, heatseeker seed 42 tick 1402)
	const GpuMeshBvh* bvhs;
	const GpuBvhNode* bvhNodes;
	// snowday puck hull points (null/0 in sphere-ball modes)
	const GpuVec3* hullPoints;
	int numHullPoints;
	// dropshot tiles (null/0 outside dropshot)
	const GpuTileShape* tiles;
	const GpuTileTables* tileTables;
	int numTiles;
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

// btQuantizedBvh::quantizeWithClamp (same math as the contact walk's helper, local
// variant taking raw min/quantization vectors)
__device__ __forceinline__ void CuRayQuantizeClamp(CuVec3 bvhMin, CuVec3 bvhMax,
	CuVec3 quantization, CuVec3 p, int isMax, uint16_t* q) {
	CuVec3 c = { fmaxf(p.x, bvhMin.x), fmaxf(p.y, bvhMin.y), fmaxf(p.z, bvhMin.z) };
	c = { fminf(c.x, bvhMax.x), fminf(c.y, bvhMax.y), fminf(c.z, bvhMax.z) };
	CuVec3 v = (c - bvhMin) * quantization; // component-wise (btVector3 scaled mul)
	if (isMax) {
		q[0] = (uint16_t)(((uint16_t)(v.x + 1.0f)) | 1);
		q[1] = (uint16_t)(((uint16_t)(v.y + 1.0f)) | 1);
		q[2] = (uint16_t)(((uint16_t)(v.z + 1.0f)) | 1);
	} else {
		q[0] = (uint16_t)(((uint16_t)v.x) & 0xfffe);
		q[1] = (uint16_t)(((uint16_t)v.y) & 0xfffe);
		q[2] = (uint16_t)(((uint16_t)v.z) & 0xfffe);
	}
}

// Mesh-independent part of the ray walk prologue (2026-09-02: hoisted out of
// the per-mesh loop; the same expressions on the same ray give the same bits)
struct CuBvhRay {
	CuVec3 source;
	CuVec3 rid;
	unsigned int sign[3];
	float lambda_max;
	CuVec3 aabbMin, aabbMax;
};

__device__ __forceinline__ CuBvhRay CuBvhRaySetup(CuVec3 raySource, CuVec3 rayTarget) {
	CuBvhRay r;
	r.source = raySource;
	CuVec3 rayDirection = rayTarget - raySource;
	rayDirection = CuSafeNormalize3(rayDirection);
	r.lambda_max = CuDot3(rayDirection, rayTarget - raySource);
	r.rid = {
		rayDirection.x == 0.0f ? 1e18f /*BT_LARGE_FLOAT*/ : 1.0f / rayDirection.x,
		rayDirection.y == 0.0f ? 1e18f : 1.0f / rayDirection.y,
		rayDirection.z == 0.0f ? 1e18f : 1.0f / rayDirection.z };
	r.sign[0] = r.rid.x < 0.0f;
	r.sign[1] = r.rid.y < 0.0f;
	r.sign[2] = r.rid.z < 0.0f;
	r.aabbMin = CuMin3(raySource, rayTarget); // (box-cast extents are zero)
	r.aabbMax = CuMax3(raySource, rayTarget);
	return r;
}

static __device__ __constant__ GpuConstMesh g_constMeshes[GPU_MAX_CONST_MESHES];
constexpr int CU_MAX_CONST_PLANES = 8;
static __device__ __constant__ GpuPlane g_constPlanes[CU_MAX_CONST_PLANES];

__device__ __forceinline__ bool CuRayOverlapsRootQuant(const GpuConstMesh& cm, const CuBvhRay& ray) {
	return (ray.aabbMin.x <= cm.rayMinLimit.x) & (ray.aabbMin.y <= cm.rayMinLimit.y) &
		(ray.aabbMin.z <= cm.rayMinLimit.z) & (ray.aabbMax.x >= cm.rayMaxLimit.x) &
		(ray.aabbMax.y >= cm.rayMaxLimit.y) & (ray.aabbMax.z >= cm.rayMaxLimit.z);
}

__device__ __forceinline__ bool CuRaySlab(const CuBvhRay& ray, CuVec3 lo, CuVec3 hi) {
	const float tmin = ((ray.sign[0] ? hi.x : lo.x) - ray.source.x) * ray.rid.x;
	const float tmax = ((ray.sign[0] ? lo.x : hi.x) - ray.source.x) * ray.rid.x;
	const float tymin = ((ray.sign[1] ? hi.y : lo.y) - ray.source.y) * ray.rid.y;
	const float tymax = ((ray.sign[1] ? lo.y : hi.y) - ray.source.y) * ray.rid.y;
	const float tzmin = ((ray.sign[2] ? hi.z : lo.z) - ray.source.z) * ray.rid.z;
	const float tzmax = ((ray.sign[2] ? lo.z : hi.z) - ray.source.z) * ray.rid.z;
	const bool xyOverlap = !(tmin > tymax) & !(tymin > tmax);
	const float xyMin = tymin > tmin ? tymin : tmin;
	const float xyMax = tymax < tmax ? tymax : tmax;
	const bool xyzOverlap = !(xyMin > tzmax) & !(tzmin > xyMax);
	const float enter = tzmin > xyMin ? tzmin : xyMin;
	const float exit = tzmax < xyMax ? tzmax : xyMax;
	return xyOverlap & xyzOverlap & (enter < ray.lambda_max) & (exit > 0.0f);
}

__device__ __forceinline__ bool CuRayQuantOverlapsBox(const uint16_t qmin[3], const uint16_t qmax[3],
	uint4 box) {
	return (qmin[0] <= (box.y >> 16)) & (qmax[0] >= (box.x & 0xffffu)) &
		(qmin[2] <= (box.z >> 16)) & (qmax[2] >= (box.y & 0xffffu)) &
		(qmin[1] <= (box.z & 0xffffu)) & (qmax[1] >= (box.x >> 16));
}

__device__ __forceinline__ float CuUnquantizeAxis(unsigned q, float quantization, float recip,
	float bvhMin) {
	const float value = (float)q;
	const float approx = value * recip;
	const float residual = fmaf(-approx, quantization, value);
	return fmaf(residual, recip, approx) + bvhMin;
}

__device__ __forceinline__ bool CuBvhRootSlab(const GpuConstMesh& cm, const CuBvhRay& ray) {
	return CuRaySlab(ray, GV(cm.rootBounds.mn), GV(cm.rootBounds.mx));
}

#define CU_RAY_NODE_WINDOW 3

template <typename F>
__device__ inline void CuBvhWalkRayPastRootSlab(const GpuBvhNode* nodes, int numNodes,
	const GpuConstMesh& cm, const CuBvhRay& ray, F&& processLeaf) {
	uint16_t qmin[3], qmax[3];
	CuRayQuantizeClamp(GV(cm.bvhAabbMin), GV(cm.bvhAabbMax), GV(cm.bvhQuantization), ray.aabbMin, 0, qmin);
	CuRayQuantizeClamp(GV(cm.bvhAabbMin), GV(cm.bvhAabbMax), GV(cm.bvhQuantization), ray.aabbMax, 1, qmax);
	const GpuBvhNode& root = cm.root;
	if (root.escapeIndexOrTriangleIndex >= 0) {
		processLeaf(root.escapeIndexOrTriangleIndex & ~((~0u) << (31 - 10)));
		return;
	}
	const CuVec3 quantization = GV(cm.bvhQuantization);
	const CuVec3 recip = GV(cm.bvhQuantizationRecip);
	const CuVec3 bvhMin = GV(cm.bvhAabbMin);
	const uint4* boxes = reinterpret_cast<const uint4*>(nodes);
	const int last = numNodes - 1;
	int next = 1;
	while (next <= last) {
		const int windowStart = next;
		unsigned reported = 0;
		int triangles[CU_RAY_NODE_WINDOW];
#pragma unroll
		for (int k = 0; k < CU_RAY_NODE_WINDOW; k++) {
			const int i = windowStart + k;
			const uint4 box = boxes[i < last ? i : last];
			const CuVec3 lo = { CuUnquantizeAxis(box.x & 0xffffu, quantization.x, recip.x, bvhMin.x),
				CuUnquantizeAxis(box.x >> 16, quantization.y, recip.y, bvhMin.y),
				CuUnquantizeAxis(box.y & 0xffffu, quantization.z, recip.z, bvhMin.z) };
			const CuVec3 hi = { CuUnquantizeAxis(box.y >> 16, quantization.x, recip.x, bvhMin.x),
				CuUnquantizeAxis(box.z & 0xffffu, quantization.y, recip.y, bvhMin.y),
				CuUnquantizeAxis(box.z >> 16, quantization.z, recip.z, bvhMin.z) };
			const int escapeOrTriangle = (int)box.w;
			const bool overlap = CuRayQuantOverlapsBox(qmin, qmax, box) & CuRaySlab(ray, lo, hi);
			const bool isLeaf = escapeOrTriangle >= 0;
			const bool visited = (i == next) & (i <= last);
			triangles[k] = (int)(escapeOrTriangle & ~((~0u) << (31 - 10)));
			reported |= (unsigned)(visited & isLeaf & overlap) << k;
			if (visited)
				next = i + (overlap | isLeaf ? 1 : -escapeOrTriangle);
		}
		for (; reported; reported &= reported - 1) {
			const int slot = __ffs(reported) - 1;
			int triangle = triangles[0];
#pragma unroll
			for (int k = 1; k < CU_RAY_NODE_WINDOW; k++)
				if (slot == k)
					triangle = triangles[k];
			processLeaf(triangle);
		}
	}
}

struct CuStaticRay {
	CuVec3 fromL, toL;
	CuBvhRay bvh;
};

__device__ __forceinline__ CuStaticRay CuStaticRaySetup(CuVec3 from, CuVec3 to) {
	const CuMat3 ident = CuIdentity3();
	const CuVec3 invOrigin = CuMat3MulVec(ident, CuNeg({ 0, 0, 0 }));
	CuStaticRay ray;
	ray.fromL = CuTransformPoint(ident, invOrigin, from);
	ray.toL = CuTransformPoint(ident, invOrigin, to);
	ray.bvh = CuBvhRaySetup(ray.fromL, ray.toL);
	return ray;
}

__device__ inline uint64_t CuRayMeshCandidates(const GpuWorldGeo& geo, const CuBvhRay& ray,
	int cbGroup, int cbMask) {
	uint64_t rootOverlaps = 0;
#pragma unroll 4
	for (int m = 0; m < geo.numMeshes; m++) {
		const GpuConstMesh& cm = g_constMeshes[m];
		if ((cm.group & cbMask) && (cbGroup & cm.mask) && cm.numNodes > 0 &&
			CuRayOverlapsRootQuant(cm, ray))
			rootOverlaps |= 1ull << m;
	}
	uint64_t candidates = 0;
	for (; rootOverlaps; rootOverlaps &= rootOverlaps - 1) {
		const int m = __ffsll((long long)rootOverlaps) - 1;
		if (CuBvhRootSlab(g_constMeshes[m], ray))
			candidates |= 1ull << m;
	}
	return candidates;
}

__device__ inline bool CuRayWalkMesh(const GpuWorldGeo& geo, int m, const CuStaticRay& ray,
	float& fraction, CuVec3& normal) {
	const GpuConstMesh& cm = g_constMeshes[m];
	const GpuVec3* tv = geo.meshTris + (size_t)cm.triStart * 3;
	CuVec3 localNormal;
	bool hit = false;
	CuBvhWalkRayPastRootSlab(geo.bvhNodes + cm.nodeStart, cm.numNodes, cm, ray.bvh, [&](int t) {
		if (CuRayTriangle(ray.fromL, ray.toL, GV(tv[t * 3]), GV(tv[t * 3 + 1]), GV(tv[t * 3 + 2]),
				fraction, localNormal))
			hit = true;
	});
	if (hit)
		normal = CuMat3MulVec(CuIdentity3(), localNormal);
	return hit;
}

constexpr unsigned CU_FULL_WARP = 0xffffffffu;

__device__ __forceinline__ CuVec3 CuShflVec3(CuVec3 v, int srcLane) {
	return { __shfl_sync(CU_FULL_WARP, v.x, srcLane), __shfl_sync(CU_FULL_WARP, v.y, srcLane),
		__shfl_sync(CU_FULL_WARP, v.z, srcLane) };
}

__device__ inline void CuRayTestStaticMeshesWarp(const GpuWorldGeo& geo, bool owner, CuVec3 from,
	CuVec3 to, int cbGroup, int cbMask, GpuRayHit& out) {
	const int warpLane = threadIdx.x & 31;
	uint64_t candidates = 0;
	if (owner)
		candidates = CuRayMeshCandidates(geo, CuStaticRaySetup(from, to).bvh, cbGroup, cbMask);
	const int count = __popcll(candidates);
	int end = count;
#pragma unroll
	for (int d = 1; d < 32; d <<= 1) {
		const int below = __shfl_up_sync(CU_FULL_WARP, end, d);
		if (warpLane >= d)
			end += below;
	}
	const int begin = end - count;
	const int total = __shfl_sync(CU_FULL_WARP, end, 31);
	out.fraction = 1.0f;
	out.hitKind = -1;
	out.normalWorld = { 0, 0, 0 };
	for (int base = 0; base < total; base += 32) {
		const int pair = base + warpLane;
		int src = 0;
#pragma unroll
		for (int step = 16; step > 0; step >>= 1)
			if (__shfl_sync(CU_FULL_WARP, end, src + step - 1) <= pair)
				src += step;
		const unsigned srcLo = __shfl_sync(CU_FULL_WARP, (unsigned)candidates, src);
		const unsigned srcHi = __shfl_sync(CU_FULL_WARP, (unsigned)(candidates >> 32), src);
		const int rank = pair - __shfl_sync(CU_FULL_WARP, begin, src);
		const CuVec3 srcFrom = CuShflVec3(from, src);
		const CuVec3 srcTo = CuShflVec3(to, src);
		float fraction = 1.0f;
		CuVec3 normal = { 0, 0, 0 };
		if (pair < total) {
			const int loCount = __popc(srcLo);
			const int mesh = rank < loCount ? (int)__fns(srcLo, 0, rank + 1)
				: 32 + (int)__fns(srcHi, 0, rank - loCount + 1);
			CuRayWalkMesh(geo, mesh, CuStaticRaySetup(srcFrom, srcTo), fraction, normal);
		}
		const int first = begin > base ? begin : base;
		const int last = end < base + 32 ? end : base + 32;
		for (int q = first; __any_sync(CU_FULL_WARP, q < last); q++) {
			const int walker = q < last ? q - base : warpLane;
			const float f = __shfl_sync(CU_FULL_WARP, fraction, walker);
			const CuVec3 n = CuShflVec3(normal, walker);
			if (q < last && f < out.fraction) {
				out.fraction = f;
				out.normalWorld = n;
				out.hitKind = 0;
			}
		}
	}
}

constexpr float CU_AXIS_PLANE_SKIP_BOUND = 1e6f;

__device__ __forceinline__ bool CuWithinAxisPlaneSkipBound(CuVec3 v) {
	return fabsf(v.x) < CU_AXIS_PLANE_SKIP_BOUND && fabsf(v.y) < CU_AXIS_PLANE_SKIP_BOUND &&
		fabsf(v.z) < CU_AXIS_PLANE_SKIP_BOUND;
}

__device__ __forceinline__ bool CuRayStaysOnAxisPlaneSide(const GpuPlane& pl, CuVec3 fromL,
	CuVec3 toL) {
	if (pl.planeConst != 0.0f || !CuWithinAxisPlaneSkipBound(GV(pl.origin)))
		return false;
	const bool zeroX = pl.normal.x == 0.0f;
	const bool zeroY = pl.normal.y == 0.0f;
	const bool zeroZ = pl.normal.z == 0.0f;
	float a, b;
	if (zeroY && zeroZ && fabsf(pl.normal.x) == 1.0f) {
		a = fromL.x;
		b = toL.x;
	} else if (zeroX && zeroZ && fabsf(pl.normal.y) == 1.0f) {
		a = fromL.y;
		b = toL.y;
	} else if (zeroX && zeroY && fabsf(pl.normal.z) == 1.0f) {
		a = fromL.z;
		b = toL.z;
	} else {
		return false;
	}
	return (a >= 0.0f && b >= 0.0f) || (a <= 0.0f && b <= 0.0f);
}

// Plane part of the static ray test: strict < against the running best
__device__ __forceinline__ void CuRayTestStaticPlanes(
	const GpuWorldGeo& geo, CuVec3 from, CuVec3 to, int cbGroup, int cbMask,
	GpuRayHit& out) {

	const CuMat3 ident = CuIdentity3();
	const bool bounded = CuWithinAxisPlaneSkipBound(from) && CuWithinAxisPlaneSkipBound(to);
	for (int p = 0; p < geo.numPlanes; p++) {
		const GpuPlane& pl = g_constPlanes[p];
		if (!((pl.group & cbMask) && (cbGroup & pl.mask)))
			continue;
		CuVec3 invOrigin = CuMat3MulVec(ident, CuNeg(GV(pl.origin)));
		CuVec3 fromL = CuTransformPoint(ident, invOrigin, from);
		CuVec3 toL = CuTransformPoint(ident, invOrigin, to);
		if (bounded && CuRayStaysOnAxisPlaneSide(pl, fromL, toL))
			continue;
		CuVec3 localNormal;
		if (CuRayPlane(fromL, toL, GV(pl.normal), pl.planeConst, out.fraction, localNormal)) {
			out.normalWorld = CuMat3MulVec(ident, localNormal);
			out.hitKind = 0;
		}
	}
}

__device__ __forceinline__ void CuRayTestStatic(
	const GpuWorldGeo& geo, CuVec3 from, CuVec3 to, int cbGroup, int cbMask,
	GpuRayHit& out) {

	out.fraction = 1.0f;
	out.hitKind = -1;
	const CuStaticRay ray = CuStaticRaySetup(from, to);
	for (uint64_t c = CuRayMeshCandidates(geo, ray.bvh, cbGroup, cbMask); c; c &= c - 1)
		if (CuRayWalkMesh(geo, __ffsll((long long)c) - 1, ray, out.fraction, out.normalWorld))
			out.hitKind = 0;
	CuRayTestStaticPlanes(geo, from, to, cbGroup, cbMask, out);
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
__device__ __forceinline__ void CuRayTestDynBodies(
	const GpuWorldGeo& geo, const GpuGridParams& grid,
	const GpuDynBody* dyn, int numDyn, int ignoreIdx,
	CuVec3 from, CuVec3 to, int cbGroup, int cbMask, GpuRayHit& out);

__device__ __forceinline__ void CuRayTestWorld(
	const GpuWorldGeo& geo, const GpuGridParams& grid,
	const GpuDynBody* dyn, int numDyn, int ignoreIdx,
	CuVec3 from, CuVec3 to, int cbGroup, int cbMask, GpuRayHit& out) {

	CuRayTestStatic(geo, from, to, cbGroup, cbMask, out);
	CuRayTestDynBodies(geo, grid, dyn, numDyn, ignoreIdx, from, to, cbGroup, cbMask, out);
}

// Dynamic-body tail of the world ray test: strict < against the running best,
// bodies in dyn[] order (factored 2026-06-11 for the cooperative wheel phase)
__device__ __forceinline__ void CuRayTestDynBodies(
	const GpuWorldGeo& geo, const GpuGridParams& grid,
	const GpuDynBody* dyn, int numDyn, int ignoreIdx,
	CuVec3 from, CuVec3 to, int cbGroup, int cbMask, GpuRayHit& out) {

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
		} else if (d.kind == CU_SHAPE_HULL) {
			// snowday puck: sphereRadius slot carries the hull collision margin
			shape = { CU_SHAPE_HULL, d.sphereRadius, { 0, 0, 0 },
				geo.hullPoints, geo.numHullPoints };
			shapeBasis = GM(d.basis);
			shapeOrigin = GV(d.origin);
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
