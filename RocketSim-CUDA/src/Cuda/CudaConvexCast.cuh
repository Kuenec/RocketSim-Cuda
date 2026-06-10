#pragma once
// 2026-06-10: Device port of btVoronoiSimplexSolver + btSubsimplexConvexCast (point
// cast = raycast against convex shapes) for ray-vs-ball(sphere) and ray-vs-car
// (compound box). Faithful to btVoronoiSimplexSolver.cpp / btSubSimplexConvexCast.cpp /
// btSphereShape.cpp / btBoxShape.h in the work tree, float precision config:
// equal-vertex threshold 0.0001f, MAX_CONVEX_CAST_ITERATIONS 32, epsilon 1e-4.

#include "CudaMath.cuh"

#ifdef __CUDACC__

namespace rscuda {

// ---------------- btVoronoiSimplexSolver ----------------

struct CuUsageBits {
	unsigned char a, b, c, d;
	__device__ void reset() { a = b = c = d = 0; }
};

struct CuClosestResult {
	CuVec3 closestPointOnSimplex;
	CuUsageBits usedVertices;
	float bary[4];
	bool degenerate;
	__device__ void reset() {
		degenerate = false;
		setBary(0, 0, 0, 0);
		usedVertices.reset();
	}
	__device__ bool isValid() const {
		return (bary[0] >= 0.0f) && (bary[1] >= 0.0f) && (bary[2] >= 0.0f) && (bary[3] >= 0.0f);
	}
	__device__ void setBary(float a, float b, float c = 0, float d = 0) {
		bary[0] = a; bary[1] = b; bary[2] = c; bary[3] = d;
	}
};

struct CuSimplexSolver {
	int numVertices;
	CuVec3 W[5], P[5], Q[5];
	CuVec3 lastW;
	CuVec3 cachedP1, cachedP2, cachedV;
	CuClosestResult cachedBC;
	bool cachedValidClosest;
	bool needsUpdate;

	__device__ void reset() {
		cachedValidClosest = false;
		numVertices = 0;
		needsUpdate = true;
		lastW = { 3.402823466e+38f, 3.402823466e+38f, 3.402823466e+38f }; // BT_LARGE_FLOAT
		cachedBC.reset();
	}

	__device__ void addVertex(CuVec3 w, CuVec3 p, CuVec3 q) {
		lastW = w;
		needsUpdate = true;
		W[numVertices] = w;
		P[numVertices] = p;
		Q[numVertices] = q;
		numVertices++;
	}

	__device__ void removeVertex(int index) {
		numVertices--;
		W[index] = W[numVertices];
		P[index] = P[numVertices];
		Q[index] = Q[numVertices];
	}

	__device__ void reduceVertices(const CuUsageBits& used) {
		if ((numVertices >= 4) && (!used.d)) removeVertex(3);
		if ((numVertices >= 3) && (!used.c)) removeVertex(2);
		if ((numVertices >= 2) && (!used.b)) removeVertex(1);
		if ((numVertices >= 1) && (!used.a)) removeVertex(0);
	}

	__device__ bool inSimplex(CuVec3 w) {
		bool found = false;
		// BT_USE_EQUAL_VERTEX_THRESHOLD, float threshold 0.0001f
		for (int i = 0; i < numVertices; i++) {
			CuVec3 d = W[i] - w;
			if (CuLength2(d) <= 0.0001f) {
				found = true;
				break;
			}
		}
		if (w.x == lastW.x && w.y == lastW.y && w.z == lastW.z)
			return true;
		return found;
	}

	// closestPtPointTriangle (btVoronoiSimplexSolver.cpp:313)
	__device__ static bool ClosestPtPointTriangle(CuVec3 p, CuVec3 a, CuVec3 b, CuVec3 c, CuClosestResult& result) {
		result.usedVertices.reset();

		CuVec3 ab = b - a;
		CuVec3 ac = c - a;
		CuVec3 ap = p - a;
		float d1 = CuDot3(ab, ap);
		float d2 = CuDot3(ac, ap);
		if (d1 <= 0.0f && d2 <= 0.0f) {
			result.closestPointOnSimplex = a;
			result.usedVertices.a = 1;
			result.setBary(1, 0, 0);
			return true;
		}

		CuVec3 bp = p - b;
		float d3 = CuDot3(ab, bp);
		float d4 = CuDot3(ac, bp);
		if (d3 >= 0.0f && d4 <= d3) {
			result.closestPointOnSimplex = b;
			result.usedVertices.b = 1;
			result.setBary(0, 1, 0);
			return true;
		}

		float vc = d1 * d4 - d3 * d2;
		if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
			float v = d1 / (d1 - d3);
			result.closestPointOnSimplex = a + ab * v; // a + v * ab (scalar mul commutes)
			result.usedVertices.a = 1;
			result.usedVertices.b = 1;
			result.setBary(1 - v, v, 0);
			return true;
		}

		CuVec3 cp = p - c;
		float d5 = CuDot3(ab, cp);
		float d6 = CuDot3(ac, cp);
		if (d6 >= 0.0f && d5 <= d6) {
			result.closestPointOnSimplex = c;
			result.usedVertices.c = 1;
			result.setBary(0, 0, 1);
			return true;
		}

		float vb = d5 * d2 - d1 * d6;
		if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
			float w = d2 / (d2 - d6);
			result.closestPointOnSimplex = a + ac * w;
			result.usedVertices.a = 1;
			result.usedVertices.c = 1;
			result.setBary(1 - w, 0, w);
			return true;
		}

		float va = d3 * d6 - d5 * d4;
		if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
			float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
			result.closestPointOnSimplex = b + (c - b) * w;
			result.usedVertices.b = 1;
			result.usedVertices.c = 1;
			result.setBary(0, 1 - w, w);
			return true;
		}

		float denom = 1.0f / (va + vb + vc);
		float v = vb * denom;
		float w = vc * denom;
		result.closestPointOnSimplex = a + ab * v + ac * w;
		result.usedVertices.a = 1;
		result.usedVertices.b = 1;
		result.usedVertices.c = 1;
		result.setBary(1 - v - w, v, w);
		return true;
	}

	// pointOutsideOfPlane (btVoronoiSimplexSolver.cpp:411), float precision branch
	__device__ static int PointOutsideOfPlane(CuVec3 p, CuVec3 a, CuVec3 b, CuVec3 c, CuVec3 d) {
		CuVec3 normal = CuCross(b - a, c - a);
		float signp = CuDot3(p - a, normal);
		float signd = CuDot3(d - a, normal);
		if (signd * signd < (1e-4f * 1e-4f))
			return -1;
		return signp * signd < 0.0f;
	}

	// closestPtPointTetrahedron (btVoronoiSimplexSolver.cpp:437)
	__device__ static bool ClosestPtPointTetrahedron(CuVec3 p, CuVec3 a, CuVec3 b, CuVec3 c, CuVec3 d, CuClosestResult& finalResult) {
		CuClosestResult tempResult;
		tempResult.reset();

		finalResult.closestPointOnSimplex = p;
		finalResult.usedVertices.reset();
		finalResult.usedVertices.a = 1;
		finalResult.usedVertices.b = 1;
		finalResult.usedVertices.c = 1;
		finalResult.usedVertices.d = 1;

		int pointOutsideABC = PointOutsideOfPlane(p, a, b, c, d);
		int pointOutsideACD = PointOutsideOfPlane(p, a, c, d, b);
		int pointOutsideADB = PointOutsideOfPlane(p, a, d, b, c);
		int pointOutsideBDC = PointOutsideOfPlane(p, b, d, c, a);

		if (pointOutsideABC < 0 || pointOutsideACD < 0 || pointOutsideADB < 0 || pointOutsideBDC < 0) {
			finalResult.degenerate = true;
			return false;
		}
		if (!pointOutsideABC && !pointOutsideACD && !pointOutsideADB && !pointOutsideBDC)
			return false;

		float bestSqDist = 3.402823466e+38f; // FLT_MAX
		if (pointOutsideABC) {
			ClosestPtPointTriangle(p, a, b, c, tempResult);
			CuVec3 q = tempResult.closestPointOnSimplex;
			CuVec3 qp = q - p;
			float sqDist = CuDot3(qp, qp);
			if (sqDist < bestSqDist) {
				bestSqDist = sqDist;
				finalResult.closestPointOnSimplex = q;
				finalResult.usedVertices.reset();
				finalResult.usedVertices.a = tempResult.usedVertices.a;
				finalResult.usedVertices.b = tempResult.usedVertices.b;
				finalResult.usedVertices.c = tempResult.usedVertices.c;
				finalResult.setBary(tempResult.bary[0], tempResult.bary[1], tempResult.bary[2], 0);
			}
		}
		if (pointOutsideACD) {
			ClosestPtPointTriangle(p, a, c, d, tempResult);
			CuVec3 q = tempResult.closestPointOnSimplex;
			CuVec3 qp = q - p;
			float sqDist = CuDot3(qp, qp);
			if (sqDist < bestSqDist) {
				bestSqDist = sqDist;
				finalResult.closestPointOnSimplex = q;
				finalResult.usedVertices.reset();
				finalResult.usedVertices.a = tempResult.usedVertices.a;
				finalResult.usedVertices.c = tempResult.usedVertices.b;
				finalResult.usedVertices.d = tempResult.usedVertices.c;
				finalResult.setBary(tempResult.bary[0], 0, tempResult.bary[1], tempResult.bary[2]);
			}
		}
		if (pointOutsideADB) {
			ClosestPtPointTriangle(p, a, d, b, tempResult);
			CuVec3 q = tempResult.closestPointOnSimplex;
			CuVec3 qp = q - p;
			float sqDist = CuDot3(qp, qp);
			if (sqDist < bestSqDist) {
				bestSqDist = sqDist;
				finalResult.closestPointOnSimplex = q;
				finalResult.usedVertices.reset();
				finalResult.usedVertices.a = tempResult.usedVertices.a;
				finalResult.usedVertices.b = tempResult.usedVertices.c;
				finalResult.usedVertices.d = tempResult.usedVertices.b;
				finalResult.setBary(tempResult.bary[0], tempResult.bary[2], 0, tempResult.bary[1]);
			}
		}
		if (pointOutsideBDC) {
			ClosestPtPointTriangle(p, b, d, c, tempResult);
			CuVec3 q = tempResult.closestPointOnSimplex;
			CuVec3 qp = q - p;
			float sqDist = CuDot3(qp, qp);
			if (sqDist < bestSqDist) {
				bestSqDist = sqDist;
				finalResult.closestPointOnSimplex = q;
				finalResult.usedVertices.reset();
				finalResult.usedVertices.b = tempResult.usedVertices.a;
				finalResult.usedVertices.c = tempResult.usedVertices.c;
				finalResult.usedVertices.d = tempResult.usedVertices.b;
				finalResult.setBary(0, tempResult.bary[0], tempResult.bary[2], tempResult.bary[1]);
			}
		}
		return true;
	}

	__device__ bool updateClosestVectorAndPoints() {
		if (needsUpdate) {
			cachedBC.reset();
			needsUpdate = false;

			switch (numVertices) {
			case 0:
				cachedValidClosest = false;
				break;
			case 1: {
				cachedP1 = P[0];
				cachedP2 = Q[0];
				cachedV = cachedP1 - cachedP2;
				cachedBC.reset();
				cachedBC.setBary(1, 0, 0, 0);
				cachedValidClosest = cachedBC.isValid();
				break;
			}
			case 2: {
				CuVec3 from = W[0];
				CuVec3 to = W[1];

				CuVec3 p = { 0, 0, 0 };
				CuVec3 diff = p - from;
				CuVec3 v = to - from;
				float t = CuDot3(v, diff);

				if (t > 0) {
					float dotVV = CuDot3(v, v);
					if (t < dotVV) {
						t /= dotVV;
						diff = diff - v * t;
						cachedBC.usedVertices.a = 1;
						cachedBC.usedVertices.b = 1;
					} else {
						t = 1;
						diff = diff - v;
						cachedBC.usedVertices.b = 1;
					}
				} else {
					t = 0;
					cachedBC.usedVertices.a = 1;
				}
				cachedBC.setBary(1 - t, t);

				cachedP1 = P[0] + (P[1] - P[0]) * t;
				cachedP2 = Q[0] + (Q[1] - Q[0]) * t;
				cachedV = cachedP1 - cachedP2;

				reduceVertices(cachedBC.usedVertices);
				cachedValidClosest = cachedBC.isValid();
				break;
			}
			case 3: {
				CuVec3 p = { 0, 0, 0 };
				ClosestPtPointTriangle(p, W[0], W[1], W[2], cachedBC);
				cachedP1 = P[0] * cachedBC.bary[0] + P[1] * cachedBC.bary[1] + P[2] * cachedBC.bary[2];
				cachedP2 = Q[0] * cachedBC.bary[0] + Q[1] * cachedBC.bary[1] + Q[2] * cachedBC.bary[2];
				cachedV = cachedP1 - cachedP2;
				reduceVertices(cachedBC.usedVertices);
				cachedValidClosest = cachedBC.isValid();
				break;
			}
			case 4: {
				CuVec3 p = { 0, 0, 0 };
				bool hasSeparation = ClosestPtPointTetrahedron(p, W[0], W[1], W[2], W[3], cachedBC);
				if (hasSeparation) {
					cachedP1 = P[0] * cachedBC.bary[0] + P[1] * cachedBC.bary[1] +
						P[2] * cachedBC.bary[2] + P[3] * cachedBC.bary[3];
					cachedP2 = Q[0] * cachedBC.bary[0] + Q[1] * cachedBC.bary[1] +
						Q[2] * cachedBC.bary[2] + Q[3] * cachedBC.bary[3];
					cachedV = cachedP1 - cachedP2;
					reduceVertices(cachedBC.usedVertices);
				} else {
					if (cachedBC.degenerate) {
						cachedValidClosest = false;
					} else {
						cachedValidClosest = true;
						cachedV = { 0, 0, 0 };
					}
					break;
				}
				cachedValidClosest = cachedBC.isValid();
				break;
			}
			default:
				cachedValidClosest = false;
			}
		}
		return cachedValidClosest;
	}

	__device__ bool closest(CuVec3& v) {
		bool success = updateClosestVectorAndPoints();
		v = cachedV;
		return success;
	}
};

// ---------------- convex shape supports ----------------

#define CU_SHAPE_POINT 0   // btSphereShape(0) cast shape, margin 0
#define CU_SHAPE_SPHERE 1  // ball
#define CU_SHAPE_BOX 2     // car hitbox child

struct CuConvexShape {
	int kind;
	float sphereRadius;          // margin == radius for spheres (and 0 for point)
	CuVec3 boxHalfExtWithMargin; // getHalfExtentsWithoutMargin() + margin, host-extracted
};

// btSphereShape::localGetSupportingVertex (btSphereShape.cpp:37): margin*normalize(dir),
// where degenerate dir uses normalize(-1,-1,-1). Note margin*vecnorm with margin 0
// produces SIGNED zeros from vecnorm - keep the multiply.
__device__ __forceinline__ CuVec3 CuSupportSphere(float margin, CuVec3 vec) {
	CuVec3 vecnorm = vec;
	if (CuLength2(vecnorm) < CU_SIMD_EPSILON * CU_SIMD_EPSILON)
		vecnorm = { -1.0f, -1.0f, -1.0f };
	vecnorm = CuNormalize3(vecnorm);
	return vecnorm * margin; // margin * vecnorm (commutative-exact)
}

// btBoxShape::localGetSupportingVertex (btBoxShape.h:47): fsel per component
__device__ __forceinline__ CuVec3 CuSupportBox(CuVec3 he, CuVec3 vec) {
	return {
		vec.x >= 0.0f ? he.x : -he.x,
		vec.y >= 0.0f ? he.y : -he.y,
		vec.z >= 0.0f ? he.z : -he.z
	};
}

__device__ __forceinline__ CuVec3 CuSupport(const CuConvexShape& s, CuVec3 dir) {
	if (s.kind == CU_SHAPE_BOX)
		return CuSupportBox(s.boxHalfExtWithMargin, dir);
	return CuSupportSphere(s.sphereRadius, dir);
}

// btVector3 * btMatrix3x3 (row-vector multiply): tdot columns, plain left-assoc sums
__device__ __forceinline__ CuVec3 CuVecMulMat(CuVec3 v, const CuMat3& m) {
	return {
		m.r[0].x * v.x + m.r[1].x * v.y + m.r[2].x * v.z,
		m.r[0].y * v.x + m.r[1].y * v.y + m.r[2].y * v.z,
		m.r[0].z * v.x + m.r[1].z * v.y + m.r[2].z * v.z
	};
}

// ---------------- btSubsimplexConvexCast::calcTimeOfImpact ----------------
// Point cast (raycast) specialization: shape A is the point (at ray from/to), shape B
// static at its world transform. Mirrors btSubSimplexConvexCast.cpp:32.
struct CuCastResult {
	float fraction;  // in: current closest (culling); out: hit fraction
	CuVec3 normal;
};

__device__ __forceinline__ bool CuSubsimplexCastRay(
	CuVec3 rayFrom, CuVec3 rayTo,
	const CuConvexShape& shapeB, const CuMat3& basisB, CuVec3 originB,
	CuCastResult& result) {

	CuSimplexSolver solver;
	solver.reset();

	const CuConvexShape pointShape = { CU_SHAPE_POINT, 0.0f, { 0, 0, 0 } };
	const CuMat3 ident = CuIdentity3();

	// fromA/toA: identity basis at ray endpoints; B static
	CuVec3 linVelA = rayTo - rayFrom;
	CuVec3 linVelB = { 0, 0, 0 };

	float lambda = 0.0f;
	CuVec3 interpA = rayFrom; // interpolated origin of A (basis identity)
	CuVec3 interpB = originB; // B is static, but CPU still re-lerps its origin (and
	                          // lerp(o,o,lambda) is NOT bitwise o) - mirror exactly

	CuVec3 r = linVelA - linVelB;

	// supportA: fromA(localGetSupportingVertex(-r * basisA)); basisA = identity
	CuVec3 supVertexA = CuTransformPoint(ident, rayFrom, CuSupport(pointShape, CuVecMulMat(CuNeg(r), ident)));
	CuVec3 supVertexB = CuTransformPoint(basisB, originB, CuSupport(shapeB, CuVecMulMat(r, basisB)));
	CuVec3 v = supVertexA - supVertexB;

	int maxIter = 32; // MAX_CONVEX_CAST_ITERATIONS (float build)

	CuVec3 n = { 0, 0, 0 };
	float dist2 = CuLength2(v);

	while ((dist2 > 0.0001f) && (maxIter--)) { // m_subSimplexCastEpsilon
		supVertexA = CuTransformPoint(ident, interpA, CuSupport(pointShape, CuVecMulMat(CuNeg(v), ident)));
		supVertexB = CuTransformPoint(basisB, interpB, CuSupport(shapeB, CuVecMulMat(v, basisB)));
		CuVec3 w = supVertexA - supVertexB;

		float VdotW = CuDot3(v, w);

		if (lambda > 1.0f)
			return false;

		if (VdotW > 0.0f) {
			float VdotR = CuDot3(v, r);
			if (VdotR >= -(CU_SIMD_EPSILON * CU_SIMD_EPSILON)) {
				return false;
			} else {
				lambda = lambda - VdotW / VdotR;
				// setInterpolate3 of BOTH origins (B from==to, still re-lerped)
				interpA = CuInterpolate3(rayFrom, rayTo, lambda);
				interpB = CuInterpolate3(originB, originB, lambda);
				w = supVertexA - supVertexB;
				n = v;
			}
		}
		if (!solver.inSimplex(w))
			solver.addVertex(w, supVertexA, supVertexB);

		if (solver.closest(v)) {
			dist2 = CuLength2(v);
		} else {
			dist2 = 0.0f;
		}
	}

	result.fraction = lambda;
	if (CuLength2(n) >= CU_SIMD_EPSILON * CU_SIMD_EPSILON)
		result.normal = CuNormalize3(n); // n.normalized()
	else
		result.normal = { 0, 0, 0 };
	return true;
}

} // namespace rscuda

#endif // __CUDACC__
