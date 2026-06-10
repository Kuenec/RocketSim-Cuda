// 2026-06-10: Device half of cuda_math_test - computes every CudaMath op on the GPU
// over fuzzed inputs and demands BIT-EXACT equality with the Bullet host reference
// (host_ref.cpp, compiled by the same compiler/flags as the physics build).
// Both-NaN results compare equal (payloads differ across architectures by design);
// the gates prove the real simulation never produces NaN.
//
// Usage: cuda_math_test [--samples N] [--trig-exhaustive]
// Exit 0 = all ops bit-exact. Exit 1 = mismatch (details printed).

#include "Cuda/CudaMath.cuh"
#include "Cuda/CudaRaycast.cuh"
#include "Cuda/CudaVehicle.cuh"
#include "Cuda/CudaCollision.cuh"
#include "Cuda/RsqrtHost.h"
#include "../../tools/cuda_math_test/test_data.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace rscuda;
using namespace mathtest;

namespace mathtest {
void HostFillDerived(Input* inputs, size_t n);
void HostCompute(const Input* inputs, Output* outputs, size_t n);
void HostTrig(const uint32_t* bits, float* sinOut, float* cosOut, size_t n);
void HostGjkDerive(GjkCase* cases, size_t n);
void HostGjkCompute(const GjkCase* cases, GjkOut* outs, size_t n);
void HostSphereTri(const GjkCase* cases, GjkOut* outs, size_t n);
}

// Sphere-vs-triangle device kernel: mirrors SphereTriangleDetector::getClosestPoints
// with triangle at identity (sphereInTr = B.inverseTimes(A): identity-canon trees)
__global__ void SphereTriKernel(const GjkCase* cases, GjkOut* outs, size_t n) {
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const GjkCase c = cases[i];
	float radius = c.boxHalfExtentsCtor.x;
	float breaking = c.maxDistSq; // stashed threshold

	CuMat3 ident = CuIdentity3();
	// inverseTimes(A): basis = B^T*A = identity; origin = (A.origin - B.origin) * B.basis
	CuVec3 centerLocal = CuVecMulMat(CuVec3{ c.boxOrigin.x, c.boxOrigin.y, c.boxOrigin.z } - CuVec3{ 0, 0, 0 }, ident);

	CuVec3 point, normal;
	float depth;
	bool has = CuSphereTriangleCollide(centerLocal, radius,
		{ c.triV[0].x, c.triV[0].y, c.triV[0].z },
		{ c.triV[1].x, c.triV[1].y, c.triV[1].z },
		{ c.triV[2].x, c.triV[2].y, c.triV[2].z },
		breaking, point, normal, depth);

	outs[i].hasContact = has;
	if (has) {
		// swapResults=false: addContactPoint(B.basis*normal, B*point, depth)
		CuVec3 nW = CuMat3MulVec(ident, normal);
		CuVec3 pW = CuTransformPoint(ident, { 0, 0, 0 }, point);
		outs[i].normalOnB = { nW.x, nW.y, nW.z };
		outs[i].pointOnB = { pW.x, pW.y, pW.z };
		outs[i].distance = depth;
	} else {
		outs[i].normalOnB = { 0, 0, 0 };
		outs[i].pointOnB = { 0, 0, 0 };
		outs[i].distance = 0;
	}
}

__global__ void GjkKernel(const GjkCase* cases, GjkOut* outs, size_t n) {
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const GjkCase c = cases[i];

	CuGjkShape box;
	box.kind = CU_SHAPE_BOX;
	box.boxHalfExtNoMargin = { c.boxImplicitHalfExt.x, c.boxImplicitHalfExt.y, c.boxImplicitHalfExt.z };
	box.margin = c.boxMargin;

	CuGjkShape tri;
	tri.kind = CU_SHAPE_TRIANGLE;
	tri.triV[0] = { c.triV[0].x, c.triV[0].y, c.triV[0].z };
	tri.triV[1] = { c.triV[1].x, c.triV[1].y, c.triV[1].z };
	tri.triV[2] = { c.triV[2].x, c.triV[2].y, c.triV[2].z };
	tri.margin = c.triMargin;

	CuMat3 basisA = { { { c.boxBasis.r[0].x, c.boxBasis.r[0].y, c.boxBasis.r[0].z },
	                    { c.boxBasis.r[1].x, c.boxBasis.r[1].y, c.boxBasis.r[1].z },
	                    { c.boxBasis.r[2].x, c.boxBasis.r[2].y, c.boxBasis.r[2].z } } };
	CuVec3 originA = { c.boxOrigin.x, c.boxOrigin.y, c.boxOrigin.z };
	CuMat3 basisB = CuIdentity3();
	CuVec3 originB = { 0, 0, 0 };

	CuGjkResult r;
	CuGjkClosestPoints(box, basisA, originA, tri, basisB, originB, c.maxDistSq, true, r);

	outs[i].hasContact = r.hasContact;
	outs[i].normalOnB = r.hasContact ? V3{ r.normalOnB.x, r.normalOnB.y, r.normalOnB.z } : V3{ 0, 0, 0 };
	outs[i].pointOnB = r.hasContact ? V3{ r.pointOnBWorld.x, r.pointOnBWorld.y, r.pointOnBWorld.z } : V3{ 0, 0, 0 };
	outs[i].distance = r.hasContact ? r.distance : 0.0f;
}

static CuVec3 CV(const V3& v) { return { v.x, v.y, v.z }; }
static __device__ CuVec3 DV(const V3& v) { return { v.x, v.y, v.z }; }
static __device__ V3 VD(const CuVec3& v) { return { v.x, v.y, v.z }; }
static __device__ CuQuat DQ(const Q4& q) { return { q.x, q.y, q.z, q.w }; }
static __device__ Q4 QD(const CuQuat& q) { return { q.x, q.y, q.z, q.w }; }
static __device__ CuMat3 DM(const M3& m) {
	return { { DV(m.r[0]), DV(m.r[1]), DV(m.r[2]) } };
}
static __device__ M3 MD(const CuMat3& m) {
	return { { VD(m.r[0]), VD(m.r[1]), VD(m.r[2]) } };
}

__global__ void ComputeKernel(const Input* inputs, Output* outputs, size_t n) {
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const Input in = inputs[i];
	Output o;

	o.dot3 = CuDot3(DV(in.a), DV(in.b));
	o.cross = VD(CuCross(DV(in.a), DV(in.b)));
	o.quatMul = QD(CuQuatMul(DQ(in.qa), DQ(in.qb)));
	o.quatDot = CuQuatDot4(DQ(in.qa), DQ(in.qb));
	o.quatNorm = QD(CuQuatSafeNormalize(DQ(in.qa)));
	o.mat3Mul = MD(CuMat3Mul(DM(in.ma), DM(in.mb)));
	o.getRot = QD(CuMat3GetRotation(DM(in.ma)));
	o.setRot = MD(CuMat3SetRotation(DQ(in.qb)));
	o.inertia = MD(CuUpdateInertiaTensor(DM(in.ma), DV(in.a)));

	CuMat3 ib;
	CuVec3 ip;
	CuIntegrateTransform(DM(in.ma), DV(in.a), DV(in.b), DV(in.w), in.dt, false, ib, ip);
	o.integBasis = MD(ib);
	o.integPos = VD(ip);

	float d = CuDot3(DV(in.a), DV(in.a));
	o.rsqrt = (d > CU_SIMD_EPSILON) ? CuRsqrtSS(d) : 0.0f;
	o.norm3 = (CuLength2(DV(in.a)) > CU_SIMD_EPSILON) ? VD(CuNormalize3(DV(in.a))) : V3{ 0, 0, 0 };
	o.safeNorm3 = VD(CuSafeNormalize3(DV(in.a)));
	o.axisAngle = (CuLength2(DV(in.a)) > CU_SIMD_EPSILON)
		? QD(CuQuatAxisAngle(DV(in.a), in.ang)) : Q4{ 0, 0, 0, 0 };

	outputs[i] = o;
}

__global__ void TrigKernel(const uint32_t* bits, float* sinOut, float* cosOut, size_t n) {
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float x;
	memcpy(&x, &bits[i], 4);
	sinOut[i] = CuSinf(x);
	cosOut[i] = CuCosf(x);
}

// glibc-exact atan2f/asinf ports (CudaVehicle.cuh) vs host glibc
__global__ void Atan2AsinKernel(const uint32_t* yBits, const uint32_t* xBits,
	float* atanOut, float* asinOut, size_t n) {
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float y, x;
	memcpy(&y, &yBits[i], 4);
	memcpy(&x, &xBits[i], 4);
	atanOut[i] = rscuda::CuAtan2f(y, x);
	asinOut[i] = rscuda::CuAsinf(y);
}

static void Check(cudaError_t e, const char* what) {
	if (e != cudaSuccess) {
		fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
		exit(2);
	}
}

// bit compare treating both-NaN as equal
static bool FloatEq(float a, float b) {
	uint32_t ua, ub;
	memcpy(&ua, &a, 4);
	memcpy(&ub, &b, 4);
	if (ua == ub) return true;
	bool aNan = (ua & 0x7F800000u) == 0x7F800000u && (ua & 0x7FFFFFu);
	bool bNan = (ub & 0x7F800000u) == 0x7F800000u && (ub & 0x7FFFFFu);
	return aNan && bNan;
}

struct FieldDef { const char* name; size_t offset; size_t count; }; // count floats
static const FieldDef FIELDS[] = {
	{ "dot3", offsetof(Output, dot3), 1 },
	{ "cross", offsetof(Output, cross), 3 },
	{ "quatMul", offsetof(Output, quatMul), 4 },
	{ "quatDot", offsetof(Output, quatDot), 1 },
	{ "quatNorm", offsetof(Output, quatNorm), 4 },
	{ "mat3Mul", offsetof(Output, mat3Mul), 9 },
	{ "getRot", offsetof(Output, getRot), 4 },
	{ "setRot", offsetof(Output, setRot), 9 },
	{ "inertia", offsetof(Output, inertia), 9 },
	{ "integBasis", offsetof(Output, integBasis), 9 },
	{ "integPos", offsetof(Output, integPos), 3 },
	{ "rsqrt", offsetof(Output, rsqrt), 1 },
	{ "norm3", offsetof(Output, norm3), 3 },
	{ "safeNorm3", offsetof(Output, safeNorm3), 3 },
	{ "axisAngle", offsetof(Output, axisAngle), 4 },
};

// Exhaustively scan +/-[0, domHi] comparing glibc sinf/cosf vs double-rounded sin/cos;
// write all mismatches as (inputBits, glibcBits) pairs sorted by input bits:
// header TFIX, sinN, cosN. Negative domain needed: steering quats use signed angles.
static void EmitTrigTable(const char* path, uint32_t domHi) {
	std::vector<TrigFixEntry> sinFix, cosFix;
	auto scan = [&](uint32_t signBit) {
		for (uint64_t u = 0; u <= domHi; u++) {
			uint32_t ub = signBit | (uint32_t)u;
			float x;
			memcpy(&x, &ub, 4);
			float hs = sinf(x), hc = cosf(x);
			float ds = (float)sin((double)x), dc = (float)cos((double)x);
			uint32_t hsb, hcb, dsb, dcb;
			memcpy(&hsb, &hs, 4); memcpy(&hcb, &hc, 4);
			memcpy(&dsb, &ds, 4); memcpy(&dcb, &dc, 4);
			if (hsb != dsb) sinFix.push_back({ ub, hsb });
			if (hcb != dcb) cosFix.push_back({ ub, hcb });
		}
	};
	scan(0);          // positive domain (already sorted ascending)
	scan(0x80000000u); // negative domain (bits sort above positives, stays sorted)
	FILE* f = fopen(path, "wb");
	if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
	uint32_t magic = 0x54464958u, sn = (uint32_t)sinFix.size(), cn = (uint32_t)cosFix.size();
	fwrite(&magic, 4, 1, f); fwrite(&sn, 4, 1, f); fwrite(&cn, 4, 1, f);
	fwrite(sinFix.data(), sizeof(TrigFixEntry), sn, f);
	fwrite(cosFix.data(), sizeof(TrigFixEntry), cn, f);
	fclose(f);
	printf("TRIG TABLE: wrote %s (%u sin fixes, %u cos fixes)\n", path, sn, cn);
}

int main(int argc, char** argv) {
	size_t samples = 4 * 1000 * 1000;
	bool trigExhaustive = false;
	const char* emitPath = nullptr;
	const char* tablePath = nullptr;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--samples") && i + 1 < argc) samples = strtoull(argv[++i], nullptr, 10);
		else if (!strcmp(argv[i], "--trig-exhaustive")) trigExhaustive = true;
		else if (!strcmp(argv[i], "--emit-trig-table") && i + 1 < argc) emitPath = argv[++i];
		else if (!strcmp(argv[i], "--trig-table") && i + 1 < argc) tablePath = argv[++i];
	}

	uint32_t domHiBits;
	{ float t = CU_ANGULAR_MOTION_THRESHOLD; memcpy(&domHiBits, &t, 4); }

	if (emitPath) {
		EmitTrigTable(emitPath, domHiBits);
		tablePath = emitPath;
	}
	if (tablePath) {
		if (!CuLoadTrigFix(tablePath)) {
			fprintf(stderr, "failed to load trig table %s\n", tablePath);
			return 2;
		}
		printf("TRIG TABLE: loaded %s\n", tablePath);
	}

	// EPA's expand() recursion needs more than the default 1KB device stack
	Check(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024), "stack limit");

	{ // RSQRTSS capture + emulation hypothesis check + upload
		static uint32_t rsqrtTab[2][4096];
		RsBuildRsqrtTable(rsqrtTab);
		uint64_t bad = RsVerifyRsqrtEmulation(rsqrtTab, 4000000);
		if (bad) {
			printf("MATHTEST rsqrt-emulation hypothesis FAILED on this CPU (%llu/4000000 mismatches)\n",
				(unsigned long long)bad);
			return 1;
		}
		if (!CuUploadRsqrtTable(rsqrtTab)) {
			fprintf(stderr, "rsqrt table upload failed\n");
			return 2;
		}
		printf("RSQRT TABLE: captured, hypothesis verified (4M samples), uploaded\n");
	}

	// ---------- op fuzz ----------
	std::vector<Input> inputs(samples);
	Gen gen(0x5EED5EEDull);
	for (size_t i = 0; i < samples; i++)
		FillInput(gen, inputs[i]);
	HostFillDerived(inputs.data(), samples);

	std::vector<Output> hostOut(samples), devOut(samples);
	HostCompute(inputs.data(), hostOut.data(), samples);

	Input* dIn; Output* dOut;
	Check(cudaMalloc(&dIn, samples * sizeof(Input)), "malloc in");
	Check(cudaMalloc(&dOut, samples * sizeof(Output)), "malloc out");
	Check(cudaMemcpy(dIn, inputs.data(), samples * sizeof(Input), cudaMemcpyHostToDevice), "upload");
	ComputeKernel<<<(unsigned)((samples + 255) / 256), 256>>>(dIn, dOut, samples);
	Check(cudaGetLastError(), "launch");
	Check(cudaDeviceSynchronize(), "sync");
	Check(cudaMemcpy(devOut.data(), dOut, samples * sizeof(Output), cudaMemcpyDeviceToHost), "download");
	cudaFree(dIn); cudaFree(dOut);

	int failures = 0;
	for (const FieldDef& f : FIELDS) {
		size_t bad = 0, firstBad = (size_t)-1;
		for (size_t i = 0; i < samples; i++) {
			const float* h = (const float*)((const char*)&hostOut[i] + f.offset);
			const float* d = (const float*)((const char*)&devOut[i] + f.offset);
			for (size_t k = 0; k < f.count; k++) {
				if (!FloatEq(h[k], d[k])) {
					if (bad == 0) firstBad = i;
					bad++;
					break;
				}
			}
		}
		if (bad) {
			failures++;
			printf("MATHTEST %-10s FAIL (%zu/%zu samples differ, first at %zu)\n", f.name, bad, samples, firstBad);
			const float* h = (const float*)((const char*)&hostOut[firstBad] + f.offset);
			const float* d = (const float*)((const char*)&devOut[firstBad] + f.offset);
			for (size_t k = 0; k < f.count; k++) {
				uint32_t hb, db;
				memcpy(&hb, &h[k], 4); memcpy(&db, &d[k], 4);
				if (hb != db)
					printf("    [%zu] host %.9g (0x%08X) dev %.9g (0x%08X)\n", k, h[k], hb, d[k], db);
			}
			const uint32_t* ib = (const uint32_t*)&inputs[firstBad];
			printf("    input qa=%08X %08X %08X %08X qb=%08X %08X %08X %08X w=%08X %08X %08X\n",
				ib[offsetof(Input, qa) / 4], ib[offsetof(Input, qa) / 4 + 1],
				ib[offsetof(Input, qa) / 4 + 2], ib[offsetof(Input, qa) / 4 + 3],
				ib[offsetof(Input, qb) / 4], ib[offsetof(Input, qb) / 4 + 1],
				ib[offsetof(Input, qb) / 4 + 2], ib[offsetof(Input, qb) / 4 + 3],
				ib[offsetof(Input, w) / 4], ib[offsetof(Input, w) / 4 + 1], ib[offsetof(Input, w) / 4 + 2]);
		} else {
			printf("MATHTEST %-10s PASS (%zu samples)\n", f.name, samples);
		}
	}

	// ---------- GJK/EPA fuzz (box vs triangle) ----------
	{
		size_t gn = samples / 4;
		std::vector<GjkCase> cases(gn);
		Gen gg(0x61D6EAull);
		for (size_t i = 0; i < gn; i++)
			FillGjkCase(gg, cases[i]);
		HostGjkDerive(cases.data(), gn);

		std::vector<GjkOut> hostOut(gn), devOut(gn);
		HostGjkCompute(cases.data(), hostOut.data(), gn);

		GjkCase* dCases;
		GjkOut* dOuts;
		Check(cudaMalloc(&dCases, gn * sizeof(GjkCase)), "gjk in");
		Check(cudaMalloc(&dOuts, gn * sizeof(GjkOut)), "gjk out");
		Check(cudaMemcpy(dCases, cases.data(), gn * sizeof(GjkCase), cudaMemcpyHostToDevice), "gjk up");
		GjkKernel<<<(unsigned)((gn + 127) / 128), 128>>>(dCases, dOuts, gn);
		Check(cudaGetLastError(), "gjk launch");
		Check(cudaMemcpy(devOut.data(), dOuts, gn * sizeof(GjkOut), cudaMemcpyDeviceToHost), "gjk dl");
		cudaFree(dCases);
		cudaFree(dOuts);

		size_t bad = 0, firstBad = (size_t)-1;
		for (size_t i = 0; i < gn; i++) {
			bool ok = hostOut[i].hasContact == devOut[i].hasContact;
			if (ok && hostOut[i].hasContact) {
				ok = FloatEq(hostOut[i].distance, devOut[i].distance) &&
					FloatEq(hostOut[i].normalOnB.x, devOut[i].normalOnB.x) &&
					FloatEq(hostOut[i].normalOnB.y, devOut[i].normalOnB.y) &&
					FloatEq(hostOut[i].normalOnB.z, devOut[i].normalOnB.z) &&
					FloatEq(hostOut[i].pointOnB.x, devOut[i].pointOnB.x) &&
					FloatEq(hostOut[i].pointOnB.y, devOut[i].pointOnB.y) &&
					FloatEq(hostOut[i].pointOnB.z, devOut[i].pointOnB.z);
			}
			if (!ok) {
				if (bad == 0) firstBad = i;
				bad++;
			}
		}
		if (bad) {
			failures++;
			uint32_t hd, dd;
			memcpy(&hd, &hostOut[firstBad].distance, 4);
			memcpy(&dd, &devOut[firstBad].distance, 4);
			printf("MATHTEST %-10s FAIL (%zu/%zu differ, first at %zu: cpuHas=%d gpuHas=%d "
				"cpuDist=%.9g(0x%08X) gpuDist=%.9g(0x%08X) cpuN=(%.9g,%.9g,%.9g) gpuN=(%.9g,%.9g,%.9g))\n",
				"gjkBoxTri", bad, gn, firstBad,
				hostOut[firstBad].hasContact, devOut[firstBad].hasContact,
				hostOut[firstBad].distance, hd, devOut[firstBad].distance, dd,
				hostOut[firstBad].normalOnB.x, hostOut[firstBad].normalOnB.y, hostOut[firstBad].normalOnB.z,
				devOut[firstBad].normalOnB.x, devOut[firstBad].normalOnB.y, devOut[firstBad].normalOnB.z);
		} else {
			printf("MATHTEST %-10s PASS (%zu samples)\n", "gjkBoxTri", gn);
		}
	}

	// ---------- sphere-vs-triangle fuzz ----------
	{
		size_t gn = samples / 4;
		std::vector<GjkCase> cases(gn);
		Gen gg(0x5713ull);
		for (size_t i = 0; i < gn; i++) {
			FillGjkCase(gg, cases[i]);
			cases[i].boxHalfExtentsCtor.x = 1.83f + gg.NextSigned() * 1.0f; // ball-ish radius
		}
		std::vector<GjkOut> hostOut(gn), devOut(gn);
		HostSphereTri(cases.data(), hostOut.data(), gn); // also stashes thresholds

		GjkCase* dCases;
		GjkOut* dOuts;
		Check(cudaMalloc(&dCases, gn * sizeof(GjkCase)), "st in");
		Check(cudaMalloc(&dOuts, gn * sizeof(GjkOut)), "st out");
		Check(cudaMemcpy(dCases, cases.data(), gn * sizeof(GjkCase), cudaMemcpyHostToDevice), "st up");
		SphereTriKernel<<<(unsigned)((gn + 127) / 128), 128>>>(dCases, dOuts, gn);
		Check(cudaGetLastError(), "st launch");
		Check(cudaMemcpy(devOut.data(), dOuts, gn * sizeof(GjkOut), cudaMemcpyDeviceToHost), "st dl");
		cudaFree(dCases);
		cudaFree(dOuts);

		size_t bad = 0, firstBad = (size_t)-1;
		for (size_t i = 0; i < gn; i++) {
			bool ok = hostOut[i].hasContact == devOut[i].hasContact;
			if (ok && hostOut[i].hasContact) {
				ok = FloatEq(hostOut[i].distance, devOut[i].distance) &&
					FloatEq(hostOut[i].normalOnB.x, devOut[i].normalOnB.x) &&
					FloatEq(hostOut[i].normalOnB.y, devOut[i].normalOnB.y) &&
					FloatEq(hostOut[i].normalOnB.z, devOut[i].normalOnB.z) &&
					FloatEq(hostOut[i].pointOnB.x, devOut[i].pointOnB.x) &&
					FloatEq(hostOut[i].pointOnB.y, devOut[i].pointOnB.y) &&
					FloatEq(hostOut[i].pointOnB.z, devOut[i].pointOnB.z);
			}
			if (!ok) {
				if (bad == 0) firstBad = i;
				bad++;
			}
		}
		if (bad) {
			failures++;
			printf("MATHTEST %-10s FAIL (%zu/%zu differ, first at %zu: cpuHas=%d gpuHas=%d cpuD=%.9g gpuD=%.9g)\n",
				"sphereTri", bad, gn, firstBad,
				hostOut[firstBad].hasContact, devOut[firstBad].hasContact,
				hostOut[firstBad].distance, devOut[firstBad].distance);
		} else {
			printf("MATHTEST %-10s PASS (%zu samples)\n", "sphereTri", gn);
		}
	}

	// ---------- trig domain check ----------
	// integrateTransform trig args lie in [0, ANGULAR_MOTION_THRESHOLD]
	const uint32_t domLo = 0;
	const uint32_t domHi = domHiBits;

	size_t trigN = trigExhaustive ? (size_t)(domHi - domLo + 1) * 2 : (size_t)16 * 1000 * 1000;
	const size_t CHUNK = 16 * 1000 * 1000;
	uint32_t* dBits; float *dSin, *dCos;
	Check(cudaMalloc(&dBits, CHUNK * 4), "malloc bits");
	Check(cudaMalloc(&dSin, CHUNK * 4), "malloc sin");
	Check(cudaMalloc(&dCos, CHUNK * 4), "malloc cos");
	std::vector<uint32_t> bits(CHUNK);
	std::vector<float> dSinH(CHUNK), dCosH(CHUNK), hSin(CHUNK), hCos(CHUNK);

	size_t sinBad = 0, cosBad = 0;
	uint64_t sinFirst = ~0ull, cosFirst = ~0ull;
	Gen tg(0x7717ull);
	for (size_t done = 0; done < trigN; done += CHUNK) {
		size_t n = (trigN - done < CHUNK) ? (trigN - done) : CHUNK;
		const uint64_t posCount = (uint64_t)(domHi - domLo + 1);
		for (size_t i = 0; i < n; i++) {
			if (trigExhaustive) {
				uint64_t k = done + i; // first all positives, then negatives
				bits[i] = (k < posCount) ? (uint32_t)(domLo + k)
				                         : (0x80000000u | (uint32_t)(domLo + (k - posCount)));
			} else {
				uint64_t r = tg.Next();
				bits[i] = ((r >> 60) & 1 ? 0x80000000u : 0) | (uint32_t)(domLo + r % posCount);
			}
		}
		Check(cudaMemcpy(dBits, bits.data(), n * 4, cudaMemcpyHostToDevice), "upload bits");
		TrigKernel<<<(unsigned)((n + 255) / 256), 256>>>(dBits, dSin, dCos, n);
		Check(cudaGetLastError(), "trig launch");
		Check(cudaMemcpy(dSinH.data(), dSin, n * 4, cudaMemcpyDeviceToHost), "dl sin");
		Check(cudaMemcpy(dCosH.data(), dCos, n * 4, cudaMemcpyDeviceToHost), "dl cos");
		HostTrig(bits.data(), hSin.data(), hCos.data(), n);
		for (size_t i = 0; i < n; i++) {
			if (!FloatEq(hSin[i], dSinH[i])) { if (!sinBad++) sinFirst = bits[i]; }
			if (!FloatEq(hCos[i], dCosH[i])) { if (!cosBad++) cosFirst = bits[i]; }
		}
	}
	cudaFree(dBits); cudaFree(dSin); cudaFree(dCos);

	auto report = [&](const char* name, size_t bad, uint64_t first) {
		if (bad) {
			failures++;
			float x; uint32_t fb = (uint32_t)first; memcpy(&x, &fb, 4);
			printf("MATHTEST %-10s FAIL (%zu/%zu differ, first input %.9g / 0x%08X)\n", name, bad, trigN, x, fb);
		} else {
			printf("MATHTEST %-10s PASS (%zu values%s)\n", name, trigN, trigExhaustive ? ", exhaustive domain" : "");
		}
	};
	report("sinf", sinBad, sinFirst);
	report("cosf", cosBad, cosFirst);

	// ---------- atan2f/asinf (glibc CORE-MATH device ports vs host glibc) ----------
	// Autoflip feeds basis components: dense [-1,1] coverage for asinf and [-1,1]^2
	// pairs for atan2f, plus full-exponent-range pairs for robustness.
	{
		size_t n2 = (size_t)16 * 1000 * 1000;
		uint32_t *dY, *dX;
		float *dAtan, *dAsin;
		Check(cudaMalloc(&dY, n2 * 4), "malloc y");
		Check(cudaMalloc(&dX, n2 * 4), "malloc x");
		Check(cudaMalloc(&dAtan, n2 * 4), "malloc atan");
		Check(cudaMalloc(&dAsin, n2 * 4), "malloc asin");
		std::vector<uint32_t> yB(n2), xB(n2);
		Gen ag(0xA7A2ull);
		for (size_t i = 0; i < n2; i++) {
			if (i % 2 == 0) {
				// unit-domain: random bits in [0, 1.0] with random signs
				yB[i] = (uint32_t)(ag.Next() % 0x3F800001ull) | ((ag.Next() & 1) << 31);
				xB[i] = (uint32_t)(ag.Next() % 0x3F800001ull) | ((ag.Next() & 1) << 31);
			} else {
				// arbitrary finite floats (clamp exponent below inf)
				yB[i] = (uint32_t)ag.Next();
				xB[i] = (uint32_t)ag.Next();
				if ((yB[i] & 0x7F800000u) == 0x7F800000u) yB[i] &= ~0x40000000u;
				if ((xB[i] & 0x7F800000u) == 0x7F800000u) xB[i] &= ~0x40000000u;
			}
		}
		Check(cudaMemcpy(dY, yB.data(), n2 * 4, cudaMemcpyHostToDevice), "up y");
		Check(cudaMemcpy(dX, xB.data(), n2 * 4, cudaMemcpyHostToDevice), "up x");
		Atan2AsinKernel<<<(unsigned)((n2 + 255) / 256), 256>>>(dY, dX, dAtan, dAsin, n2);
		Check(cudaGetLastError(), "atan2 launch");
		std::vector<float> gAtan(n2), gAsin(n2);
		Check(cudaMemcpy(gAtan.data(), dAtan, n2 * 4, cudaMemcpyDeviceToHost), "dl atan");
		Check(cudaMemcpy(gAsin.data(), dAsin, n2 * 4, cudaMemcpyDeviceToHost), "dl asin");
		size_t atanBad = 0, asinBad = 0;
		uint64_t atanFirst = ~0ull, asinFirst = ~0ull;
		for (size_t i = 0; i < n2; i++) {
			float y, x;
			memcpy(&y, &yB[i], 4);
			memcpy(&x, &xB[i], 4);
			if (!FloatEq(atan2f(y, x), gAtan[i])) { if (!atanBad++) atanFirst = ((uint64_t)yB[i] << 32) | xB[i]; }
			if (!FloatEq(asinf(y), gAsin[i])) { if (!asinBad++) asinFirst = yB[i]; }
		}
		cudaFree(dY); cudaFree(dX); cudaFree(dAtan); cudaFree(dAsin);
		if (atanBad) {
			failures++;
			printf("MATHTEST %-10s FAIL (%zu/%zu differ, first y/x bits 0x%016llX)\n",
				"atan2f", atanBad, n2, (unsigned long long)atanFirst);
		} else
			printf("MATHTEST %-10s PASS (%zu values)\n", "atan2f", n2);
		if (asinBad) {
			failures++;
			printf("MATHTEST %-10s FAIL (%zu/%zu differ, first input bits 0x%08X)\n",
				"asinf", asinBad, n2, (uint32_t)asinFirst);
		} else
			printf("MATHTEST %-10s PASS (%zu values)\n", "asinf", n2);
	}

	if (failures) {
		printf("MATHTEST: %d op(s) FAILED\n", failures);
		return 1;
	}
	printf("MATHTEST: ALL PASS\n");
	return 0;
}
