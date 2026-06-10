// RocketSim WORKSPACE benchmark + validation gate driver (rocketsim_bench).
//
// Forked 2026-06-10 from tools/bench/main.cpp (which stays frozen as the reference bench
// source). This copy links the RocketSim-CUDA work tree and carries workspace-only
// features (--profile, --gpu-step). Gate behavior in CPU mode must remain identical to
// the reference bench: same args -> same STATE_HASH.
//
// Determinism notes (2026-06-10, verified against RocketSim v2.2.1 source):
//  - Arena::_cars is std::unordered_set<Car*> (pointer-keyed): iteration order varies across
//    runs under ASLR. This bench therefore keeps its own insertion-ordered car list for input
//    assignment and state hashing, and never relies on arena container order.
//  - Car demolition respawn calls Math::RandInt(.., seed=-1) which uses a wall-clock-seeded
//    engine (Math.cpp GetRandEngine). Gates run with DemoMode::DISABLED so no time-seeded
//    randomness can enter the simulation.
//  - ResetToRandomKickoff uses std::shuffle + std::default_random_engine (both
//    implementation-defined) and arena container order; not used. Spawns are set explicitly.
//  - All bench-side randomness comes from splitmix64 (fully defined here), and generated
//    floats are exact power-of-two quantizations, so the input schedule is identical on
//    every platform/compiler.

#include "RocketSim.h"
#include "CollisionMeshFile/CollisionMeshFile.h"
#include "DataStream/DataStreamIn.h"
#include "DataStream/DataStreamOut.h"
#include "Profiling.h"

#ifdef RS_CUDA
#include "Cuda/CudaStep.h"
#include "Cuda/PhaseCapture.h"
#endif

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace RocketSim;

// Known-good soccar mesh hashes, copied from MeshHashSet in src/RocketSim.cpp (v2.2.1)
static const uint32_t KNOWN_SOCCAR_MESH_HASHES[16] = {
	0xA160BAF9, 0x2811EEE8, 0xB81AC8B9, 0x760358D3,
	0x73AE4940, 0x918F4A4E, 0x1F8EE550, 0x255BA8C1,
	0x14B84668, 0xEC759EBF, 0x94FB0D5C, 0xDEA07102,
	0xBD4FBEA8, 0x39A47F63, 0x3D79D25D, 0xD84C7A68
};

struct Rng { // splitmix64
	uint64_t s;
	explicit Rng(uint64_t seed) : s(seed) {}
	uint64_t Next() {
		s += 0x9E3779B97F4A7C15ull;
		uint64_t z = s;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
		return z ^ (z >> 31);
	}
	// Exact [-1, 1] in 1/1024 steps
	float NextSigned() { return (float)((int64_t)(Next() % 2049) - 1024) / 1024.0f; }
	bool Chance(uint32_t percent) { return (Next() % 100) < percent; }
};

static uint64_t MixSeed(uint64_t a, uint64_t b) {
	Rng r(a ^ (b * 0x9E3779B97F4A7C15ull));
	return r.Next();
}

static uint64_t Fnv1a64(const void* data, size_t len, uint64_t h) {
	const unsigned char* p = (const unsigned char*)data;
	for (size_t i = 0; i < len; i++) {
		h ^= p[i];
		h *= 0x100000001B3ull;
	}
	return h;
}
static constexpr uint64_t FNV_BASIS = 0xCBF29CE484222325ull;

struct Args {
	std::string meshDir;
	int arenas = 1;
	int ticks = 0;
	int cars = 2;
	uint64_t seed = 1337;
	bool hasExpectedHash = false;
	uint64_t expectedHash = 0;
	bool dumpArenaHashes = false;   // diagnostic: print per-arena state hash
	std::string dumpStateDir;       // diagnostic: write per-arena serialized state blobs
	bool profile = false;           // print per-phase timing breakdown after the run
	bool gpuStep = false;           // GPU produces the simulation result (monolithic kernel)
	int gpuCompare = 0;             // lockstep-validate ported GPU phases for N ticks
	std::string trigTable = "glibc_trig_fix.bin"; // libm correction table (see CudaMath.cuh)
};

static void PrintUsage(const char* argv0) {
	fprintf(stderr,
		"Usage: %s --mesh-dir <collision_meshes dir> --ticks N [--arenas N] [--cars N] "
		"[--seed N] [--expected-state-hash H] [--profile] [--gpu-step]\n", argv0);
}

static bool ParseArgs(int argc, char** argv, Args& out) {
	for (int i = 1; i < argc; i++) {
		auto need = [&](const char* name) -> const char* {
			if (i + 1 >= argc) {
				fprintf(stderr, "Missing value for %s\n", name);
				return nullptr;
			}
			return argv[++i];
		};
		if (!strcmp(argv[i], "--mesh-dir")) {
			const char* v = need("--mesh-dir"); if (!v) return false;
			out.meshDir = v;
		} else if (!strcmp(argv[i], "--arenas")) {
			const char* v = need("--arenas"); if (!v) return false;
			out.arenas = atoi(v);
		} else if (!strcmp(argv[i], "--ticks")) {
			const char* v = need("--ticks"); if (!v) return false;
			out.ticks = atoi(v);
		} else if (!strcmp(argv[i], "--cars")) {
			const char* v = need("--cars"); if (!v) return false;
			out.cars = atoi(v);
		} else if (!strcmp(argv[i], "--seed")) {
			const char* v = need("--seed"); if (!v) return false;
			out.seed = strtoull(v, nullptr, 10);
		} else if (!strcmp(argv[i], "--expected-state-hash")) {
			const char* v = need("--expected-state-hash"); if (!v) return false;
			out.expectedHash = strtoull(v, nullptr, 10);
			out.hasExpectedHash = true;
		} else if (!strcmp(argv[i], "--profile")) {
			out.profile = true;
		} else if (!strcmp(argv[i], "--gpu-step")) {
			out.gpuStep = true;
		} else if (!strcmp(argv[i], "--gpu-compare")) {
			const char* v = need("--gpu-compare"); if (!v) return false;
			out.gpuCompare = atoi(v);
		} else if (!strcmp(argv[i], "--trig-table")) {
			const char* v = need("--trig-table"); if (!v) return false;
			out.trigTable = v;
		} else if (!strcmp(argv[i], "--dump-arena-hashes")) {
			out.dumpArenaHashes = true;
		} else if (!strcmp(argv[i], "--dump-state-dir")) {
			const char* v = need("--dump-state-dir"); if (!v) return false;
			out.dumpStateDir = v;
		} else {
			fprintf(stderr, "Unknown argument: %s\n", argv[i]);
			return false;
		}
	}
	if (out.meshDir.empty() || out.ticks <= 0 || out.arenas <= 0 || out.cars <= 0) {
		fprintf(stderr, "Required: --mesh-dir, --ticks > 0, --arenas > 0, --cars > 0\n");
		return false;
	}
	return true;
}

// Verify the soccar meshes byte-for-byte against the known v2.2.1 hash set before Init,
// so a wrong/corrupt asset can never silently shift the baseline.
static bool VerifyMeshes(const std::string& meshDir) {
	std::filesystem::path soccarDir = std::filesystem::path(meshDir) / "soccar";
	if (!std::filesystem::exists(soccarDir)) {
		fprintf(stderr, "MESHES: FAIL (missing directory %s)\n", soccarDir.string().c_str());
		return false;
	}
	std::set<std::filesystem::path> files; // sorted for a stable report
	for (auto& entry : std::filesystem::directory_iterator(soccarDir))
		if (entry.path().extension() == ".cmf")
			files.insert(entry.path());

	std::set<uint32_t> known(KNOWN_SOCCAR_MESH_HASHES, KNOWN_SOCCAR_MESH_HASHES + 16);
	std::set<uint32_t> found;
	for (auto& f : files) {
		DataStreamIn stream(f, false);
		CollisionMeshFile meshFile = {};
		meshFile.ReadFromStream(stream, true, f.string());
		if (!known.count(meshFile.hash)) {
			fprintf(stderr, "MESHES: FAIL (%s has unknown hash 0x%X)\n", f.string().c_str(), meshFile.hash);
			return false;
		}
		found.insert(meshFile.hash);
	}
	if (found.size() != 16) {
		fprintf(stderr, "MESHES: FAIL (have %zu/16 distinct known soccar meshes in %s)\n",
			found.size(), soccarDir.string().c_str());
		return false;
	}
	printf("MESHES: OK (16/16 soccar)\n");
	return true;
}

struct CarSlot {
	Car* car;
	Rng rng;
	int holdTicks = 0;
};

static CarControls RollControls(Rng& rng) {
	CarControls c = {};
	c.throttle = rng.NextSigned();
	c.steer = rng.NextSigned();
	c.pitch = rng.NextSigned();
	c.yaw = rng.NextSigned();
	c.roll = rng.NextSigned();
	c.jump = rng.Chance(10);
	c.boost = rng.Chance(30);
	c.handbrake = rng.Chance(15);
	return c;
}

static void UpdateControls(CarSlot& slot) {
	if (slot.holdTicks == 0) {
		slot.car->controls = RollControls(slot.rng);
		slot.holdTicks = 2 + (int)(slot.rng.Next() % 59); // change every 2-60 ticks
	}
	slot.holdTicks--;
}

static constexpr float HALF_PI = 1.5707963267948966f;

// Identical arena/car setup for CPU and GPU paths - this is state initialization, not physics.
static Arena* SetupArena(const Args& args, int arenaIdx, std::vector<CarSlot>& slots) {
	Arena* arena = Arena::Create(GameMode::SOCCAR);

	// Demos disabled: upstream demo respawn is wall-clock-seeded (see header comment)
	MutatorConfig mc = arena->GetMutatorConfig();
	mc.demoMode = DemoMode::DISABLED;
	arena->SetMutatorConfig(mc);

	Rng spawnRng(MixSeed(args.seed, (uint64_t)arenaIdx));
	slots.clear();
	slots.reserve(args.cars);
	for (int i = 0; i < args.cars; i++) {
		Team team = (i % 2 == 0) ? Team::BLUE : Team::ORANGE;
		Car* car = arena->AddCar(team);

		int pairIdx = i / 2;
		float ySign = (team == Team::BLUE) ? -1.0f : 1.0f;
		CarState cs = {};
		cs.pos = Vec(
			(float)((pairIdx % 5) - 2) * 800.0f + spawnRng.NextSigned() * 256.0f,
			ySign * (2000.0f + (float)(pairIdx / 5) * 600.0f + spawnRng.NextSigned() * 256.0f),
			RLConst::CAR_SPAWN_REST_Z);
		cs.rotMat = Angle(ySign * -HALF_PI + spawnRng.NextSigned() * 0.25f, 0, 0).ToRotMat();
		car->SetState(cs);

		slots.push_back(CarSlot{ car, Rng(MixSeed(MixSeed(args.seed, (uint64_t)arenaIdx), 0xCA0ull + (uint64_t)i)) });
	}
	// Ball stays at its deterministic default rest state (center, BALL_REST_Z)
	return arena;
}

// Hash final state in bench-defined order (never arena container order)
static uint64_t HashArena(Arena* arena, std::vector<CarSlot>& slots, DataStreamOut& out) {
	out.Write<uint64_t>(arena->tickCount);
	for (CarSlot& slot : slots) {
		out.Write<uint32_t>(slot.car->id);
		out.Write<uint8_t>((uint8_t)slot.car->team);
		CarState cs = slot.car->GetState();
		cs.Serialize(out);
	}
	BallState bs = arena->ball->GetState();
	// 2026-06-10: dsInfo.lastDamageTick has no initializer upstream ("Only valid if
	// hasDamaged") and is never written outside dropshot, but BallState::Serialize
	// writes it regardless - without this mask the hash ingests uninitialized heap
	// bytes (observed: stable-gate hash flipped between runs, 4-byte diff at the
	// lastDamageTick offset of the ball block).
	if (!bs.dsInfo.hasDamaged)
		bs.dsInfo.lastDamageTick = 0;
	bs.Serialize(out);
	for (BoostPad* pad : arena->GetBoostPads()) {
		BoostPadState ps = pad->GetState();
		ps.Serialize(out);
	}
	return Fnv1a64(out.data.data(), out.data.size(), FNV_BASIS);
}

static void EmitArenaDiagnostics(const Args& args, int arenaIdx, uint64_t arenaHash, const DataStreamOut& out) {
	if (args.dumpArenaHashes)
		printf("ARENA %d HASH: %" PRIu64 "\n", arenaIdx, arenaHash);
	if (!args.dumpStateDir.empty()) {
		std::filesystem::create_directories(args.dumpStateDir);
		auto blobPath = std::filesystem::path(args.dumpStateDir) / ("arena_" + std::to_string(arenaIdx) + ".bin");
		std::ofstream blob(blobPath, std::ios::binary);
		blob.write((const char*)out.data.data(), out.data.size());
	}
}

// ============================== CPU PATH ==============================

static uint64_t RunCpu(const Args& args, double& elapsedS) {
	uint64_t combinedHash = FNV_BASIS;
	for (int arenaIdx = 0; arenaIdx < args.arenas; arenaIdx++) {
		std::vector<CarSlot> slots;
		Arena* arena = SetupArena(args, arenaIdx, slots);

		auto t0 = std::chrono::steady_clock::now();
		for (int tick = 0; tick < args.ticks; tick++) {
			for (CarSlot& slot : slots)
				UpdateControls(slot);
			arena->Step(1);
		}
		auto t1 = std::chrono::steady_clock::now();
		elapsedS += std::chrono::duration<double>(t1 - t0).count();

		DataStreamOut out;
		uint64_t arenaHash = HashArena(arena, slots, out);
		combinedHash = Fnv1a64(&arenaHash, sizeof(arenaHash), combinedHash);
		EmitArenaDiagnostics(args, arenaIdx, arenaHash, out);

		delete arena;
	}
	return combinedHash;
}

// ============================== GPU PATH ==============================

#ifdef RS_CUDA

static rscuda::GpuVec3 ToGpu(const Vec& v) { return { v.x, v.y, v.z }; }
static Vec FromGpu(const rscuda::GpuVec3& v) { return Vec(v.x, v.y, v.z); }

static rscuda::GpuVec3 ToGpuB(const btVector3& v) { return { v.x(), v.y(), v.z() }; }
static btVector3 FromGpuB(const rscuda::GpuVec3& v) { return btVector3(v.x, v.y, v.z); }

// GpuMat3 rows = btMatrix3x3 rows (row-major, BULLET basis - not RocketSim RotMat)
static rscuda::GpuMat3 ToGpuB(const btMatrix3x3& m) {
	return { { ToGpuB(m[0]), ToGpuB(m[1]), ToGpuB(m[2]) } };
}
static btMatrix3x3 FromGpuB(const rscuda::GpuMat3& m) {
	return btMatrix3x3(
		m.row[0].x, m.row[0].y, m.row[0].z,
		m.row[1].x, m.row[1].y, m.row[1].z,
		m.row[2].x, m.row[2].y, m.row[2].z);
}

// Full bullet-unit dynamic state of a rigid body -> GPU mirror
static void SnapshotBody(const btRigidBody& rb, rscuda::GpuBodyState& b) {
	const btTransform& t = rb.getWorldTransform();
	b.basis = ToGpuB(t.getBasis());
	b.pos = ToGpuB(t.getOrigin());
	b.linVel = ToGpuB(rb.getLinearVelocity());
	b.angVel = ToGpuB(rb.getAngularVelocity());
	b.totalForce = ToGpuB(rb.getTotalForce());
	b.totalTorque = ToGpuB(rb.getTotalTorque());
	b.invInertiaWorld = ToGpuB(rb.getInvInertiaTensorWorld());
	b.activationState = rb.getActivationState();
	b.deactivationTime = rb.getDeactivationTime();
}

static void WritebackBody(btRigidBody& rb, const rscuda::GpuBodyState& b) {
	btTransform t;
	t.setBasis(FromGpuB(b.basis));
	t.setOrigin(FromGpuB(b.pos));
	rb.setWorldTransform(t);
	rb.setLinearVelocity(FromGpuB(b.linVel));
	rb.setAngularVelocity(FromGpuB(b.angVel));
	rb.updateInertiaTensor(); // same result as GPU's carried tensor when basis matches
	rb.forceActivationState(b.activationState);
	rb.setDeactivationTime(b.deactivationTime);
}

static rscuda::GpuBodyConsts MakeBodyConsts(const btRigidBody& rb, float dt) {
	rscuda::GpuBodyConsts c = {};
	c.invMass = rb.getInvMass();
	c.invInertiaLocal = ToGpuB(rb.getInvInertiaDiagLocal());
	// CPU recomputes btPow(1-damping, dt) every applyDamping call with constant args;
	// computing it once here with the same expression gives the identical float
	c.linDampFactor = powf(1.0f - rb.getLinearDamping(), dt);
	c.angDampFactor = powf(1.0f - rb.getAngularDamping(), dt);
	c.linSleepThreshSq = rb.getLinearSleepingThreshold() * rb.getLinearSleepingThreshold();
	c.angSleepThreshSq = rb.getAngularSleepingThreshold() * rb.getAngularSleepingThreshold();
	c.noRot = rb.m_noRot;
	return c;
}

static rscuda::GpuCarControls ToGpu(const CarControls& c) {
	return { c.throttle, c.steer, c.pitch, c.yaw, c.roll,
		(uint8_t)c.jump, (uint8_t)c.boost, (uint8_t)c.handbrake };
}
static CarControls FromGpu(const rscuda::GpuCarControls& g) {
	CarControls c = {};
	c.throttle = g.throttle; c.steer = g.steer;
	c.pitch = g.pitch; c.yaw = g.yaw; c.roll = g.roll;
	c.jump = g.jump; c.boost = g.boost; c.handbrake = g.handbrake;
	return c;
}

static void SnapshotCar(Car* car, rscuda::GpuBodyState& body, rscuda::GpuCarGameState& g) {
	CarState cs = car->GetState();
	SnapshotBody(car->_rigidBody, body);

	g = {};
	g.tickCountSinceUpdate = cs.tickCountSinceUpdate;
	g.isOnGround = cs.isOnGround;
	for (int w = 0; w < 4; w++) g.wheelsWithContact[w] = cs.wheelsWithContact[w];
	g.hasJumped = cs.hasJumped; g.hasDoubleJumped = cs.hasDoubleJumped; g.hasFlipped = cs.hasFlipped;
	g.flipRelTorque = ToGpu(cs.flipRelTorque);
	g.jumpTime = cs.jumpTime; g.flipTime = cs.flipTime;
	g.isFlipping = cs.isFlipping; g.isJumping = cs.isJumping;
	g.airTime = cs.airTime; g.airTimeSinceJump = cs.airTimeSinceJump;
	g.boost = cs.boost; g.timeSinceBoosted = cs.timeSinceBoosted;
	g.isBoosting = cs.isBoosting; g.boostingTime = cs.boostingTime;
	g.isSupersonic = cs.isSupersonic; g.supersonicTime = cs.supersonicTime;
	g.handbrakeVal = cs.handbrakeVal;
	g.isAutoFlipping = cs.isAutoFlipping;
	g.autoFlipTimer = cs.autoFlipTimer; g.autoFlipTorqueScale = cs.autoFlipTorqueScale;
	g.worldContact_hasContact = cs.worldContact.hasContact;
	g.worldContact_normal = ToGpu(cs.worldContact.contactNormal);
	g.carContact_otherCarID = cs.carContact.otherCarID;
	g.carContact_cooldownTimer = cs.carContact.cooldownTimer;
	g.isDemoed = cs.isDemoed; g.demoRespawnTimer = cs.demoRespawnTimer;
	g.ballHitInfo.isValid = cs.ballHitInfo.isValid;
	g.ballHitInfo.relativePosOnBall = ToGpu(cs.ballHitInfo.relativePosOnBall);
	g.ballHitInfo.ballPos = ToGpu(cs.ballHitInfo.ballPos);
	g.ballHitInfo.extraHitVel = ToGpu(cs.ballHitInfo.extraHitVel);
	g.ballHitInfo.tickCountWhenHit = cs.ballHitInfo.tickCountWhenHit;
	g.ballHitInfo.tickCountWhenExtraImpulseApplied = cs.ballHitInfo.tickCountWhenExtraImpulseApplied;
	g.lastControls = ToGpu(cs.lastControls);
	g.velocityImpulseCache = ToGpu(car->_velocityImpulseCache);
	g.carID = car->id;
	g.team = (uint8_t)car->team;
}

static void WritebackCar(Car* car, const rscuda::GpuBodyState& body, const rscuda::GpuCarGameState& g) {
	WritebackBody(car->_rigidBody, body);
	CarState& cs = car->_internalState;
	// Car::GetState refreshes pos/vel/angVel from the body but NOT rotMat (that happens
	// in _PostTickUpdate, which doesn't run in GPU mode) - sync it here
	cs.rotMat = (RotMat)car->_rigidBody.getWorldTransform().getBasis();
	cs.isOnGround = g.isOnGround;
	for (int w = 0; w < 4; w++) cs.wheelsWithContact[w] = g.wheelsWithContact[w];
	cs.hasJumped = g.hasJumped; cs.hasDoubleJumped = g.hasDoubleJumped; cs.hasFlipped = g.hasFlipped;
	cs.flipRelTorque = FromGpu(g.flipRelTorque);
	cs.jumpTime = g.jumpTime; cs.flipTime = g.flipTime;
	cs.isFlipping = g.isFlipping; cs.isJumping = g.isJumping;
	cs.airTime = g.airTime; cs.airTimeSinceJump = g.airTimeSinceJump;
	cs.boost = g.boost; cs.timeSinceBoosted = g.timeSinceBoosted;
	cs.isBoosting = g.isBoosting; cs.boostingTime = g.boostingTime;
	cs.isSupersonic = g.isSupersonic; cs.supersonicTime = g.supersonicTime;
	cs.handbrakeVal = g.handbrakeVal;
	cs.isAutoFlipping = g.isAutoFlipping;
	cs.autoFlipTimer = g.autoFlipTimer; cs.autoFlipTorqueScale = g.autoFlipTorqueScale;
	cs.worldContact.hasContact = g.worldContact_hasContact;
	cs.worldContact.contactNormal = FromGpu(g.worldContact_normal);
	cs.carContact.otherCarID = g.carContact_otherCarID;
	cs.carContact.cooldownTimer = g.carContact_cooldownTimer;
	cs.isDemoed = g.isDemoed; cs.demoRespawnTimer = g.demoRespawnTimer;
	cs.ballHitInfo.isValid = g.ballHitInfo.isValid;
	cs.ballHitInfo.relativePosOnBall = FromGpu(g.ballHitInfo.relativePosOnBall);
	cs.ballHitInfo.ballPos = FromGpu(g.ballHitInfo.ballPos);
	cs.ballHitInfo.extraHitVel = FromGpu(g.ballHitInfo.extraHitVel);
	cs.ballHitInfo.tickCountWhenHit = g.ballHitInfo.tickCountWhenHit;
	cs.ballHitInfo.tickCountWhenExtraImpulseApplied = g.ballHitInfo.tickCountWhenExtraImpulseApplied;
	cs.lastControls = FromGpu(g.lastControls);
	cs.tickCountSinceUpdate = g.tickCountSinceUpdate;
	car->_velocityImpulseCache = FromGpu(g.velocityImpulseCache);
}

static constexpr float TICK_TIME = 1.0f / 120.0f;

static void RequireTrigTable(const Args& args) {
	if (!rscuda::LoadTrigFix(args.trigTable.c_str())) {
		fprintf(stderr, "GPU: FAIL - missing/corrupt trig correction table '%s'.\n"
			"Generate it once on this machine with:\n"
			"  ./build/cuda_math_test --samples 1000 --emit-trig-table %s\n",
			args.trigTable.c_str(), args.trigTable.c_str());
		exit(2);
	}
}

static void UploadConsts(rscuda::GpuBatch* batch, Arena* arena, std::vector<CarSlot>& slots) {
	std::vector<rscuda::GpuBodyConsts> consts(1 + slots.size());
	consts[0] = MakeBodyConsts(arena->ball->_rigidBody, TICK_TIME);
	for (size_t c = 0; c < slots.size(); c++)
		consts[1 + c] = MakeBodyConsts(slots[c].car->_rigidBody, TICK_TIME);
	rscuda::UploadBodyConsts(batch, consts.data());
}

// ---- lockstep phase validation (--gpu-compare N) ----
// Per tick: run the CPU step with capture enabled, then for each ported phase upload
// the CPU pre-phase state, run JUST that phase on the GPU, and bit-compare the result
// against the CPU post-phase capture. Localizes any divergence to (tick, phase, body,
// field, bits).

static rscuda::GpuBodyState SnapToGpu(const rscap::BodySnap& s) {
	rscuda::GpuBodyState b;
	for (int i = 0; i < 3; i++) {
		b.basis.row[i] = { s.basis[i * 3], s.basis[i * 3 + 1], s.basis[i * 3 + 2] };
		b.invInertiaWorld.row[i] = { s.invInertiaWorld[i * 3], s.invInertiaWorld[i * 3 + 1], s.invInertiaWorld[i * 3 + 2] };
	}
	b.pos = { s.pos[0], s.pos[1], s.pos[2] };
	b.linVel = { s.linVel[0], s.linVel[1], s.linVel[2] };
	b.angVel = { s.angVel[0], s.angVel[1], s.angVel[2] };
	b.totalForce = { s.totalForce[0], s.totalForce[1], s.totalForce[2] };
	b.totalTorque = { s.totalTorque[0], s.totalTorque[1], s.totalTorque[2] };
	b.activationState = s.activationState;
	b.deactivationTime = s.deactivationTime;
	return b;
}

// returns index of first differing float (-1 if equal); NaN==NaN
static int BodyStateDiff(const rscuda::GpuBodyState& a, const rscuda::GpuBodyState& b) {
	const uint32_t* ua = (const uint32_t*)&a;
	const uint32_t* ub = (const uint32_t*)&b;
	for (size_t i = 0; i < sizeof(a) / 4; i++)
		if (ua[i] != ub[i])
			return (int)i;
	return -1;
}

static int RunGpuCompare(const Args& args) {
	if (!rscuda::Available()) {
		fprintf(stderr, "GPU_COMPARE: FAIL (no usable CUDA device)\n");
		return 2;
	}
	RequireTrigTable(args);

	std::vector<CarSlot> slots;
	Arena* arena = SetupArena(args, 0, slots);
	const int numPads = (int)arena->GetBoostPads().size();
	const int numBodies = 1 + args.cars;

	rscuda::GpuBatch* batch = rscuda::CreateBatch(1, args.cars, numPads);
	UploadConsts(batch, arena, slots);

	// body index mapping: our order is [ball, car0, car1...]
	std::vector<const void*> rbOrder(numBodies);
	rbOrder[0] = &arena->ball->_rigidBody;
	for (int c = 0; c < args.cars; c++)
		rbOrder[1 + c] = &slots[c].car->_rigidBody;
	auto mapSnaps = [&](const std::vector<rscap::BodySnap>& snaps, std::vector<rscuda::GpuBodyState>& out) {
		for (const rscap::BodySnap& s : snaps) {
			for (int i = 0; i < numBodies; i++) {
				if (rbOrder[i] == s.rb) {
					out[i] = SnapToGpu(s);
					break;
				}
			}
		}
	};

	// (uploaded once; cars/pads device buffers are untouched by these phases)
	std::vector<rscuda::GpuCarGameState> carsDummy(args.cars);
	std::vector<rscuda::GpuPadState> padsDummy(numPads);
	std::vector<rscuda::GpuBodyState> preState(numBodies), refState(numBodies), gotState(numBodies);

	struct Segment {
		rscap::Point pre, post;
		int phase;
		const char* name;
	};
	const Segment SEGMENTS[] = {
		{ rscap::PRE_BULLET, rscap::AFTER_PREDICT, rscuda::GPU_PHASE_PREDICT, "predict" },
		{ rscap::AFTER_SOLVE, rscap::AFTER_INTEGRATE, rscuda::GPU_PHASE_INTEGRATE, "integrate" },
		{ rscap::AFTER_INTEGRATE, rscap::AFTER_ACTIVATION, rscuda::GPU_PHASE_ACTIVATION, "activation" },
	};

	for (int tick = 0; tick < args.gpuCompare; tick++) {
		for (CarSlot& slot : slots)
			UpdateControls(slot);
		rscap::enabled = true;
		arena->Step(1);
		rscap::enabled = false;

		for (const Segment& seg : SEGMENTS) {
			mapSnaps(rscap::points[seg.pre], preState);
			mapSnaps(rscap::points[seg.post], refState);
			rscuda::UploadArenaState(batch, 0, preState.data(), carsDummy.data(), padsDummy.data());
			rscuda::RunPhaseSegment(batch, seg.phase, seg.phase);
			rscuda::DownloadArenaState(batch, 0, gotState.data(), carsDummy.data(), padsDummy.data());

			for (int bIdx = 0; bIdx < numBodies; bIdx++) {
				int d = BodyStateDiff(refState[bIdx], gotState[bIdx]);
				if (d >= 0) {
					const uint32_t* cr = (const uint32_t*)&refState[bIdx];
					const uint32_t* gr = (const uint32_t*)&gotState[bIdx];
					printf("GPU_COMPARE: FAIL tick %d phase %s body %d (%s) float[%d]: cpu 0x%08X gpu 0x%08X\n",
						tick, seg.name, bIdx, bIdx == 0 ? "ball" : "car", d, cr[d], gr[d]);
					return 1;
				}
			}
		}
	}

	printf("GPU_COMPARE: PASS (%d ticks, phases: predict, integrate, activation; %d bodies bit-exact)\n",
		args.gpuCompare, numBodies);
	rscuda::DestroyBatch(batch);
	delete arena;
	return 0;
}

static uint64_t RunGpu(const Args& args, double& elapsedS) {
	if (!rscuda::Available()) {
		fprintf(stderr, "GPU_STEP: FAIL (no usable CUDA device)\n");
		exit(2);
	}

	// Set up all arenas (CPU-side state init only - no physics is stepped on CPU)
	std::vector<Arena*> arenas(args.arenas);
	std::vector<std::vector<CarSlot>> slotsPerArena(args.arenas);
	for (int a = 0; a < args.arenas; a++)
		arenas[a] = SetupArena(args, a, slotsPerArena[a]);

	const int numPads = (int)arenas[0]->GetBoostPads().size();
	rscuda::GpuBatch* batch = rscuda::CreateBatch(args.arenas, args.cars, numPads);
	RequireTrigTable(args);
	UploadConsts(batch, arenas[0], slotsPerArena[0]);

	// Precompute the full control schedule (identical Rng sequence to the CPU path)
	std::vector<rscuda::GpuCarControls> schedule((size_t)args.arenas * args.cars * args.ticks);
	for (int a = 0; a < args.arenas; a++) {
		for (int c = 0; c < args.cars; c++) {
			Rng rng = slotsPerArena[a][c].rng;
			int hold = 0;
			CarControls cur = {};
			for (int t = 0; t < args.ticks; t++) {
				if (hold == 0) {
					cur = RollControls(rng);
					hold = 2 + (int)(rng.Next() % 59);
				}
				hold--;
				schedule[((size_t)a * args.cars + c) * args.ticks + t] = ToGpu(cur);
			}
		}
	}

	auto t0 = std::chrono::steady_clock::now();

	// Upload initial state
	std::vector<rscuda::GpuBodyState> bodies(1 + args.cars);
	std::vector<rscuda::GpuCarGameState> cars(args.cars);
	std::vector<rscuda::GpuPadState> pads(numPads);
	for (int a = 0; a < args.arenas; a++) {
		Ball* ball = arenas[a]->ball;
		SnapshotBody(ball->_rigidBody, bodies[0]);
		for (int c = 0; c < args.cars; c++)
			SnapshotCar(slotsPerArena[a][c].car, bodies[1 + c], cars[c]);
		auto& padList = arenas[a]->GetBoostPads();
		for (int p = 0; p < numPads; p++) {
			BoostPadState ps = padList[p]->GetState();
			pads[p].isActive = ps.isActive;
			pads[p].cooldown = ps.cooldown;
			pads[p].prevLockedCarID = ps.prevLockedCarID;
			pads[p].lockedCarIdx = -1;
			for (int c = 0; c < args.cars; c++)
				if (ps.curLockedCar == slotsPerArena[a][c].car)
					pads[p].lockedCarIdx = c;
		}
		rscuda::UploadArenaState(batch, a, bodies.data(), cars.data(), pads.data());
	}
	rscuda::UploadControlSchedule(batch, schedule.data(), args.ticks);

	// THE step: one kernel launch runs every tick of every arena
	rscuda::StepTicks(batch, args.ticks);

	// Download final state and write it back into the arenas (for hashing/inspection)
	uint64_t combinedHash = FNV_BASIS;
	for (int a = 0; a < args.arenas; a++) {
		rscuda::DownloadArenaState(batch, a, bodies.data(), cars.data(), pads.data());
		WritebackBody(arenas[a]->ball->_rigidBody, bodies[0]);
		for (int c = 0; c < args.cars; c++)
			WritebackCar(slotsPerArena[a][c].car, bodies[1 + c], cars[c]);
		auto& padList = arenas[a]->GetBoostPads();
		for (int p = 0; p < numPads; p++) {
			BoostPadState ps = padList[p]->GetState();
			ps.isActive = pads[p].isActive;
			ps.cooldown = pads[p].cooldown;
			ps.prevLockedCarID = pads[p].prevLockedCarID;
			ps.curLockedCar = (pads[p].lockedCarIdx >= 0) ? slotsPerArena[a][pads[p].lockedCarIdx].car : nullptr;
			padList[p]->SetState(ps);
		}
		arenas[a]->tickCount += args.ticks;
	}

	auto t1 = std::chrono::steady_clock::now();
	elapsedS += std::chrono::duration<double>(t1 - t0).count();

	for (int a = 0; a < args.arenas; a++) {
		DataStreamOut out;
		uint64_t arenaHash = HashArena(arenas[a], slotsPerArena[a], out);
		combinedHash = Fnv1a64(&arenaHash, sizeof(arenaHash), combinedHash);
		EmitArenaDiagnostics(args, a, arenaHash, out);
		delete arenas[a];
	}

	rscuda::DestroyBatch(batch);
	return combinedHash;
}

#endif // RS_CUDA

int main(int argc, char** argv) {
	Args args;
	if (!ParseArgs(argc, argv, args)) {
		PrintUsage(argv[0]);
		return 2;
	}

	if (!VerifyMeshes(args.meshDir))
		return 2;
	RocketSim::Init(args.meshDir, true);

	rsprof::enabled = args.profile;

	if (args.gpuCompare > 0) {
#ifdef RS_CUDA
		return RunGpuCompare(args);
#else
		fprintf(stderr, "GPU_COMPARE: FAIL (built without CUDA toolkit)\n");
		return 2;
#endif
	}

	double elapsedS = 0;
	uint64_t combinedHash;
	if (args.gpuStep) {
#ifdef RS_CUDA
		printf("GPU_STEP: ON (monolithic kernel produces the simulation result)\n");
		combinedHash = RunGpu(args, elapsedS);
#else
		fprintf(stderr, "GPU_STEP: FAIL (this binary was built without the CUDA toolkit; "
			"reconfigure with CMake on a machine with nvcc)\n");
		return 2;
#endif
	} else {
		combinedHash = RunCpu(args, elapsedS);
	}

	if (args.profile) {
		uint64_t phaseSumNs = 0;
		for (int p = 0; p < rsprof::PHASE_COUNT; p++)
			phaseSumNs += rsprof::totalNs[p];
		printf("PROFILE (totals over all arenas; %% of summed phase time):\n");
		for (int p = 0; p < rsprof::PHASE_COUNT; p++)
			printf("  %-14s %10.3f ms  %5.1f%%\n", rsprof::PHASE_NAMES[p],
				rsprof::totalNs[p] / 1e6, phaseSumNs ? 100.0 * rsprof::totalNs[p] / phaseSumNs : 0.0);
		printf("  phase_sum      %10.3f ms\n", phaseSumNs / 1e6);
	}

	uint64_t totalSteps = (uint64_t)args.arenas * (uint64_t)args.ticks;
	printf("ARENAS: %d\nTICKS: %d\nCARS: %d\nSEED: %" PRIu64 "\n", args.arenas, args.ticks, args.cars, args.seed);
	printf("STATE_HASH: %" PRIu64 "\n", combinedHash);
	printf("ELAPSED_S: %.3f\n", elapsedS);
	printf("SPS: %.3f\n", (double)totalSteps / elapsedS);

	if (args.hasExpectedHash) {
		if (combinedHash == args.expectedHash) {
			printf("GATE: PASS\n");
		} else {
			printf("GATE: FAIL (expected %" PRIu64 ", got %" PRIu64 ")\n", args.expectedHash, combinedHash);
			return 1;
		}
	}
	return 0;
}
