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
#include "../libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btStridingMeshInterface.h"
#include "../libsrc/bullet3-3.24/BulletCollision/BroadphaseCollision/btRSBroadphase.h"

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
	int rayFuzz = 0;                // validate device static raycast vs CPU for N rays
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
		} else if (!strcmp(argv[i], "--ray-fuzz")) {
			const char* v = need("--ray-fuzz"); if (!v) return false;
			out.rayFuzz = atoi(v);
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

// Extract the arena's static collision geometry (mesh triangles + planes, with their
// broadphase filters) for upload. _worldCollisionRBs is in creation order: meshes
// first, then planes - same order the device ray test iterates.
static void ExtractWorldGeo(Arena* arena,
	std::vector<rscuda::GpuPlane>& planes,
	std::vector<rscuda::GpuMeshInfo>& meshes,
	std::vector<rscuda::GpuVec3>& tris) {
	for (btRigidBody* rb : arena->_worldCollisionRBs) {
		const btCollisionShape* shape = rb->getCollisionShape();
		const btBroadphaseProxy* proxy = rb->getBroadphaseProxy();
		if (shape->getShapeType() == TRIANGLE_MESH_SHAPE_PROXYTYPE) {
			auto* meshShape = (const btBvhTriangleMeshShape*)shape;
			const unsigned char* vertBase;
			int numVerts, vertStride;
			const unsigned char* idxBase;
			int idxStride, numFaces;
			meshShape->getMeshInterface()->getLockedReadOnlyVertexIndexBase(
				&vertBase, numVerts, vertStride, &idxBase, idxStride, numFaces);
			rscuda::GpuMeshInfo mi;
			mi.triStart = (int)(tris.size() / 3);
			mi.triCount = numFaces;
			mi.group = proxy->m_collisionFilterGroup;
			mi.mask = proxy->m_collisionFilterMask;
			meshes.push_back(mi);
			for (int f = 0; f < numFaces; f++) {
				const int* idx = (const int*)(idxBase + (size_t)f * idxStride);
				for (int c = 0; c < 3; c++) {
					const float* v = (const float*)(vertBase + (size_t)idx[c] * vertStride);
					tris.push_back({ v[0], v[1], v[2] });
				}
			}
		} else if (shape->getShapeType() == STATIC_PLANE_PROXYTYPE) {
			auto* planeShape = (const btStaticPlaneShape*)shape;
			rscuda::GpuPlane pl;
			pl.normal = ToGpuB(planeShape->getPlaneNormal());
			pl.planeConst = planeShape->getPlaneConstant();
			pl.origin = ToGpuB(rb->getWorldTransform().getOrigin());
			pl.group = proxy->m_collisionFilterGroup;
			pl.mask = proxy->m_collisionFilterMask;
			planes.push_back(pl);
		}
	}
}

// ---- static-world raycast validation (--ray-fuzz N) ----
// Random rays through the arena volume; CPU bullet rayTest (filtered to statics) vs
// the device CuRayTestStatic, bitwise.
static int RunRayFuzz(const Args& args, int numRays) {
	std::vector<CarSlot> slots;
	Arena* arena = SetupArena(args, 0, slots);

	std::vector<rscuda::GpuPlane> planes;
	std::vector<rscuda::GpuMeshInfo> meshes;
	std::vector<rscuda::GpuVec3> tris;
	ExtractWorldGeo(arena, planes, meshes, tris);
	printf("RAY_FUZZ: world geo: %zu planes, %zu meshes, %zu tris\n",
		planes.size(), meshes.size(), tris.size() / 3);

	rscuda::GpuBatch* batch = rscuda::CreateBatch(1, args.cars, (int)arena->GetBoostPads().size());
	rscuda::UploadWorldGeo(batch, planes.data(), (int)planes.size(),
		meshes.data(), (int)meshes.size(), tris.data(), (int)(tris.size() / 3));

	// Dynamic candidates: ball (sphere) + cars (compound box), with live transforms
	// and proxy aabbMin (drives btRSBroadphase voxel cell membership)
	std::vector<rscuda::GpuDynBody> dyn;
	{
		rscuda::GpuDynBody d = {};
		btRigidBody& rb = arena->ball->_rigidBody;
		d.kind = 1; // sphere
		d.sphereRadius = ((btSphereShape*)rb.getCollisionShape())->getRadius();
		d.basis = ToGpuB(rb.getWorldTransform().getBasis());
		d.origin = ToGpuB(rb.getWorldTransform().getOrigin());
		d.aabbMin = ToGpuB(rb.getBroadphaseProxy()->m_aabbMin);
		d.group = rb.getBroadphaseProxy()->m_collisionFilterGroup;
		d.mask = rb.getBroadphaseProxy()->m_collisionFilterMask;
		d.hasContactResponse = rb.hasContactResponse();
		dyn.push_back(d);
	}
	for (CarSlot& slot : slots) {
		rscuda::GpuDynBody d = {};
		btRigidBody& rb = slot.car->_rigidBody;
		d.kind = 2; // compound box
		const btBoxShape& box = slot.car->_childHitboxShape;
		btVector3 he = box.getHalfExtentsWithoutMargin();
		float m = box.getMargin();
		d.boxHalfExtWithMargin = { he.x() + m, he.y() + m, he.z() + m };
		d.childOffset = ToGpuB(slot.car->_compoundShape.getChildTransform(0).getOrigin());
		d.basis = ToGpuB(rb.getWorldTransform().getBasis());
		d.origin = ToGpuB(rb.getWorldTransform().getOrigin());
		d.aabbMin = ToGpuB(rb.getBroadphaseProxy()->m_aabbMin);
		d.group = rb.getBroadphaseProxy()->m_collisionFilterGroup;
		d.mask = rb.getBroadphaseProxy()->m_collisionFilterMask;
		d.hasContactResponse = rb.hasContactResponse();
		dyn.push_back(d);
	}

	rscuda::GpuGridParams grid = {};
	{
		auto* bp = (btRSBroadphase*)arena->_bulletWorldParams.broadphase;
		grid.minPos = ToGpuB(bp->minPos);
		grid.cellSize = bp->cellSize;
		grid.cellSizeSq = bp->cellSizeSq;
		grid.cellsX = bp->cellsX;
		grid.cellsY = bp->cellsY;
		grid.cellsZ = bp->cellsZ;
	}

	Rng rng(MixSeed(args.seed, 0x52A1ull));
	std::vector<rscuda::GpuVec3> fromTo((size_t)numRays * 2);
	for (int i = 0; i < numRays; i++) {
		// positions across the arena volume (BT units; arena ~ +-82 x +-102 x 0..41),
		// every 4th ray aimed near the ball or a car so dynamic shapes get real coverage
		float x, y, z;
		uint64_t pick = rng.Next() % 4;
		if (pick == 0) {
			const rscuda::GpuDynBody& target = dyn[rng.Next() % dyn.size()];
			x = target.origin.x + rng.NextSigned() * 4.0f;
			y = target.origin.y + rng.NextSigned() * 4.0f;
			z = target.origin.z + (rng.NextSigned() * 0.5f + 0.5f) * 4.0f;
		} else {
			x = rng.NextSigned() * 90.0f;
			y = rng.NextSigned() * 110.0f;
			z = (rng.NextSigned() * 0.5f + 0.5f) * 45.0f;
		}
		// direction: downward-biased (suspension-like) or fully random; varied length
		float dx = rng.NextSigned(), dy = rng.NextSigned();
		float dz = (rng.Next() % 4 == 0) ? rng.NextSigned() : -(float)(rng.Next() % 1025) / 1024.0f;
		float len = 0.25f + (float)(rng.Next() % 1025) / 1024.0f * ((rng.Next() % 8 == 0) ? 40.0f : 2.0f);
		fromTo[i * 2] = { x, y, z };
		fromTo[i * 2 + 1] = { x + dx * len, y + dy * len, z + dz * len };
	}

	std::vector<rscuda::GpuRayFuzzResult> gpuRes(numRays);
	rscuda::RunRayFuzz(batch, fromTo.data(), numRays, btBroadphaseProxy::DefaultFilter,
		btBroadphaseProxy::AllFilter, dyn.data(), (int)dyn.size(), -1, grid, gpuRes.data());

	int bad = 0;
	for (int i = 0; i < numRays; i++) {
		btVector3 from(fromTo[i * 2].x, fromTo[i * 2].y, fromTo[i * 2].z);
		btVector3 to(fromTo[i * 2 + 1].x, fromTo[i * 2 + 1].y, fromTo[i * 2 + 1].z);
		btCollisionWorld::ClosestRayResultCallback cb(from, to, nullptr);
		arena->_bulletWorld.rayTest(from, to, cb);

		bool cpuHit = cb.hasHit();
		bool gpuHit = gpuRes[i].hitKind >= 0;
		uint32_t cf, gf;
		memcpy(&cf, &cb.m_closestHitFraction, 4);
		memcpy(&gf, &gpuRes[i].fraction, 4);
		bool fracOk = !cpuHit || cf == gf;
		bool normOk = true;
		if (cpuHit && gpuHit) {
			float cn[3] = { cb.m_hitNormalWorld.x(), cb.m_hitNormalWorld.y(), cb.m_hitNormalWorld.z() };
			float gn[3] = { gpuRes[i].normal.x, gpuRes[i].normal.y, gpuRes[i].normal.z };
			normOk = memcmp(cn, gn, 12) == 0;
		}
		if (cpuHit != gpuHit || !fracOk || !normOk) {
			if (bad < 5) {
				int cpuShapeType = cpuHit ? cb.m_collisionObject->getCollisionShape()->getShapeType() : -1;
				printf("RAY_FUZZ: MISMATCH ray %d from(%g,%g,%g) to(%g,%g,%g): "
					"cpuHit=%d gpuHit=%d cpuFrac=0x%08X gpuFrac=0x%08X cpuShape=%d\n"
					"  cpuN(%.9g,%.9g,%.9g) gpuN(%.9g,%.9g,%.9g)\n",
					i, from.x(), from.y(), from.z(), to.x(), to.y(), to.z(),
					cpuHit, gpuHit, cf, gf, cpuShapeType,
					cb.m_hitNormalWorld.x(), cb.m_hitNormalWorld.y(), cb.m_hitNormalWorld.z(),
					gpuRes[i].normal.x, gpuRes[i].normal.y, gpuRes[i].normal.z);
			}
			bad++;
		}
	}
	rscuda::DestroyBatch(batch);
	delete arena;
	if (bad) {
		printf("RAY_FUZZ: FAIL (%d/%d rays differ)\n", bad, numRays);
		return 1;
	}
	printf("RAY_FUZZ: PASS (%d rays bit-exact: hit/fraction/normal)\n", numRays);
	return 0;
}

static rscuda::GpuCurve ExtractCurve(const LinearPieceCurve& c) {
	rscuda::GpuCurve g = {};
	int i = 0;
	for (auto& p : c.valueMappings) {
		g.xs[i] = p.first;
		g.ys[i] = p.second;
		i++;
	}
	g.n = i;
	return g;
}

static rscuda::GpuCarConsts ExtractCarConsts(Arena* arena, Car* car) {
	using namespace RLConst;
	rscuda::GpuCarConsts cc = {};
	for (int i = 0; i < 4; i++) {
		auto& wi = car->_bulletVehicle.m_wheelInfo[i];
		auto& wc = cc.wheels[i];
		wc.connectionPointCS = ToGpuB(wi.m_chassisConnectionPointCS);
		wc.suspensionRestLength = wi.getSuspensionRestLength();
		wc.wheelRadius = wi.m_wheelsRadius;
		wc.maxSuspensionTravelCm = wi.m_maxSuspensionTravelCm;
		wc.suspensionStiffness = wi.m_suspensionStiffness;
		wc.dampingCompression = wi.m_wheelsDampingCompression;
		wc.dampingRelaxation = wi.m_wheelsDampingRelaxation;
		wc.suspensionForceScale = wi.m_suspensionForceScale;
	}
	cc.wheelDirectionCS = ToGpuB(car->_bulletVehicle.m_wheelInfo[0].m_wheelDirectionCS);
	cc.wheelAxleCS = ToGpuB(car->_bulletVehicle.m_wheelInfo[0].m_wheelAxleCS);
	cc.massBT = CAR_MASS_BT;
	cc.frictionScale = car->_rigidBody.getMass() / 3; // calcFrictionImpulses expression
	cc.dodgeDeadzone = car->config.dodgeDeadzone;
	cc.rayCastGroup = btBroadphaseProxy::DefaultFilter | CollisionMasks::DROPSHOT_FLOOR;
	cc.erp = arena->_bulletWorld.getSolverInfo().m_erp;
	cc.uuToBT = UU_TO_BT;
	cc.btToUU = BT_TO_UU;

	const MutatorConfig& mc = arena->GetMutatorConfig();
	cc.jumpAccel = mc.jumpAccel;
	cc.jumpImmediateForce = mc.jumpImmediateForce;
	cc.jumpImmediateForceRL = JUMP_IMMEDIATE_FORCE;
	cc.boostAccelGround = mc.boostAccelGround;
	cc.boostAccelAir = mc.boostAccelAir;
	cc.boostUsedPerSecond = mc.boostUsedPerSecond;
	cc.rechargeBoostEnabled = mc.rechargeBoostEnabled;
	cc.rechargeBoostDelay = mc.rechargeBoostDelay;
	cc.rechargeBoostPerSecond = mc.rechargeBoostPerSecond;
	cc.unlimitedFlips = mc.unlimitedFlips;
	cc.unlimitedDoubleJumps = mc.unlimitedDoubleJumps;

	cc.powerslideRiseRate = POWERSLIDE_RISE_RATE;
	cc.powerslideFallRate = POWERSLIDE_FALL_RATE;
	cc.throttleDeadzone = THROTTLE_DEADZONE;
	cc.stoppingForwardVel = STOPPING_FORWARD_VEL;
	cc.coastingBrakeFactor = COASTING_BRAKE_FACTOR;
	cc.brakingNoThrottleSpeedThresh = BRAKING_NO_THROTTLE_SPEED_THRESH;
	cc.throttleAirAccel = THROTTLE_AIR_ACCEL;
	cc.jumpMinTime = JUMP_MIN_TIME;
	cc.jumpMaxTime = JUMP_MAX_TIME;
	cc.jumpResetTimePad = JUMP_RESET_TIME_PAD;
	cc.doublejumpMaxDelay = DOUBLEJUMP_MAX_DELAY;
	cc.flipZDamp120 = FLIP_Z_DAMP_120;
	cc.flipZDampStart = FLIP_Z_DAMP_START;
	cc.flipZDampEnd = FLIP_Z_DAMP_END;
	cc.flipTorqueTime = FLIP_TORQUE_TIME;
	cc.flipPitchlockExtraTime = FLIP_PITCHLOCK_EXTRA_TIME;
	cc.flipInitialVelScale = FLIP_INITIAL_VEL_SCALE;
	cc.flipTorqueX = FLIP_TORQUE_X;
	cc.flipTorqueY = FLIP_TORQUE_Y;
	cc.flipForwardImpulseMaxSpeedScale = FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE;
	cc.flipSideImpulseMaxSpeedScale = FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE;
	cc.flipBackwardImpulseMaxSpeedScale = FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE;
	cc.flipBackwardImpulseScaleX = FLIP_BACKWARD_IMPULSE_SCALE_X;
	cc.carTorqueScale = CAR_TORQUE_SCALE;
	cc.carAirControlTorque = ToGpu(CAR_AIR_CONTROL_TORQUE);
	cc.carAirControlDamping = ToGpu(CAR_AIR_CONTROL_DAMPING);
	cc.carAutoflipImpulse = CAR_AUTOFLIP_IMPULSE;
	cc.carAutoflipTorque = CAR_AUTOFLIP_TORQUE;
	cc.carAutoflipTime = CAR_AUTOFLIP_TIME;
	cc.carAutoflipNormZThresh = CAR_AUTOFLIP_NORMZ_THRESH;
	cc.carAutoflipRollThresh = CAR_AUTOFLIP_ROLL_THRESH;
	cc.carAutorollForce = CAR_AUTOROLL_FORCE;
	cc.carAutorollTorque = CAR_AUTOROLL_TORQUE;
	cc.boostMinTime = BOOST_MIN_TIME;
	cc.boostMax = BOOST_MAX;
	cc.carMaxSpeed = CAR_MAX_SPEED;
	cc.suspensionSubtraction = BTVehicle::SUSPENSION_SUBTRACTION;

	cc.throttleTorqueBT = THROTTLE_TORQUE_AMOUNT * UU_TO_BT;
	cc.brakeTorqueBT = BRAKE_TORQUE_AMOUNT * UU_TO_BT;
	cc.gravityZBT = GRAVITY_Z * UU_TO_BT;
	cc.flipZDampFactor = powf(1 - FLIP_Z_DAMP_120, TICK_TIME / (1 / 120.f));

	cc.driveSpeedTorqueFactor = ExtractCurve(DRIVE_SPEED_TORQUE_FACTOR_CURVE);
	cc.steerAngleFromSpeed = ExtractCurve(STEER_ANGLE_FROM_SPEED_CURVE);
	cc.powerslideSteerAngleFromSpeed = ExtractCurve(POWERSLIDE_STEER_ANGLE_FROM_SPEED_CURVE);
	cc.nonStickyFrictionFactor = ExtractCurve(NON_STICKY_FRICTION_FACTOR_CURVE);
	cc.latFriction = ExtractCurve(LAT_FRICTION_CURVE);
	cc.longFriction = ExtractCurve(LONG_FRICTION_CURVE);
	cc.handbrakeLatFrictionFactor = ExtractCurve(HANDBRAKE_LAT_FRICTION_FACTOR_CURVE);
	cc.handbrakeLongFrictionFactor = ExtractCurve(HANDBRAKE_LONG_FRICTION_FACTOR_CURVE);
	return cc;
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

	// world geo + car consts + dyn template table for the car pre-tick segment
	{
		std::vector<rscuda::GpuPlane> planes;
		std::vector<rscuda::GpuMeshInfo> meshes;
		std::vector<rscuda::GpuVec3> tris;
		ExtractWorldGeo(arena, planes, meshes, tris);
		rscuda::UploadWorldGeo(batch, planes.data(), (int)planes.size(),
			meshes.data(), (int)meshes.size(), tris.data(), (int)(tris.size() / 3));
		rscuda::UploadCarConsts(batch, ExtractCarConsts(arena, slots[0].car));
	}
	rscuda::GpuGridParams grid = {};
	{
		auto* bp = (btRSBroadphase*)arena->_bulletWorldParams.broadphase;
		grid.minPos = ToGpuB(bp->minPos);
		grid.cellSize = bp->cellSize;
		grid.cellSizeSq = bp->cellSizeSq;
		grid.cellsX = bp->cellsX;
		grid.cellsY = bp->cellsY;
		grid.cellsZ = bp->cellsZ;
	}
	// rb -> shape/filter template (transforms/dynamics filled per snap)
	struct DynTemplate { const void* rb; rscuda::GpuDynBody base; };
	std::vector<DynTemplate> dynTemplates;
	{
		DynTemplate t = {};
		btRigidBody& rb = arena->ball->_rigidBody;
		t.rb = &rb;
		t.base.kind = 1;
		t.base.sphereRadius = ((btSphereShape*)rb.getCollisionShape())->getRadius();
		t.base.group = rb.getBroadphaseProxy()->m_collisionFilterGroup;
		t.base.mask = rb.getBroadphaseProxy()->m_collisionFilterMask;
		dynTemplates.push_back(t);
		for (CarSlot& slot : slots) {
			DynTemplate c = {};
			c.rb = &slot.car->_rigidBody;
			c.base.kind = 2;
			const btBoxShape& box = slot.car->_childHitboxShape;
			btVector3 he = box.getHalfExtentsWithoutMargin();
			float m = box.getMargin();
			c.base.boxHalfExtWithMargin = { he.x() + m, he.y() + m, he.z() + m };
			c.base.childOffset = ToGpuB(slot.car->_compoundShape.getChildTransform(0).getOrigin());
			c.base.group = slot.car->_rigidBody.getBroadphaseProxy()->m_collisionFilterGroup;
			c.base.mask = slot.car->_rigidBody.getBroadphaseProxy()->m_collisionFilterMask;
			dynTemplates.push_back(c);
		}
	}

	for (int tick = 0; tick < args.gpuCompare; tick++) {
		for (CarSlot& slot : slots)
			UpdateControls(slot);
		rscap::carSnaps.clear();
		rscap::enabled = true;
		arena->Step(1);
		rscap::enabled = false;

		// ---- car pre-tick segment (one compare per car, in CPU execution order) ----
		for (size_t k = 0; k + 2 < rscap::carSnaps.size(); k += 3) {
			const rscap::CarPreTickSnap& entry = rscap::carSnaps[k];
			const rscap::CarPreTickSnap& afterFirst = rscap::carSnaps[k + 1];
			const rscap::CarPreTickSnap& exitSnap = rscap::carSnaps[k + 2];

			// build dyn array in entry.dyn order, merging shape templates
			rscuda::GpuDynBody dyn[8] = {};
			int selfIdx = -1;
			for (int i = 0; i < entry.numDyn; i++) {
				const rscap::DynSnap& ds = entry.dyn[i];
				for (auto& t : dynTemplates)
					if (t.rb == ds.rb)
						dyn[i] = t.base;
				memcpy(&dyn[i].basis, ds.basis, 36);
				memcpy(&dyn[i].origin, ds.origin, 12);
				memcpy(&dyn[i].aabbMin, ds.aabbMin, 12);
				memcpy(&dyn[i].linVel, ds.linVel, 12);
				memcpy(&dyn[i].angVel, ds.angVel, 12);
				memcpy(&dyn[i].invInertiaWorld, ds.invInertiaWorld, 36);
				memcpy(&dyn[i].invInertiaDiagLocal, ds.invInertiaDiagLocal, 12);
				dyn[i].invMass = ds.invMass;
				dyn[i].hasContactResponse = ds.hasContactResponse;
				if (ds.rb == entry.body.rb)
					selfIdx = i;
			}

			rscuda::GpuCarTickState in = {};
			memcpy(&in.basis, entry.body.basis, 36);
			memcpy(&in.origin, entry.body.pos, 12);
			memcpy(&in.linVel, entry.body.linVel, 12);
			memcpy(&in.angVel, entry.body.angVel, 12);
			memcpy(&in.totalForce, entry.body.totalForce, 12);
			memcpy(&in.totalTorque, entry.body.totalTorque, 12);
			memcpy(&in.invInertiaWorld, entry.body.invInertiaWorld, 36);
			memcpy(&in.invInertiaDiagLocal, entry.dyn[selfIdx].invInertiaDiagLocal, 12);
			in.invMass = entry.dyn[selfIdx].invMass;
			in.solverTimeStep = entry.solverTimeStep;

			auto fillState = [](rscuda::GpuCarGameState& g, const rscap::CarPreTickSnap& s) {
				g.handbrakeVal = s.handbrakeVal;
				g.boost = s.boost;
				g.boostingTime = s.boostingTime;
				g.timeSinceBoosted = s.timeSinceBoosted;
				g.jumpTime = s.jumpTime;
				g.flipTime = s.flipTime;
				g.airTime = s.airTime;
				g.airTimeSinceJump = s.airTimeSinceJump;
				g.autoFlipTimer = s.autoFlipTimer;
				g.autoFlipTorqueScale = s.autoFlipTorqueScale;
				memcpy(&g.flipRelTorque, s.flipRelTorque, 12);
				memcpy(&g.worldContact_normal, s.worldContactNormal, 12);
				g.isOnGround = s.isOnGround;
				for (int w = 0; w < 4; w++) g.wheelsWithContact[w] = s.wheelsWithContact[w];
				g.hasJumped = s.hasJumped;
				g.hasDoubleJumped = s.hasDoubleJumped;
				g.hasFlipped = s.hasFlipped;
				g.isFlipping = s.isFlipping;
				g.isJumping = s.isJumping;
				g.isBoosting = s.isBoosting;
				g.isAutoFlipping = s.isAutoFlipping;
				g.worldContact_hasContact = s.worldContactHasContact;
				g.isDemoed = s.isDemoed;
			};
			fillState(in.st, entry);
			in.st.lastControls.jump = entry.lastCtrlJump;
			in.controls.throttle = entry.ctrlThrottle;
			in.controls.steer = entry.ctrlSteer;
			in.controls.pitch = entry.ctrlPitch;
			in.controls.yaw = entry.ctrlYaw;
			in.controls.roll = entry.ctrlRoll;
			in.controls.jump = entry.ctrlJump;
			in.controls.boost = entry.ctrlBoost;
			in.controls.handbrake = entry.ctrlHandbrake;
			for (int w = 0; w < 4; w++) {
				in.wheels[w].steerAngle = entry.wheels[w].steerAngle;
				in.wheels[w].engineForce = entry.wheels[w].engineForce;
				in.wheels[w].brake = entry.wheels[w].brake;
				in.wheels[w].latFriction = entry.wheels[w].latFriction;
				in.wheels[w].longFriction = entry.wheels[w].longFriction;
				in.wheels[w].suspensionLength = entry.wheels[w].suspensionLength;
				in.wheels[w].extraPushback = entry.wheels[w].extraPushback;
			}

			rscuda::GpuCarTickState out;
			rscuda::GpuWheelScratchOut wOut[4];
			rscuda::RunCarPreTickOnce(batch, in, dyn, entry.numDyn, selfIdx, grid, TICK_TIME, out, wOut);

			// compare against exit snap
			auto failF = [&](const char* what, int wheelIdx, float cpu, float gpu) -> bool {
				uint32_t cb, gb;
				memcpy(&cb, &cpu, 4);
				memcpy(&gb, &gpu, 4);
				if (cb == gb) return false;
				printf("GPU_COMPARE: FAIL tick %d car %u pretick %s%s%d: cpu %.9g (0x%08X) gpu %.9g (0x%08X)\n",
					tick, entry.carId, what, wheelIdx >= 0 ? " wheel " : "", wheelIdx >= 0 ? wheelIdx : 0,
					cpu, cb, gpu, gb);
				return true;
			};
			bool bad = false;
			bad |= failF("dbgForwardSpeedUU", -1, afterFirst.forwardSpeedUU, out.dbgForwardSpeedUU);
			const float* cv;
			cv = exitSnap.body.linVel;
			bad |= failF("linVel.x", -1, cv[0], out.linVel.x) || failF("linVel.y", -1, cv[1], out.linVel.y) || failF("linVel.z", -1, cv[2], out.linVel.z);
			cv = exitSnap.body.angVel;
			bad |= failF("angVel.x", -1, cv[0], out.angVel.x) || failF("angVel.y", -1, cv[1], out.angVel.y) || failF("angVel.z", -1, cv[2], out.angVel.z);
			cv = exitSnap.body.totalForce;
			bad |= failF("force.x", -1, cv[0], out.totalForce.x) || failF("force.y", -1, cv[1], out.totalForce.y) || failF("force.z", -1, cv[2], out.totalForce.z);
			cv = exitSnap.body.totalTorque;
			bad |= failF("torque.x", -1, cv[0], out.totalTorque.x) || failF("torque.y", -1, cv[1], out.totalTorque.y) || failF("torque.z", -1, cv[2], out.totalTorque.z);
			bad |= failF("handbrakeVal", -1, exitSnap.handbrakeVal, out.st.handbrakeVal);
			bad |= failF("boost", -1, exitSnap.boost, out.st.boost);
			bad |= failF("boostingTime", -1, exitSnap.boostingTime, out.st.boostingTime);
			bad |= failF("timeSinceBoosted", -1, exitSnap.timeSinceBoosted, out.st.timeSinceBoosted);
			bad |= failF("jumpTime", -1, exitSnap.jumpTime, out.st.jumpTime);
			bad |= failF("flipTime", -1, exitSnap.flipTime, out.st.flipTime);
			bad |= failF("airTime", -1, exitSnap.airTime, out.st.airTime);
			bad |= failF("airTimeSinceJump", -1, exitSnap.airTimeSinceJump, out.st.airTimeSinceJump);
			bad |= failF("autoFlipTimer", -1, exitSnap.autoFlipTimer, out.st.autoFlipTimer);
			bad |= failF("flipRelTorque.x", -1, exitSnap.flipRelTorque[0], out.st.flipRelTorque.x);
			bad |= failF("flipRelTorque.y", -1, exitSnap.flipRelTorque[1], out.st.flipRelTorque.y);
			auto failB = [&](const char* what, int cpu, int gpu) -> bool {
				if (cpu == gpu) return false;
				printf("GPU_COMPARE: FAIL tick %d car %u pretick %s: cpu %d gpu %d\n", tick, entry.carId, what, cpu, gpu);
				return true;
			};
			bad |= failB("isOnGround", exitSnap.isOnGround, out.st.isOnGround);
			bad |= failB("hasJumped", exitSnap.hasJumped, out.st.hasJumped);
			bad |= failB("hasDoubleJumped", exitSnap.hasDoubleJumped, out.st.hasDoubleJumped);
			bad |= failB("hasFlipped", exitSnap.hasFlipped, out.st.hasFlipped);
			bad |= failB("isFlipping", exitSnap.isFlipping, out.st.isFlipping);
			bad |= failB("isJumping", exitSnap.isJumping, out.st.isJumping);
			bad |= failB("isBoosting", exitSnap.isBoosting, out.st.isBoosting);
			bad |= failB("isAutoFlipping", exitSnap.isAutoFlipping, out.st.isAutoFlipping);
			for (int w = 0; w < 4; w++) {
				bad |= failF("steerAngle", w, exitSnap.wheels[w].steerAngle, out.wheels[w].steerAngle);
				bad |= failF("engineForce", w, exitSnap.wheels[w].engineForce, out.wheels[w].engineForce);
				bad |= failF("brake", w, exitSnap.wheels[w].brake, out.wheels[w].brake);
				bad |= failF("latFriction", w, exitSnap.wheels[w].latFriction, out.wheels[w].latFriction);
				bad |= failF("longFriction", w, exitSnap.wheels[w].longFriction, out.wheels[w].longFriction);
				bad |= failF("suspensionLength", w, exitSnap.wheels[w].suspensionLength, out.wheels[w].suspensionLength);
				// localize with after-vehicleFirst scratch
				bad |= failF("scr.contactNormal.x", w, afterFirst.wheels[w].contactNormalWS[0], wOut[w].contactNormalWS.x);
				bad |= failF("scr.contactNormal.y", w, afterFirst.wheels[w].contactNormalWS[1], wOut[w].contactNormalWS.y);
				bad |= failF("scr.contactNormal.z", w, afterFirst.wheels[w].contactNormalWS[2], wOut[w].contactNormalWS.z);
				bad |= failF("scr.contactPoint.x", w, afterFirst.wheels[w].contactPointWS[0], wOut[w].contactPointWS.x);
				bad |= failF("scr.impulse.x", w, afterFirst.wheels[w].impulse[0], wOut[w].impulse.x);
				bad |= failF("scr.impulse.y", w, afterFirst.wheels[w].impulse[1], wOut[w].impulse.y);
				bad |= failF("scr.impulse.z", w, afterFirst.wheels[w].impulse[2], wOut[w].impulse.z);
				if (memcmp(&afterFirst.wheels[w].extraPushback, &wOut[w].extraPushback, 4) != 0) {
					// derive CPU-side branch inputs from the capture for localization
					btVector3 hp(afterFirst.wheels[w].hardPointWS[0], afterFirst.wheels[w].hardPointWS[1], afterFirst.wheels[w].hardPointWS[2]);
					btVector3 cp(afterFirst.wheels[w].contactPointWS[0], afterFirst.wheels[w].contactPointWS[1], afterFirst.wheels[w].contactPointWS[2]);
					btVector3 up(entry.body.basis[2], entry.body.basis[5], entry.body.basis[8]);
					float cpuTrace = (hp - cp).dot(up);
					printf("  pushback dbg wheel %d: cpuTrace %.9g gpuTrace %.9g gpuThresh %.9g gpuResolveRaw %.9g groundStatic cpu=%d gpu=%d\n",
						w, cpuTrace, wOut[w].dbgWheelTraceLen, wOut[w].dbgPushThresh, wOut[w].dbgResolveRaw,
						afterFirst.wheels[w].groundIsStatic, wOut[w].groundIdx == -1 ? 1 : (wOut[w].groundIdx >= 0 ? 0 : -1));
				}
				bad |= failF("scr.extraPushback", w, afterFirst.wheels[w].extraPushback, wOut[w].extraPushback);
				bad |= failB("scr.isInContact", afterFirst.wheels[w].isInContact, wOut[w].isInContact);
			}
			if (bad)
				return 1;
		}

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

	printf("GPU_COMPARE: PASS (%d ticks, phases: car-pretick, predict, integrate, activation; %d bodies bit-exact)\n",
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
	if (args.rayFuzz > 0) {
#ifdef RS_CUDA
		return RunRayFuzz(args, args.rayFuzz);
#else
		fprintf(stderr, "RAY_FUZZ: FAIL (built without CUDA toolkit)\n");
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
