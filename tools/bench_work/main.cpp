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
//    engine upstream (Math.cpp GetRandEngine). The work copy adds Math::SeedRandEngine; both
//    paths pin the engine to a per-arena seed (DemoRngSeed) right before stepping, so
//    --demo-mode normal/on-contact runs are deterministic. The default stays DISABLED.
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
#include "../libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btTriangleInfoMap.h"
#include "../libsrc/bullet3-3.24/BulletCollision/BroadphaseCollision/btRSBroadphase.h"
#include "../libsrc/bullet3-3.24/BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
#include "../libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btConvexHullShape.h"

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
	bool phaseProf = false;         // print per-phase GPU cycle breakdown after the run
	int gpuCompare = 0;             // lockstep-validate ported GPU phases for N ticks
	int rayFuzz = 0;                // validate device static raycast vs CPU for N rays
	int manifoldDump = 0;           // diagnostic: print CPU collision manifolds for N ticks
	int arenaOffset = 0;            // diagnostic: arena index offset (reproduce arena K standalone)
	bool noBallMotion = false;      // diagnostic: skip the gpu-compare ball-motion injection
	GameMode gameMode = GameMode::SOCCAR;
	DemoMode demoMode = DemoMode::DISABLED;
	// chunked GPU stepping (RL decision-step model): constant per-car controls per
	// chunk via UploadConstantControls + one StepTicks(N) launch per chunk.
	// Controls roll exactly every N ticks on both paths so results are comparable.
	int gpuChunk = 0;
};

// Demo respawns draw Math::RandInt(0, 4, -1) from the engine; both paths pin it to this
// per-arena seed before stepping (CPU: Math::SeedRandEngine, GPU: UploadRngState)
static uint64_t DemoRngSeed(const Args& args, int arenaIdx) {
	return MixSeed(args.seed, (uint64_t)(arenaIdx + args.arenaOffset)) ^ 0xDE30ull;
}

static void PrintUsage(const char* argv0) {
	fprintf(stderr,
		"Usage: %s --mesh-dir <collision_meshes dir> --ticks N [--arenas N] [--cars N] "
		"[--seed N] [--expected-state-hash H] [--profile] [--gpu-step] "
		"[--game-mode M] [--demo-mode disabled|normal|on-contact]\n", argv0);
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
		} else if (!strcmp(argv[i], "--phase-prof")) {
			out.phaseProf = true;
		} else if (!strcmp(argv[i], "--gpu-compare")) {
			const char* v = need("--gpu-compare"); if (!v) return false;
			out.gpuCompare = atoi(v);
		} else if (!strcmp(argv[i], "--ray-fuzz")) {
			const char* v = need("--ray-fuzz"); if (!v) return false;
			out.rayFuzz = atoi(v);
		} else if (!strcmp(argv[i], "--manifold-dump")) {
			const char* v = need("--manifold-dump"); if (!v) return false;
			out.manifoldDump = atoi(v);
		} else if (!strcmp(argv[i], "--arena-offset")) {
			const char* v = need("--arena-offset"); if (!v) return false;
			out.arenaOffset = atoi(v);
		} else if (!strcmp(argv[i], "--no-ball-motion")) {
			out.noBallMotion = true;
		} else if (!strcmp(argv[i], "--game-mode")) {
			const char* v = need("--game-mode"); if (!v) return false;
			if (!strcmp(v, "soccar")) out.gameMode = GameMode::SOCCAR;
			else if (!strcmp(v, "heatseeker")) out.gameMode = GameMode::HEATSEEKER;
			else if (!strcmp(v, "hoops")) out.gameMode = GameMode::HOOPS;
			else if (!strcmp(v, "snowday")) out.gameMode = GameMode::SNOWDAY;
			else if (!strcmp(v, "dropshot")) out.gameMode = GameMode::DROPSHOT;
			else { fprintf(stderr, "Unknown game mode: %s\n", v); return false; }
		} else if (!strcmp(argv[i], "--gpu-chunk")) {
			const char* v = need("--gpu-chunk"); if (!v) return false;
			out.gpuChunk = atoi(v);
		} else if (!strcmp(argv[i], "--demo-mode")) {
			const char* v = need("--demo-mode"); if (!v) return false;
			if (!strcmp(v, "disabled")) out.demoMode = DemoMode::DISABLED;
			else if (!strcmp(v, "normal")) out.demoMode = DemoMode::NORMAL;
			else if (!strcmp(v, "on-contact")) out.demoMode = DemoMode::ON_CONTACT;
			else { fprintf(stderr, "Unknown demo mode: %s\n", v); return false; }
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

static void UpdateControls(CarSlot& slot, int chunkAlign = 0) {
	if (slot.holdTicks == 0) {
		slot.car->controls = RollControls(slot.rng);
		slot.holdTicks = chunkAlign > 0
			? chunkAlign // chunked mode: controls constant within each chunk
			: 2 + (int)(slot.rng.Next() % 59); // change every 2-60 ticks
	}
	slot.holdTicks--;
}

static constexpr float HALF_PI = 1.5707963267948966f;

// Identical arena/car setup for CPU and GPU paths - this is state initialization, not physics.
static Arena* SetupArena(const Args& args, int arenaIdxIn, std::vector<CarSlot>& slots) {
	int arenaIdx = arenaIdxIn + args.arenaOffset;
	Arena* arena = Arena::Create(args.gameMode);

	// Demo respawn randomness is pinned per arena via DemoRngSeed (see header comment)
	MutatorConfig mc = arena->GetMutatorConfig();
	mc.demoMode = args.demoMode;
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
	if (args.gameMode == GameMode::HEATSEEKER) {
		// The arena ctor randomizes the heatseeker ball's y-sign with the
		// wall-clock-seeded engine; override with a seed-derived deterministic state.
		// yTargetDir starts nonzero so the steering path is exercised from tick 0.
		float ySign = (spawnRng.Next() & 1) ? 1.0f : -1.0f;
		Vec scale = Vec(1, ySign, 1);
		BallState bs = BallState();
		bs.pos = RLConst::Heatseeker::BALL_START_POS * scale;
		bs.vel = RLConst::Heatseeker::BALL_START_VEL * scale;
		bs.hsInfo.yTargetDir = ySign;
		arena->ball->SetState(bs);
	} else if (args.gameMode == GameMode::SNOWDAY) {
		// ResetToRandomKickoff's "don't freeze" puck init: the ctor leaves the ball
		// asleep at the soccar rest height with zero velocity, where it never falls
		// to puck rest height; the epsilon z-vel keeps it awake (deterministic)
		BallState bs = BallState();
		bs.vel.z = FLT_EPSILON;
		arena->ball->SetState(bs);
	} else if (args.gameMode == GameMode::HOOPS) {
		// Per-arena coin: half the arenas keep the frozen kickoff state (covers the
		// launch-delay impulse), half get a seed-derived live ball (covers car-ball
		// hits incl. the hoops zScale branch, net mesh and wall plane contacts -
		// random-control cars essentially never reach a center-resting ball)
		if (spawnRng.Next() & 1) {
			BallState bs = BallState();
			float ang = spawnRng.NextSigned() * (float)M_PI;
			bs.pos.z = 400.0f;
			bs.vel = Vec(cosf(ang) * 1300.0f, sinf(ang) * 1300.0f, 600.0f);
			arena->ball->SetState(bs);
		}
	} else if (args.gameMode == GameMode::DROPSHOT) {
		// Per-arena 3-way: frozen kickoff (launch + center-tile damage), live ball
		// (car hits / charge), or a vertical slam onto a seed-derived tile spot -
		// the slam re-damages the same tile on the rebound (> MIN_DAMAGE_INTERVAL
		// apart), BREAKING it: covers response removal, the deferred post-tick
		// sync, and the ball falling through the hole
		uint64_t variant = spawnRng.Next() % 3;
		if (variant == 1) {
			BallState bs = BallState();
			float ang = spawnRng.NextSigned() * (float)M_PI;
			bs.pos.z = 400.0f;
			bs.vel = Vec(cosf(ang) * 1300.0f, sinf(ang) * 1300.0f, 600.0f);
			arena->ball->SetState(bs);
		} else if (variant == 2) {
			BallState bs = BallState();
			float ySign = (spawnRng.Next() & 1) ? 1.0f : -1.0f;
			bs.pos = Vec(spawnRng.NextSigned() * 2000.0f,
				ySign * (1500.0f + spawnRng.NextSigned() * 1000.0f), 600.0f);
			bs.vel = Vec(0, 0, -1200.0f);
			arena->ball->SetState(bs);
		}
	}
	// Soccar: ball stays at its deterministic default rest state (center, BALL_REST_Z)
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
	// Dropshot tile damage states (DropshotTilesState has no upstream serialization,
	// but tile state IS simulation state - include it in the gate hash)
	if (arena->gameMode == GameMode::DROPSHOT)
		for (int t = 0; t < RLConst::Dropshot::TEAM_AMOUNT; t++)
			for (int i = 0; i < RLConst::Dropshot::NUM_TILES_PER_TEAM; i++)
				out.Write<uint8_t>(arena->_dropshotTilesState.states[t][i].damageState);
	return Fnv1a64(out.data.data(), out.data.size(), FNV_BASIS);
}

static void EmitCarDiag(const char* tag, std::vector<CarSlot>& slots) {
	for (size_t c = 0; c < slots.size(); c++) {
		const CarState& cs = slots[c].car->_internalState;
		printf("DIAG %s car%zu: whenHit=%llu whenExtra=%llu ballPos=(%.9g,%.9g,%.9g) wcNormal=(%.9g,%.9g,%.9g) vel=(%.9g,%.9g,%.9g)\n",
			tag, c, (unsigned long long)cs.ballHitInfo.tickCountWhenHit,
			(unsigned long long)cs.ballHitInfo.tickCountWhenExtraImpulseApplied,
			cs.ballHitInfo.ballPos.x, cs.ballHitInfo.ballPos.y, cs.ballHitInfo.ballPos.z,
			cs.worldContact.contactNormal.x, cs.worldContact.contactNormal.y, cs.worldContact.contactNormal.z,
			cs.vel.x, cs.vel.y, cs.vel.z);
	}
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
	// Create ALL arenas up front, mirroring RunGpu's allocation sequence exactly.
	// Arena::Step's car pre-tick iteration follows _cars (a pointer-keyed
	// unordered_set), so the heap layout at car creation is PHYSICS-RELEVANT when
	// cars interact during pre-ticks (e.g. one car riding on another). Identical
	// allocation histories give both processes identical _cars orders, which the
	// GPU then replicates via the uploaded carOrder.
	std::vector<Arena*> arenas(args.arenas);
	std::vector<std::vector<CarSlot>> slotsPerArena(args.arenas);
	for (int a = 0; a < args.arenas; a++)
		arenas[a] = SetupArena(args, a, slotsPerArena[a]);

	uint64_t combinedHash = FNV_BASIS;
	for (int arenaIdx = 0; arenaIdx < args.arenas; arenaIdx++) {
		Arena* arena = arenas[arenaIdx];
		std::vector<CarSlot>& slots = slotsPerArena[arenaIdx];

		// Pin the demo-respawn engine per arena (the Arena ctor may draw from it)
		Math::SeedRandEngine(DemoRngSeed(args, arenaIdx));

		auto t0 = std::chrono::steady_clock::now();
		for (int tick = 0; tick < args.ticks; tick++) {
			for (CarSlot& slot : slots)
				UpdateControls(slot, args.gpuChunk);
			arena->Step(1);
		}
		auto t1 = std::chrono::steady_clock::now();
		elapsedS += std::chrono::duration<double>(t1 - t0).count();

		DataStreamOut out;
		uint64_t arenaHash = HashArena(arena, slots, out);
		combinedHash = Fnv1a64(&arenaHash, sizeof(arenaHash), combinedHash);
		EmitArenaDiagnostics(args, arenaIdx, arenaHash, out);
		if (args.dumpArenaHashes)
			EmitCarDiag("cpu", slots);
	}
	for (Arena* arena : arenas)
		delete arena;
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
	// btRigidBody::setGravity: m_gravity (force) = acceleration * (1/invMass)
	c.gravityForce = ToGpuB(rb.getGravity() * (1.0f / rb.getInvMass()));
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
	g.rotMat = ToGpuB((btMatrix3x3)cs.rotMat);
	g.noContactResponse = car->_rigidBody.hasContactResponse() ? 0 : 1;
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
	for (int w = 0; w < 4; w++) {
		const btWheelInfoRL& wi = car->_bulletVehicle.m_wheelInfo[w];
		g.wheels[w].steerAngle = wi.m_steerAngle;
		g.wheels[w].engineForce = wi.m_engineForce;
		g.wheels[w].brake = wi.m_brake;
		g.wheels[w].latFriction = wi.m_latFriction;
		g.wheels[w].longFriction = wi.m_longFriction;
		g.wheels[w].suspensionLength = wi.m_raycastInfo.m_suspensionLength;
		g.wheels[w].extraPushback = wi.m_extraPushback;
	}
}

static void WritebackCar(Car* car, const rscuda::GpuBodyState& body, const rscuda::GpuCarGameState& g) {
	WritebackBody(car->_rigidBody, body);
	CarState& cs = car->_internalState;
	// CarState::rotMat is the device-carried post-tick cache, NOT the live basis -
	// they differ for a car that is demoed right now (post-tick skips the refresh)
	cs.rotMat = (RotMat)FromGpuB(g.rotMat);
	{ // mirror noContactResponse back into the body's collision flags
		int flags = car->_rigidBody.getCollisionFlags();
		if (g.noContactResponse)
			flags |= btCollisionObject::CF_NO_CONTACT_RESPONSE;
		else
			flags &= ~btCollisionObject::CF_NO_CONTACT_RESPONSE;
		car->_rigidBody.setCollisionFlags(flags);
	}
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
	for (int w = 0; w < 4; w++) {
		btWheelInfoRL& wi = car->_bulletVehicle.m_wheelInfo[w];
		wi.m_steerAngle = g.wheels[w].steerAngle;
		wi.m_engineForce = g.wheels[w].engineForce;
		wi.m_brake = g.wheels[w].brake;
		wi.m_latFriction = g.wheels[w].latFriction;
		wi.m_longFriction = g.wheels[w].longFriction;
		wi.m_raycastInfo.m_suspensionLength = g.wheels[w].suspensionLength;
		wi.m_extraPushback = g.wheels[w].extraPushback;
	}
}

static constexpr float TICK_TIME = 1.0f / 120.0f;

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

// Extract per-mesh BVH arrays + internal-edge triangle info for upload. Mesh order and
// triangle indexing match ExtractWorldGeo (creation order), so triInfos for mesh i live
// at [meshes[i].triStart, +triCount). Absent info-map entries get hasInfo=0: the CPU
// adjustInternalEdgeContacts returns immediately when find() misses, so the device must
// skip those triangles too.
static void ExtractMeshBvhs(Arena* arena,
	std::vector<rscuda::GpuMeshBvh>& bvhs,
	std::vector<rscuda::GpuBvhNode>& nodes,
	std::vector<rscuda::GpuBvhSubtree>& subtrees,
	std::vector<rscuda::GpuTriInfo>& triInfos,
	std::vector<float>& margins) {
	for (btRigidBody* rb : arena->_worldCollisionRBs) {
		btCollisionShape* shape = rb->getCollisionShape();
		if (shape->getShapeType() != TRIANGLE_MESH_SHAPE_PROXYTYPE)
			continue;
		auto* meshShape = (btBvhTriangleMeshShape*)shape;
		btOptimizedBvh* bvh = meshShape->getOptimizedBvh();

		rscuda::GpuMeshBvh mb;
		mb.nodeStart = (int)nodes.size();
		mb.subtreeStart = (int)subtrees.size();
		mb.numNodes = bvh->getQuantizedNodeArray().size();
		mb.bvhAabbMin = ToGpuB(bvh->getBvhAabbMin());
		mb.bvhAabbMax = ToGpuB(bvh->getBvhAabbMax());
		mb.bvhQuantization = ToGpuB(bvh->getBvhQuantization());

		const QuantizedNodeArray& na = bvh->getQuantizedNodeArray();
		for (int i = 0; i < na.size(); i++) {
			rscuda::GpuBvhNode n;
			for (int c = 0; c < 3; c++) {
				n.qmin[c] = na[i].m_quantizedAabbMin[c];
				n.qmax[c] = na[i].m_quantizedAabbMax[c];
			}
			n.escapeIndexOrTriangleIndex = na[i].m_escapeIndexOrTriangleIndex;
			nodes.push_back(n);
		}
		const BvhSubtreeInfoArray& sa = bvh->getSubtreeInfoArray();
		mb.numSubtrees = sa.size();
		for (int i = 0; i < sa.size(); i++) {
			rscuda::GpuBvhSubtree s;
			for (int c = 0; c < 3; c++) {
				s.qmin[c] = sa[i].m_quantizedAabbMin[c];
				s.qmax[c] = sa[i].m_quantizedAabbMax[c];
			}
			s.rootNodeIndex = sa[i].m_rootNodeIndex;
			s.subtreeSize = sa[i].m_subtreeSize;
			subtrees.push_back(s);
		}
		bvhs.push_back(mb);
		margins.push_back(meshShape->getMargin());

		const unsigned char* vertBase;
		int numVerts, vertStride;
		const unsigned char* idxBase;
		int idxStride, numFaces;
		meshShape->getMeshInterface()->getLockedReadOnlyVertexIndexBase(
			&vertBase, numVerts, vertStride, &idxBase, idxStride, numFaces);
		const btTriangleInfoMap* infoMap = meshShape->getTriangleInfoMap();
		for (int f = 0; f < numFaces; f++) {
			rscuda::GpuTriInfo ti = {};
			// btGetHash(partId=0, triIdx) = (0 << 21) | triIdx = triIdx
			const btTriangleInfo* info = infoMap ? infoMap->find(f) : nullptr;
			if (info) {
				ti.hasInfo = 1;
				ti.flags = info->m_flags;
				ti.edgeV0V1Angle = info->m_edgeV0V1Angle;
				ti.edgeV1V2Angle = info->m_edgeV1V2Angle;
				ti.edgeV2V0Angle = info->m_edgeV2V0Angle;
			}
			triInfos.push_back(ti);
		}
	}
}

// Extract collision-orchestration data: collide consts, static object table in
// broadphase handle order, per-cell static CSR lists, and the initial dyn proxy
// state. Mesh/plane shapeIdx match ExtractWorldGeo order (both creation order).
static rscuda::GpuCurve ExtractCurve(const LinearPieceCurve& c); // defined below

static void ExtractCollideData(Arena* arena, const std::vector<CarSlot>& slots,
	rscuda::GpuCollideConsts& cc,
	std::vector<rscuda::GpuStaticObj>& statics,
	std::vector<int>& cellStart, std::vector<int>& cellList,
	std::vector<rscuda::GpuDynProxy>& proxies, int& regCounter) {
	using namespace RLConst;
	auto* bp = (btRSBroadphase*)arena->_bulletWorldParams.broadphase;

	// static table (handle order) + handle -> static index map
	std::vector<int> staticIdxOfHandle(bp->m_LastHandleIndex + 1, -1);
	int meshCount = 0, planeCount = 0;
	for (int h = 0; h <= bp->m_LastHandleIndex; h++) {
		btRSBroadphaseProxy* p = &bp->m_pHandles[h];
		if (!p->m_clientObject || !p->isStatic)
			continue;
		auto* obj = (btCollisionObject*)p->m_clientObject;
		rscuda::GpuStaticObj so = {};
		if (p->shapeType == TRIANGLE_MESH_SHAPE_PROXYTYPE) {
			so.kind = 0;
			so.shapeIdx = meshCount++;
		} else if (p->shapeType == CONVEX_HULL_SHAPE_PROXYTYPE) {
			so.kind = 2; // dropshot tile
			so.shapeIdx = obj->getUserIndex2(); // total tile index 0..139
		} else {
			so.kind = 1;
			so.shapeIdx = planeCount++;
		}
		so.aabbMin = ToGpuB(p->m_aabbMin);
		so.aabbMax = ToGpuB(p->m_aabbMax);
		{
			btVector3 smn, smx;
			obj->getCollisionShape()->getAabb(obj->getWorldTransform(), smn, smx);
			so.shapeAabbMin = ToGpuB(smn);
			so.shapeAabbMax = ToGpuB(smx);
		}
		so.group = p->m_collisionFilterGroup;
		so.mask = p->m_collisionFilterMask;
		so.friction = obj->getFriction();
		so.restitution = obj->getRestitution();
		so.breakingThreshold = obj->getCollisionShape()->getContactBreakingThreshold(gContactBreakingThreshold);
		staticIdxOfHandle[h] = (int)statics.size();
		statics.push_back(so);
	}

	// per-cell static lists (CSR), preserving CPU smear insertion order
	cellStart.resize(bp->totalCells + 1);
	int total = 0;
	for (int c = 0; c < bp->totalCells; c++) {
		cellStart[c] = total;
		total += (int)bp->cells[c].staticHandles.size();
	}
	cellStart[bp->totalCells] = total;
	cellList.reserve(total);
	for (int c = 0; c < bp->totalCells; c++)
		for (auto* sh : bp->cells[c].staticHandles)
			cellList.push_back(staticIdxOfHandle[(int)(sh - bp->m_pHandles)]);

	// initial dyn proxy state (creation order: ball, cars). regSeq = creation order.
	regCounter = 0;
	auto addProxy = [&](btRigidBody* rb) {
		auto* p = (btRSBroadphaseProxy*)rb->getBroadphaseProxy();
		rscuda::GpuDynProxy dp = {};
		dp.aabbMin = ToGpuB(p->m_aabbMin);
		dp.aabbMax = ToGpuB(p->m_aabbMax);
		dp.cellIdx = p->cellIdx;
		dp.i = p->iIdx;
		dp.j = p->jIdx;
		dp.k = p->kIdx;
		dp.regSeq = ++regCounter;
		proxies.push_back(dp);
	};
	addProxy(&arena->ball->_rigidBody);
	for (const CarSlot& slot : slots)
		addProxy(&slot.car->_rigidBody);

	// collide consts
	cc = {};
	Car* car = slots[0].car;
	const btCompoundShape& comp = car->_compoundShape;
	cc.carLocalAabbMin = ToGpuB(comp.getLocalAabbMin());
	cc.carLocalAabbMax = ToGpuB(comp.getLocalAabbMax());
	cc.carCompoundMargin = comp.getMargin();
	btCollisionShape* ballShape = arena->ball->_rigidBody.getCollisionShape();
	if (arena->ball->IsSphere()) {
		cc.ballRadius = ((btSphereShape*)ballShape)->getRadius();
	} else {
		// snowday puck: btConvexHullShape (points uploaded via UploadBallHull).
		// m_localAabb is protected; recompute it exactly as recalcLocalAabb does
		// (6-direction batched support extrema +- margin, margin included ONCE here
		// and added AGAIN by getAabb - bullet quirk, replicated on device)
		auto* hull = (btConvexHullShape*)ballShape;
		cc.ballIsHull = 1;
		cc.ballHullMargin = hull->getMargin();
		static const btVector3 dirs[6] = {
			btVector3(1, 0, 0), btVector3(0, 1, 0), btVector3(0, 0, 1),
			btVector3(-1, 0, 0), btVector3(0, -1, 0), btVector3(0, 0, -1) };
		btVector3 sup[6] = {};
		hull->batchedUnitVectorGetSupportingVertexWithoutMargin(dirs, sup, 6);
		btVector3 lmn, lmx;
		for (int i = 0; i < 3; i++) {
			lmx[i] = sup[i][i] + hull->getMargin();
			lmn[i] = sup[i + 3][i] - hull->getMargin();
		}
		cc.ballHullLocalAabbMin = ToGpuB(lmn);
		cc.ballHullLocalAabbMax = ToGpuB(lmx);
	}
	cc.puckGroundStickForce = Snowday::PUCK_GROUND_STICK_FORCE;
	cc.ballLaunchDelay = (arena->gameMode == GameMode::DROPSHOT)
		? Dropshot::BALL_LAUNCH_DELAY : BALL_HOOPS_LAUNCH_DELAY;
	cc.ballLaunchZVel = (arena->gameMode == GameMode::DROPSHOT)
		? Dropshot::BALL_LAUNCH_Z_VEL : BALL_HOOPS_LAUNCH_Z_VEL;
	cc.ballMassBT = arena->ball->GetMass();
	cc.hoopsZScaleGround = BALL_CAR_EXTRA_IMPULSE_Z_SCALE_HOOPS_GROUND;
	cc.hoopsZScaleThresh = BALL_CAR_EXTRA_IMPULSE_Z_SCALE_HOOPS_NORMAL_Z_THRESH;
	cc.dsMinDownwardSpeedToDamage = Dropshot::MIN_DOWNWARD_SPEED_TO_DAMAGE;
	cc.dsMinChargeHitSpeed = Dropshot::MIN_CHARGE_HIT_SPEED;
	cc.dsMinAbsorbedForCharge = Dropshot::MIN_ABSORBED_FORCE_FOR_CHARGE;
	cc.dsMinAbsorbedForSupercharge = Dropshot::MIN_ABSORBED_FORCE_FOR_SUPERCHARGE;
	cc.dsMinDamageInterval = Dropshot::MIN_DAMAGE_INTERVAL;
	if (arena->gameMode == GameMode::DROPSHOT) {
		cc.numTiles = (int)arena->_worldDropshotTileRBs.size();
		cc.tileMargin = arena->_worldDropshotTileRBs[0]->getCollisionShape()->getMargin();
	}
	const btBoxShape& box = car->_childHitboxShape;
	cc.carBoxHalfExtNoMargin = ToGpuB(box.getHalfExtentsWithoutMargin());
	cc.carBoxMargin = box.getMargin();
	cc.carChildOffset = ToGpuB(car->_compoundShape.getChildTransform(0).getOrigin());
	cc.ballBreakingThreshold = ballShape->getContactBreakingThreshold(gContactBreakingThreshold);
	cc.carBreakingThreshold = comp.getContactBreakingThreshold(gContactBreakingThreshold);
	cc.ballFriction = arena->ball->_rigidBody.getFriction();
	cc.ballRestitution = arena->ball->_rigidBody.getRestitution();
	cc.carFriction = car->_rigidBody.getFriction();
	cc.carRestitution = car->_rigidBody.getRestitution();
	cc.ballFilterGroup = arena->ball->_rigidBody.getBroadphaseProxy()->m_collisionFilterGroup;
	cc.ballFilterMask = arena->ball->_rigidBody.getBroadphaseProxy()->m_collisionFilterMask;
	cc.carFilterGroup = car->_rigidBody.getBroadphaseProxy()->m_collisionFilterGroup;
	cc.carFilterMask = car->_rigidBody.getBroadphaseProxy()->m_collisionFilterMask;
	cc.numStatics = (int)statics.size();
	cc.gameMode = (int)arena->gameMode;
	const MutatorConfig& mc = arena->GetMutatorConfig();
	cc.demoMode = (int)mc.demoMode;
	cc.enableTeamDemos = mc.enableTeamDemos;
	cc.respawnDelay = mc.respawnDelay;
	cc.bumpCooldownTime = mc.bumpCooldownTime;
	cc.bumpForceScale = mc.bumpForceScale;
	cc.ballHitExtraForceScale = mc.ballHitExtraForceScale;
	cc.carWorldFriction = mc.carWorldFriction;
	cc.carWorldRestitution = mc.carWorldRestitution;
	cc.carBallFriction = CARBALL_COLLISION_FRICTION;
	cc.carBallRestitution = CARBALL_COLLISION_RESTITUTION;
	cc.carCarFriction = CARCAR_COLLISION_FRICTION;
	cc.carCarRestitution = CARCAR_COLLISION_RESTITUTION;
	cc.bumpMinForwardDist = BUMP_MIN_FORWARD_DIST;
	cc.ballCarExtraZScale = BALL_CAR_EXTRA_IMPULSE_Z_SCALE;
	cc.ballCarExtraForwardScale = BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE;
	cc.ballCarExtraMaxDeltaVelUU = BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL_UU;
	cc.ballExtraImpulseFactorCurve = ExtractCurve(BALL_CAR_EXTRA_IMPULSE_FACTOR_CURVE);
	cc.bumpVelGroundCurve = ExtractCurve(BUMP_VEL_AMOUNT_GROUND_CURVE);
	cc.bumpVelAirCurve = ExtractCurve(BUMP_VEL_AMOUNT_AIR_CURVE);
	cc.bumpUpwardVelCurve = ExtractCurve(BUMP_UPWARD_VEL_AMOUNT_CURVE);
	cc.hsTargetY = Heatseeker::TARGET_Y;
	cc.hsTargetZ = Heatseeker::TARGET_Z;
	cc.hsHorizontalBlend = Heatseeker::HORIZONTAL_BLEND;
	cc.hsVerticalBlend = Heatseeker::VERTICAL_BLEND;
	cc.hsSpeedBlend = Heatseeker::SPEED_BLEND;
	cc.hsMaxTurnPitch = Heatseeker::MAX_TURN_PITCH;
	cc.hsMaxSpeed = Heatseeker::MAX_SPEED;
	cc.hsTargetSpeedIncrement = Heatseeker::TARGET_SPEED_INCREMENT;
	cc.hsMinSpeedupInterval = Heatseeker::MIN_SPEEDUP_INTERVAL;
	cc.hsWallBounceChangeYThresh = Heatseeker::WALL_BOUNCE_CHANGE_Y_THRESH;
	cc.hsWallBounceChangeYNormal = Heatseeker::WALL_BOUNCE_CHANGE_Y_NORMAL;
	cc.hsWallBounceForceScale = Heatseeker::WALL_BOUNCE_FORCE_SCALE;
	cc.hsWallBounceUpFrac = Heatseeker::WALL_BOUNCE_UP_FRAC;
	cc.arenaExtentY = ARENA_EXTENT_Y;
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
	{
		std::vector<rscuda::GpuMeshBvh> bvhs;
		std::vector<rscuda::GpuBvhNode> bvhNodes;
		std::vector<rscuda::GpuBvhSubtree> bvhSubtrees;
		std::vector<rscuda::GpuTriInfo> triInfos;
		std::vector<float> meshMargins;
		ExtractMeshBvhs(arena, bvhs, bvhNodes, bvhSubtrees, triInfos, meshMargins);
		rscuda::UploadMeshBvhs(batch, bvhs.data(), (int)bvhs.size(),
			bvhNodes.data(), (int)bvhNodes.size(),
			bvhSubtrees.data(), (int)bvhSubtrees.size(),
			triInfos.data(), (int)triInfos.size(), meshMargins.data());
	}

	// Dynamic candidates: ball (sphere) + cars (compound box), with live transforms
	// and proxy aabbMin (drives btRSBroadphase voxel cell membership)
	std::vector<rscuda::GpuDynBody> dyn;
	{
		rscuda::GpuDynBody d = {};
		btRigidBody& rb = arena->ball->_rigidBody;
		if (arena->ball->IsSphere()) {
			d.kind = 1; // sphere
			d.sphereRadius = ((btSphereShape*)rb.getCollisionShape())->getRadius();
		} else {
			d.kind = 4; // hull (snowday puck); sphereRadius slot = margin
			d.sphereRadius = rb.getCollisionShape()->getMargin();
			auto* hull = (btConvexHullShape*)arena->ball->_collisionShape;
			std::vector<rscuda::GpuVec3> pts(hull->getNumPoints());
			for (int i = 0; i < hull->getNumPoints(); i++)
				pts[i] = ToGpuB(hull->getUnscaledPoints()[i]);
			rscuda::UploadBallHull(batch, pts.data(), (int)pts.size());
		}
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

static void ExtractTiles(Arena* arena,
	std::vector<rscuda::GpuTileShape>& tiles, rscuda::GpuTileTables& tables);

static int RunGpuCompare(const Args& args) {
	if (!rscuda::Available()) {
		fprintf(stderr, "GPU_COMPARE: FAIL (no usable CUDA device)\n");
		return 2;
	}

	std::vector<CarSlot> slots;
	Arena* arena = SetupArena(args, 0, slots);
	const int numPads = (int)arena->GetBoostPads().size();
	const int numBodies = 1 + args.cars;

	if (!args.noBallMotion) { // wake the ball with seed-derived motion so collision compare
	  // covers ball-world and car-ball contacts (kickoff ball is asleep -> no manifolds)
		BallState bs = arena->ball->GetState();
		Rng rng(MixSeed(args.seed, 0xBA11BA11ull));
		bs.vel = Vec(rng.NextSigned() * 1000.f, rng.NextSigned() * 1000.f,
			fabsf(rng.NextSigned()) * 800.f);
		bs.angVel = Vec(rng.NextSigned() * 2.f, rng.NextSigned() * 2.f, rng.NextSigned() * 2.f);
		arena->ball->SetState(bs);
		arena->ball->_rigidBody.activate(true);
	}

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

		std::vector<rscuda::GpuMeshBvh> bvhs;
		std::vector<rscuda::GpuBvhNode> bvhNodes;
		std::vector<rscuda::GpuBvhSubtree> bvhSubtrees;
		std::vector<rscuda::GpuTriInfo> triInfos;
		std::vector<float> meshMargins;
		ExtractMeshBvhs(arena, bvhs, bvhNodes, bvhSubtrees, triInfos, meshMargins);
		rscuda::UploadMeshBvhs(batch, bvhs.data(), (int)bvhs.size(),
			bvhNodes.data(), (int)bvhNodes.size(),
			bvhSubtrees.data(), (int)bvhSubtrees.size(),
			triInfos.data(), (int)triInfos.size(), meshMargins.data());
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

	// Statics' proxy aabbs/cell smear change ONCE at the first updateAabbs
	// (forceUpdateAllAabbs re-runs them through setAabb WITH the +-0.02 expansion,
	// which createProxy lacked). Apply that update now so the extracted static
	// state matches what every tick's pair-gen will actually see. Idempotent for
	// statics; dyn proxies are re-extracted as the lockstep baseline below.
	arena->_bulletWorld.updateAabbs();

	// collision orchestration data + host-lockstep proxy state (regSeq maintained by
	// the CPU re-registration rule: cellIdx change && >1 dyn proxies)
	std::vector<rscuda::GpuDynProxy> hostProxies;
	int hostRegCounter = 0;
	{
		rscuda::GpuCollideConsts ccons;
		std::vector<rscuda::GpuStaticObj> statics;
		std::vector<int> cellStart, cellList;
		ExtractCollideData(arena, slots, ccons, statics, cellStart, cellList,
			hostProxies, hostRegCounter);
		// TEMP diagnostic
		{
			int c0 = hostProxies[0].cellIdx;
			printf("DBG: ball cell %d statics [%d..%d):", c0, cellStart[c0], cellStart[c0 + 1]);
			for (int it = cellStart[c0]; it < cellStart[c0 + 1]; it++)
				printf(" %d", cellList[it]);
			printf("\n  totalCells=%zu listLen=%zu numStatics=%zu ballGroup=%d ballMask=%d s16(g=%d,m=%d,kind=%d) aabb z[%.3f,%.3f]\n",
				cellStart.size() - 1, cellList.size(), statics.size(),
				ccons.ballFilterGroup, ccons.ballFilterMask,
				statics.size() > 16 ? statics[16].group : 0,
				statics.size() > 16 ? statics[16].mask : 0,
				statics.size() > 16 ? statics[16].kind : -1,
				statics.size() > 16 ? statics[16].aabbMin.z : 0,
				statics.size() > 16 ? statics[16].aabbMax.z : 0);
		}
		rscuda::UploadCollideData(batch, ccons, statics.data(), (int)statics.size(),
			cellStart.data(), (int)cellStart.size() - 1,
			cellList.data(), (int)cellList.size(), grid);
		if (!arena->ball->IsSphere()) { // snowday puck hull
			auto* hull = (btConvexHullShape*)arena->ball->_collisionShape;
			std::vector<rscuda::GpuVec3> pts(hull->getNumPoints());
			for (int i = 0; i < hull->getNumPoints(); i++)
				pts[i] = ToGpuB(hull->getUnscaledPoints()[i]);
			rscuda::UploadBallHull(batch, pts.data(), (int)pts.size());
		}
		if (args.gameMode == GameMode::DROPSHOT) { // tile shapes + per-arena state
			std::vector<rscuda::GpuTileShape> tiles;
			rscuda::GpuTileTables tables;
			ExtractTiles(arena, tiles, tables);
			rscuda::UploadTiles(batch, tiles.data(), (int)tiles.size(), tables);
		}
	}
	// CPU manifold body -> device id (>=0 dyn slot, <0 static handle -1-h)
	auto collideBodyId = [&](const void* obj) -> int {
		if (obj == &arena->ball->_rigidBody) return 0;
		for (size_t c2 = 0; c2 < slots.size(); c2++)
			if (obj == &slots[c2].car->_rigidBody) return 1 + (int)c2;
		for (size_t w = 0; w < arena->_worldCollisionRBs.size(); w++)
			if (obj == arena->_worldCollisionRBs[w]) return -1 - (int)w;
		return -1000000;
	};

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
				bad |= failF("scr.contactPoint.y", w, afterFirst.wheels[w].contactPointWS[1], wOut[w].contactPointWS.y);
				bad |= failF("scr.contactPoint.z", w, afterFirst.wheels[w].contactPointWS[2], wOut[w].contactPointWS.z);
				bad |= failF("scr.hardPoint.x", w, afterFirst.wheels[w].hardPointWS[0], wOut[w].hardPointWS.x);
				bad |= failF("scr.hardPoint.y", w, afterFirst.wheels[w].hardPointWS[1], wOut[w].hardPointWS.y);
				bad |= failF("scr.hardPoint.z", w, afterFirst.wheels[w].hardPointWS[2], wOut[w].hardPointWS.z);
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

		// ---- collision segment: proxies + manifolds vs CPU AFTER_COLLISION ----
		{
			mapSnaps(rscap::points[rscap::AFTER_PREDICT], preState);
			rscuda::UploadArenaState(batch, 0, preState.data(), carsDummy.data(), padsDummy.data());
			rscuda::UploadDynProxies(batch, 0, hostProxies.data(), hostRegCounter);
			rscuda::SetTickCount(batch, arena->tickCount - 1); // value during the tick
			rscuda::RunPhaseSegment(batch, rscuda::GPU_PHASE_COLLISION, rscuda::GPU_PHASE_COLLISION);

			std::vector<rscuda::GpuManifoldOut> gm(GPU_MAX_MANIFOLDS);
			int gn = 0;
			std::vector<rscuda::GpuDynProxy> gp(numBodies);
			rscuda::GpuBallGameState gball;
			rscuda::DownloadCollideResults(batch, 0, gm.data(), &gn, gp.data(), &gball);

			bool bad = false;
			auto cpuProxy = [&](int d) -> const rscap::ProxySnap* {
				for (const auto& s : rscap::proxies)
					if (s.rb == rbOrder[d])
						return &s;
				return nullptr;
			};
			for (int d = 0; d < numBodies; d++) {
				const rscap::ProxySnap* ps = cpuProxy(d);
				if (!ps)
					continue;
				if (memcmp(ps->aabbMin, &gp[d].aabbMin, 12) || memcmp(ps->aabbMax, &gp[d].aabbMax, 12) ||
					ps->cellIdx != gp[d].cellIdx || ps->iIdx != gp[d].i ||
					ps->jIdx != gp[d].j || ps->kIdx != gp[d].k) {
					printf("GPU_COMPARE: FAIL tick %d collision proxy %d:\n"
						"  cpu aabbMin(%.9g,%.9g,%.9g) aabbMax(%.9g,%.9g,%.9g) cell %d ijk(%d,%d,%d)\n"
						"  gpu aabbMin(%.9g,%.9g,%.9g) aabbMax(%.9g,%.9g,%.9g) cell %d ijk(%d,%d,%d)\n",
						tick, d,
						ps->aabbMin[0], ps->aabbMin[1], ps->aabbMin[2],
						ps->aabbMax[0], ps->aabbMax[1], ps->aabbMax[2],
						ps->cellIdx, ps->iIdx, ps->jIdx, ps->kIdx,
						gp[d].aabbMin.x, gp[d].aabbMin.y, gp[d].aabbMin.z,
						gp[d].aabbMax.x, gp[d].aabbMax.y, gp[d].aabbMax.z,
						gp[d].cellIdx, gp[d].i, gp[d].j, gp[d].k);
					bad = true;
				}
			}

			if ((int)rscap::manifolds.size() != gn) {
				printf("GPU_COMPARE: FAIL tick %d collision manifold count: cpu %zu gpu %d\n",
					tick, rscap::manifolds.size(), gn);
				for (size_t i = 0; i < rscap::manifolds.size(); i++)
					printf("  cpu m%zu: %d vs %d (%d pts)\n", i,
						collideBodyId(rscap::manifolds[i].body0),
						collideBodyId(rscap::manifolds[i].body1),
						rscap::manifolds[i].numContacts);
				for (int i = 0; i < gn; i++)
					printf("  gpu m%d: %d vs %d (%d pts)\n", i, gm[i].bodyA, gm[i].bodyB, gm[i].numContacts);
				int dbg[2 * 96 + 1];
				rscuda::DownloadCollideDebug(batch, 0, dbg);
				printf("  gpu pairs (%d):", dbg[0]);
				for (int i = 0; i < dbg[0] && i < 96; i++)
					printf(" (%d,%d)", dbg[1 + 2 * i], dbg[2 + 2 * i]);
				printf("\n");
				float pz;
				memcpy(&pz, &dbg[106], 4);
				printf("  gpu d0: cell %d list[%d..%d) first %d g %d m %d aabbMin.z %.9g nb %d\n",
					dbg[100], dbg[101], dbg[102], dbg[103], dbg[104], dbg[105], pz, dbg[107]);
				for (int b2 = 0; b2 < numBodies; b2++)
					printf("  body %d activation(AFTER_PREDICT)=%d deact=%.6g\n",
						b2, preState[b2].activationState, preState[b2].deactivationTime);
				bad = true;
			}
			int nCmp = (int)rscap::manifolds.size() < gn ? (int)rscap::manifolds.size() : gn;
			for (int i = 0; i < nCmp && !bad; i++) {
				const rscap::ManifoldSnap& cm = rscap::manifolds[i];
				int cA = collideBodyId(cm.body0), cB = collideBodyId(cm.body1);
				const rscuda::GpuManifoldOut& g = gm[i];
				if (cA != g.bodyA || cB != g.bodyB || cm.numContacts != g.numContacts) {
					printf("GPU_COMPARE: FAIL tick %d collision manifold %d: cpu (%d vs %d, %d pts) gpu (%d vs %d, %d pts)\n",
						tick, i, cA, cB, cm.numContacts, g.bodyA, g.bodyB, g.numContacts);
					for (int c = 0; c < cm.numContacts && c < 4; c++)
						printf("  cpu pt%d d=%.9g n=(%.9g,%.9g,%.9g) wB=(%.9g,%.9g,%.9g) part(%d,%d) idx(%d,%d)\n",
							c, cm.pts[c].distance, cm.pts[c].normalWorldOnB[0], cm.pts[c].normalWorldOnB[1],
							cm.pts[c].normalWorldOnB[2], cm.pts[c].positionWorldOnB[0], cm.pts[c].positionWorldOnB[1],
							cm.pts[c].positionWorldOnB[2], cm.pts[c].partId0, cm.pts[c].partId1,
							cm.pts[c].index0, cm.pts[c].index1);
					for (int c = 0; c < g.numContacts && c < 4; c++)
						printf("  gpu pt%d d=%.9g n=(%.9g,%.9g,%.9g) wB=(%.9g,%.9g,%.9g) part(%d,%d) idx(%d,%d)\n",
							c, g.pts[c].distance, g.pts[c].normalWorldOnB.x, g.pts[c].normalWorldOnB.y,
							g.pts[c].normalWorldOnB.z, g.pts[c].positionWorldOnB.x, g.pts[c].positionWorldOnB.y,
							g.pts[c].positionWorldOnB.z, g.pts[c].partId0, g.pts[c].partId1,
							g.pts[c].index0, g.pts[c].index1);
					{
						int dbg2[2 * 96 + 1];
						rscuda::DownloadCollideDebug(batch, 0, dbg2);
						printf("  gpu last car-mesh walk: handle %d leaves %d triAabbPass %d contactsAdded %d\n",
							dbg2[110], dbg2[111], dbg2[112], dbg2[113]);
						float dd, bb2;
						memcpy(&dd, &dbg2[122], 4);
						memcpy(&bb2, &dbg2[123], 4);
						printf("  gpu last car-plane: handle %d hit %d dist %.9g breaking %.9g\n",
							dbg2[120], dbg2[121], dd, bb2);
					}
					bad = true;
					break;
				}
				for (int c = 0; c < cm.numContacts && c < 4; c++) {
					const rscap::ContactSnap& cp = cm.pts[c];
					const rscuda::GpuContactOut& gc = g.pts[c];
					auto cmpV = [&](const char* what, const float* cv, const rscuda::GpuVec3& gv) {
						if (memcmp(cv, &gv, 12) != 0) {
							printf("GPU_COMPARE: FAIL tick %d collision m%d pt%d %s: cpu (%.9g,%.9g,%.9g) gpu (%.9g,%.9g,%.9g)\n",
								tick, i, c, what, cv[0], cv[1], cv[2], gv.x, gv.y, gv.z);
							bad = true;
						}
					};
					auto cmpF = [&](const char* what, float cv, float gv) {
						if (memcmp(&cv, &gv, 4) != 0) {
							printf("GPU_COMPARE: FAIL tick %d collision m%d pt%d %s: cpu %.9g gpu %.9g\n",
								tick, i, c, what, cv, gv);
							bad = true;
						}
					};
					cmpV("localPointA", cp.localPointA, gc.localPointA);
					cmpV("localPointB", cp.localPointB, gc.localPointB);
					cmpV("positionWorldOnA", cp.positionWorldOnA, gc.positionWorldOnA);
					cmpV("positionWorldOnB", cp.positionWorldOnB, gc.positionWorldOnB);
					cmpV("normalWorldOnB", cp.normalWorldOnB, gc.normalWorldOnB);
					cmpF("distance", cp.distance, gc.distance);
					cmpF("combinedFriction", cp.combinedFriction, gc.combinedFriction);
					cmpF("combinedRestitution", cp.combinedRestitution, gc.combinedRestitution);
					if (cp.partId0 != gc.partId0 || cp.partId1 != gc.partId1 ||
						cp.index0 != gc.index0 || cp.index1 != gc.index1 ||
						(int)cp.isSpecial != (int)gc.isSpecial) {
						printf("GPU_COMPARE: FAIL tick %d collision m%d pt%d ids: cpu part(%d,%d) idx(%d,%d) sp%d gpu part(%d,%d) idx(%d,%d) sp%d\n",
							tick, i, c, cp.partId0, cp.partId1, cp.index0, cp.index1, cp.isSpecial,
							gc.partId0, gc.partId1, gc.index0, gc.index1, gc.isSpecial);
						bad = true;
					}
				}
			}
			if (bad)
				return 1;

			// ---- solve segment: chained collision+solve from AFTER_PREDICT,
			// compared against CPU AFTER_SOLVE body snaps (velocities, transforms
			// [split impulse], activation [island wake/sleep]) ----
			{
				mapSnaps(rscap::points[rscap::AFTER_PREDICT], preState);
				mapSnaps(rscap::points[rscap::AFTER_SOLVE], refState);
				rscuda::UploadArenaState(batch, 0, preState.data(), carsDummy.data(), padsDummy.data());
				rscuda::UploadDynProxies(batch, 0, hostProxies.data(), hostRegCounter);
				rscuda::RunPhaseSegment(batch, rscuda::GPU_PHASE_COLLISION, rscuda::GPU_PHASE_SOLVE);
				rscuda::DownloadArenaState(batch, 0, gotState.data(), carsDummy.data(), padsDummy.data());
				for (int bIdx = 0; bIdx < numBodies; bIdx++) {
					int df = BodyStateDiff(refState[bIdx], gotState[bIdx]);
					if (df >= 0) {
						const uint32_t* cr = (const uint32_t*)&refState[bIdx];
						const uint32_t* gr = (const uint32_t*)&gotState[bIdx];
						printf("GPU_COMPARE: FAIL tick %d phase solve body %d (%s) float[%d]: cpu 0x%08X gpu 0x%08X\n",
							tick, bIdx, bIdx == 0 ? "ball" : "car", df, cr[df], gr[df]);
						for (size_t mi = 0; mi < rscap::manifolds.size(); mi++) {
							const rscap::ManifoldSnap& cm = rscap::manifolds[mi];
							printf("  manifold %zu: %d vs %d, %d pts", mi,
								collideBodyId(cm.body0), collideBodyId(cm.body1), cm.numContacts);
							for (int c = 0; c < cm.numContacts && c < 4; c++)
								printf(" [d=%.6g sp%d]", cm.pts[c].distance, cm.pts[c].isSpecial);
							printf("\n");
						}
						for (int b2 = 0; b2 < numBodies; b2++)
							printf("  body %d act(predict)=%d act(cpu solve)=%d act(gpu)=%d\n", b2,
								preState[b2].activationState, refState[b2].activationState,
								gotState[b2].activationState);
						// row-level diff: CPU capture vs GPU scratch
						{
							std::vector<rscuda::GpuSolverRowDl> gc(404), gf(404);
							int nc = 0, nf = 0;
							rscuda::DownloadSolveDebug(batch, 0, gc.data(), &nc, gf.data(), &nf);
							printf("  rows: cpu %zu contact / %zu friction; gpu %d / %d\n",
								rscap::solverContactRows.size(), rscap::solverFrictionRows.size(), nc, nf);
							auto dumpRows = [&](const char* kind,
								const std::vector<rscap::SolverRowSnap>& cpu,
								const std::vector<rscuda::GpuSolverRowDl>& gpu, int ng) {
								for (size_t r = 0; r < cpu.size() && (int)r < ng; r++) {
									const rscap::SolverRowSnap& c = cpu[r];
									const rscuda::GpuSolverRowDl& g = gpu[r];
									auto neq = [](float a2, float b2) { return memcmp(&a2, &b2, 4) != 0; };
									bool diff = c.bodyA != g.bodyA || c.bodyB != g.bodyB ||
										neq(c.jacDiagABInv, g.jacDiagABInv) || neq(c.rhs, g.rhs) ||
										neq(c.rhsPenetration, g.rhsPenetration) ||
										neq(c.appliedImpulse, g.appliedImpulse) ||
										neq(c.normal1[0], g.normal1.x) || neq(c.normal1[1], g.normal1.y) ||
										neq(c.normal1[2], g.normal1.z) ||
										(int)c.isSpecial != (int)g.isSpecial;
									if (diff)
										printf("  %s row %zu DIFF: cpu(b %d/%d jac %.9g rhs %.9g rhsP %.9g app %.9g n %.9g,%.9g,%.9g sp%d) gpu(b %d/%d jac %.9g rhs %.9g rhsP %.9g app %.9g n %.9g,%.9g,%.9g sp%d)\n",
											kind, r, c.bodyA, c.bodyB, c.jacDiagABInv, c.rhs, c.rhsPenetration,
											c.appliedImpulse, c.normal1[0], c.normal1[1], c.normal1[2], c.isSpecial,
											g.bodyA, g.bodyB, g.jacDiagABInv, g.rhs, g.rhsPenetration,
											g.appliedImpulse, g.normal1.x, g.normal1.y, g.normal1.z, g.isSpecial);
								}
							};
							dumpRows("contact", rscap::solverContactRows, gc, nc);
							dumpRows("friction", rscap::solverFrictionRows, gf, nf);
						}
						return 1;
					}
				}
			}

			// advance host proxy lockstep state from CPU truth (regSeq rule: cell
			// change && >1 dyn proxies; handle order)
			for (int d = 0; d < numBodies; d++) {
				const rscap::ProxySnap* ps = cpuProxy(d);
				if (!ps)
					continue;
				bool cellChanged = hostProxies[d].cellIdx != ps->cellIdx;
				memcpy(&hostProxies[d].aabbMin, ps->aabbMin, 12);
				memcpy(&hostProxies[d].aabbMax, ps->aabbMax, 12);
				hostProxies[d].cellIdx = ps->cellIdx;
				hostProxies[d].i = ps->iIdx;
				hostProxies[d].j = ps->jIdx;
				hostProxies[d].k = ps->kIdx;
				if (cellChanged && numBodies > 1)
					hostProxies[d].regSeq = ++hostRegCounter;
			}
		}
	}

	printf("GPU_COMPARE: PASS (%d ticks, phases: car-pretick, predict, collision, solve, integrate, activation; %d bodies bit-exact)\n",
		args.gpuCompare, numBodies);
	rscuda::DestroyBatch(batch);
	delete arena;
	return 0;
}

// Game-tail constants + pad configs + dyn shape templates for the full GPU step
static void ExtractTailData(Arena* arena, std::vector<CarSlot>& slots,
	rscuda::GpuTailConsts& tc, std::vector<rscuda::GpuPadConfig>& padConfigs,
	std::vector<rscuda::GpuDynBody>& dynTemplates) {
	using namespace RLConst;
	tc = {};
	tc.supersonicStartSpeed = SUPERSONIC_START_SPEED;
	tc.supersonicMaintainMinSpeed = SUPERSONIC_MAINTAIN_MIN_SPEED;
	tc.supersonicMaintainMaxTime = SUPERSONIC_MAINTAIN_MAX_TIME;
	tc.carMaxSpeedBT = CAR_MAX_SPEED * UU_TO_BT;
	tc.carMaxAngSpeed = CAR_MAX_ANG_SPEED;
	const MutatorConfig& mc = arena->GetMutatorConfig();
	tc.ballMaxSpeedBT = mc.ballMaxSpeed * UU_TO_BT;
	tc.ballMaxAngSpeed = BALL_MAX_ANG_SPEED;
	tc.boostMax = BOOST_MAX;
	tc.boostAmountBig = BoostPads::BOOST_AMOUNT_BIG;
	tc.boostAmountSmall = BoostPads::BOOST_AMOUNT_SMALL;
	tc.padCooldownBig = mc.boostPadCooldown_Big;
	tc.padCooldownSmall = mc.boostPadCooldown_Small;
	tc.padGridExtentZ = BoostPadGrid::EXTENT_Z;
	tc.padGridCellSizeX = BoostPadGrid::CELL_SIZE_X;
	tc.padGridCellSizeY = BoostPadGrid::CELL_SIZE_Y;
	tc.padGridCellsX = BoostPadGrid::CELLS_X;
	tc.padGridCellsY = BoostPadGrid::CELLS_Y;
	tc.cylHeightBT = BoostPads::CYL_HEIGHT * UU_TO_BT;

	{ // Car::Respawn spawn table (mode-appropriate) + spawn constants
		const CarSpawnPos* respawns;
		switch (arena->gameMode) {
		case GameMode::HOOPS: respawns = CAR_RESPAWN_LOCATIONS_HOOPS; break;
		case GameMode::DROPSHOT: respawns = CAR_RESPAWN_LOCATIONS_DROPSHOT; break;
		default: respawns = CAR_RESPAWN_LOCATIONS_SOCCAR;
		}
		for (int i = 0; i < CAR_RESPAWN_LOCATION_AMOUNT; i++) {
			tc.respawnX[i] = respawns[i].x;
			tc.respawnY[i] = respawns[i].y;
			tc.respawnYaw[i] = respawns[i].yawAng;
		}
		tc.carRespawnZ = CAR_RESPAWN_Z;
		tc.carSpawnBoostAmount = mc.carSpawnBoostAmount;
	}

	const auto& padList = arena->GetBoostPads();
	for (BoostPad* pad : padList) {
		rscuda::GpuPadConfig pc = {};
		pc.posBT = ToGpu(pad->_posBT);
		pc.boxMinBT = ToGpu(pad->_boxMinBT);
		pc.boxMaxBT = ToGpu(pad->_boxMaxBT);
		pc.cylRadBT = (pad->config.isBig ? BoostPads::CYL_RAD_BIG : BoostPads::CYL_RAD_SMALL) * UU_TO_BT;
		pc.isBig = pad->config.isBig;
		padConfigs.push_back(pc);
	}
	for (int i = 0; i < BoostPadGrid::CELLS_X; i++) {
		for (int j = 0; j < BoostPadGrid::CELLS_Y; j++) {
			BoostPad* p = arena->_boostPadGrid.pads[i][j];
			int idx = -1;
			if (p)
				for (size_t k = 0; k < padList.size(); k++)
					if (padList[k] == p) { idx = (int)k; break; }
			tc.padGridMap[i * BoostPadGrid::CELLS_Y + j] = (int16_t)idx;
		}
	}

	{ // dyn shape/filter templates (transforms filled per tick on device)
		rscuda::GpuDynBody t = {};
		btRigidBody& rb = arena->ball->_rigidBody;
		if (arena->ball->IsSphere()) {
			t.kind = 1;
			t.sphereRadius = ((btSphereShape*)rb.getCollisionShape())->getRadius();
		} else {
			t.kind = 4; // CU_SHAPE_HULL; sphereRadius slot = collision margin
			t.sphereRadius = rb.getCollisionShape()->getMargin();
		}
		t.group = rb.getBroadphaseProxy()->m_collisionFilterGroup;
		t.mask = rb.getBroadphaseProxy()->m_collisionFilterMask;
		t.invMass = rb.getInvMass();
		t.invInertiaDiagLocal = ToGpuB(rb.getInvInertiaDiagLocal());
		t.hasContactResponse = rb.hasContactResponse();
		dynTemplates.push_back(t);
		for (CarSlot& slot : slots) {
			rscuda::GpuDynBody c = {};
			btRigidBody& crb = slot.car->_rigidBody;
			c.kind = 2;
			const btBoxShape& box = slot.car->_childHitboxShape;
			btVector3 he = box.getHalfExtentsWithoutMargin();
			float m = box.getMargin();
			c.boxHalfExtWithMargin = { he.x() + m, he.y() + m, he.z() + m };
			c.childOffset = ToGpuB(slot.car->_compoundShape.getChildTransform(0).getOrigin());
			c.group = crb.getBroadphaseProxy()->m_collisionFilterGroup;
			c.mask = crb.getBroadphaseProxy()->m_collisionFilterMask;
			c.invMass = crb.getInvMass();
			c.invInertiaDiagLocal = ToGpuB(crb.getInvInertiaDiagLocal());
			c.hasContactResponse = crb.hasContactResponse();
			dynTemplates.push_back(c);
		}
	}
}

// Dropshot tile shapes (6-pt hulls, world-space points; rb origin (0,0,-margin)) +
// neighbor tables. Local aabbs recomputed exactly as recalcLocalAabb (see hull note).
static void ExtractTiles(Arena* arena,
	std::vector<rscuda::GpuTileShape>& tiles, rscuda::GpuTileTables& tables) {
	for (btRigidBody* rb : arena->_worldDropshotTileRBs) {
		auto* hull = (btConvexHullShape*)rb->getCollisionShape();
		rscuda::GpuTileShape ts = {};
		for (int i = 0; i < 6; i++)
			ts.pts[i] = ToGpuB(hull->getUnscaledPoints()[i]);
		static const btVector3 dirs[6] = {
			btVector3(1, 0, 0), btVector3(0, 1, 0), btVector3(0, 0, 1),
			btVector3(-1, 0, 0), btVector3(0, -1, 0), btVector3(0, 0, -1) };
		btVector3 sup[6] = {};
		hull->batchedUnitVectorGetSupportingVertexWithoutMargin(dirs, sup, 6);
		btVector3 lmn, lmx;
		for (int i = 0; i < 3; i++) {
			lmx[i] = sup[i][i] + hull->getMargin();
			lmn[i] = sup[i + 3][i] - hull->getMargin();
		}
		ts.localAabbMin = ToGpuB(lmn);
		ts.localAabbMax = ToGpuB(lmx);
		ts.origin = ToGpuB(rb->getWorldTransform().getOrigin());
		tiles.push_back(ts);
	}
	tables = {};
	for (int i = 0; i < RLConst::Dropshot::NUM_TILES_PER_TEAM; i++) {
		auto n1 = DropshotTiles::GetNeighborIndices(i, 2);
		auto n2 = DropshotTiles::GetNeighborIndices(i, 3);
		tables.n1Cnt[i] = (int8_t)n1.size();
		for (size_t k = 0; k < n1.size(); k++)
			tables.n1[i][k] = (int8_t)n1[k];
		tables.n2Cnt[i] = (int8_t)n2.size();
		for (size_t k = 0; k < n2.size(); k++)
			tables.n2[i][k] = (int8_t)n2[k];
	}
}

static uint64_t RunGpu(const Args& args, double& elapsedS) {
	// Set up all arenas FIRST (state init only - no physics is stepped on CPU).
	// This must happen before ANY CUDA call so the allocation history (and with it
	// the physics-relevant _cars iteration order) matches RunCpu exactly.
	std::vector<Arena*> arenas(args.arenas);
	std::vector<std::vector<CarSlot>> slotsPerArena(args.arenas);
	for (int a = 0; a < args.arenas; a++)
		arenas[a] = SetupArena(args, a, slotsPerArena[a]);

	if (!rscuda::Available()) {
		fprintf(stderr, "GPU_STEP: FAIL (no usable CUDA device)\n");
		exit(2);
	}

	const int numPads = (int)arenas[0]->GetBoostPads().size();
	rscuda::GpuBatch* batch = rscuda::CreateBatch(args.arenas, args.cars, numPads);
	UploadConsts(batch, arenas[0], slotsPerArena[0]);
	rscuda::UploadCarConsts(batch, ExtractCarConsts(arenas[0], slotsPerArena[0][0].car));

	// world geometry + collision orchestration data (identical across arenas).
	// updateAabbs() first: statics reach their steady-state proxy aabbs/cell smear.
	for (int a = 0; a < args.arenas; a++)
		arenas[a]->_bulletWorld.updateAabbs();
	{
		std::vector<rscuda::GpuPlane> planes;
		std::vector<rscuda::GpuMeshInfo> meshes;
		std::vector<rscuda::GpuVec3> tris;
		ExtractWorldGeo(arenas[0], planes, meshes, tris);
		rscuda::UploadWorldGeo(batch, planes.data(), (int)planes.size(),
			meshes.data(), (int)meshes.size(), tris.data(), (int)(tris.size() / 3));
		std::vector<rscuda::GpuMeshBvh> bvhs;
		std::vector<rscuda::GpuBvhNode> bvhNodes;
		std::vector<rscuda::GpuBvhSubtree> bvhSubtrees;
		std::vector<rscuda::GpuTriInfo> triInfos;
		std::vector<float> meshMargins;
		ExtractMeshBvhs(arenas[0], bvhs, bvhNodes, bvhSubtrees, triInfos, meshMargins);
		rscuda::UploadMeshBvhs(batch, bvhs.data(), (int)bvhs.size(),
			bvhNodes.data(), (int)bvhNodes.size(),
			bvhSubtrees.data(), (int)bvhSubtrees.size(),
			triInfos.data(), (int)triInfos.size(), meshMargins.data());

		rscuda::GpuGridParams grid = {};
		auto* bp = (btRSBroadphase*)arenas[0]->_bulletWorldParams.broadphase;
		grid.minPos = ToGpuB(bp->minPos);
		grid.cellSize = bp->cellSize;
		grid.cellSizeSq = bp->cellSizeSq;
		grid.cellsX = bp->cellsX;
		grid.cellsY = bp->cellsY;
		grid.cellsZ = bp->cellsZ;
		rscuda::GpuCollideConsts ccons;
		std::vector<rscuda::GpuStaticObj> statics;
		std::vector<int> cellStart, cellList;
		std::vector<rscuda::GpuDynProxy> proxies0;
		int regCounter0 = 0;
		ExtractCollideData(arenas[0], slotsPerArena[0], ccons, statics, cellStart, cellList,
			proxies0, regCounter0);
		rscuda::UploadCollideData(batch, ccons, statics.data(), (int)statics.size(),
			cellStart.data(), (int)cellStart.size() - 1,
			cellList.data(), (int)cellList.size(), grid);

		rscuda::GpuTailConsts tc;
		std::vector<rscuda::GpuPadConfig> padConfigs;
		std::vector<rscuda::GpuDynBody> dynTemplates;
		ExtractTailData(arenas[0], slotsPerArena[0], tc, padConfigs, dynTemplates);
		rscuda::UploadTailData(batch, tc, padConfigs.data(), numPads, dynTemplates.data());

		if (!arenas[0]->ball->IsSphere()) { // snowday puck hull points
			auto* hull = (btConvexHullShape*)arenas[0]->ball->_collisionShape;
			std::vector<rscuda::GpuVec3> pts(hull->getNumPoints());
			for (int i = 0; i < hull->getNumPoints(); i++)
				pts[i] = ToGpuB(hull->getUnscaledPoints()[i]);
			rscuda::UploadBallHull(batch, pts.data(), (int)pts.size());
		}
		if (args.gameMode == GameMode::DROPSHOT) {
			std::vector<rscuda::GpuTileShape> tiles;
			rscuda::GpuTileTables tables;
			ExtractTiles(arenas[0], tiles, tables);
			rscuda::UploadTiles(batch, tiles.data(), (int)tiles.size(), tables);
		}
	}
	// per-arena: dyn proxies, ball game state, car iteration order (_cars set order)
	for (int a = 0; a < args.arenas; a++) {
		rscuda::GpuCollideConsts ccUnused;
		std::vector<rscuda::GpuStaticObj> sUnused;
		std::vector<int> aUnused, bUnused;
		std::vector<rscuda::GpuDynProxy> proxies;
		int regCounter = 0;
		ExtractCollideData(arenas[a], slotsPerArena[a], ccUnused, sUnused, aUnused, bUnused,
			proxies, regCounter);
		rscuda::UploadDynProxies(batch, a, proxies.data(), regCounter);
		rscuda::GpuBallGameState bs = {};
		bs.velocityImpulseCache = ToGpu(arenas[a]->ball->_velocityImpulseCache);
		bs.tickCountSinceUpdate = arenas[a]->ball->_internalState.tickCountSinceUpdate;
		bs.hsYTargetDir = arenas[a]->ball->_internalState.hsInfo.yTargetDir;
		bs.hsCurTargetSpeed = arenas[a]->ball->_internalState.hsInfo.curTargetSpeed;
		bs.hsTimeSinceHit = arenas[a]->ball->_internalState.hsInfo.timeSinceHit;
		bs.groundStickApplied = arenas[a]->ball->_groundStickApplied;
		const auto& dsi = arenas[a]->ball->_internalState.dsInfo;
		bs.dsChargeLevel = dsi.chargeLevel;
		bs.dsAccumulatedHitForce = dsi.accumulatedHitForce;
		bs.dsYTargetDir = dsi.yTargetDir;
		bs.dsHasDamaged = dsi.hasDamaged;
		bs.dsLastDamageTick = dsi.hasDamaged ? dsi.lastDamageTick : 0;
		rscuda::UploadBallState(batch, a, bs);
		std::vector<int> order;
		for (Car* car : arenas[a]->_cars)
			for (int c = 0; c < args.cars; c++)
				if (slotsPerArena[a][c].car == car)
					order.push_back(c);
		rscuda::UploadCarOrder(batch, a, order.data());
		// minstd_rand0 post-seed state: libstdc++ takes seed % 2147483647, 0 -> 1
		// (RunCpu pins the same engine via Math::SeedRandEngine with this seed)
		uint32_t rngState = (uint32_t)(DemoRngSeed(args, a) % 2147483647ull);
		rscuda::UploadRngState(batch, a, rngState == 0 ? 1u : rngState);
	}
	rscuda::SetTickCount(batch, 0);

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
					hold = args.gpuChunk > 0 ? args.gpuChunk : 2 + (int)(rng.Next() % 59);
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
	if (args.phaseProf)
		rscuda::EnablePhaseProfiling(batch);
	if (args.gpuChunk > 0) {
		// RL decision-step model: constant controls per chunk, one launch per chunk
		std::vector<rscuda::GpuCarControls> chunkControls((size_t)args.arenas * args.cars);
		for (int t0 = 0; t0 < args.ticks; t0 += args.gpuChunk) {
			int n = args.gpuChunk < (args.ticks - t0) ? args.gpuChunk : (args.ticks - t0);
			for (int a = 0; a < args.arenas; a++)
				for (int c = 0; c < args.cars; c++)
					chunkControls[(size_t)a * args.cars + c] =
						schedule[((size_t)a * args.cars + c) * args.ticks + t0];
			rscuda::UploadConstantControls(batch, chunkControls.data());
			rscuda::StepTicks(batch, n);
		}
	} else {
		rscuda::UploadControlSchedule(batch, schedule.data(), args.ticks);

		// THE step: one kernel launch runs every tick of every arena
		rscuda::StepTicks(batch, args.ticks);
	}

	if (args.phaseProf) {
		static const char* PHASE_NAMES[rscuda::GPU_PHASE_COUNT] = {
			"game_pre", "predict", "collision", "solve",
			"integrate", "activation", "game_post" };
		unsigned long long pc[rscuda::GPU_PHASE_PROF_SLOTS];
		rscuda::GetPhaseCycles(batch, pc);
		unsigned long long total = 0;
		for (int i = 0; i < rscuda::GPU_PHASE_COUNT; i++) total += pc[i];
		for (int i = 0; i < rscuda::GPU_PHASE_COUNT; i++)
			printf("PHASE %-10s %6.2f%%  (%llu cycles)\n", PHASE_NAMES[i],
				total ? 100.0 * pc[i] / total : 0.0, pc[i]);
	}

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
		rscuda::GpuBallGameState bs;
		rscuda::DownloadBallState(batch, a, &bs);
		arenas[a]->ball->_velocityImpulseCache = FromGpu(bs.velocityImpulseCache);
		arenas[a]->ball->_internalState.tickCountSinceUpdate = bs.tickCountSinceUpdate;
		arenas[a]->ball->_internalState.hsInfo.yTargetDir = bs.hsYTargetDir;
		arenas[a]->ball->_internalState.hsInfo.curTargetSpeed = bs.hsCurTargetSpeed;
		arenas[a]->ball->_internalState.hsInfo.timeSinceHit = bs.hsTimeSinceHit;
		arenas[a]->ball->_groundStickApplied = bs.groundStickApplied;
		auto& dsi = arenas[a]->ball->_internalState.dsInfo;
		dsi.chargeLevel = bs.dsChargeLevel;
		dsi.accumulatedHitForce = bs.dsAccumulatedHitForce;
		dsi.yTargetDir = bs.dsYTargetDir;
		dsi.hasDamaged = bs.dsHasDamaged;
		if (bs.dsHasDamaged)
			dsi.lastDamageTick = bs.dsLastDamageTick;
		if (args.gameMode == GameMode::DROPSHOT) {
			constexpr int NT = RLConst::Dropshot::TEAM_AMOUNT * RLConst::Dropshot::NUM_TILES_PER_TEAM;
			uint8_t dmg[NT], nr[NT];
			rscuda::DownloadTileState(batch, a, dmg, nr);
			for (int t = 0; t < RLConst::Dropshot::TEAM_AMOUNT; t++)
				for (int i = 0; i < RLConst::Dropshot::NUM_TILES_PER_TEAM; i++)
					arenas[a]->_dropshotTilesState.states[t][i].damageState =
						dmg[t * RLConst::Dropshot::NUM_TILES_PER_TEAM + i];
			// mirror the device response flags onto the tile bodies (writeback fidelity)
			for (int i = 0; i < NT; i++) {
				auto* rb = arenas[a]->_worldDropshotTileRBs[i];
				if (nr[i])
					rb->m_collisionFlags |= btCollisionObject::CF_NO_CONTACT_RESPONSE;
				else
					rb->m_collisionFlags &= ~btCollisionObject::CF_NO_CONTACT_RESPONSE;
			}
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
		if (args.dumpArenaHashes)
			EmitCarDiag("gpu", slotsPerArena[a]);
		delete arenas[a];
	}

	rscuda::DestroyBatch(batch);
	return combinedHash;
}

// Diagnostic: print the CPU dispatcher's manifolds (creation order) each tick, with
// bodies mapped to names. Ground truth for the GPU collision orchestrator port.
static int RunManifoldDump(const Args& args) {
	std::vector<CarSlot> slots;
	Arena* arena = SetupArena(args, 0, slots);

	auto bodyName = [&](const void* obj) -> std::string {
		if (obj == &arena->ball->_rigidBody) return "ball";
		for (size_t c = 0; c < slots.size(); c++)
			if (obj == &slots[c].car->_rigidBody) return "car" + std::to_string(c);
		for (size_t w = 0; w < arena->_worldCollisionRBs.size(); w++)
			if (obj == arena->_worldCollisionRBs[w])
				return (arena->_worldCollisionRBs[w]->getCollisionShape()->getShapeType()
					== TRIANGLE_MESH_SHAPE_PROXYTYPE ? "mesh" : "plane") + std::to_string(w);
		return "?";
	};

	for (int tick = 0; tick < args.manifoldDump; tick++) {
		for (CarSlot& slot : slots)
			UpdateControls(slot);
		rscap::carSnaps.clear();
		rscap::enabled = true;
		arena->Step(1);
		rscap::enabled = false;

		bool any = false;
		for (const rscap::ManifoldSnap& m : rscap::manifolds)
			if (m.numContacts) any = true;
		if (!any)
			continue;
		printf("tick %d: %zu manifolds\n", tick, rscap::manifolds.size());
		for (size_t i = 0; i < rscap::manifolds.size(); i++) {
			const rscap::ManifoldSnap& m = rscap::manifolds[i];
			printf("  [%zu] %s vs %s: %d pts\n", i,
				bodyName(m.body0).c_str(), bodyName(m.body1).c_str(), m.numContacts);
			for (int c = 0; c < m.numContacts; c++) {
				const rscap::ContactSnap& p = m.pts[c];
				printf("    pt%d d=%.9g n=(%.9g,%.9g,%.9g) wB=(%.9g,%.9g,%.9g) "
					"fr=%.9g re=%.9g part(%d,%d) idx(%d,%d)%s\n",
					c, p.distance, p.normalWorldOnB[0], p.normalWorldOnB[1], p.normalWorldOnB[2],
					p.positionWorldOnB[0], p.positionWorldOnB[1], p.positionWorldOnB[2],
					p.combinedFriction, p.combinedRestitution,
					p.partId0, p.partId1, p.index0, p.index1,
					p.isSpecial ? " SPECIAL" : "");
			}
		}
	}
	delete arena;
	return 0;
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
	if (args.manifoldDump > 0) {
#ifdef RS_CUDA
		return RunManifoldDump(args);
#else
		fprintf(stderr, "MANIFOLD_DUMP: FAIL (built without CUDA toolkit)\n");
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
