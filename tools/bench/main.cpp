// RocketSim CPU benchmark + validation gate driver.
//
// One source, compiled twice by the top-level CMakeLists:
//   rocketsim_reference_bench -> links the pristine RocketSim-ORIGINAL tree (gate producer, frozen)
//   rocketsim_bench           -> links the RocketSim-CUDA work tree (must reproduce reference hashes)
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
};

static void PrintUsage(const char* argv0) {
	fprintf(stderr,
		"Usage: %s --mesh-dir <collision_meshes dir> --ticks N [--arenas N] [--cars N] "
		"[--seed N] [--expected-state-hash H]\n", argv0);
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

static void UpdateControls(CarSlot& slot) {
	if (slot.holdTicks == 0) {
		CarControls c = {};
		c.throttle = slot.rng.NextSigned();
		c.steer = slot.rng.NextSigned();
		c.pitch = slot.rng.NextSigned();
		c.yaw = slot.rng.NextSigned();
		c.roll = slot.rng.NextSigned();
		c.jump = slot.rng.Chance(10);
		c.boost = slot.rng.Chance(30);
		c.handbrake = slot.rng.Chance(15);
		slot.car->controls = c;
		slot.holdTicks = 2 + (int)(slot.rng.Next() % 59); // change every 2-60 ticks
	}
	slot.holdTicks--;
}

int main(int argc, char** argv) {
	Args args;
	if (!ParseArgs(argc, argv, args)) {
		PrintUsage(argv[0]);
		return 2;
	}

	if (!VerifyMeshes(args.meshDir))
		return 2;
	RocketSim::Init(args.meshDir, true);

	constexpr float HALF_PI = 1.5707963267948966f;

	uint64_t combinedHash = FNV_BASIS;
	double elapsedS = 0;

	for (int arenaIdx = 0; arenaIdx < args.arenas; arenaIdx++) {
		Arena* arena = Arena::Create(GameMode::SOCCAR);

		// Demos disabled: upstream demo respawn is wall-clock-seeded (see header comment)
		MutatorConfig mc = arena->GetMutatorConfig();
		mc.demoMode = DemoMode::DISABLED;
		arena->SetMutatorConfig(mc);

		Rng spawnRng(MixSeed(args.seed, (uint64_t)arenaIdx));
		std::vector<CarSlot> slots;
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

		auto t0 = std::chrono::steady_clock::now();
		for (int tick = 0; tick < args.ticks; tick++) {
			for (CarSlot& slot : slots)
				UpdateControls(slot);
			arena->Step(1);
		}
		auto t1 = std::chrono::steady_clock::now();
		elapsedS += std::chrono::duration<double>(t1 - t0).count();

		// Hash final state in bench-defined order (never arena container order)
		DataStreamOut out;
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

		uint64_t arenaHash = Fnv1a64(out.data.data(), out.data.size(), FNV_BASIS);
		combinedHash = Fnv1a64(&arenaHash, sizeof(arenaHash), combinedHash);

		if (args.dumpArenaHashes)
			printf("ARENA %d HASH: %" PRIu64 "\n", arenaIdx, arenaHash);
		if (!args.dumpStateDir.empty()) {
			std::filesystem::create_directories(args.dumpStateDir);
			auto blobPath = std::filesystem::path(args.dumpStateDir) / ("arena_" + std::to_string(arenaIdx) + ".bin");
			std::ofstream blob(blobPath, std::ios::binary);
			blob.write((const char*)out.data.data(), out.data.size());
		}

		delete arena;
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
