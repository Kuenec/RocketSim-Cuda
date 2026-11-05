# RocketSim-CUDA

A CUDA port of [RocketSim](https://github.com/ZealanL/RocketSim) 2.2.1 that is bit-exact with the CPU simulation.

A single kernel launch simulates every tick of every arena in a batch. That covers the broadphase, the narrowphase (GJK/EPA, sphere, box, convex hull, triangle mesh, plane), the sequential-impulse solver, vehicle suspension and wheel raycasts, and all game logic. The CPU does no physics.

## Features

- Game modes: Soccar, Hoops, Heatseeker, Snowday and Dropshot.
- Demolitions and respawns, with deterministic on-device RNG.
- Boost pads, bumps, supersonic, jumps, flips, dodges and autoflip.
- Asynchronous stepping, so the CPU can work while the GPU simulates.
- CPU-only builds still work when no CUDA toolkit is found.

## Exactness

GPU results match the CPU build bit for bit. Each validation gate hashes the complete serialized end state of every arena, including the cars, the ball, boost pads and Dropshot tiles. The gates cover every game mode and include 20,000-tick runs, where a single ulp of drift would change the final hash.

Matching the CPU required reproducing:

- glibc's `sinf`, `cosf`, `atan2f` and `asinf`.
- `RSQRTSS` and Bullet's SSE evaluation order.
- Signed-zero propagation and the tie-breaking order in BVH and convex-hull queries.

The CPU side is compiled with `-ffp-contract=off` and the GPU side with `--fmad=false`, so both sides use plain IEEE multiply and add. Do not change these flags. The reference is the Linux x86-64 glibc build; other compilers or math libraries produce a different, but equally self-consistent, reference.

## Building

Requires CMake 3.20+, a C++20 compiler and, for the GPU path, the CUDA toolkit. The CUDA architecture defaults to `native`.

```sh
cmake -B build -G Ninja
cmake --build build
```

When CUDA is found, the `RocketSim` static library defines `RS_CUDA=1` and includes the GPU path. Set `-DRS_GPU_PHASE_PROF=ON` to compile in the kernel's per-phase profiler.

## Usage

`rscuda::GpuArenaBatch` (`src/Cuda/GpuArenaBridge.h`) steps a group of regular `Arena` objects on the GPU. The arenas must share a game mode, car count and tick count, and use the default car config.

```cpp
RocketSim::Init("collision_meshes");

std::vector<RocketSim::Arena*> arenas;
for (int i = 0; i < 1000; i++) {
	auto arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
	arena->AddCar(RocketSim::Team::BLUE);
	arena->AddCar(RocketSim::Team::ORANGE);
	arenas.push_back(arena);
}

rscuda::GpuArenaBatch batch(arenas, 1337);
batch.Step(8);
batch.Writeback();
```

`Step` reads each car's controls and holds them for the given number of ticks. `Writeback` copies the results back into the arenas. After a reset or state change, call `UploadArena(i)`.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | RocketSim 2.2.1 |
| `src/Cuda/` | GPU port: `CudaStep.cu` kernel, collision, solver, vehicle, raycast and game-logic headers, `GpuArenaBridge` host API |
| `libsrc/bullet3-3.24/` | Bullet subset used by RocketSim |

Collision meshes are not included. Dump them from your own copy of Rocket League.

## License

MIT, see [LICENSE](LICENSE). Based on RocketSim by ZealanL.
