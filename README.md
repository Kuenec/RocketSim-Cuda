# RocketSim-CUDA V3

RocketSim-CUDA V3 is a C++ and CUDA port of [RocketSim](https://github.com/ZealanL/RocketSim) v0.3.5, the Rust Rocket League physics simulator by ZealanL and VirxEC. It runs batches of arenas on either a C++ CPU backend or a CUDA GPU backend behind one C++ API, and both backends produce results that are bit-identical to the unmodified upstream Rust simulator.

## Exactness guarantee

The reference is the pinned `rocketsim` crate (tag `v0.3.5`, commit `88bd345`) built unmodified in release mode on the reference machine (Linux x86-64, glibc 2.44 CachyOS x86-64-v4 build). For every gate scenario, the harness steps the reference and this port side by side and compares, after every compared tick, every field of the serialized arena state bit for bit: cars (including the internal contact cache, timers, flags, and demo attribution), ball and its Heatseeker and Dropshot state, boost pads (cooldown and ordered membership), Dropshot tiles, tick count, RNG state, the last tick's event list, mutators, and goal state.

To achieve this, the port reproduces glam 0.34 SIMD operation order, Rust `f32::min`/`max` lowering, fastrand wyrand, Bullet's broadphase, manifold, island and solver ordering, and the installed glibc `sinf`, `cosf`, `acosf`, `atan2f`, `atan2`, `hypot` and `powf` results. The device copies of those libm routines are checked against the host glibc over all 2^32 inputs for the one-argument functions and 2^30 structured samples for the two-argument functions (`build/cmake/rsc_libm_check`). All device code is compiled with `--fmad=false`, IEEE division and square root, and no flush-to-zero; host code uses `-ffp-contract=off`.

Results are only guaranteed against a reference built on a machine with the same glibc math build. NaN payload bits of the device libm differ from glibc for NaN inputs; this has not been observed to reach simulation state.

## Features

- Game modes: Soccar, Hoops, Heatseeker, Snowday, Dropshot, and TheVoid.
- 0 to 8 cars per arena, uneven teams, and all seven car bodies (Octane, Dominus, Plank, Breakout, Hybrid, Merc, Psyclops), uniform within a batch.
- Demo modes Normal, OnContact, and Disabled; runtime mutator changes; boost pads with upstream pad order and membership; Dropshot tiles.
- Full state get and set: car state (core or full snapshot), ball state, pad state, tile state, kickoff resets with or without seeds.
- Arena copies within and across batches and backends, and raw snapshots that include hidden physics state.
- Exact upstream per-tick event lists (goals via `is_ball_scored`, ball touches, bumps, demos, boost pickups, landings, world hits) plus per-step event counters.
- Synchronous and asynchronous stepping of N ticks.

## Build

Requirements: CMake 3.28+, Ninja, a C++20 compiler, and the CUDA toolkit (tested with CUDA 13.4 and GCC 16 on an RTX 5070). The Rust harness additionally needs a Rust toolchain.

```bash
cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
```

```bash
cmake --build build/cmake
```

`-DRSC_ENABLE_CUDA=OFF` builds the CPU backend only. The library target is `rocketsim_cuda` (static); `rocketsim_cuda_c` is a shared C ABI used by the verification harness.

Collision meshes are not included. Place the `.cmf` files dumped from Rocket League in `collision_meshes/soccar`, `collision_meshes/hoops`, and `collision_meshes/dropshot`, as for upstream RocketSim.

## Usage

```cpp
#include "rocketsim_cuda/rocketsim_cuda.hpp"

using namespace rocketsim_cuda;

auto meshes = MeshSet::load_directory("collision_meshes");
BatchConfig config;
config.game_mode = GameMode::Soccar;
config.mutators = default_mutators(GameMode::Soccar);
config.cars = {{Team::Blue, car_body_config(CarBody::Octane)},
               {Team::Orange, car_body_config(CarBody::Octane)}};
config.rng_seeds.resize(1024);
for (uint64_t i = 0; i < config.rng_seeds.size(); ++i) {
  config.rng_seeds[i] = i;
}

ArenaBatch batch(Backend::Cuda, meshes, config);
for (uint32_t a = 0; a < batch.num_arenas(); ++a) {
  batch.reset_to_random_kickoff(a, a);
}
std::vector<CarControls> controls(batch.num_arenas() * batch.num_cars());
batch.set_all_controls(controls);
batch.step(8);
CarState car = batch.get_car_state(0, 0);
EventCounts events = batch.get_event_counts(0);
```

Switching `Backend::Cuda` to `Backend::Cpu` runs the same physics on the CPU with identical results.

## Verification

```bash
tools/run_gates.sh
```

The script builds the library and the Rust harness, runs the quick libm check, and runs every gate scenario on both backends against the upstream crate. It prints one PASS or FAIL line per scenario and backend with event coverage, and ends with `ALL GATES PASS` or `GATES FAIL`. Arguments are passed to the harness, for example `--backend cuda`, `--filter hoops`, or `--quick`. Scenarios cover every mode with meshes, 1, 2, 3, 4, 6, and 8 cars, all bodies, all demo modes, 20,000-tick single-arena runs, 128-arena batches, seeded random controls, controls from the `rocketsim_test` recordings, kickoffs, scripted goals and head-on collisions, mid-run state sets, runtime mutator changes, pad state restores, clones, snapshots, and cross-batch and cross-backend copies. Dropshot scenarios report BLOCKED until Dropshot meshes are present.

## Benchmarks

```bash
build/harness-target/release/rsv3_harness bench --targets upstream,cpu,cuda --arenas 100,1000,10000
```

Soccar 1v1, Octane vs Octane, new random controls every 8 ticks, measured after a 1,200-tick warm-up over 5 runs of 240 ticks (arena-ticks per second, median with min to max). Upstream and the CPU backend run on one pinned core of a Ryzen 7 7700X; the CUDA backend runs on an RTX 5070.

| Arenas | Upstream Rust v0.3.5, 1 thread | C++ CPU backend, 1 thread | CUDA backend | CUDA vs upstream |
| --- | --- | --- | --- | --- |
| 100 | 1.03M (0.98M to 1.12M) | 0.68M (0.65M to 0.74M) | 0.55M (0.49M to 0.61M) | 0.5x |
| 1,000 | 0.85M (0.80M to 0.91M) | 0.65M (0.62M to 0.69M) | 2.34M (2.08M to 2.47M) | 2.8x |
| 10,000 | 0.74M (0.73M to 0.77M) | 0.60M (0.59M to 0.64M) | 6.78M (6.61M to 7.06M) | 9.1x |

The CUDA backend runs one arena per thread and picks the block size from the batch size, so it needs thousands of arenas to fill the GPU. The C++ CPU backend runs at about 65 to 80 percent of the upstream Rust speed on one core. It uses portable scalar code where upstream uses SSE vector instructions, which is the likely cause; it has not been profiled in detail yet.

## License

MIT, see [LICENSE](LICENSE). Based on RocketSim by ZealanL, ported to Rust by VirxEC. Some math routines are derived from glibc (LGPL-2.1-or-later) and CORE-MATH, and the physics derive from Bullet Physics (zlib); see [NOTICE](NOTICE).
