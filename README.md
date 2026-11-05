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
- A device-resident CUDA backend: every tick, reset, state change, mutator change, and copy runs as a kernel on the GPU, and arena state never travels to the host unless you ask for it. Controls can be read from device memory, kickoff resets can be selected by a device-side mask, and the full observable state can be exported straight into device buffers, so a GPU policy can drive a batch with no host transfers.

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

For a training loop that stays on the GPU, pass device pointers (for example from libtorch tensors) and use the batch's CUDA stream for ordering:

```cpp
batch.set_all_controls_device(actions);
batch.step_async(8);
StateBuffers out;
out.balls = device_balls;
out.cars = device_cars;
out.infos = device_infos;
out.event_counts = device_events;
batch.export_state(out, MemorySpace::Device);
batch.reset_to_random_kickoff_masked(device_done_mask, device_seeds,
                                     MemorySpace::Device);
cudaStream_t stream = static_cast<cudaStream_t>(batch.cuda_stream());
```

All of these calls are queued on the batch's stream in order. Device inputs (actions, masks, seeds) must be fully written before the call that reads them, so finish or order any copy or kernel that produces them against the batch's stream first; likewise, synchronize with that stream (or call `batch.sync()`) before another stream reads the exported buffers.

## Verification

```bash
tools/run_gates.sh
```

The script builds the library and the Rust harness, runs the quick libm check, and runs every gate scenario on both backends against the upstream crate. It prints one PASS or FAIL line per scenario and backend with event coverage, and ends with `ALL GATES PASS` or `GATES FAIL`. Arguments are passed to the harness, for example `--backend cuda`, `--filter hoops`, or `--quick`. Scenarios cover every mode with meshes, 1, 2, 3, 4, 6, and 8 cars, all bodies, all demo modes, 20,000-tick single-arena runs, 128-arena batches, seeded random controls, controls from the `rocketsim_test` recordings, kickoffs, scripted goals and head-on collisions, mid-run state sets, runtime mutator changes, pad state restores, clones, snapshots, and cross-batch and cross-backend copies. Dropshot scenarios report BLOCKED until Dropshot meshes are present.

## Benchmarks

```bash
build/harness-target/release/rsv3_harness bench --targets upstream,cpu,cuda --arenas 100,1000,10000
```

Soccar 1v1, Octane vs Octane, new random controls every 8 ticks, measured after a 1,200-tick warm-up over 5 runs of 240 ticks (arena-ticks per second, median with min to max). Upstream and the CPU backend run on one pinned core of a Ryzen 7 7700X. The CUDA backend runs on an RTX 5070 in the device-resident loop: controls are read from device memory, steps are asynchronous, and the ball, car, arena-info, and event-count state of every arena is exported into device buffers after every 8-tick action, with one host synchronization per measured run.

| Arenas | Upstream Rust v0.3.5, 1 thread | C++ CPU backend, 1 thread | CUDA backend | CUDA vs upstream |
| --- | --- | --- | --- | --- |
| 100 | 1.03M (0.98M to 1.12M) | 0.66M (0.64M to 0.73M) | 0.54M (0.48M to 0.59M) | 0.5x |
| 1,000 | 0.85M (0.80M to 0.91M) | 0.62M (0.60M to 0.68M) | 2.25M (1.96M to 2.37M) | 2.7x |
| 10,000 | 0.74M (0.73M to 0.77M) | 0.59M (0.59M to 0.62M) | 6.80M (6.45M to 6.86M) | 9.2x |

The CUDA backend runs one arena per thread and picks the block size from the batch size, so it needs thousands of arenas to fill the GPU. The C++ CPU backend runs at about 65 to 80 percent of the upstream Rust speed on one core. It uses portable scalar code where upstream uses SSE vector instructions, which is the likely cause; it has not been profiled in detail yet.

## License

MIT, see [LICENSE](LICENSE). Based on RocketSim by ZealanL, ported to Rust by VirxEC. Some math routines are derived from glibc (LGPL-2.1-or-later) and CORE-MATH, and the physics derive from Bullet Physics (zlib); see [NOTICE](NOTICE).
