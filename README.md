# RocketSim-CUDA V3

RocketSim-CUDA V3 is a C++ and CUDA port of [RocketSim](https://github.com/ZealanL/RocketSim) v0.3.5, the Rust Rocket League physics simulator by ZealanL and VirxEC. It runs batches of arenas on a C++ CPU backend, a CUDA GPU backend, or a hybrid backend that splits one batch across both, behind one C++ API, and every backend produces results that are bit-identical to the unmodified upstream Rust simulator.

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
- Synchronous and asynchronous stepping of N ticks; the CPU backend steps arenas on a thread pool (`BatchConfig::cpu_threads`, all hardware threads by default), which cannot change results because arenas are independent.
- A device-resident CUDA backend: every tick, reset, state change, mutator change, and copy runs as a kernel on the GPU, and arena state never travels to the host unless you ask for it. Controls can be read from device memory, kickoff resets can be selected by a device-side mask, and the full observable state can be exported straight into device buffers, so a GPU policy can drive a batch with no host transfers.
- A hybrid backend for the highest total throughput: the GPU steps the first arenas while the CPU thread pool steps the rest at the same time, and the split moves at runtime so both finish together. It has the same API as the CUDA backend, including device-memory controls, masks, and exports; only the CPU arenas' rows cross the PCIe bus, through pinned staging buffers.

## Build

Requirements: CMake 3.28+, Ninja, a C++20 compiler, and the CUDA toolkit (tested with CUDA 13.4 and GCC 16 on an RTX 5070). The Rust harness additionally needs a Rust toolchain.

```bash
cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
```

```bash
cmake --build build/cmake
```

`-DRSC_ENABLE_CUDA=OFF` builds the CPU backend only. `-DRSC_NATIVE_ARCH=ON` compiles host code for the build machine's CPU (`-march=native`), which makes the CPU backend about 6 percent faster on the reference machine without changing results, but produces binaries that may not run on older CPUs. The library target is `rocketsim_cuda` (static); `rocketsim_cuda_c` is a shared C ABI used by the verification harness.

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

The hybrid backend takes the same calls. Construct the batch with `Backend::Hybrid`:

```cpp
ArenaBatch batch(Backend::Hybrid, meshes, config);
```

The first arenas run on the GPU and the rest on the CPU thread pool (`BatchConfig::cpu_threads`). `BatchConfig::hybrid_cpu_arenas` sets how many arenas start on the CPU (0 starts with half). After every 16 steps the batch compares the GPU kernel time with the CPU pool time and moves arenas at the boundary between the two halfway toward the split where both take equally long, which can put a whole small batch on the CPU or a whole large one on the GPU. Moving an arena copies its full state, so results stay bit-identical; arena indices never change. `step_async` launches the GPU arenas, steps the CPU arenas on the calling thread and the pool, and returns while the GPU may still be running. Calls that move CPU arenas' rows between host and device memory (device controls, device masks and seeds, device exports) wait for the batch's stream before they return.

## Verification

```bash
tools/run_gates.sh
```

The script builds the library and the Rust harness, runs the quick libm check, and runs every gate scenario on every backend against the upstream crate. It prints one PASS or FAIL line per scenario and backend with event coverage, and ends with `ALL GATES PASS` or `GATES FAIL`. Arguments are passed to the harness, for example `--backend cuda`, `--backend hybrid`, `--filter hoops`, or `--quick`; by default every scenario runs on the CPU, CUDA, and hybrid backends, and the hybrid runs start with the last half of each batch (rounded up) on the CPU and rebalance as they run, so they also check arena moves between the GPU and the CPU. Scenarios cover every mode, 0, 1, 2, 3, 4, 6, and 8 cars, all bodies, all demo modes, 20,000-tick single-arena runs, batches of 128 to 8,192 arenas, seeded random controls, controls from the `rocketsim_test` recordings, kickoffs, scripted goals and head-on collisions, mid-run state sets, runtime mutator changes, pad state restores, clones, snapshots, and cross-batch and cross-backend copies. Scenarios for a mode whose meshes are missing report BLOCKED.

## Benchmarks

```bash
build/harness-target/release/rsv3_harness bench --targets upstream,cpu,cpu-mt,cuda,hybrid --arenas 100,1000,10000,20000 --hybrid-cpu-share 0.45
```

Soccar 1v1, Octane vs Octane, new random controls every 8 ticks, measured after a 1,200-tick warm-up over 5 runs of 240 ticks (arena-ticks per second, median with min to max). Upstream and the single-thread CPU backend run on one core of a Ryzen 7 7700X (8 cores, 16 threads); the all-cores CPU column uses 16 threads and varies by about 10 percent between runs with normal desktop load. The CUDA backend runs on an RTX 5070 in the device-resident loop: controls are read from device memory, steps are asynchronous, and the ball, car, arena-info, and event-count state of every arena is exported into device buffers after every 8-tick action, with one host synchronization per measured run. The hybrid column runs the same loop starting with 45 percent of the arenas on the CPU thread pool.

| Arenas | Upstream Rust v0.3.5, 1 thread | C++ CPU, 1 thread | C++ CPU, all cores | CUDA | Hybrid |
| --- | --- | --- | --- | --- | --- |
| 100 | 1.03M (0.98M to 1.11M) | 0.73M (0.70M to 0.78M) | 5.20M (4.81M to 5.70M) | 1.39M (1.26M to 1.44M) | 4.16M (3.29M to 4.39M) |
| 1,000 | 0.89M (0.87M to 0.95M) | 0.70M (0.67M to 0.73M) | 6.69M (6.53M to 7.12M) | 6.60M (5.97M to 6.65M) | 8.78M (7.29M to 8.92M) |
| 10,000 | 0.74M (0.74M to 0.78M) | 0.64M (0.63M to 0.67M) | 6.47M (5.59M to 6.82M) | 14.24M (14.05M to 14.88M) | 17.13M (16.33M to 17.26M) |
| 20,000 | 0.75M (0.74M to 0.78M) | 0.64M (0.63M to 0.67M) | 6.35M (6.20M to 6.67M) | 14.82M (14.48M to 15.60M) | 18.89M (18.30M to 20.03M) |

With `-DRSC_NATIVE_ARCH=ON` the single-thread CPU backend measures 0.78M, 0.72M, and 0.67M at 100, 1,000, and 10,000 arenas, and the all-cores CPU backend 5.24M, 6.74M, and 6.63M.

The CUDA backend gives each arena a group of 2 to 32 GPU threads, chosen so the whole batch stays resident: 32 threads per arena for small batches, halved as the batch grows (8 at 1,000 arenas and 2 at 10,000 on an RTX 5070). The threads of a group, and of all groups in a warp, split the parts of a tick that are independent in upstream's algorithm: the wheel rays (one per thread, each culling the mesh tree with its own ray), the per-wheel suspension and friction stages of the vehicle update, the car-versus-triangle tests and their 47 candidate point pairs across the whole warp, boost pads, broadphase pair planning, and per-body integration. Work whose order matters in upstream, such as manifold updates, the constraint solver, event recording, and impulse accumulation, runs in upstream order on one thread per group or car, and every parallel stage is checked by the gates against the upstream crate. Car updates run in parallel only when no car is demolished and no wheel touches another car or the ball; otherwise the group's first thread runs the original sequential loop. When a batch needs more thread blocks than the GPU holds at once, each step starts the blocks that took longest in the previous step first, which shortens the last wave. The CUDA backend pays off with hundreds of arenas and more, and when the policy also runs on the GPU, since it avoids host transfers; the hybrid backend balances each batch between the GPU and the CPU cores and is the fastest backend from about 1,000 arenas up, while for small batches it moves every arena to the CPU and runs at the CPU backend's speed. The single-thread C++ CPU backend runs at about 70 to 87 percent of upstream speed; it uses portable scalar code for most vector math where upstream uses SSE vector instructions.

## License

MIT, see [LICENSE](LICENSE). Based on RocketSim by ZealanL, ported to Rust by VirxEC. Some math routines are derived from glibc (LGPL-2.1-or-later) and CORE-MATH, and the physics derive from Bullet Physics (zlib); see [NOTICE](NOTICE).
