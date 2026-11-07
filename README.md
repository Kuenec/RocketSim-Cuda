# RocketSim-CUDA V3

A C++ and CUDA port of [RocketSim](https://github.com/ZealanL/RocketSim) v0.3.5 that steps batches of arenas on the CPU, the GPU, or both, with results bit-identical to upstream.

## Exactness

Every backend is gated against the unmodified `rocketsim` crate (tag `v0.3.5`, `88bd345`) by comparing the full arena state, including hidden physics state, bit for bit after every tick. Results match a reference built against the same glibc math library; tested on Linux x86-64.

## Features

- All game modes: Soccar, Hoops, Heatseeker, Snowday, Dropshot, and TheVoid.
- 0 to 8 cars per arena, uneven teams, and all car bodies.
- Full state get and set, kickoff resets, runtime mutators, arena copies, and per-tick events.
- `Cpu`, `Cuda`, and `Hybrid` backends behind one API. `Cuda` keeps all state on the GPU and accepts device pointers, so a GPU policy needs no host transfers.

## Build

Requires CMake 3.28+, Ninja, a C++20 compiler, and the CUDA toolkit. The verification harness also needs Rust.

```bash
cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
```

```bash
cmake --build build/cmake
```

Use `-DRSC_ENABLE_CUDA=OFF` for a CPU-only build. Collision meshes are not included; place the `.cmf` files dumped from Rocket League in `collision_meshes/soccar`, `collision_meshes/hoops`, and `collision_meshes/dropshot`.

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
```

Device-memory calls (`set_all_controls_device`, `export_state`, `reset_to_random_kickoff_masked`) and `step_async` run in order on `batch.cuda_stream()`.

## Verification

```bash
tools/run_gates.sh
```

Runs every gate scenario on every backend against the upstream crate and ends with `ALL GATES PASS` or `GATES FAIL`.

## Benchmarks

Soccar 1v1 with random controls, in millions of arena-ticks per second (median of 5 runs), on a Ryzen 7 7700X and an RTX 5070. CUDA and hybrid read controls from and export state to GPU memory every 8 ticks.

| Arenas | Upstream, 1 thread | CPU, 1 thread | CPU, 16 threads | CUDA | Hybrid |
| --- | --- | --- | --- | --- | --- |
| 100 | 0.99 | 0.99 | 5.35 | 1.37 | 4.68 |
| 1,000 | 0.74 | 0.81 | 7.17 | 5.83 | 6.64 |
| 10,000 | 0.70 | 0.81 | 7.03 | 14.16 | 16.66 |
| 20,000 | 0.71 | 0.82 | 7.27 | 14.55 | 18.59 |

```bash
build/harness-target/release/rsv3_harness bench --targets upstream,cpu,cpu-mt,cuda,hybrid --arenas 100,1000,10000,20000 --hybrid-cpu-share 0.45
```

## License

MIT, see [LICENSE](LICENSE). Based on RocketSim by ZealanL and VirxEC. Third-party notices are in [NOTICE](NOTICE).
