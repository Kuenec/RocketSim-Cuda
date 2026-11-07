# RocketSim-CUDA V3

A bit-exact C++ and CUDA port of [RocketSim](https://github.com/ZealanL/RocketSim) v0.3.5. It steps batches of arenas on the CPU, the GPU, or both, supports every game mode and car body with up to 8 cars per arena, and is verified bit for bit against the upstream crate after every tick.

## Build

```bash
cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
```

```bash
cmake --build build/cmake
```

Requires CMake 3.28+, a C++20 compiler, and CUDA; `-DRSC_ENABLE_CUDA=OFF` builds the CPU backend only. Place the `.cmf` collision meshes in `collision_meshes/soccar`, `hoops`, and `dropshot`.

## Usage

```cpp
#include "rocketsim_cuda/rocketsim_cuda.hpp"

using namespace rocketsim_cuda;

BatchConfig config;
config.game_mode = GameMode::Soccar;
config.mutators = default_mutators(GameMode::Soccar);
config.cars = {{Team::Blue, car_body_config(CarBody::Octane)},
               {Team::Orange, car_body_config(CarBody::Octane)}};
config.rng_seeds = {0, 1, 2, 3};

ArenaBatch batch(Backend::Cuda, MeshSet::load_directory("collision_meshes"),
                 config);
std::vector<CarControls> controls(batch.num_arenas() * batch.num_cars());
batch.set_all_controls(controls);
batch.step(8);
CarState car = batch.get_car_state(0, 0);
```

## Verification

```bash
tools/run_gates.sh
```

Needs Rust. Compares every backend against the upstream crate and prints `ALL GATES PASS` or `GATES FAIL`.

## Performance

Soccar 1v1, millions of arena-ticks per second, Ryzen 7 7700X and RTX 5070.

| Arenas | Upstream, 1 thread | CPU, 1 thread | CPU, 16 threads | CUDA | Hybrid |
| --- | --- | --- | --- | --- | --- |
| 100 | 0.99 | 0.99 | 5.35 | 1.37 | 4.68 |
| 1,000 | 0.74 | 0.81 | 7.17 | 5.83 | 6.64 |
| 10,000 | 0.70 | 0.81 | 7.03 | 14.16 | 16.66 |
| 20,000 | 0.71 | 0.82 | 7.27 | 14.55 | 18.59 |

## License

MIT. Based on RocketSim by ZealanL and VirxEC; see [NOTICE](NOTICE).
