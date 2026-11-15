# RocketSim-CUDA V3

A bit-exact C++ and CUDA port of [RocketSim](https://github.com/ZealanL/RocketSim) v0.3.6. It steps batches of arenas on the CPU, the GPU, or both, supports every game mode and car body with up to 8 cars per arena, and is verified bit for bit against the upstream crate after every tick.

## Build

```bash
cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
```

```bash
cmake --build build/cmake
```

Requires CMake 3.28+, a C++20 compiler, and CUDA; `-DRSC_ENABLE_CUDA=OFF` builds the CPU backend only. Place the `.cmf` collision meshes in `collision_meshes/soccar`, `hoops`, and `dropshot`.

The default build matches upstream built with its default features. `-DRSC_NATIVE_MATH=ON` matches upstream's `native-math` feature instead: integer rotators, the sine table, and the engine's own `acos`, `atan2`, `pow`, `sin`, and `cos`. The two modes produce different results from each other, as they do upstream.

`-DRSC_BOX_TRIANGLE_GJK=ON` matches upstream's `box-triangle-gjk` feature: arena triangle contacts of the car box and the car impact poses of bumps and demos use the GJK and EPA pair detector instead of the separating-axis tests. It combines with either math mode. On the GPU these pairs are computed by one lane per arena, so this mode steps more slowly there; the default build is unaffected.

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

`sleep_ball(arena)` stops the ball and puts it to sleep, and `wake_ball(arena)` activates it and clears its deactivation timer, as `Arena::sleep_ball` and `Arena::wake_ball` do upstream.

The body and contact access of upstream's `sim/physics.rs` is available per arena, in native Bullet units (one unit is 50 Unreal units):

```cpp
uint32_t bodies = batch.physics_body_count(0);
uint32_t ball = batch.ball_body_id(0);
std::optional<uint32_t> car_body = batch.car_body_id(0, 0);
BodyInfo info = batch.body_info(0, ball);
Vec3 top = info.world_point({0.0f, 0.0f, 1.0f});
BodyState state = batch.body_state(0, ball);
batch.set_body_state(0, ball, state);
std::vector<ContactManifold> manifolds = batch.contact_manifolds(0);
batch.set_contact_manifolds(0, manifolds);
```

Body ids index the arena's body array: the static arena bodies first, then the ball and the cars. The setters check every value before they write and throw `StateError` with the same cases as upstream's `StateError`, leaving the arena unchanged. One difference from upstream: the static bodies are shared by all arenas of a batch, so `set_body_state` on a static body throws `std::logic_error`.

## Verification

```bash
tools/run_gates.sh
```

Needs Rust. Compares every backend against the upstream crate and prints `ALL GATES PASS` or `GATES FAIL`. `--mode native-math` gates the engine-math build against upstream built with `native-math`, `--mode gjk` and `--mode native-math-gjk` gate the GJK build against upstream built with `box-triangle-gjk` alone and together with `native-math`, and `--mode all` gates all four; each mode has its own build directories.

## Performance

Soccar 1v1, millions of arena-ticks per second, Ryzen 7 7700X and RTX 5070.

| Arenas | Upstream, 1 thread | CPU, 1 thread | CPU, 16 threads | CUDA | Hybrid |
| --- | --- | --- | --- | --- | --- |
| 100 | 0.98 | 0.99 | 6.90 | 1.26 | 5.06 |
| 1,000 | 0.74 | 0.85 | 8.61 | 5.84 | 8.83 |
| 10,000 | 0.72 | 0.84 | 7.42 | 13.46 | 17.06 |
| 20,000 | 0.72 | 0.83 | 7.32 | 13.58 | 17.53 |

## License

MIT. Based on RocketSim by ZealanL and VirxEC; see [NOTICE](NOTICE).
