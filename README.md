# RocketSim-CUDA

A port of the **entire** [RocketSim](https://github.com/ZealanL/RocketSim) physics
simulation — including the Bullet physics engine as RocketSim exercises it — to a
single CUDA kernel, **bit-exact** against the CPU reference.

When running with `--gpu-step`, one kernel launch simulates every tick of every arena
in the batch. The CPU does no physics at all: the host thread sleeps (blocking sync)
while the GPU produces the complete simulation result — broadphase, narrowphase
(GJK/EPA, sphere/box/convex-hull/triangle-mesh/plane), the sequential-impulse solver,
vehicle suspension and wheel raycasts, and all game-layer logic.

## Status

| Feature | On GPU | Gated bit-exact |
|---|---|---|
| Soccar | ✅ | ✅ |
| Heatseeker (ball steering, hit speedup, wall bounce) | ✅ | ✅ |
| Snowday (convex-hull puck, ground stick) | ✅ | ✅ |
| Hoops (net, kickoff launch, zScale hit branch) | ✅ | ✅ |
| Dropshot (stateful 140-tile floor, damage/charge, fall-through) | ✅ | ✅ |
| Demolitions + respawns (deterministic RNG) | ✅ | ✅ |
| Boost pads, bumps, supersonic, jumps/flips/dodges, autoflip | ✅ | ✅ |

## Exactness

"Bit-exact" here is literal, not approximate:

- Every validation gate hashes (FNV-1a) the **complete serialized end state** of every
  arena: each car's position, rotation matrix, velocities, every timer and flag, ball
  state including mode-specific info, boost pads, and dropshot tile damage. A single
  flipped bit anywhere fails the gate.
- Gates include 20,000-tick single-arena runs, where one ulp of floating-point drift
  early on would avalanche into a completely different end state.
- The chain to the original: the frozen reference bench, built from the untouched
  upstream source (`RocketSim-ORIGINAL/`), produces the **same hashes** as the work
  CPU on the soccar gates. The work tree's simulation code is the original's, with
  only inert instrumentation and a deterministic RNG seeding hook added. GPU results
  hash-identical to the work CPU on every gate.
- Supporting evidence beyond the gates: 200k-raycast fuzz per geometry mode compared
  bitwise (hit/fraction/normal); device `sinf`/`cosf` verified against glibc over
  **all 2³² float inputs**; standalone GJK/EPA and sphere-triangle fuzz versus the
  real Bullet implementations.

Achieving this required reproducing, on the GPU, things like Bullet's SSE expression
trees, `RSQRTSS` instruction emulation, glibc's exact `sinf`/`cosf`/`atan2f`/`asinf`,
signed-zero propagation through identity transforms, BVH ray-walk visitation order
(equal-fraction ties keep the first-visited triangle), the SSE `maxDot` tie-break in
convex-hull supports, and `mergesSimulationIslands` semantics for no-contact-response
bodies.

Two honest qualifications:

1. **The reference is this machine's CPU build** (Linux x86-64, glibc, SSE float
   behavior), bugs and FP quirks included. A different compiler or libm produces a
   different — equally self-consistent — reference.
2. Gates validate what they exercise, which is very broad, but one branch
   (`DemoMode::NORMAL`'s supersonic check) never fired under random controls; it is a
   1:1 port whose both outcomes are otherwise gate-covered.

## Performance

Soccar, 1000 ticks, 2 cars per arena, RTX 5070 vs one CPU thread.
SPS = arena-ticks simulated per second.

| Arenas | GPU SPS | CPU SPS | Speedup |
|---|---|---|---|
| 100 | 494,916 | 57,597 | 8.6× |
| 200 | 638,402 | ~57,500 | 11.1× |
| 500 | 1,059,946 | 57,505 | 18.4× |
| 1,000 | 1,146,923 | 57,336 | 20.0× |
| 2,000 | 1,331,558 | ~57,500 | 23.2× |
| 5,000 | 1,375,483 | ~57,500 | 23.9× |
| 10,000 | 1,412,098 | ~57,500 | 24.6× |

The CPU runs arenas sequentially, so its throughput is flat; the GPU scales with
batch size and saturates around 1.4M SPS (~24×) on this card. 10,000 simultaneous
arenas use roughly 2.5 GB of VRAM.

## Building

Requires CMake, Ninja, a C++20 compiler, and the CUDA toolkit (arch defaults to
`native`). GPU math is compiled with `--fmad=false` to match the CPU's
`-ffp-contract=off` IEEE semantics.

```sh
cmake -B build -G Ninja
ninja -C build
```

Produces `rocketsim_bench` (the work-tree bench + validation driver),
`rocketsim_reference_bench` (frozen, built from `RocketSim-ORIGINAL/`), and
`cuda_math_test` (math validation, e.g. `--trig-exhaustive-full`).

## Running the gates

```sh
# CPU reference / GPU result — same args, same hash expected
./build/rocketsim_bench --mesh-dir collision_meshes --ticks 1000 --arenas 128 --cars 2 \
    --seed 1337 --expected-state-hash 3380507951560469826
./build/rocketsim_bench --mesh-dir collision_meshes --ticks 1000 --arenas 128 --cars 2 \
    --seed 1337 --expected-state-hash 3380507951560469826 --gpu-step
```

All established gates (CPU = reference, GPU passes every one with the identical hash;
hashes are machine-local, established on the reference machine):

| Mode / config | Hash |
|---|---|
| soccar 128×2×1000 s1337 | 3380507951560469826 |
| soccar 1×2×20000 s1337 | 11764255410181881605 |
| heatseeker 1×2×20000 s1337 | 9090346742586108841 |
| heatseeker 64×4×1000 s1337 | 6365120802482066310 |
| demos on-contact 64×4×1000 s1337 | 16798083503057895723 |
| demos on-contact 8×6×3000 s4242 | 2807978879122602574 |
| snowday 1×2×20000 s1337 | 6720894011815016031 |
| snowday 32×4×1500 s4242 | 16993491767979963113 |
| snowday + demos 8×6×2000 s99 | 3415898148067774285 |
| hoops 1×2×20000 s1337 | 14674870656341860422 |
| hoops 32×6×2000 s4242 | 14326563309044877142 |
| hoops + demos 8×6×3000 s99 | 6767940556749896631 |
| dropshot 1×2×20000 s1337 | 16797750564250741040 |
| dropshot 32×6×2500 s4242 | 18379725107633245205 |
| dropshot + demos 8×6×3000 s99 | 16801832455181733896 |

Other validation modes: `--ray-fuzz N` (bitwise raycast compare per mode),
`--gpu-compare N` (per-phase lockstep replay against CPU captures),
`--demo-mode {disabled,normal,on-contact}`, `--game-mode
{soccar,heatseeker,hoops,snowday,dropshot}`.

## Assets

`collision_meshes/` holds the arena collision meshes (`soccar/`, `hoops/`,
`dropshot/` — the hoops set matches RocketSim's built-in known-hash list; snowday and
heatseeker reuse the soccar meshes).

## Repository layout

- `RocketSim-CUDA/` — the work tree. Device code lives in `src/Cuda/`
  (`CudaStep.cu` monolithic kernel, `CudaCollision.cuh` narrowphase,
  `CudaSolver.cuh` constraint solver, `CudaVehicle.cuh` suspension/wheels,
  `CudaCollideWorld.cuh` collision orchestration + game callbacks,
  `CudaGameTail.cuh` game-layer pre/post tick, `CudaRaycast.cuh`,
  `CudaConvexCast.cuh`, `CudaMath.cuh` glibc-exact math).
- `RocketSim-ORIGINAL/` — untouched upstream source; never modified, used to build
  the frozen reference bench.
- `tools/bench_work/` — the bench/validation driver; `tools/bench/` — frozen
  reference copy; `tools/cuda_math_test/` — math validation.
- `Persistent_Memory.md` — living engineering log: specs, root-cause histories,
  gate provenance.
