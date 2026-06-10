# RocketSim CUDA Project Requirements

## Goal

Port the ENTIRE RocketSim + Bullet physics stack to CUDA so that ALL simulation code runs on the GPU. No hybrids. No CPU fallbacks. Every phase, broadphase, narrowphase, solver, integration, damping, island management, RocketSim car updates, boost pads, ball logic, EVERYTHING must execute on the GPU via CUDA.

---

## WHAT "100% CUDA" MEANS - READ THIS CAREFULLY

100% CUDA means: when the benchmark runs a physics tick, a GPU kernel executes the ENTIRE physics step and produces the FINAL result that the simulation uses. The CPU does NOT compute any physics.

### What COUNTS as CUDA:
- A `__global__` kernel that runs on the GPU and produces the physics result that `Arena::Step()` would normally produce on CPU and that result is what the simulation ACTUALLY USES for the next tick.
- The GPU kernel must handle ALL of: predict, AABB update, broadphase, narrowphase (GJK/EPA/box-plane/sphere-plane/compound-mesh), solver setup, solver iterations, solver finish, integration, vehicle dynamics (wheel raycasts + suspension + friction), activation state, AND all RocketSim game logic (car pre-tick, car post-tick, ball updates, boost pad pickup/cooldown).
- The kernel's output must match the CPU reference validation hashes (with acceptable FP drift that is minimized as much as possible).

### What DOES NOT count as CUDA:
- `.cuh` header files that compile but are never called from any kernel. Dead code is not CUDA.
- GPU kernels that run but whose results are THROWN AWAY while the CPU result is used instead. Shadow kernels are not CUDA.
- A hybrid approach where the GPU handles some phases (predict, integrate, activation) and the CPU handles other phases (collision, solver setup). Hybrids are stepping stones, not the goal.
- Reporting inflated "CUDA %" based on what percentage of wall-clock time is spent in GPU phases. CUDA % is based on whether the GPU produces the FULL simulation result.
- Writing lots of GPU code files to inflate a LOC count. LOC only matters if the code EXECUTES on GPU and PRODUCES the result the simulation uses.

### How to measure CUDA %:
CUDA is either 0% or 100%. There is no in-between for the final deliverable.
- 0% = the CPU is still computing any part of the physics step, and the simulation uses that CPU result.
- 100% = the GPU computes the ENTIRE physics step, and the simulation uses ONLY the GPU result. The CPU does zero physics work.
- During development, intermediate states are acceptable (e.g., "GPU handles predict+integrate, CPU handles collision+solver"). But these are stepping stones. The project is NOT complete until CUDA = 100%.
- In every message you will state what percentage you think you are to 100% CUDA but that is so the user can get a sense of where you are at in development.

---

## BEHAVIORAL REQUIREMENTS FOR THE AI ASSISTANT

### DO NOT take shortcuts. Specifically:
1. **DO NOT write GPU code that isn't connected to the simulation path.** Every GPU function you write must be callable from the monolithic kernel, and the monolithic kernel must be the path that produces the simulation result.
2. **DO NOT inflate CUDA % with meaningless metrics.** If the CPU is still doing collision detection, CUDA is not 100%. Period. Don't report "56% CUDA by time-weighted phases" or "33% CUDA by LOC" — report honestly: "the GPU handles X phases, the CPU still handles Y phases, CUDA is not yet 100%."
3. **DO NOT write `.cuh` headers that nobody calls and count them as progress.** A file that compiles but isn't called is dead code. It contributes nothing.
4. **DO NOT avoid the hard work.** The hard work is: making the monolithic GPU kernel produce output that matches the CPU reference hash. This requires debugging GJK/EPA, manifold management, solver setup, vehicle dynamics, and RocketSim game logic ON THE GPU until the output is correct. This is tedious and difficult. Do it anyway.
5. **DO NOT lie about progress.** If the GPU kernel produces wrong output, say so. If the CPU is still doing physics work, say so. If you don't know how to fix something, say so. Honest status reporting is more valuable than inflated numbers.
6. **DO NOT redefine metrics to make progress look better than it is.** Deliverable CUDA % is binary: 0% or 100%. During development you can report intermediate progress like "GPU handles predict+integrate+activation, working on collision next"

### DO focus on:
1. **Making the monolithic GPU kernel (`--gpu-step`) produce the correct reference hash.** This is the ONLY path to 100% CUDA. Every session should make progress toward this goal.
2. **Debugging GPU physics output against CPU reference.** Compare per-body state after each phase. Find where the GPU diverges from CPU. Fix it. Repeat.
3. **Matching Bullet's exact algorithms on GPU.** The btVector3 operation order technique (dot3 grouping, angular multiply associativity) is proven to work for the PGS solver. Apply it to ALL GPU physics code.
4. **Testing incrementally.** Don't try to make everything work at once. Get predict correct first. Then add collision. Then solver. Verify each phase matches CPU before adding the next.
5. **Keeping the CPU hybrid path working** as a fallback for validation, but always pushing toward replacing it with the GPU path.

---

## Non-Negotiable Constraints

1. **Reference clone is untouched.**
   - Location: `RocketSim-ORIGINAL/` (cloned from https://github.com/ZealanL/RocketSim)
   - Never modify any file in this directory.

2. **Exactness is verified against the reference, not against itself.**
   - `RocketSim-Cuda/` Started as an exact copy of the reference.
   - Every change must be validated against the reference hash gates.
   - Workspace-vs-workspace agreement is not sufficient.

3. **Validation gates (deterministic), established 2026-06-10 on this Linux machine:**
   - Stable gate: 128 arenas / 1000 ticks / 2 cars / seed 1337 -> hash `3380507951560469826` (verified identical across 5 fresh-process runs)
   - Hard gate: 1 arena / 20000 ticks / 2 cars / seed 1337 -> hash `11764255410181881605` (verified identical across 6 fresh-process runs)
   - Commands (from repo root, after `cmake -B build -G Ninja && ninja -C build`):
     ```
     ./build/rocketsim_bench --mesh-dir collision_meshes --ticks 1000  --arenas 128 --cars 2 --seed 1337 --expected-state-hash 3380507951560469826
     ./build/rocketsim_bench --mesh-dir collision_meshes --ticks 20000 --arenas 1   --cars 2 --seed 1337 --expected-state-hash 11764255410181881605
     ```
     (`rocketsim_reference_bench` takes the same args and must produce the same hashes; prints `GATE: PASS/FAIL`, exit code 0/1.)
   - Both gates must pass after every change.
   - FP drift from GPU execution is acceptable IF it is minimized and the simulation behavior is negligibly different. New deterministic hashes for the GPU path are acceptable if the physics behavior is correct.

4. **Full CUDA — no hybrids accepted as final state.**
   - The end state is 100% GPU execution of all physics and game logic.
   - CPU-GPU hybrids (e.g. GPU raycasts + CPU solver) are only acceptable as intermediate stepping stones, never as the final deliverable.
   - Every Bullet phase must be ported: broadphase overlap detection, narrowphase contact generation, constraint solving (PGS iterations), integration (transform updates), damping, island management, activation state, everything.
   - Every RocketSim phase must be ported: car pre-tick updates, wheel raycasts, vehicle dynamics, boost pad collision/updates, ball updates, post-tick updates.
   - "CUDA ownership" means the GPU produces the result that the simulation uses. Shadow/dry-run GPU kernels that run alongside CPU do not count.

5. **Build with Ninja.**
   - This machine (Linux/CachyOS): CMake 4.3.3 + Ninja 1.13.2 + GCC 16.1.1, top-level `CMakeLists.txt`.
   - Commands: `cmake -B build -G Ninja` then `ninja -C build` (Release/-O3 by default).
   - `-ffp-contract=off` is pinned on all targets so the CPU reference defines plain
     mul+add FP semantics the GPU port can reproduce exactly (nvcc `-fmad=false`).
   - CUDA toolkit installed 2026-06-10: nvcc 13.3 at `/opt/cuda/bin/nvcc` (pacman `cuda` package).
     Host compiler MUST be g++-15 (`NVCC_CCBIN=/usr/bin/g++-15` is set system-wide via
     `/etc/profile.d/cuda.sh`, which also adds `/opt/cuda/bin` to PATH on login shells);
     system g++ 16 is unsupported by nvcc 13.3.
   - GPU: NVIDIA GeForce RTX 5070 (Blackwell, sm_120), 48 SMs, 12227 MiB, driver 610.43.02.
     End-to-end verified: trivial mul+add kernel compiled with `-ccbin g++-15 -fmad=false`,
     ran on the GPU, results exact.

---

## Workspace Layout

```
RocketSim-ORIGINAL/   # pristine upstream v2.2.1 @ f80f9d90 (NEVER modify; gitignored, re-clone per .gitignore note if missing)
RocketSim-CUDA/       # working copy (modify here; ALL GPU kernel code goes in src/Cuda/)
  libsrc/bullet3-3.24/  # Bullet physics (compiled as C++, not by NVCC)
tools/bench/main.cpp  # ONE bench source compiled twice: rocketsim_reference_bench (frozen now
                      #   that the baseline is set) and rocketsim_bench (workspace bench; fork
                      #   into its own source file when GPU flags like --gpu-step are added)
collision_meshes/     # soccar .cmf assets (gitignored; see Asset Provenance below)
CMakeLists.txt        # top-level build: rocketsim_ref + rocketsim_work libs, both benches
build/                # ninja output (gitignored)
```

---

## Metrics

- **Exactness**: % of validation gates passing (0%, 50%, or 100%)
- **CUDA**: Does the GPU produce the FULL simulation result? (0% during development, 100% when complete)
- **Throughput**: workspace SPS / reference SPS (must reach 100%+) you will also tell the user the SPS value.

---

## Priority Order (strict, top to bottom)

1. **Exactness** — Physics output must be as close as possible to the untouched CPU reference. FP drift from GPU execution is expected and acceptable, but must be minimized. Every change must be validated. A fast but incorrect simulation is worthless.

2. **CUDA completion** — This is the entire point of the project: move ALL simulation code to the GPU. Full CUDA ownership doesn't just satisfy the goal — it also raises the throughput ceiling by enabling GPU parallelism across arenas and eliminating per-tick CPU-GPU round trips. During the porting process, throughput may temporarily drop below the CPU reference due to transfer overhead, kernel launch costs, and incomplete batching. That is expected and acceptable as a transitional cost. What matters is that the code is on the GPU and produces the result the simulation uses.

3. **Throughput** — Once exactness is preserved and CUDA coverage is complete, maximize SPS. With the full simulation on GPU, the architecture can batch 128+ arenas into single kernel launches, eliminate host-device transfers mid-tick, and exploit massive parallelism. The long-term throughput should significantly exceed the CPU reference. But throughput is never a reason to compromise exactness or leave code on the CPU.

---

## Completion Standard

The project is complete ONLY when ALL THREE are true:
- Exactness = 100% (both gates pass, or new deterministic GPU hashes with negligible FP drift)
- CUDA = 100% (the GPU kernel produces the ENTIRE physics step result, CPU does zero physics)
- Throughput >= 100% (faster than or equal to CPU reference)

---

## Git Workflow

- **Remote**: https://github.com/Kuenec/RocketSim-Cuda
- **Branch**: Push directly to `Fable`
- **Author**: All commits under the user's git config — NO co-authoring, NO `Co-Authored-By` lines Username: Kuenec Email: Kuenec44@gmail.com
- **Commit after every kept change** that passes both gates. Don't batch up multiple changes.
- **Push to GitHub after every commit.** The remote must always reflect the latest working state.
- **Commits**: All commits Must be short and Consise about what you changed
- **`main` and all other remote branches are a dead failed experiment. Never read from, merge, or push to them. This folder + `Fable` is the only truth.**

---

## Current State (2026-06-10)

**Status: Exactness 100% (both gates pass) | CUDA 0% (no GPU code yet) | Throughput 100% (workspace == reference, identical CPU code)**

### Done
- Repo initialized at project root on branch `Fable` (fresh history, not based on old `main`), pushed to the remote. Work copy verified file-identical to pristine v2.2.1 before import.
- Soccar collision meshes obtained and verified: all 16 `.cmf` hashes match the known-good set hardcoded in upstream `src/RocketSim.cpp` (`MeshHashSet`). Bench re-verifies them on every run before Init.
- Bench tools built and gates established (values + commands above).

### Baseline reference throughput (single thread, this machine: CachyOS Linux, GCC 16.1.1 -O3 -ffp-contract=off)
| Gate | Reference SPS | Workspace SPS |
|------|--------------|----------------|
| Stable (128a/1000t) | ~58,000 | ~58,300 |
| Hard (1a/20000t) | ~242,000 | ~242,600 |
SPS varies a few percent run to run; the hash is the gate, SPS is informational.

### Key decisions (with evidence)
1. **Gates run with `DemoMode::DISABLED`.** Upstream demolition respawn calls `Math::RandInt(.., seed=-1)` -> `Math::GetRandEngine()`, which is seeded from wall-clock + thread id (`src/Math/Math.cpp`). Any demo would inject non-reproducible randomness. Demo LOGIC must still be ported to GPU; the gates just cannot cover the respawn path. Bump/demo checks still execute every tick.
2. **Bench masks `BallState.dsInfo.lastDamageTick` when `!hasDamaged`.** That field has NO initializer upstream and is only written in dropshot, but `BallState::Serialize` writes it regardless. Unmasked, the state hash ingests uninitialized heap bytes — diagnosed as the root cause of a nondeterministic stable gate (per-arena hash dump + 4-byte blob diff at exactly that field's offset; only soccar-mode read of the field is inside a `gameMode == DROPSHOT` guard, so physics is unaffected).
3. **No `ResetToRandomKickoff`, explicit spawns instead.** It iterates `_cars` (a pointer-keyed `unordered_set`, ASLR-order) and uses `std::shuffle`/`std::default_random_engine` (implementation-defined). Bench sets car states explicitly.
4. **Bench randomness is splitmix64 with power-of-two float quantization** — input schedule is bit-identical on any platform/compiler. Never use std distributions for gate-relevant randomness.
5. **`unordered_set<Car*>` iteration-order hazard.** `Arena::Step` iterates `_cars` in pointer order for pre/post-tick. With the current gates this is provably benign (hashes stable across fresh processes), but any future change that makes per-car work order-dependent (e.g. same-tick boost pad contests) can break determinism. The GPU port must define a FIXED car order (by car id) and validate against the gates.
6. **`-ffp-contract=off` everywhere** so CPU semantics are plain IEEE mul+add, matchable on GPU with `nvcc -fmad=false`.

### Asset Provenance
- Meshes downloaded from `https://github.com/heresfilly09/rocket-league-collision-meshes` (plain `.cmf`), verified against upstream's known hash set. Game-derived, so NOT committed; re-download to `collision_meshes/soccar/` and the bench's `MESHES: OK (16/16 soccar)` line confirms integrity.

### GPU port progress (CUDA 0% — kernel does not yet produce correct sim output)
Done so far:
- Workspace bench forked to `tools/bench_work/main.cpp` with `--profile` and `--gpu-step`.
  Phase profile on hard gate: car_pre_tick 50% (vehicle sim lives in Car::_PreTickUpdate,
  NOT bullet updateActions), bt_collision 35%, everything else small.
- `src/Cuda/`: GpuTypes.h (POD mirrors), CudaStep.h/.cu (batch bridge + monolithic kernel
  scaffold: ONE launch runs all ticks x all arenas; control schedule precomputed on host
  and uploaded once), CudaMath.cuh (bit-exact device math, see below).
- `--gpu-step` machinery works end-to-end (upload -> kernel -> download -> hash),
  deterministic; phases are still empty so the hash is wrong vs gates (expected, honest).

**Bit-exact device math layer (VERIFIED, the cornerstone):**
- Bullet on this host compiles with BT_USE_SSE_IN_API: SSE kernels fix specific
  summation ASSOCIATIONS that differ from scalar fallbacks. CudaMath.cuh encodes the
  SSE trees: dot3 = (xx+yy)+zz; quat dot4 = (xx+zz)+(yy+ww); quat mul lanes
  (A0-B1)+(A1+A2); mat mul (a+b)+c; setRotation per-lane trees derived from the SSE
  shuffles, INCLUDING a final "+0.0f" on off-diagonals (the SSE row-add of v1000/v0100/
  v0010 canonicalizes -0 to +0 — found via fuzz, 196/4M samples).
- glibc sinf/cosf are NOT double-rounded sin/cos for ~0.08%/0.002% of inputs. Fix:
  empirical correction table measured from THIS machine's libm:
  `cuda_math_test --emit-trig-table glibc_trig_fix.bin` (exhaustive scan of
  [0, ANGULAR_MOTION_THRESHOLD], 803336 sin + 17882 cos entries, ~6.5MB, gitignored —
  regenerate per machine). Device CuSinf/CuCosf: double-rounded base + binary-search
  substitution. `CuLoadTrigFix(path)` binds it (per-TU symbol).
- `cuda_math_test --samples 4000000 --trig-table glibc_trig_fix.bin --trig-exhaustive`:
  ALL PASS — 11 op families bit-exact over 4M fuzzed samples each; sinf/cosf bit-exact
  over the ENTIRE 1.06e9-value domain. This binary is the regression guard for the math
  layer; run it after any CudaMath change.
- btVector3::normalize() uses _mm_rsqrt_ss + 1 Newton-Raphson step (CPU-vendor-specific
  estimate table!). NOT yet needed (predict/integrate/activation avoid it; quat
  normalize is exact sqrt+div). WILL be needed for Ball::_FinishPhysicsTick
  (vel.normalized()), suspension, solver friction. Plan: emulate via host-built
  RSQRTSS table (2x4096 mantissa entries + exponent arithmetic) with startup self-check.
- Body facts for predict/integrate/activation (verified in source): gravity applied as
  force pre-step (stepSimulation: applyGravity, then clearForces after); velocity
  integration happens in SOLVER setup/finish (even contactless bodies form single-body
  islands -> gravity lands via solver finish, NOT predict); predict = applyDamping +
  interpolation-transform write (interp transform is write-only in RocketSim - skip);
  ball linearDamping = ballDrag=0.03 -> btPow(0.97,dt) PRECOMPUTED host-side (constant);
  cars have zero damping; no CCD (thresholds 0); no additionalDamping; activation per
  updateDeactivation/wantsSleeping incl. the odd setActivationState(0); sleeping
  thresholds are bullet defaults (lin 0.8, ang 1.0), gDeactivationTime=2.
- CUDA arch gotcha: CMAKE_CUDA_ARCHITECTURES must be set BEFORE enable_language(CUDA)
  or it locks to compiler default (was 75); now `native` (sm_120 here).

### Next Steps (in order)
1. Implement kernel phases predict/integrate/activation using CudaMath; add CPU capture
   hooks (work copy, diagnostic) at phase boundaries + `--gpu-compare` lockstep mode in
   bench_work: per tick upload CPU pre-bullet state, run GPU tick prefix, bit-compare
   at each phase point, report first divergence (tick/phase/body/field/bits).
2. Then per profile order: vehicle+car logic (50%), collision (35%), solver, ball/pads.
3. RSQRTSS emulation when first normalize() call site is ported.

### Open Questions / Blockers
- None currently.

---