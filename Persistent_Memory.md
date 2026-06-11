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

## Current State (2026-06-11)

**Status: ALL FIVE GAME MODES (soccar, heatseeker, snowday, hoops, dropshot) + demolitions run 100% ON GPU, every mode gated bit-exact vs the work-CPU reference (15 gates + 4x ray-fuzz). Throughput GOAL MET: GPU 668k SPS @128 arenas, 1.19M @1024 vs CPU 58.3k single-thread (11-20x); 1024-arena cross-check bit-exact. GGL-ATTENTION RL-framework integration COMPLETE and validated (see section below). Remaining: nothing blocked.**

### GGL-ATTENTION integration (2026-06-11, repo /home/kue/Projects/GGL-ATTENTION, branch `rocketsim-cuda`, commits 1da1342/25b49b9/20b7bae, not pushed)
- RocketSim-CUDA vendored at `GigaLearnCPP/RLGymCPP/RocketSim/`; libtorch 2.10.0+cu130 at `GigaLearnCPP/libtorch/` (gitignored, re-download from pytorch.org). Builds: `build/` (GigaLearnBot trainer, needs `-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15`), `build-rlgym/` (gpusim_validate).
- GPU stepping: `RLGymCPP/src/RLGymCPP/GpuSim/GpuEnvStepper` groups arenas by car count into `rscuda::GpuArenaBatch`es (bridge: `src/Cuda/GpuArenaBridge.h/.cpp`); EnvSet GPU branches pool parse/prep/post-step on g_ThreadPool; first half-step runs WITHOUT writeback (nothing reads arena state between halves), second half writes back via Download + pooled per-arena scatter + FinishWriteback. Bump/demo events derived from carContact-cooldown-increase/isDemoed transitions. CRITICAL bridge invariant: device car slots = creation order (ascending car->id); `_cars` set order is uploaded separately as the carOrder permutation.
- First-half chunk is DELIBERATELY synchronous: sim kernels + policy inference share one GPU; async interleaving measured inference 0.9s -> 3.3s.
- Exactness regression guard: `build-rlgym/gpusim_validate <meshes> [arenas] [carsPerTeam] [steps] [demoMode]` (per-step full-state hashes CPU vs GPU; passes 8x2v2x600, 64x2v2x1000 incl 37/37 bump events, 16x3v3x600, 1x2v2x1500 demo-mode). Bench `--gpu-chunk N` in RocketSim-CUDA covers the chunked-stepping model engine-side.
- Trainer flags added: `--no-metrics` (no embedded-python wandb; system python lacks wandb), `--cpu-sim` (pre-CUDA physics), `--games N`. Meshes via `RS_MESHES` or `./collision_meshes` symlink (gitignored). KeyPressDetector fixed (spun on non-TTY stdin, 31M perror lines/90s).
- PERF FINDING (measured, 150s runs, RTX 5070 + 24-core CPU, 1v1, 700k ts/iter): overall SPS GPU-sim 76k@500/85k@2000-4000 games vs CPU-sim 124k. GPU path shares the GPU with inference+PPO (learn ~3.3s/iter caps overall at ~167k). On THIS machine CPU sim trains faster; GPU sim wins only when arena count is huge relative to CPU cores or the CPU is otherwise busy. Training-state caveat: smoke runs left throwaway checkpoints in GGL `checkpoints/` (random-policy, ~100M steps) — delete before real training.
- PROFILE (GPUSIM_PROF=1 env var, 2026-06-11, 2000 1v1 games, per 100 decision steps): sim kernels 1.03s (64%), prep+parse+postStep 0.41s, upload/download/scatter/events 0.19s. Kernel is SPILL-BOUND: 255 regs/thread + ~25KB stack/thread. __launch_bounds__ caps (128/~170 regs): 2x occupancy but 23-32% SLOWER — documented in CudaStep.cu, do not redo.
- KERNEL OPTIMIZATION ROUND 1 (2026-06-11, COMMITTED): per-phase clock64 profiling added (bench `--phase-prof`, `EnablePhaseProfiling`/`GetPhaseCycles`) revealed GAME_PRE = 83.7% of kernel cycles, and GAME_PRE/COLLISION/SOLVE/GAME_POST all ran on thread 0 only (31 idle lanes). Fix: `CuCarPreTickWheel` (CudaVehicle.cuh) extracts the per-wheel transform+raycast+friction slice (provably independent per wheel — each writes only W[i]); the kernel stages per-car io/dyn tables in __shared__ and runs the four wheels on threads 0-3 (`wheelsPrecomputed` param on CuCarPreTick keeps the debug path). Cars stay sequential in _cars order. RESULTS: standalone 1.23M -> 2.77M SPS @2048 1v1 (peak 3.14M @1024, ~55x single-thread CPU; README table updated); ALL 18 GPU gates re-PASS bit-exact; gpusim_validate 64x1000 re-PASS (37/37 events). In-trainer: kernel 1.03 -> 0.60s/100 steps, collection 3.88 -> 2.94s/iter @2000 games (collection SPS 177k -> 212k). New phase split @2048: game_pre 63%, collision 12%, game_post 12%, solve 10%.
- GPU-vs-CPU-sim training gap REMAINING (user goal via /goal: optimize RocketSim-CUDA as far as possible while staying bit-exact): GPU collection 2.94s vs CPU 1.89s @2000 games (pre-round-2 numbers). Remaining work: (a) RAY PARALLELIZATION (plan below), (b) collision static-pair parallelization, (c) GPU-resident obs/action-mask build in GGL (CPU bookkeeping ~1.07s/iter + transfers ~0.3s).
- KERNEL ROUND 2 (2026-06-11, COMMITTED, 18/18 gates + gpusim_validate 16x3v3 PASS): GAME_POST car post-tick/finish one-car-per-thread (own-state only, body const in post-tick; grid checks stay sequential in _cars order — reorder proof in kernel comment); CuPadPostTick split into CuPadPostTickPad (parallel per pad) + CuPadGrantBoost (sequential in pad order; grant decision is boost-independent so split is exact); pads-pre strided; gravity per-body after ball-pre (ACTIVE_TAG dependency). game_post 24.5B->19.6B cycles; 2.97M SPS @2048. Phase split now: game_pre 65% (148B, rays dominate), collision 13%, solve 10%, game_post 9%.
- KERNEL ROUND 3 (2026-06-11, COMMITTED + SYNCED TO GGL): wheel raycasts run 4 wheels x 8 lanes via mesh-level striding (CuRayTestStaticMeshes lane/stride + lex-min (fraction, meshIdx) reduce = exact sequential tie-break; CuWheelRayPrep/Epilogue split; CuRayTestDynBodies factored). 18/18 gates + 4x200k ray-fuzz + gpusim_validate PASS. @2048 1v1: 2.97M -> 3.38M SPS (session total 1.23M -> 3.38M, 2.75x). GGL: overall 84k -> 104k SPS (CPU-sim 124k). Phase split now: game_pre 53% (98B; lane balance limited by biggest mesh), collision 17% (31B), solve 14% (26B), game_post 12% (21B).
- COLLISION PARALLELIZATION: BUILT, VALIDATED, AND **REJECTED BY MEASUREMENT** (2026-06-11). The design below was fully implemented (template-mode pair loop: PREPASS slot assignment replaying create/release swap-with-last; parallel per-dyn-body static narrowphase; sequential dyn-dyn) and passed ALL 18 gates hash-identical - the correctness analysis held. But perf: NEUTRAL at 1v1 (one heavy car-mesh pair dominates a lane anyway) and **-17% at 512x6** (650k vs 787k SPS: prepass replay + template register pressure + per-lane full-pair-scan overhead outweigh the parallelism). REVERTED to sequential CuCollideTick. Do not redo without a fundamentally cheaper slot-assignment scheme.
- ENGINE OPTIMIZATION: **SATURATED BY MEASUREMENT (end of 2026-06-11).** All identified levers landed, rejected by measurement, or shown sequential-by-definition:
  LANDED (3): 4-thread wheel phase; game_post car/pad split; 8-lane mesh-strided rays. Final: 3.34-3.38M SPS @2048 1v1 (2.75x in one day, ~59x one CPU thread), 800k @512x6, 18/18 gates + 4x200k ray-fuzz + gpusim_validate bit-exact throughout.
  REJECTED BY MEASUREMENT (5): (1) __launch_bounds__ register caps: -23-32%. (2) Collision pair parallelization (full prepass/slot design, gates-green): neutral @1v1, -17% @512x6. (3) Subtree-gated ray striding: timing probe shows the ENTIRE 8-lane mesh-walk wall is 16.6B cycles (9% of kernel; game_pre is 79B even with rays deleted) -> rebalancing bound ~2-3%, below the overhead floor demonstrated by (2). (4) Solver iteration parallelization: iterations cost only 1.2B cycles (0.7%) AND are sequential by PGS definition. (5) Solver setup parallelization: timing probe shows row construction = 2.4B cycles (1.3%); the other ~21B of solve is island/union-find/body bookkeeping on <=3 bodies - an inherently sequential latency chain.
  WHAT REMAINS in the kernel is sequential bit-exactness itself: per-car ordered pre/post chains (game_pre 79B sans rays), island/body bookkeeping (solve ~21B), single-ray BVH walk latency (16.6B, already 8-lane), pad-grid ordering. Further gains require relaxing exactness (out of scope by user constraint) or different hardware.
  The big remaining TRAINING-SPS lever is FRAMEWORK-side, not engine-side: GGL GPU-resident obs/action-mask build (~1.0s/iter CPU bookkeeping + 0.3s transfers vs the ~1.0s collection gap to CPU sim's 124k overall).
- GPU-DIRECT GAMESTATE: **DONE, VALIDATED, COMMITTED** (2026-06-11, GGL bca22ce + bridge synced to engine repo). FillGameState mirrors UpdateFromArena from GPU buffers (shared field-fillers with the writeback - zero drift); WB_LITE syncs only ball body/state + car ballHitInfo + tickCount (tracker/IsBallScored work unchanged); resets ScatterArena-then-set; GPUSIM_GS_VALIDATE harness bit-compares vs the old path - PASSES (multi-minute runs, zero mismatches). PLUS: all GPU-path pooled phases chunked (numThreads*2 jobs instead of per-arena - dispatch was ~0.7s/iter!). RESULT @2000 games: collection 2.94 -> 2.01s, collection SPS 341k, OVERALL 110k vs CPU 124k. ASYNC CHUNK OVERLAP: **DONE, VALIDATED, COMMITTED** (engine StepTicksAsync/SyncBatch/StepTicksSubset + arenaList kernel param; bridge StepAsync/Sync/StepSubset with inFlightTicks separated from pendingTicks - a mid-window ScatterArena must not advance tickCount for undownloaded ticks, the harness caught this as a 15-tick deltaTime; EnvSet launches chunk7 after the WB_LITE chunk, FinishChunkAsync syncs BEFORE inference and subset-re-steps reset arenas with re-gathered controls). GPUSIM_GS_VALIDATE: zero mismatches incl. resets. **RESULT @2000 games: collection 1.85s (CPU: 1.9s - GPU now collects faster), overall 122.1k vs CPU 124.6k = parity. @3000 games: GPU collection 1.46s vs CPU 4.9s (CPU degrades past ~2000 arenas). Day total: GPU-sim 76k -> 122k overall (+61%), collection SPS 138k -> 399k, bit-exact at every step (18 engine gates + ray-fuzz + GameState field-exact harness).** Remaining ideas if ever needed: obs-on-GPU (the original full plan - removes inference H2D + remaining postStep), multi-GPU split (sim on second GPU).
- ARENA-COUNT SCALING (2026-06-11): 10k arenas initially CUDA-OOM'd - root cause was the EPA device stack reserve (allocated per resident thread, ~74k on the 5070: 64KB = ~4.5GB VRAM). 16KB fails ALL gates; 32KB passes 18/18 bit-exact and freed ~2.3GB (committed both repos). RESULTS, GPU path: 2000 games collection 1.85s / overall 122.1k; 5000: 1.51s / 117.5k; 10000: 1.7-1.8s / 121.9k - collection stays ~1.5-1.8s at every scale (overall is learner-capped ~120k) while CPU-sim degrades hard (1.9s @2000 -> 4.9s @3000 -> impractical @10k). GPU sim is now the better backend at every count >= 2000.
- PPO LEARN-PHASE AUDIT (2026-06-11, GGL be50b7f): the loop is ALREADY CLEAN - enableSpeedOptimizations (on in their config) defers metric .item() syncs, preloads batches, pins buffers. GPU experience buffer (CPU-seeded shuffle kept = identical minibatches/gradients) measured NEUTRAL (learn 3.34s unchanged) -> ships default-OFF (expBufferOnGpu). The learn phase is genuine GPU compute for the 11.7M model; NO exact-math-preserving speedup exists there. Remaining SPS levers are structural only: pipelined collection-during-learn (1-iteration staleness, needs user sign-off) or a second GPU.
- ORIGINAL PLAN (executed above): keep obs/rewards/parsers as CPU classes reading GameState - that is the RLGym portability boundary - but build GameState straight from the downloaded GPU buffers, deleting the per-step Arena scatter + Arena->GameState double conversion; saves ~0.28s/100 steps = ~0.5s/iter @2000 games; with it, the 7-tick chunk can also overlap post-step bookkeeping since Arena objects are no longer touched mid-step):
  1. Bridge: expose download scratch (BodiesScratch/PadsScratch/BallScratch alongside CurCars); add a download-only writeback mode (Download + FinishWriteback + events, NO WritebackArena scatter).
  2. RLGymCPP: GameState::UpdateFromGpu (or GpuEnvStepper::FillGameState) mirroring UpdateFromArena field-for-field: per player = carId/team from carLists + CarState composed from GpuBodyState (pos/rotMat/vel/angVel x BT_TO_UU exactly as GpuArenaBridge WritebackCar + Car::GetState compose today) + GpuCarGameState (boost, isOnGround, isDemoed, wheels, ballHitInfo -> ballTouchedStep/Tick, lastControls); ball = BallState from body 0 + GpuBallGameState (incl hsInfo/dsInfo); pads via boostPadIndexMap from GpuPadState (isActive/cooldown); goalScored = reimplement Arena::IsBallScored as pure fn of gs.ball pos + gameMode (Arena object is STALE mid-episode in this mode).
  3. EnvSet GPU path: _PostStepUpdate uses UpdateFromGpu; eventTrackers[i]->Update(arena) READS ARENA (stale!) - port GameEventTracker to read GameState (check what it reads first; shots/saves may already derive from ballHitInfo+ball state).
  4. RESET PATH CRITICAL: state setters write the Arena then MarkDirty->UploadArena uploads FULL Arena state. With no per-step scatter the Arena's non-setter fields (esp. PADS cooldowns, ball internal state) are STALE. Fix: on reset, first WritebackArena(idx) from the last download (data is in the scratch), THEN run the setter, THEN upload. Resets are rare so this is cheap.
  5. VALIDATION: env-gated mode (GPUSIM_GS_VALIDATE) runs BOTH paths per step (scatter+UpdateFromArena vs UpdateFromGpu) and bit-compares every GameState field across steps incl. resets/demos/touches; also extend gpusim_validate. Only then remove the old path from the hot loop (keep for the validator).
  Projected: collection 2.9 -> ~2.4s/iter directly; ~2.0-2.1s with chunk7-overlap-bookkeeping enabled afterwards; overall ~110-120k vs CPU 124k - obs-on-GPU remains the final step past CPU if the user later wants it.
- COLLISION PARALLELIZATION DESIGN (kept for reference - implementation was correct, perf was not):
  FACTS: CuCollideTick = UpdateAabbs + GenPairs (pairs[] ordered statics-first) + sequential pair loop. ml.create appends header BEFORE narrowphase in every branch; ml.release(mIdx) = SWAP-WITH-LAST + POP (btCollisionDispatcher semantics, comment at CudaCollideWorld.cuh:358) - so manifold array order depends on the create/release SEQUENCE, and the solver consumes that order. Releases happen for: car-static !overlap (line ~865), car-car !overlapNested / !overlapOuter (~991/994). Ball-car creation is LAZY (aabb test before create, ~889). ALL create/release decisions are cheap predicates (activity, aabb math) - no BVH walks.
  CALLBACK ISOLATION: CuContactAddedCallback writes per pair type: ball-static -> ball state only (snowday groundStick guard ORDER-SENSITIVE between ball pairs; dropshot tileDamage with two-tiles-same-tick break logic reading earlier pairs' writes); car-static -> own car state (worldContact for autoflip); dyn-dyn (ball-car, car-car) -> both bodies' states (ballHit/carContact bump data). Manifold refresh/AddContactPoint touch only the pair's own manifold.
  SCHEME (bit-exact): Phase A thread 0: UpdateAabbs + GenPairs (unchanged). Phase B thread 0 PRE-PASS: replay the pair loop's cheap predicates only, performing ml.create header inits and ml.release swap-pops, recording slot[pi] (final manifold index or -1) - released pairs never produce contacts so final-slot filling is sound. Phase C parallel: STATIC pairs only, lane = dyn body index, each lane processes ITS pairs in pair order (preserves ball-pair callback order incl. tiles/stick); narrowphase + AddContactPoint + callback + refresh write into ml.arr[slot[pi]]. Cross-lane writes are disjoint (own manifold + own body state). Phase D thread 0: dyn-dyn pairs sequentially (cheap: single GJK/boxbox, no mesh walk) + numManifolds=ml.n. Statics-before-dyndyn pair order makes the C/D barrier match CPU callback order.
  NOTE: EPA/GJK use per-thread stack (64KB device stack limit already set) - lanes multiply local memory, acceptable at 7 lanes.
- ORIGINAL RAY PLAN (mesh-stride variant EXECUTED in round 3; subtree variant remains optional):
  KEY INVARIANTS (CudaRaycast.cuh): CuBvhWalkRay visits leaves in ASCENDING node index order (curIndex monotonic), does NOT prune by current-best fraction (lambda_max is full ray length) -> the visited-leaf SET and each leaf's CuRayTriangle result are order-independent; ties broken by visit order via strict < in CuRayTriangle ("first visited wins"). CuRayTestStatic iterates meshes then planes, sharing out.fraction with strict < (cross-mesh pruning is perf-only, cannot change the winner).
  THEREFORE bit-exact parallel scheme: each lane walks a subset of BVH SUBTREES (GpuBvhSubtree, already used by CuConvexVsMesh; flattened (mesh,subtree) index strided over lanes), each candidate hit carries (fraction, absoluteNodeIndex); lex-min reduction (float compares are exact) reproduces the sequential winner INCLUDING ties; mesh order folds into the key via absolute node index per mesh + mesh index; planes (cheap) then dynamic bodies stay sequential on lane 0 afterwards, seeded with the reduced best (strict < preserved). Need a subtree-ranged CuBvhWalkRay variant (walk nodes [root, root+size), same escape logic — mirror CuConvexVsMesh's subtree walk).
  KERNEL RESTRUCTURE: wheel phase becomes wheels x 8 lanes (bodyIdx: wheel = bodyIdx/8, lane = bodyIdx%8, 32 threads = 4 wheels). Split CuCarPreTickWheel: transform (lane 0) -> sync -> cooperative static-mesh ray (8 lanes + reduction) -> lane 0: planes + dyn bodies + CuCalcFrictionImpulse. Per-wheel shared staging extends sW[4] (ray endpoints from transform, reduced hit for friction). Block-wide __syncthreads at sub-phase boundaries is simplest and correct. MUST re-run all 18 gates + 4x ray-fuzz (the bench has ray-fuzz modes) + gpusim_validate after.
  ALSO PENDING: collision static-vs-dyn pairs are create-manifold-BEFORE-narrowphase (verified for tile/ball/car branches; dyn-dyn branch NOT yet read — check before parallelizing); per-dyn-body grouping is the safe grain (callbacks write per-body state: snowday groundStick + dropshot tile damage are ball-pair-only; car-car bump callbacks stay sequential).

2026-06-10: THE GPU KERNEL OWNS THE ENTIRE SIMULATION STEP. `--gpu-step` runs every
phase of every tick on-device (one launch per run) and reproduces the CPU reference
bit-exactly: stable gate (128 arenas x 1000 ticks, hash 3380507951560469826) PASS,
hard gate (20000 ticks, hash 11764255410181881605) PASS, plus an untested-config
cross-check (32 arenas x 4 cars x 2000 ticks seed 909090) hash-identical to CPU.
CPU gates still PASS (reference untouched).

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
1. **Soccar/heatseeker gates run with `DemoMode::DISABLED` (the bench default); demo gates pin the respawn RNG.** Upstream demolition respawn calls `Math::RandInt(.., seed=-1)` -> `Math::GetRandEngine()` (wall-clock + thread-id seeded). The work copy adds `Math::SeedRandEngine(uint64)`; the bench seeds it per arena (`DemoRngSeed = MixSeed(seed, arenaIdx + arenaOffset) ^ 0xDE30`) right after SetupArena, and the GPU path uploads the matching minstd_rand0 state (`s % 2147483647`, 0 -> 1) via `UploadRngState`. `--demo-mode {disabled,normal,on-contact}` selects the mutator; demos are now fully deterministic and gate-covered.
2. **Bench masks `BallState.dsInfo.lastDamageTick` when `!hasDamaged`.** That field has NO initializer upstream and is only written in dropshot, but `BallState::Serialize` writes it regardless. Unmasked, the state hash ingests uninitialized heap bytes — diagnosed as the root cause of a nondeterministic stable gate (per-arena hash dump + 4-byte blob diff at exactly that field's offset; only soccar-mode read of the field is inside a `gameMode == DROPSHOT` guard, so physics is unaffected).
3. **No `ResetToRandomKickoff`, explicit spawns instead.** It iterates `_cars` (a pointer-keyed `unordered_set`, ASLR-order) and uses `std::shuffle`/`std::default_random_engine` (implementation-defined). Bench sets car states explicitly.
4. **Bench randomness is splitmix64 with power-of-two float quantization** — input schedule is bit-identical on any platform/compiler. Never use std distributions for gate-relevant randomness.
5. **`unordered_set<Car*>` iteration-order hazard.** `Arena::Step` iterates `_cars` in pointer order for pre/post-tick. With the current gates this is provably benign (hashes stable across fresh processes), but any future change that makes per-car work order-dependent (e.g. same-tick boost pad contests) can break determinism. The GPU port must define a FIXED car order (by car id) and validate against the gates.
6. **`-ffp-contract=off` everywhere** so CPU semantics are plain IEEE mul+add, matchable on GPU with `nvcc -fmad=false`.

### Asset Provenance
- Meshes downloaded from `https://github.com/heresfilly09/rocket-league-collision-meshes` (plain `.cmf`), verified against upstream's known hash set. Game-derived, so NOT committed; re-download to `collision_meshes/soccar/` and the bench's `MESHES: OK (16/16 soccar)` line confirms integrity.

### GPU port progress (CUDA 100% — complete; history below)
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

**PORTED + VALIDATED BIT-EXACT (2026-06-10):** kernel phases predict (damping),
integrate (transform + inertia tensor), activation (deactivation/sleeping) — verified by
`--gpu-compare 20000` lockstep: every tick, CPU phase-boundary captures (rscap, hooks in
work btDiscreteDynamicsWorld.cpp) vs GPU phase segments, full GpuBodyState bitwise equal
for ball+2 cars. Validation command:
  `./build/rocketsim_bench --mesh-dir collision_meshes --ticks 1 --gpu-compare 20000 --cars 2 --seed 1337`
GPU bridge now carries BULLET-unit body state (basis/pos/vel/angVel/forces/invInertiaWorld/
activation); bench_work has SnapshotBody/WritebackBody + MakeBodyConsts (damping factors
precomputed host-side; sleep thresholds squared; noRot). Kernel supports phase windows
(RunPhaseSegment) for isolation; production StepTicks runs the full range in one launch.
`--gpu-step`/`--gpu-compare` REQUIRE glibc_trig_fix.bin (actionable error if missing).

**Vehicle port in progress (task: car pre-tick = 50% of CPU time). Decoded semantics
(2026-06-10, all from work-tree source):**
- RSQRTSS emulation DONE+VERIFIED (RsqrtHost.cpp capture, 12-bit mantissa + exponent
  parity hypothesis holds on this CPU, 4M-sample check at every init; CuNormalize3 /
  CuSafeNormalize3 / CuQuatAxisAngle in CudaMath, all fuzz-PASS). Trig table extended
  to negative domain (steering quats; 1.6M sin + 36k cos entries) - REGENERATE with
  cuda_math_test --emit-trig-table if missing.
- Suspension raycast chain: Car::_PreTickUpdate -> btVehicleRL::updateVehicleFirst
  (updateWheelTransform per wheel -> rayCast per wheel -> calcFrictionImpulses), then
  car game logic, then updateVehicleSecond (updateSuspension applyImpulse + friction
  applyImpulse). Cross-tick persistent wheel state: m_steerAngle, m_engineForce,
  m_brake, m_latFriction, m_longFriction (calcFrictionImpulses in tick N reads values
  set by _UpdateWheels in tick N-1!), m_suspensionForceScale.
- btDefaultVehicleRaycaster::castRay (RS-modified): ClosestRayResultCallback(from, to,
  ignoreObj=chassis); group = DefaultFilter|addedMask, mask = AllFilter. After closest
  hit found: requires btRigidBody + hasContactResponse() else returns NO HIT (a closest
  demoed car masks walls behind it!). Result normal gets normalize() (rsqrt path).
- needsCollision: (proxyGroup & cbMask) && (cbGroup & proxyMask), ignoreObj excluded.
  Soccar ray candidates: 16 meshes (static, identity transform -> local==world),
  4 planes (identity basis, origin=posBT; floor n=+z @0, ceiling n=-z @height,
  side walls n=+-x), ball (sphere), other cars (compound->box). Static bodies:
  group=StaticFilter(2), mask=All^Static.
- Mesh ray: rayTestSingleInternal concave branch -> btTriangleRaycastCallback::
  processTriangle (exact math read: plane-side products, distance=dist_a/(dist_a-dist_b),
  STRICT < hitFraction, edge_tolerance=-1e-4*len2(n), three cross/dot containment tests,
  normal.normalize(), backface flip if dist_a<=0). Fraction culling across triangles/
  objects is min-equivalent (brute force OK) EXCEPT exact ties: first-tested wins ->
  BVH/dbvt order tie risk, accepted for now (compare harness will catch).
- Plane ray: concave generic branch synthesizes 2 triangles from ray AABB in object-
  local space: btPlaneSpace1(n) tangents (uses btRecipSqrt = exact 1/sqrt, NOT rsqrtss),
  radius=halfExtents.length(), projectedCenter; tris ((+t0+t1),(+t0-t1),(-t0-t1)) and
  ((-t0-t1),(-t0+t1),(+t0+t1)) -> same processTriangle.
- Ball/car ray: rayTestSingleInternal convex branch: btSubsimplexConvexCast with point
  shape (margin 0) vs shape, needs btVoronoiSimplexSolver port (closest-pt barycentric);
  sphere support = normalized(dir)*radius (WITH margin path); box support = sign-select
  getHalfExtentsWithMargin; compound loops children (car: 1 box child with
  hitboxPosOffset transform). castResult.m_fraction starts at current closest; accept
  if normal len2>1e-4 && fraction strictly <; normal normalize()d.
- resolveSingleCollision + resolveSingleBilateral (btContactConstraint.cpp) still to
  read/port (used by extraPushback + side friction impulse).

**DONE (2026-06-10): static-world device raycast bit-exact.** CudaRaycast.cuh
(CuRayTestStatic: meshes brute-force + planes synthesized-quad), world geo extraction/
upload in bench_work + CudaStep (ExtractWorldGeo: 16 meshes/8020 tris/4 planes with
proxy filters). Validated: `--ray-fuzz 200000` PASS on seeds 1337 & 99 (hit/fraction/
normal bitwise vs CPU world rayTest filtered to statics).
KEY LESSON: identity-basis transforms are NOT bitwise no-ops - bullet's mat*vec dot
tree (1*x+0*y)+0*z canonicalizes -0 to +0; CPU applies it to ray endpoints and hit
normals (colObjWorldTransform). Device must run the same trees (CuMat3MulVec/
CuTransformPoint with explicit identity). Found via ray-fuzz: 4% mismatches, all and
only +-0 sign on normal components.
NOTE: RocketSim uses a custom btRSBroadphase ("voxel broadphase", warns on rays >7.4uu)
- relevant for the collision-phase port later.

**DONE (2026-06-10): FULL-world device raycast bit-exact** (`--ray-fuzz 200000` PASS,
seeds 1337+99, statics + ball sphere + car compound box). CudaConvexCast.cuh: faithful
btVoronoiSimplexSolver + btSubsimplexConvexCast (point cast, 32 iters, eps 1e-4,
equal-vertex 1e-4) + sphere/box supports. CRITICAL LESSONS:
- CPU re-lerps BOTH transforms each lambda update; lerp(o,o,lambda) = o*(1-l)+o*l is
  NOT bitwise o -> must interpolate the static shape's origin too.
- btRSBroadphase voxel grid GATES candidates: short rays (len^2 < cellSizeSq) test ONLY
  proxies registered in rayFrom's cell. Statics smear +-1 cell (brute force provably
  equivalent); DYNAMICS register in 3x3x3 around their aabbMin's cell -> device gate:
  |cellIdx(rayFrom) - cellIdx(dyn aabbMin)| <= 1 per axis (CuGetCellIndices replicates
  (pos-min)*(1/cellSize) trunc+clamp). CPU MISSES geometrically-hit dynamics outside
  the gate - reference behavior, bugs included. GpuDynBody carries aabbMin; production
  kernel must track proxy aabbMin per body (updated in collision phase / updateAabbs,
  NOT every position change - aabbMin lags to last updateAabbs!).
- Grid params live in btRSBroadphase (public): minPos/cellSize/cells dims; same grid
  drives COLLISION pair generation (calculateOverlappingPairs) for task 10.

**DONE: CudaVehicle.cuh constraint helpers** (CuResolveSingleCollision/Bilateral,
CuApplyImpulse, CuComputeImpulseDenominator, CuVelocityInLocalPoint) - trees from
btContactConstraint.cpp/btRigidBody.h/btJacobianEntry.h. erp: extract solverInfo.m_erp
host-side (default 0.2; check Arena overrides).

**Car pre-tick port: FULL INVENTORY DONE (all of Car.cpp + btVehicleRL read).**
Structure: demo branch (gates: disabled) -> updateVehicleFirst (wheel transforms via
steering axis-angle quat * column-assembled basis2, fwd=up x right then normalize()
[rsqrt]; rayCast per wheel [castRay semantics: ignore chassis, group Default|
DROPSHOT_FLOOR, hasContactResponse no-hit rule, normal normalize(); suspension length
clamp via std::clamp; relVel via denominator>0.1 branch; extraPushback =
resolveSingleCollision(...,applyImpulses=false)/numWheels when static ground and
wheelTraceLen < restLen+radius-SUSPENSION_SUBTRACTION]; calcFrictionImpulses [axleDir
from wheel basis getColumn(right), project off surface normal, safeNormalized;
forwardDir = surfNormal x axleDir safeNormalized; sideImpulse = resolveSingleBilateral;
rollingFriction: engineForce ? -engine/frictionScale : brake path with relVel
threshold rounding (timeStep>1/80 branch; 120Hz -> skipped) * 113.73963f clamp +-brake;
impulse = (fwd*rolling*longFric + axle*side*latFric) * (mass/3)]) -> wheelsWithContact/
isOnGround(>=3) -> _UpdateWheels [handbrake rise/fall clamp; throttle/brake state
machine (deadzone, STOPPING_FORWARD_VEL, BRAKING_NO_THROTTLE_SPEED_THRESH); engine =
throttle*THROTTLE_TORQUE*UU_TO_BT*driveSpeedScale(/4 if airborne); steer from speed
curves blended by handbrakeVal, applied to wheels 0,1; friction curve inputs from
(angVel x wheelDelta + vel)*BT_TO_UU dotted with lat/long dirs, baseFriction>5 ratio;
handbrake factor curves; non-sticky scale by normal.z when throttle==0; sticky force:
upwardsDirFromWheelContacts [sum contact normals, safeNormalized], fullStick,
scale 0.5 + (1-|up.z|), applyCentralForce] -> air torque (if <3 wheels) [flip torque:
applyTorque(invInertiaWorld.INVERSE() * basis * dodgeTorque); air control torque+
damping trees; THROTTLE_AIR_ACCEL central force] -> _UpdateJump [impulse/force
sequence, JUMP_PRE_MIN_ACCEL_SCALE 0.62] -> _UpdateAutoFlip [needs Angle::FromRotMat =
ATAN2F (exact port needed!); CAR_AUTOFLIP_* consts; angVel += fwd*torque*scale*dt] ->
_UpdateDoubleJumpOrFlip [dodgeDir safeNormalized 2D, Vec::Normalized() (CHECK its
tree), flip impulse scaling consts, flipRelTorque /tickTimeScale(=1 at 120Hz),
powf(1-FLIP_Z_DAMP_120, 1.0) z-damp (constant, precompute host)] -> _UpdateAutoRoll
(if throttle && (contact 1..3 wheels || worldContact)) [cross/dot/clamp trees, central
force + applyTorque(invInertiaWorld.inverse()*...)] -> worldContact.hasContact=false
-> updateVehicleSecond [updateSuspension: force=(rest-len)*stiffness*clippedInv;
damping rate compress/relax; *suspensionForceScale; max(0); applyImpulse(normal*
(force*dt + extraPushback), contactPoint-origin); friction: applyImpulse(impulse*dt,
contactOffset - upDir*(upDir dot contactOffset))] -> _UpdateBoost [state machine,
BOOST_MIN_TIME, applyCentralForce fwd*boostAccel(Ground|Air)*UU_TO_BT*CAR_MASS].
DECODED ADDITIONS: btMatrix3x3::inverse = cofac(r1,c1,r2,c2)=m[r1][c1]*m[r2][c2]-
m[r1][c2]*m[r2][c1]; co=(cofac(1,1,2,2),cofac(1,2,2,0),cofac(1,0,2,1)); det=row0.dot(co)
[SSE dot3 tree]; s=1/det; elements = cofac(...)*s per btMatrix3x3.h:1093.
Vec::Normalized() (RocketSim Vec): length = sqrtf(LengthSq) with LengthSq =
x*x+y*y+z*z+_w*_w (plain left-assoc, INCLUDES _w - different tree from bullet dot3!);
if len > FLT_EPSILON^2: operator/(float) = element-wise DIRECT division (x/val incl _w,
NOT reciprocal-multiply - differs from bullet operator/); else Vec() zero.
applyCentralForce: m_totalForce += force * m_linearFactor; applyTorque: m_totalTorque
+= torque * m_angularFactor (element mul by (1,1,1) - keep).
updateWheelTransform origin uses PREVIOUS tick's m_raycastInfo.m_suspensionLength ->
suspensionLength is another persistent GpuWheelState field (origin is physics-dead,
basis is what friction reads, but replicate both).
getForwardVector = basis COLUMN 1 (indexForward=1); getUpVector = COLUMN 2.
STILL TO DECODE: getEulerYPR/getEulerZYX (autoflip needs roll: ATAN2F+ASINF exact ports
needed - bounded domain, fuzz-verify; only on autoflip trigger path),
applyTorque/applyCentralForce accumulator trees (m_totalForce += force*m_linearFactor,
m_totalTorque += torque*m_angularFactor - confirm in btRigidBody.h), LinearPieceCurve
data extraction (RLConst curves -> arrays; GetOutput lerp tree read: Math.cpp:5),
wheel consts extraction (connection points = config offsets *UU_TO_BT with restLength
-= MAX_SUSPENSION_TRAVEL, m_suspensionRestLength1 = addWheel's value, radius per
front/back, SUSPENSION_SUBTRACTION const in RLConst::BTVehicle).
GpuWheelState persists: steerAngle, engineForce, brake, latFriction, longFriction,
suspensionForceScale (+ raycast scratch is intra-tick).
Capture plan: hooks in Car::_PreTickUpdate around updateVehicleFirst / game block /
updateVehicleSecond / _UpdateBoost; compare wheel scratch + body forces + car state.

**DONE (2026-06-10): CAR PRE-TICK PHASE BIT-EXACT ON GPU** - the 50%-of-CPU-time phase.
`--gpu-compare 20000` PASS (also 10k @ seed 99, 10k @ 4 cars seed 555): per car per tick,
CPU captures (rscap::CaptureCarPreTick hooks at entry/after-updateVehicleFirst/exit in
Car::_PreTickUpdate) vs device CuCarPreTick (CudaVehicle.cuh) - velocities, force/torque
accumulators, all CarState scalars/bools, wheel persists, and wheel scratch bit-equal.
Bench: ExtractCarConsts (RLConst scalars/folded products/curves via ExtractCurve,
mutators, wheel consts from live btWheelInfoRL, erp, frictionScale=mass/3) +
RunCarPreTickOnce single-shot kernel (CudaStep). Found+fixed root causes:
- setCoordinateSystem(1,2,0) OVERRIDES vehicle axis defaults: indexForward=0 (basis
  COLUMN 0), indexRight=1, indexUp=2 - affects forwardSpeed, basis2 column assembly,
  friction axleDir (column 1).
- m_extraPushback is CROSS-TICK PERSISTENT (only written on no-hit (0) or threshold
  fire; uncompressed contact keeps stale value) -> moved into GpuWheelState.
- Double jump uses RLConst::JUMP_IMMEDIATE_FORCE, NOT mutatorConfig.jumpImmediateForce.
- Mat*vec scaled-by-const trees: (inv * v) * scale, scale AFTER the matrix multiply.
CAVEAT: autoflip atan2f/asinf double-rounded stand-ins (CuAtan2f/CuAsinf) - not yet
proven to trigger in compare runs; fuzz atan2/asin vs glibc still pending (bounded
domain |args|<=~1).

**Collision phase decoded so far (2026-06-10):**
- btRSBroadphase::calculateOverlappingPairs: ALL activePairs REMOVED then re-added EVERY
  tick -> algorithms+manifolds are PER-TICK (no warm starting, no cross-tick contact
  state!) - massive port simplification. Pair candidates per dynamic proxy (handle
  order = creation: statics, ball, cars): statics in proxy's cellIdx cell + dyns in
  same cell (if numDynProxies>1); aabbOverlap = TestAabbAgainstAabb2; findPair dedups;
  ADD ORDER (-> dispatch -> solver row order) = iteration order, replicate exactly.
- updateSingleAabb (per ACTIVE object only - inactive keep STALE proxy AABB/cells):
  AABB = shape getAabb(worldTransform) +- gContactBreakingThreshold(0.02) UNIONED with
  getAabb(interpolationWorldTransform) +- 0.02 (m_useContinuous default true).
  CORRECTION to earlier note: the interpolation transform written by predict IS read
  here - but predict+collision are in-tick adjacent, so it can be recomputed as tick
  scratch (integrateTransform of post-damping state); no cross-tick storage. Proxy
  aabbMin/aabbMax + cell membership DO persist across ticks (GpuBodyState must carry).
  setAabb early-outs if AABB unchanged; dyn cell membership only changes when aabbMin's
  cellIdx changes (and only if numDynProxies>1).
- Narrowphase algo map for soccar pairs: sphere/box-vs-trimesh = btConvexConcave
  (per-triangle GJK+EPA via btConvexTriangleCallback + btInternalEdgeUtility adjust),
  compound(car) wraps child box, box-box = btBoxBoxCollisionAlgorithm (SAT detector),
  box-sphere = btConvexConvex (GJK+EPA), box/sphere-vs-plane = btConvexPlane.
  Contact callback: CF_CUSTOM_MATERIAL_CALLBACK -> Arena::_BulletContactAddedCallback
  (ball hit info, bumps, worldContact.hasContact). Manifold mechanics per tick:
  btPersistentManifold add/refresh/4-point reduction.

- btConvexPlaneCollisionAlgorithm::processCollision READ: support pt vs plane in local
  spaces (transform inverse/compose trees), distance vs breaking threshold (+closestPoint
  threshold 0), addContactPoint(planeWorldNormal, projected pt, dist); polyhedral
  (box) perturbation loop when numContacts < m_minimumPointsPerturbationThreshold:
  perturbeRot quats around plane-space tangents, angle = gContactBreakingThreshold /
  getAngularMotionDisc clamped pi/8 (check CreateFunc defaults for iters/threshold).
- btManifoldResult::addContactPoint READ: depth>breakingThreshold reject; pointA =
  pointInWorld + normal*depth; localA/B via invXform; getCacheEntry -> replace or add;
  combined friction/restitution (RS-MODIFIED: static pair -> min(friction)/
  max(restitution), dynamic pair -> a*b); btPlaneSpace1 for lateral friction dirs;
  partId/index stored (swap-aware); gContactAddedCallback when CF_CUSTOM_MATERIAL
  (always: ball+cars have it) AFTER insert, with obj0/obj1 swap-normalized.

- btPersistentManifold READ: ROCKETSIM CHANGE - getCacheEntry returns -1 ALWAYS (no
  point replacement; every contact adds). addManifoldPoint: when cache full (4),
  insertIndex = sortCachedPoints(new) [keeps deepest + maximizes area via cross
  length2 of 4 candidate triangles; gContactCalcArea3Points default?], else append.
  Manifold contactBreakingThreshold = min over both shapes of
  shape->getContactBreakingThreshold(gContactBreakingThreshold=0.02) =
  getAngularMotionDisc()*0.02 (PER-SHAPE; compute host-side per pair type).
  refreshContactPoints: still to read tail (prunes pts beyond threshold by recomputed
  distance; called post-processCollision by compound/concave paths?).

- Arena::_BulletContactAddedCallback READ (Arena.cpp:245): skip if either lacks contact
  response; swap so lower userIndex is A (car=?<ball userindex values: BT_USERINFO_TYPE_CAR
  vs BALL - check constants); car-ball -> ball->_OnHit(relBallPos local pt *BT_TO_UU,
  combinedFriction/Restitution, tickCount); car-car -> OVERRIDE manifold friction=
  CARCAR_COLLISION_FRICTION restitution=CARCAR..., two-way bump test (cooldown by
  carContact, vel toward other, speed compare, bumper localPoint.x*BT_TO_UU >
  BUMP_MIN_FORWARD_DIST, demo mode branches - READ TAIL pending Arena.cpp:360-470);
  car-world -> _BtCallback_OnCarWorldCollision (sets worldContact - read tail);
  ball-world -> ball->_OnWorldCollision + m_isSpecial=true (non-snowday).
  THEN btAdjustInternalEdgeContacts always (swap-aware args).
- Bump/demo + car-world callback tails READ (Arena.cpp:360-415): demo decision by
  demoMode (gates: DISABLED -> isDemo=false) + team check; bump: baseScale from
  BUMP_VEL_AMOUNT_GROUND/AIR curve(speedTowardsOtherCar), hitUpDir = other isOnGround ?
  car2 GetUpDir : (0,0,1); bumpImpulse = velDir*baseScale + hitUpDir*UPWARD_curve*
  mutator.bumpForceScale (Vec trees); car2->_velocityImpulseCache += bumpImpulse*
  UU_TO_BT; car1 carContact={car2 id, mutator.bumpCooldownTime}. Car-world:
  worldContact={true, m_normalWorldOnB}; manifold friction=mutator.carWorldFriction,
  restitution=mutator.carWorldRestitution (overrides AFTER combine).
- Arena ctor: btRSBroadphase(minPos*UU_TO_BT, maxPos*UU_TO_BT, maxAABBLen*UU_TO_BT*
  cellSizeMult(1 heavy), pairCache=btHashedOverlappingPairCache, maxObjects).
  STILL TO READ: Ball::_OnHit (extra impulse + BALL_CAR_EXTRA_IMPULSE curve +
  ballHitInfo writes), Ball::_OnWorldCollision, btAdjustInternalEdgeContacts
  (btInternalEdgeUtility.cpp), GJK (btGjkPairDetector), EPA (btGjkEpa2),
  btConvexConcave, btBoxBox, btCompound, dispatcher needsCollision/order,
  refreshContactPoints tail, sortCachedPoints tail + gContactCalcArea3Points,
  convex-plane CreateFunc perturbation defaults, solver handoff.

**CONFIRMED FINAL narrowphase inventory for soccar (config map + CreateFunc defaults
verified in source 2026-06-10):**
- PERTURBATION LOOPS ARE DEAD: convexConvex CreateFunc m_numPerturbationIterations=0;
  convexPlane CreateFunc iters=1 but threshold=0 (numContacts<0 never) -> single
  contact per algo invocation, no perturbation anywhere.
- box-box (car-car) -> btBoxBoxCollisionAlgorithm (btBoxBoxDetector ODE SAT ~400 ln).
- sphere-box (ball-car child) -> btSphereBoxCollisionAlgorithm (analytic ~200 ln).
- sphere-triangle (ball-mesh tris) -> btSphereTriangleCollisionAlgorithm +
  SphereTriangleDetector (analytic ~200 ln).
- box-triangle (car-mesh tris) -> convexConvex PLAIN path: single
  btGjkPairDetector::getClosestPoints (+ btGjkEpaPenetrationDepthSolver/btGjkEpa2 when
  penetrating); box has NO convexPolyhedron (initializePolyhedralFeatures never called
  in RS) so polyhedral SAT/clipping paths are SKIPPED. input.m_maximumDistanceSquared =
  (marginA+marginB+manifold breaking threshold)^2.
- convex-plane (ball/car vs 4 planes) -> READ DONE, single support contact.
- Ball::_OnHit READ: friction/restitution overrides CARBALL_*; ballHitInfo writes;
  extra impulse guard (tickCount > lastApplied+1 || lastApplied > tickCount); RS-Vec
  trees: hitDir=(relPos*Vec(1,1,Z_SCALE)).Normalized(), forward adjustment, addedVel =
  (hitDir*relSpeed)*CURVE(relSpeed)*mutator.ballHitExtraForceScale ->
  ball._velocityImpulseCache += *UU_TO_BT. Ball world collision: SOCCAR NO-OP except
  manifold m_isSpecial=true (check m_isSpecial consumer in solver!).
- SphereTriangleDetector READ (RS uses VirxEC fast variants!): collide() in TRIANGLE
  LOCAL space (sphereInTr = transformB.inverseTimes(transformA)); tri normal cross,
  l2>=eps^2, normal /= sqrt(l2) [recip-mul], distanceFromPlane sign-flip, inside-plane
  if < radius+breakingThresh; face containment via BARYCENTRIC pointInTriangle
  (u x w . n / nLenSq trees); else closestPointTriangle (embree-style, SAME structure
  as voronoi ClosestPtPointTriangle but RETURNS POINT: a + v*ab forms); contact:
  contactToCentre dist^2 < (r+thresh)^2 -> depth=-(radius-distance),
  normal=normalize(contactToCentre) [RSQRT] (or tri normal + depth=-radius if
  dist2<=eps); output world: transformB.getBasis()*normal, transformB*point,
  addContactPoint. swapResults variant flips.
- btSphereBoxCollisionAlgorithm READ: sphere center in box local (invXform); closest
  pt = per-axis min/max clamp vs halfExtentsWithoutMargin; normal = rel - closest;
  reject if dist2 > (radius+boxMargin+maxContactDistance)^2; if dist2<=SIMD_EPSILON:
  deep face fallback getSpherePenetration (6-face min dist, axis normals, sets
  closestPoint on face) distance = -minDist; else distance=length, normal/=distance
  [recip-mul]; pointOnBox = closest + normal*boxMargin -> world via m44T(pointOnBox),
  normal via basis*normal; penetrationDepth = distance - (radius+boxMargin);
  addContactPoint(normal, pointOnBox, depth); refreshContactPoints if contacts.

- Dispatcher glue READ: needsCollision = (body0 active || body1 active) &&
  checkCollideWith both ways (default true - no custom overrides in RS);
  dispatchAllCollisionPairs -> processAllOverlappingPairs over the hashed pair cache
  ARRAY (insertion order; pairs rebuilt each tick so order = RS broadphase add order);
  defaultNearCallback: needsCollision -> find/create algo from
  m_doubleDispatchContactPoints[type0][type1] -> algo->processCollision(wrappers).
  needsResponse (solver-side): both haveContactResponse && not both static.

- btGjkPairDetector::getClosestPointsNonVirtual FULLY READ (btGjkPairDetector.cpp:688):
  positionOffset = (originA+originB)*0.5 subtracted from both local transforms (!);
  margins from getMarginNonVirtual (box getMarginNV 0.04, triangle getMarginNV 0.04,
  sphere margin=radius); init m_cachedSeparatingAxis=(0,1,0); REL_ERROR2=1e-6 (float);
  loop: support dirs = (-axis)*basisA [vec*mat tdot] / axis*basisB; supports
  localGetSupportVertexWithoutMarginNonVirtual (box: fsel on implicitShapeDimensions;
  sphere: (0,0,0); triangle: read case still pending - dot-select of 3 verts);
  pWorld=localTransA(pInA); w=p-q; delta=axis.dot(w); exits: delta>0 && delta^2 >
  sqDist*maxDistSq (deg10); inSimplex(w) (deg1); f0=sqDist-delta <= f1=sqDist*REL_ERROR2
  (deg2 if f0<=0 else deg11); addVertex; !closest (deg3); newAxis.len2<REL_ERROR2
  (deg6, axis=newAxis); !improving prevSq-sq <= eps*prevSq (deg12, axis NOT updated);
  iter>1000; fullSimplex (deg13, numVertices==4). checkSimplex path: compute_points,
  normalInB=axis; lenSqr<0.0001 -> deg5; if lenSqr>eps^2: rlen=1/sqrt(lenSqr),
  normalInB*=rlen, s=sqrt(sqDist), pointOnA -= axis*(marginA/s), pointOnB +=
  axis*(marginB/s), distance=(1/rlen)-margin, isValid. Penetration: if !isValid ||
  (catchDegeneracies(1) && pdSolver && degenerate && distance+margin<0.01): axis.setZero,
  EPA calcPenDepth(simplexSolver, shapes, localTransA/B, axis, tmpA, tmpB); valid2:
  tmpNormal=tmpB-tmpA (fallback axis if tiny), /=sqrt(lenSqr), distance2=-(tmpA-tmpB)
  .length(); replace if !isValid||distance2<distance (method3); EPA-fail fallback:
  if axis.len2>0: distance2=(tmpA-tmpB).length()-margin; replace if less: points -=/+=
  axis*margins, normal=axis normalize() [RSQRT]. Final accept: isValid && (distance<0
  || distance^2 < maxDistSq); m_fixContactNormalDirection(1): aabb-center diff
  (getAabb both shapes at local transforms!) dot normal < 0 -> normal *= -1;
  output.addContactPoint(normalInB, pointOnB+positionOffset, distance).

### Next Steps (in order)
1. SOLVER PORT (btSequentialImpulseConstraintSolver + islands) - READING DONE for
   the orchestration shell (2026-06-10), port next. Decoded so far:
   - calculateSimulationIslands: islandManager->updateActivationState: island tag =
     sequential index over NON-static objects in m_collisionObjects order (=> ball 0,
     car0 1, car1 2; statics tag -1/companion -2) + initUnionFind + findUnions over
     the PAIR ARRAY (not manifolds! aabb-overlap pairs merge islands when both
     mergesSimulationIslands) -> storeIslandActivationState (tag=find(index),
     m_sz = collision-object array index). No constraints/predictive manifolds in RS.
   - solveConstraints: m_sortedConstraints empty -> setup(solverInfo,0,0);
     buildAndProcessIslands: buildIslands: unionFind.sortIslands(); per island:
     all-sleeping (no ACTIVE_TAG/DISABLE_DEACTIVATION member) -> ALL set
     ISLAND_SLEEPING; else sleeping members -> WANTS_DEACTIVATION + deactTime=0
     (the island WAKE mechanism). Then m_islandmanifold built from dispatcher
     manifolds IN ORDER (no numContacts filter since m_deterministicOverlappingPairs
     false) filtered: either body !ISLAND_SLEEPING && needsResponse (hasContactResponse
     both); kinematic-activate quirks skipped (no kinematics).
   - processIslands: m_islandmanifold.quickSort(by islandId only - bullet quickSort
     order must be replicated for ties!); per island (unionFind element order):
     bodies pushed in m_sz order; if !islandSleeping: callback->processIsland.
   - InplaceSolverIslandCallback: minimumSolverBatchSize=128 default -> islands
     ACCUMULATE (bodies/manifolds appended) and solve in ONE solveGroup at final
     processConstraints() (our scale never exceeds 128) -> single batched
     solveGroup(all active islands' bodies, all their manifolds in sorted order).
   - solverInfo: defaults numIterations=10, sor=1, erp=0.2, globalCfm=0, linearSlop=0,
     warmstartingFactor=0.85, restitutionVelocityThreshold=0.2, splitImpulse=true,
     splitImpulseTurnErp=0.1, minimumSolverBatchSize=128, leastSquaresResidualThreshold=0,
     solverMode = SOLVER_USE_WARMSTARTING | SOLVER_SIMD. ARENA OVERRIDES:
     m_splitImpulsePenetrationThreshold = 1.0e30f (=> rhs/rhsPenetration SPLIT path for
     ALL contacts: rhs = velocityImpulse only, rhsPenetration = penetrationImpulse),
     m_erp2 = 0.8. m_timeStep set by stepSimulation flow (1/60 DEFAULT on tick 0
     until set - the car pre-tick capture showed solverTimeStep semantics; VERIFY
     where it's assigned when wiring).
   - RESOLVE FUNCTIONS: BT_ALLOW_SSE4 is MSVC-only -> Linux GCC uses the _sse2
     variants (USE_SIMD=1 via btSolverBody.h). Trees to replicate:
     btSimdDot3(a,b) = m0 + (m1 + m2) [x + (y + z) ORDER, unlike (x+y)+z elsewhere!].
     RowGeneric_sse2: deltaImpulse = rhs - applied*cfm; deltaVel1Dotn = dot3(n1,
     dLinA) + dot3(rp1xn, dAngA); same B; deltaImpulse -= dv1*jacInv; -= dv2*jacInv;
     sum = applied + deltaImpulse; if sum < lower: deltaImpulse = lower-applied,
     applied = lower; (elif) sum < upper? keep : deltaImpulse = upper-applied,
     applied = upper [mask logic = exactly two nested selects]; then dLin += (n *
     invMass) * deltaImpulse, dAng += angComp * deltaImpulse for A and B.
     RowLowerLimit_sse2: same minus the upper clamp. SplitPenetrationImpulse_sse2:
     same shape on push/turn velocities with rhsPenetration (read its exact body +
     btSolverBody.h SIMD internalApplyImpulse/getVelocityInLocalPointNoDelta/
     writebackVelocityAndTransform trees when porting - USE_SIMD structs use SSE!).
   - solveGroup = Setup + Iterations + Finish:
     SETUP: convertBodies (companionId=-1 for all; pool reserve(n+1) resize(0); per
     body getOrInitSolverBody [pool in bodies[] order; statics->fixed body shared];
     initSolverBody: m_externalForceImpulse = totalForce*invMass*dt [THE force->vel
     integration], m_externalTorqueImpulse = totalTorque*invInertiaWorld*dt; no gyro
     flags in RS) -> convertContacts in manifold order (per contact dist <=
     processingThreshold(LARGE): expand contact constraint; m_frictionIndex =
     frictionPool.size(); setupContactConstraint:
       relaxation=sor=1; invTimeStep; cfm=0*invTimeStep; erp=0.8;
       torqueAxis0 = rel_pos1 x n; angCompA = invInertiaWorld*axis*angFactor(1);
       torqueAxis1 = rel_pos2 x n; angCompB = invInertiaWorld*(-axis)*angFactor;
       denom0/1 = invMass + n.dot(angComp x rel_pos) [B: (-angCompB) x rel_pos2];
       jacDiagABInv = relaxation/(denom0+denom1+cfm);
       contactNormal1 = n, relpos1CrossNormal = torqueAxis0; normal2 = -n,
       relpos2CrossNormal = -torqueAxis1 (fixed body: zeros);
       penetration = dist + linearSlop(0); rel_vel via rb->getVelocityInLocalPoint
       (REAL body vels, not solver); restitution = restitutionCurve(rel_vel,
       combinedRestitution, 0.2): |rel_vel| < thresh -> 0 else rest*-rel_vel; clamp<=0->0;
       WARMSTARTING ON: appliedImpulse = cp.m_appliedImpulse(0 fresh)*0.85 ->
       internalApplyImpulse with 0 (replicate arithmetic anyway); appliedPush=0;
       vel1Dotn = n1.dot(linVel+extForceImp) + rp1xn.dot(angVel+extTorqueImp) [SOLVER
       body vels]; rel_vel = vel1Dotn+vel2Dotn; velocityError = restitution - rel_vel;
       positionalError = penetration>0 ? 0 : -penetration*erp*invTimeStep;
       Arena threshold 1e30 => rhs = velocityImpulse = velErr*jacInv, rhsPenetration =
       posErr*jacInv; m_cfm = cfm*jacInv (0); lower=0, upper=1e10;
       RS: solverConstraint.m_isSpecial = cp.m_isSpecial)
     + RS isSpecial accumulation into NON-STATIC bodies' m_specialResolveInfo
       {num++, friction=cp.combined (LAST wins), restitution=cp, totalNormal +=
       n, totalDist += rel_pos(of that body).length()}
     + convertContactInner: no rolling friction (combined=0); friction dir:
       vel = solverA.getVelocityInLocalPointNoDelta(rel_pos1) - B(...) [uses
       linVel+extForce? READ btSolverBody fn]; lateralDir1 = vel - n*rel_vel;
       lat_rel_vel = len2; if > SIMD_EPSILON: dir1 *= 1/sqrt (btSqrt!) ->
       addFrictionConstraint(dir1, desiredVel=0, cfmSlip=0) else planeSpace1(n) ->
       addFrictionConstraint(dir1); ONE friction row (no 2-dir flag);
       setFrictionConstraintImpulse: frictionPool[m_frictionIndex].appliedImpulse=0.
       (READ addFrictionConstraint tail: rhs computation w/ desiredVelocity.)
     + per body in bodies[] order with specialResolveInfo.num>0: convertContactSpecial:
       distance = totalDist/num; normal = totalNormal/num (UNNORMALIZED avg);
       synthetic btManifoldPoint {dist, normal, lastFriction, lastRestitution,
       isSpecial=false}; frictionIndex = contactPool.size() BEFORE expand; bodyA =
       the body, bodyB = FIXED body (created on demand); rel_pos1 = normal*-distance,
       rel_pos2 = 0; setupContactConstraint + convertContactInner (collObj1=NULL).
       QUIRK: the synthetic constraint's m_frictionIndex is STALE GARBAGE from the
       pool slot (never assigned) -> setFrictionConstraintImpulse writes 0 to a stale
       friction index - VALUE-BENIGN (all appliedImpulses already 0 fresh); skip on
       device but remember. THEN body->m_specialResolveInfo = {} (cleared).
     ITERATIONS: solveGroupCacheFriendlySplitImpulseIterations FIRST (splitImpulse
     on): numIterations loop over ALL contact rows IN ORDER (orderTmpConstraintPool =
     identity; NO isSpecial skip here!) resolveSplitPenetrationImpulse; early-break
     when residual^2max <= 0 (threshold) - exact FP compare; then main loop
     iteration 0..9 (NO early exit): solveSingleIteration: (no joints) NON-interleave
     path: contact rows in order SKIPPING m_isSpecial; then friction rows in order:
     totalImpulse = contactPool[m_frictionIndex].appliedImpulse; if > 0: lower=-fric*
     total, upper=+fric*total, resolveRowGeneric. (Special contacts' friction rows
     skip via totalImpulse==0.)
     FINISH: writeBackContacts (warmstarting on): pt->m_appliedImpulse etc. (manifold
     pts - released next tick, no sim effect; RS skips null pts) -> writeBackBodies:
     writebackVelocityAndTransform(dt, splitImpulseTurnErp=0.1) [READ btSolverBody.h:
     linVel += deltaLin, angVel += deltaAng; transform from push/turn velocities via
     btTransformUtil::integrateTransform if push/turn nonzero] -> body setLinearVelocity
     (m_linearVelocity + m_externalForceImpulse) / setAngularVelocity(+extTorque) ->
     setWorldTransform(solverBody.m_worldTransform) [split impulse: position correction!]
     -> companionId=-1. Pools resizeNoInitialize(0).
   - solverInfo.m_timeStep = timeStep assigned in internalSingleStepSimulation
     (btDiscreteDynamicsWorld.cpp:439) BEFORE solveConstraints each tick = 1/120
     always during bullet step (the 1/60 default only matters for the car pre-tick
     which reads it BEFORE the first bullet step - already handled/validated).
   - btSolverBody::getVelocityInLocalPointNoDelta = (linVel + extForceImp) +
     (angVel + extTorqueImp).cross(rel_pos) (zero for fixed body).
     writebackVelocityAndTransform: linVel += deltaLin; angVel += deltaAng; if any
     push/turn component != 0: integrateTransform(NoRot if m_noRot)(worldTransform,
     pushVel, turnVel * 0.1, dt) -> m_worldTransform (POSITION CORRECTION).
   - SplitPenetration_sse2: early `if (!c.m_rhsPenetration) return 0` (exact FP !=0
     test); deltaImpulse = rhsPen - appliedPush*cfm; dotn trees on PUSH/TURN
     velocities; LOWER clamp only (upper mask computed but unused, same as scalar);
     push += (n*invMass)*delta, turn += angComp*delta.
   - setupFrictionConstraint (full): appliedImpulse=0 (NO friction warmstart);
     jacDiagABInv = relaxation/(denom0+denom1) [NO cfm]; vel1Dotn = n1.dot(linVel+
     extForceImp) + rp1xn.dot(angVel [NO extTorque!]); velocityError = desired(0) -
     rel_vel; rhs = velErr*jacInv (no friction anchor flag); rhsPenetration=0;
     cfm=cfmSlip(0); lower=-friction, upper=+friction (overwritten per iteration
     by +-fric*totalImpulse anyway). addFrictionConstraint: expand + m_frictionIndex
     = contact idx + setup.
   - btSolverBody internalApplyImpulse(linComp, angComp, mag): deltaLin += linComp*
     mag, deltaAng += angComp*mag (SSE per-lane = scalar-identical).
   - rb->getVelocityInLocalPoint(rel) = m_linearVelocity + m_angularVelocity.cross(rel).
   SOLVER PORTED + VALIDATED (commit 5154fe6): CudaSolver.cuh (CuSolveTick: union-
   find islands [path compression, quickSortPairs replicating btAlignedObjectArray
   quickSortInternal mid-pivot], island sleep/wake, manifold filtering+sort,
   batched solveGroup: solver body pool, convertContacts + RS special accumulation +
   convertContactSpecial, split-impulse iterations [early-break residual<=0], 10
   main iterations [contacts skip isSpecial, friction limits from owning contact],
   finish writeback incl. split-impulse position integration). GPU_PHASE_SOLVE in
   the kernel consumes the SAME-RUN collide outputs (manifolds + NEW pairs buffer);
   CuSolveCtx scratch lives in GLOBAL memory (dSolveScratch - 250KB would blow the
   64KB device stack: illegal-memory-access root-caused). gpu-compare runs a chained
   AFTER_PREDICT->(COLLISION+SOLVE) segment vs AFTER_SOLVE snaps: PASS 20000 ticks
   x seeds 1337/42/7777/123456789; gates PASS.
   BUGS root-caused during validation (recorded for posterity):
   - initSolverBody extTorqueImpulse: totalTorque * invInertiaWorld = btVector3 *
     btMatrix3x3 = COLUMN dots (CuVecMulMat); invInertiaWorld is NOT bitwise
     symmetric -> row-dot order gave 1-4 ulp angVel drift.
   - gResolveSingleConstraintRowGeneric_sse2 caches cpAppliedImp AT ENTRY: BOTH
     clamp deltas (lower-applied / upper-applied) use the ORIGINAL applied impulse,
     not the lower-clamp-updated one. Wrong order made friction rows diverge wildly
     once limits clamped (seed 42 tick 980).
   DIAGNOSTIC INFRA (keep): rscap::CaptureSolver hook in solveGroup (between
   iterations and finish; pools are public) captures contact/friction rows + solver
   bodies; rscuda::DownloadSolveDebug dumps the GPU scratch; the bench solve-FAIL
   block diffs them row by row. Same pattern for future phases.
   GAME TAIL + ASSEMBLY (DONE, commit 4fbd3de): CudaGameTail.cuh (ball sleep check
   [exact ==0 length2], applyGravity [active only; gravityForce = acc*(1/invMass)
   in GpuBodyConsts], clearForces, car post-tick [supersonic, contact cooldown,
   lastControls], car/ball finish [impulse caches + rsqrt-normalized vel clamps],
   pad pre/grid-check/post [grid int-truncation indices, prevLockedCarID box-vs-
   cylinder split, Vec all-component compares]). GpuStepArenasKernel runs phases
   GAME_PRE(0)..GAME_POST(6) per tick: car pre-ticks SEQUENTIAL in the CPU's
   per-arena _cars unordered_set order (UploadCarOrder permutation; later cars see
   earlier cars' impulses; ground-body friction impulses written back), controls
   from the uploaded schedule, solverTimeStep = 1/60 on tick 0 else 1/120.
   Bench RunGpu uploads everything (geo/BVH/collide/tail/proxies/ball state/car
   order/schedule) after one updateAabbs(); downloads + writes back into arenas
   (incl. wheel persists, ball velocityImpulseCache + tickCountSinceUpdate) and
   hashes with the SAME HashArena as the CPU path.
   LAST BUG (stable gate FAIL 128 arenas -> arena 2 only): static CuContactBodyInfo
   used origin=0 for ALL statics; correct for meshes but PLANES carry their position
   in the rb origin (side wall x=-81.92). localPointB landed in the wrong frame and
   refresh pruned the contact via the 2D-distance check. The floor plane (origin
   0,0,0) masked it through every prior validation. Diagnosed with --arena-offset/
   --no-ball-motion flags replaying arena 2 standalone under the per-phase compare.
   GOAL EXPANDED (2026-06-10, user directive: "no exceptions - everything CUDA"):
   port ALL remaining game modes + demolitions to device, each validated with NEW
   CPU-reference gates built the same way (work-CPU vs work-GPU bit-exact).
   Standard: the CPU performs ZERO simulation computation; host does only file
   I/O, upload, download. Already true for soccar (ball tickCountSinceUpdate moved
   on-device, commit 92bbed7; host thread sleeps at 0% CPU during the kernel -
   blocking-sync flag, commit 8df3f10).
   ORDER (tasks #13-#17):
   1. HEATSEEKER (#13, IN PROGRESS): no new assets/shapes. READ so far:
      - Ball::_PreTickUpdate HS branch (steer ball toward goal target: Angle::FromVec
        [atan2f - CuAtan2f covers], deltaAngle = GetDeltaTo (READ NEXT), blend yaw/
        pitch by speedRatio*tickTime*BLENDs, NormalizeFix [WrapNormalizeFloat =
        DOUBLE fmod of floats - exact, port direct], clamp pitch +-MAX_TURN_PITCH,
        RoundAngleUE3 [int quantize (val*TO_INTS)>>2 & 0x3FFF, * BACK_TO_RADIANS],
        GetForwardVec [cosf/sinf FULL RANGE! see below], newVel = dir*speed blend,
        m_linearVelocity = newVel*UU_TO_BT, timeSinceHit += tickTime).
      - _OnHit HS branch + _OnWorldCollision HS wall bounce: read earlier (memory).
      - Consts: RLConst::Heatseeker (INITIAL_TARGET_SPEED 2900, INCREMENT 85,
        TARGET_Y 5120, TARGET_Z 320, blends 1.45/0.78/0.3, MAX_TURN_PITCH
        7000*pi/2^15, MAX_SPEED 4600, WALL_BOUNCE_*).
      - Arena ctor HEATSEEKER: ball start pos/vel = Heatseeker::BALL_START_* with
        WALL-CLOCK-random y sign -> heatseeker gates must SetState the ball
        explicitly (seed-derived sign) in SetupArena.
      - DONE (commit 42b4da2): glibc sinf/cosf ported to device FULL-RANGE
        (CudaMath.cuh CuSinf/CuCosf rewritten: glibc s_sinf/cosf generic algorithm
        + the FMA-build contraction the host IFUNC actually dispatches to
        (__sinf_fma; proven by disassembly + an exhaustive HOST reference matching
        all 8.56e9 evals): explicit fma() on every poly mul-add and reduce_fast's
        x - n*hpi. GPU exhaustive: ALL 2^32 finite floats PASS for both. Gates
        PASS. cuda_math_test gained --trig-exhaustive-full.
        CLEANUP PENDING: the old TrigFix table machinery (g_trigFix/CuTrigLookup/
        CuLoadTrigFix/--trig-table/RequireTrigTable/--emit-trig-table +
        glibc_trig_fix.bin) is now UNUSED dead weight - remove it all, then re-run
        gates + a 20k compare.
      HEATSEEKER PORT WRITTEN (uncommitted at this checkpoint): bench --game-mode
      flag + deterministic HS ball init (SetState w/ seed-derived y-sign +
      yTargetDir set from tick 0 for steering coverage); GpuBallGameState += hs
      fields (uploaded/written back incl. hsInfo for the hash); GpuCollideConsts
      += hs* consts (extraction in ExtractCollideData); CudaGameTail.cuh: CuAngle
      helpers (CuWrapNormalizeFloat [double fmod exact], CuAngleFromVec,
      CuRoundAngleUE3, CuAngleForwardVec) + CuBallPreTickHS; HS branches in
      CuContactAddedCallback (_OnHit speedup block AFTER the cooldown-else-RETURN
      restructure [CPU returns from _OnHit on cooldown], wall bounce in ball-world)
      + kernel GAME_PRE hook after pads. FIXED latent bug: SNOWDAY enum is 3 not 4
      (isSpecial gate used 4=DROPSHOT). GameMode: SOCCAR0 HOOPS1 HEATSEEKER2
      SNOWDAY3 DROPSHOT4.
      VALIDATION: soccar gates PASS; HS 5000x4ar/20000x1ar seed1337 MATCH;
      HS 5000x8ar seed42 DIVERGES at arena0 tick 1402 -> per-phase compare
      (--game-mode heatseeker --no-ball-motion) pinpointed CAR PRE-TICK wheel-2
      RAY: contact POINT + hardPoint match BITWISE, only the NORMAL differs ~1e-4.
      ROOT CAUSE IDENTIFIED: ray hit a triangle EDGE; two triangles return the
      BIT-IDENTICAL fraction; strict < tie-break keeps the FIRST-VISITED triangle.
      CPU visits triangles in btQuantizedBvh RAY-walk order
      (reportRayOverlappingNodex); CuRayTestStatic predates the BVH port and scans
      mesh triangles LINEARLY -> different tie winner -> different normal.
      FIXED + HEATSEEKER VALIDATED: CuBvhWalkRay in CudaRaycast.cuh ports
      btQuantizedBvh::walkStacklessQuantizedTreeAgainstRay (RAYAABB2 path: linear
      depth-first node scan, quantized-box test + btRayAabb2 slab test with
      safeNormalized inverse direction, escape-index skips); GpuWorldGeo carries
      bvhs/bvhNodes (set by UploadMeshBvhs; GpuMeshBvh grew numNodes); ray fuzz
      callers updated (RunRayFuzz now uploads BVHs too).
      RESULTS: ray-fuzz 200k PASS; the failing trajectory (HS seed42 tick 1402)
      per-phase PASS; SOCCAR gates PASS + 20k compare PASS (the ray walk changed
      every wheel ray path - revalidated); HEATSEEKER: 8ar x5000 seed42 MATCH,
      20000x1 seed1337 MATCH (hash 9090346742586108841), 64ar x1000 x4cars MATCH
      (hash 6365120802482066310). HEATSEEKER GATES (work-bench CPU = reference):
        --game-mode heatseeker --ticks 20000 --arenas 1 --cars 2 --seed 1337
          -> 9090346742586108841
        --game-mode heatseeker --ticks 1000 --arenas 64 --cars 4 --seed 1337
          -> 6365120802482066310
      TASK #13 HEATSEEKER: COMPLETE.
   2. DEMOLITIONS (#14, COMPLETE):
      - Car::Demolish = isDemoed=1 + demoRespawnTimer (ALREADY on device via bump).
      - Car::_PreTickUpdate head: if isDemoed { timer = RS_MAX(timer-dt, 0);
        if timer==0 Respawn(); body.activationState = DISABLE_SIMULATION (RAW
        write); collisionFlags |= NO_CONTACT_RESPONSE } else { ACTIVE_TAG (RAW -
        cars NEVER sleep on CPU! device kernel must add this raw write for
        non-demoed cars too); flags &= ~NO_RESPONSE }; then `if (isDemoed) return`
        RE-EVALUATED: a just-respawned car (isDemoed now false) CONTINUES the
        pre-tick with a DISABLE_SIMULATION body + NO_RESPONSE flags STILL SET this
        tick (quirk - port exactly). => GpuCarGameState needs a separate
        noContactResponse flag (collide hasResponse/callback + solver hasResponse
        must use IT, not isDemoed).
      - Car::Respawn(gameMode, seed=-1, boost=mc.carSpawnBoostAmount): idx =
        Math::RandInt(0, 4, -1) = engine() % 4 where engine = thread_local
        libstdc++ std::default_random_engine = minstd_rand0 LCG
        (x = x*16807 % 2147483647; .seed(s): x = s % 2147483647, 0 -> 1).
        spawn tables CAR_RESPAWN_LOCATIONS_{SOCCAR,HOOPS,DROPSHOT} (4 entries
        {x, y, yawAng}), pos = (x, y * (blue?1:-1), CAR_RESPAWN_Z=36); rotMat =
        Angle(yawAng + (blue?0:M_PI as DOUBLE add), 0, 0).ToRotMat() =
        btMatrix3x3::setEulerYPR(yaw, -0.f, -0.f) = setEulerZYX(-0, -0, yaw):
        ci=cos(-0) si=sin(-0)=-0 cj=cos(-0) sj=-0 ch=cos(yaw) sh=sin(yaw);
        cc=ci*ch cs=ci*sh sc=si*ch ss=si*sh; rows: [cj*ch, sj*sc-cs, sj*cc+ss /
        cj*sh, sj*ss+cc, sj*cs-sc / -sj, cj*si, cj*ci] - SIGNED ZEROS flow, port
        verbatim w/ exact CuSinf/CuCosf. Then Car::SetState: body origin =
        pos*UU_TO_BT, basis = rotMat, linVel = 0, angVel = 0,
        updateInertiaTensor() (CuUpdateInertiaTensor!); _velocityImpulseCache = 0;
        _internalState = CarState() defaults [pos.z=17 CAR_SPAWN_REST_Z(unused),
        isOnGround=TRUE, boost=param, all timers/flags 0, ballHitInfo defaults
        (tickCountWhenHit = UINT64_MAX per observed default), lastControls = {}],
        tickCountSinceUpdate = 0. WHEELS (btVehicleRL persists) NOT reset (quirk).
        carID/team keep.
      TASK #14 DEMOLITIONS: COMPLETE (all gates PASS bit-exact).
      IMPLEMENTED: Math::SeedRandEngine work-copy hook; bench --demo-mode
      {disabled,normal,on-contact} + DemoRngSeed helper (RunCpu seeds the engine
      per arena after SetupArena; RunGpu uploads the post-.seed() minstd state via
      new UploadRngState / GpuBatch::dRngStates); GpuTailConsts += respawnX/Y/Yaw[4]
      + carRespawnZ + carSpawnBoostAmount (ExtractTailData fills mode-appropriate
      CAR_RESPAWN_LOCATIONS_*); device CuMinstdNext + CuCarRespawn (CudaGameTail -
      setEulerZYX signed-zero trees, SetState semantics: NO activate(), NO
      deactivationTime reset, wheels persist); kernel GAME_PRE demo block (timer
      RS_MAX countdown, respawn, RAW DISABLE_SIMULATION + noContactResponse=1 for
      demoed / RAW ACTIVE_TAG + 0 for non-demoed, re-evaluated isDemoed skip so a
      just-respawned car continues pre-tick with the disabled body - CPU quirk);
      hasResponse switched isDemoed -> noContactResponse in CuContactAddedCallback
      head + CuSolveTick hasResponse; per-car live dyn[] override
      hasContactResponse = !noContactResponse (wheel-ray acceptance rule);
      SnapshotCar/WritebackCar carry noContactResponse (mirrors body collision
      flags both ways).
      ROOT CAUSES FIXED while validating (8x6x3000 seed4242 on-contact diverged):
      (1) **CarState::rotMat staleness**: rotMat is refreshed from the live basis
      only in the post-tick, which demoed cars SKIP - so a car demoed at hash time
      carries a STALE rotMat while its body basis moved during the demo tick.
      WritebackCar used the live basis -> mismatch. Fix: GpuCarGameState grew
      rotMat; CuCarPostTick writes st.rotMat = b.basis after the demo guard;
      CuCarRespawn sets it to the spawn basis; WritebackCar uses g.rotMat.
      (2) **mergesSimulationIslands**: btSimulationIslandManager::findUnions only
      unites pairs where BOTH bodies have mergesSimulationIslands() = non-static
      AND no CF_NO_CONTACT_RESPONSE. The device union-find united every dyn-dyn
      pair, so a freshly-respawned (still no-response, DISABLE_SIMULATION) car
      with a stale overlap pair got pulled into an ACTIVE island -> the solver
      integrated its pre-tick external forces (air torque/boost) into vel/angVel;
      the CPU leaves it in a skipped singleton island (buildAndProcessIslands
      skips islands with no isActive() body). Fix: CuSolveTick union step now
      requires !noContactResponse on both car slots (ball always merges).
      Diagnosis evidence: state-blob bisection -> first diverging tick 1126;
      blob decode (WriteMultiple writes a uint32 count; Vec=16B, RotMat=48B)
      -> vel/angVel of the respawned car; device printf bracket: vel=0 after
      CuCarRespawn AND after pre-tick, NONZERO after CuSolveTick, pair list
      showed the demoed car's pair {21,25} -> island merge confirmed.
      Bullet semantics verified for DISABLE_SIMULATION (no device change needed):
      predictUnconstraintMotion damps unconditionally; applyGravity/integrate/
      island-processing use isActive(); setActivationState guarded writes;
      updateDeactivation/wantsSleeping do NOT special-case state 5 (timer keeps
      accumulating - replicated); forceUpdateAllAabbs=true so the teleported
      proxy updates the same tick; CPU vehicle raycaster rejects hits on
      no-response bodies (live dyn override covers); BoostPadGrid isDemoed guard
      + post-tick/finish isDemoed guards were already ported.
      DEMO GATES (work-bench CPU = reference, all PASS on GPU):
        --ticks 1000 --arenas 64 --cars 4 --seed 1337 --demo-mode on-contact
          -> 16798083503057895723
        --ticks 3000 --arenas 8 --cars 6 --seed 4242 --demo-mode on-contact
          -> 2807978879122602574
        --ticks 1000 --arenas 8 --cars 6 --seed 4242 --demo-mode on-contact
          -> 9086356441874091534   (hashes cars MID-DEMOLITION: stale rotMat,
                                    frozen body, DISABLE_SIMULATION covered)
        --ticks 2000 --arenas 32 --cars 4 --seed 909090 --demo-mode on-contact
          -> 11661070273966666902
        --ticks 2000 --arenas 16 --cars 6 --seed 777 --demo-mode normal
          -> 1608027089116060330
      All soccar + heatseeker gates re-PASS (CPU and GPU).
      RESIDUAL COVERAGE NOTE: no random-controls run produced a SUPERSONIC bumper
      contact, so DemoMode::NORMAL's supersonic branch always fell through to the
      bump path (normal hash == disabled hash for the tested cfgs). The branch is
      a 1:1 port (isDemo = car1.isSupersonic) and both downstream paths are
      gate-covered; a supersonic-demo gate would need a crafted control schedule.
      KNOWN LIMITATION: --gpu-compare lockstep harness mispairs rscap captures for
      demoed cars (phase-1/2 captures missing during demo, phase-0 entry state is
      pre-respawn) - prints garbage diffs; fix only if needed for future demo
      debugging (state-blob bisection + device printf worked better anyway).
   3. SNOWDAY (#15, COMPLETE - all gates PASS bit-exact):
      The puck is a btConvexHullShape (40 points: 20-pt circles at z = +-PUCK_HEIGHT/2,
      radius = mutator ballRadius = PUCK_RADIUS 114.25; mass 50, friction 0.1,
      restitution 0.3 via MutatorConfig). Device CU_SHAPE_HULL (=4) added everywhere:
      - Support: btVector3::maxDot takes the SSE _maxdot_large path for 40 points;
        for finite inputs it reduces to FIRST-index-of-max with
        dot = (x*dx + y*dy) + z*dz - ported scalar (CuSupportHullNoMargin); with-margin
        variant adds margin*normalize(dir) ((-1,-1,-1) degenerate fallback)
        (CudaConvexCast.cuh CuSupportHull). Plugged into CuSupport (subsimplex ray
        cast), CuSupportNoMargin (GJK/EPA), CuConvexPlaneCollide, and the
        CuConvexVsMesh early-out (virtual localGetSupportingVertex semantics).
      - AABB: btPolyhedralConvexAabbCachingShape::getAabb = btTransformAabb over the
        cached local aabb (which ALREADY contains one margin from recalcLocalAabb -
        margin counted twice, bullet quirk, replicated). m_localAabb is protected;
        the bench recomputes it exactly via the public
        batchedUnitVectorGetSupportingVertexWithoutMargin 6-direction form.
        CuHullAabbAt (broadphase/car-ball gates) + CuShapeAabb hull branch (mesh-walk
        bounds + GJK fix-normal).
      - Narrowphase dispatch: ball-vs-mesh takes the convexConvex-GJK-per-triangle
        path (same as the car box, NOT sphere-triangle); ball-vs-plane via
        CuConvexPlaneCollide with hull support; car-vs-ball GJK box-vs-hull with
        maximumDistanceSquared margin = hull collision margin (0.04), NOT radius.
      - Game logic: Ball::_PreTickUpdate snowday resets _groundStickApplied (GAME_PRE
        hook); _OnWorldCollision applies ONE applyCentralForce(-normal *
        PUCK_GROUND_STICK_FORCE=70) per tick (callback ball-world branch);
        groundStickApplied carried in GpuBallGameState (not hashed, not serialized);
        isSpecial already gated on != SNOWDAY.
      - Hull points uploaded once via new UploadBallHull (GpuWorldGeo.hullPoints);
        GpuDynBody kind 4 reuses the sphereRadius slot as the margin.
      - BENCH QUIRK FOUND: the Arena CTOR leaves the ball at BallState defaults
        (z=93.15, vel=0) -> in snowday it sleeps instantly and never falls; the
        vel.z=FLT_EPSILON "don't freeze" init lives in ResetToRandomKickoff (never
        called by the bench). SetupArena now applies that init for snowday -
        without it the first "PASS" was vacuous (puck frozen all run; caught by
        decoding the state blob: ball pos never left 93.15).
      SNOWDAY GATES (work-bench CPU = reference, GPU PASS):
        --game-mode snowday --ticks 20000 --arenas 1 --cars 2 --seed 1337
          -> 6720894011815016031
        --game-mode snowday --ticks 1500 --arenas 32 --cars 4 --seed 4242
          -> 16993491767979963113
        --game-mode snowday --ticks 2000 --arenas 8 --cars 6 --seed 99
          --demo-mode on-contact -> 3415898148067774285  (puck + demolitions)
      Ray-fuzz 200k PASS with the hull ball (--game-mode snowday) AND soccar 200k
      re-PASS. All 12 prior gates (soccar/heatseeker/demo, CPU+GPU) re-PASS.
   4. HOOPS (#16, COMPLETE - all gates PASS bit-exact):
      ASSETS RESOLVED (2026-06-10): user supplied maps.zip with embedded-byte-array
      headers (maps/{soccar,hoops,dropshot,throwback}.h + generate_maps.py, raw
      .cmf bytes + MeshSpan offsets). Converted hoops (12) + dropshot (2) back to
      .cmf in collision_meshes/{hoops,dropshot}/. PROOF: the soccar header's 16
      meshes are byte-identical (sha256) to the existing trusted soccar files, and
      all 12 hoops files hash-match the known MeshHashSet in src/RocketSim.cpp
      (CollisionMeshFile::UpdateHash replicated in python: float->uint32 truncate,
      2x mueller mix, shift-xor accumulate). collision_meshes/soccar was NOT
      touched (directory_iterator order feeds creation order -> gates).
      PORT (small - statics machinery was already generic): hoops arena = 12
      meshes (505-vert net detected, HOOPS_NET group/mask - flows through generic
      static extraction) + 6 planes (incl. Y walls); hoops pads use the same
      BoostPadGrid (generic extraction); CAR_RESPAWN_LOCATIONS_HOOPS already
      mode-dispatched. New device code: CuBallPreTickLaunch (CudaGameTail -
      Ball::_PreTickUpdate hoops/dropshot kickoff launch: tickCountSinceUpdate*dt
      crossing the delay, frozen-ball check [vel/angVel zero + pos.To2D zero in
      UU], applyCentralImpulse((0,0,L)*mass*UU_TO_BT) * linearFactor * invMass,
      GUARDED setActivationState(ACTIVE_TAG)); hoops _OnHit zScale branch in the
      callback (isOnGround && CACHED rotMat.up.z [= rotMat.row[2].z, RotMat is
      column-major] > thresh -> Z_SCALE_HOOPS_GROUND). GpuCollideConsts +=
      ballLaunchDelay/ballLaunchZVel/ballMassBT/hoopsZScaleGround/hoopsZScaleThresh.
      BENCH COVERAGE: random-control cars essentially NEVER reach a center-resting
      ball (soccar/snowday configs: 0 hits; heatseeker hits come from the moving
      ball). SetupArena hoops: per-arena coin (spawnRng) - half the arenas keep
      the frozen kickoff (launch coverage), half get a seed-derived live ball
      (pos.z=400, vel=1300@angle + 600z) -> 7-10 car-ball hits per 32x6x2000 run
      (zScale branch + net/wall contacts exercised).
      HOOPS GATES (work-bench CPU = reference, GPU PASS):
        --game-mode hoops --ticks 20000 --arenas 1 --cars 2 --seed 1337
          -> 14674870656341860422
        --game-mode hoops --ticks 2000 --arenas 32 --cars 6 --seed 4242
          -> 14326563309044877142
        --game-mode hoops --ticks 3000 --arenas 8 --cars 6 --seed 99
          --demo-mode on-contact -> 6767940556749896631
      Hoops ray-fuzz 200k PASS (new meshes + net BVHs). All 18 prior gates re-PASS.
   5. DROPSHOT (#17, COMPLETE - all gates PASS bit-exact). ALL FIVE GAME MODES
      NOW RUN 100% ON GPU.
      ARENA: 2 meshes + floor plane z=1.5 (DROPSHOT_FLOOR - collides with CARS
      only; car group = DefaultFilter|DROPSHOT_FLOOR) + ceiling, NO side walls,
      NO boost pads (numPads=0 flows through generically), 140 static tiles
      (6-pt btConvexHullShapes, world x/y baked into points, rb origin
      (0,0,-margin), group=mask=DROPSHOT_TILE - ball-only via filters).
      DEVICE: GpuStaticObj kind 2 = tile (shapeIdx = userIndex2 total index);
      GpuTileShape (pts[6] + cached local aabb + origin) + GpuTileTables
      (neighbor radius-2 <=7 / radius-3 <=20 lists) in GpuWorldGeo via
      UploadTiles (also allocs per-arena dTileDamage/dTileNoResponse);
      ball-vs-tile = plain convexConvex: pair (tile, ball) -> manifold
      (tile, ball) = ml.create(-1-h0, 0), GJK tile-hull-vs-sphere (reuses the
      snowday CU_SHAPE_HULL machinery), result order matches manifold
      (swap=false), callback gets userIndex-swapped (ball, tile), NO isSpecial;
      CuBodyHasResponse (callback head) + solver hasResponse treat BROKEN tiles
      as no-response (flags state separate from damage: re-synced only in
      GAME_POST when dsLastDamageTick == tickCount - SetDropshotTilesState's
      deferred timing so the ball never falls through a tile broken this tick);
      _OnDropshotTileCollision ported in the callback (broken-skip,
      MIN_DAMAGE_INTERVAL via (tickCount-lastDamageTick)*dt, downward-speed
      gate, charge side gate [team0 tiles all y<0 -> sign by teamIdx], neighbor
      break radius = chargeLevel, dsInfo reset); _OnHit dropshot charge branch
      (velIntoBall accumulation -> chargeLevel 2/3 + yTargetDir);
      dsInfo carried in GpuBallGameState (serialized+hashed via BallState);
      launch delay shared with hoops (CuBallPreTickLaunch). Bench: tile
      extraction (points + recalcLocalAabb replication + neighbor tables via
      DropshotTiles::GetNeighborIndices), tile damage ADDED TO THE GATE HASH
      (DropshotTilesState has no upstream serialization), tile damage + response
      flags downloaded and mirrored onto the arena at writeback, wheels now
      written back in WritebackCar, dropshot 3-way per-arena ball init (frozen
      kickoff / live ball / vertical slam - the slam re-damages the same tile on
      rebound, BREAKING it: covers response removal + fall-through; natural play
      produced damage but ZERO broken tiles).
      *** ROOT CAUSE FOUND while validating (32x6x2500 arena 0 diverged @ tick
      591): Arena::Step's car pre-tick iteration follows _cars (POINTER-keyed
      unordered_set) - the order is ALLOCATION-LAYOUT-DEPENDENT and becomes
      PHYSICS-RELEVANT when a car rides on another car (wheel rays hit the other
      chassis; the ground car's mid-tick state depends on whether it pre-ticked
      first). RunCpu created arenas one-at-a-time interleaved with
      stepping/deletion while RunGpu created them all upfront after CUDA init ->
      different heap layouts -> different _cars orders -> the GPU's uploaded
      carOrder replicated the GPU-PROCESS order, not the CPU-gate-run order.
      FIX: RunCpu now creates ALL arenas upfront and RunGpu creates arenas
      BEFORE any CUDA call - identical allocation histories give identical
      _cars orders in both processes (empirically reproducible; same-binary CPU
      runs always reproduce). All 12 prior gate hashes were UNCHANGED by the
      restructure (those configs were order-insensitive). Diagnosis chain:
      state-blob bisect -> hidden-state-extended blob (wheels/vic/tile flags -
      still matched) -> lockstep phase compare passed (all phases correct from
      CPU inputs) -> pair caches matched -> input fingerprint of the diverging
      car's pre-tick: own state identical, GROUND CAR's velocity was pre- vs
      post-its-own-pre-tick -> ORDER. (Also fixed: the gpu-compare harness now
      uploads hull/tile data - it crashed on kind-2 statics with null tiles.)
      DROPSHOT GATES (work-bench CPU = reference, GPU PASS; tile damage hashed):
        --game-mode dropshot --ticks 20000 --arenas 1 --cars 2 --seed 1337
          -> 16797750564250741040
        --game-mode dropshot --ticks 2500 --arenas 32 --cars 6 --seed 4242
          -> 18379725107633245205  (5 arenas end with BROKEN tiles + fall-through)
        --game-mode dropshot --ticks 3000 --arenas 8 --cars 6 --seed 99
          --demo-mode on-contact -> 16801832455181733896
      All 12 prior gates re-PASS (CPU + GPU) + ray-fuzz 200k x4 modes PASS.
   THROUGHPUT: GOAL MET (2026-06-10 re-measure): GPU 668k SPS @128 arenas / 1.12M
   @256 / 1.08M @512 / 1.19M @1024 (1000 ticks, 2 cars, soccar) vs CPU 58.3k
   single-thread = 11-20x. The old "10.6k SPS" figure was stale. 1024-arena
   cross-check CPU vs GPU bit-exact (300 ticks seed 777).
   Cleanup decision: the debug[100+/110+/120+] diagnostic channels in
   CudaCollideWorld.cuh STAY - they are null-gated compare-harness infrastructure
   that hoops/dropshot debugging will need when assets arrive.
   Validation plan: chain GPU segment AFTER_PREDICT -> AFTER_SOLVE (collide phase
   already bit-exact feeds the solver in the same kernel run; compare body snaps at
   AFTER_SOLVE: vels + worldTransform [split impulse moves positions!] + the
   manifold appliedImpulses optional). rscap AFTER_SOLVE body snaps already exist.
2. (DONE - collision phase; narrowphase notes kept below for reference)
   PORTED + VALIDATED (CudaCollision.cuh): manifold mechanics (add/sort4/remove/
   refresh, RS no-replacement), CuAddContactPoint (combine rules, plane-space friction
   dirs, swap semantics), sphere-triangle collide, sphere-box getSphereDistance,
   convex-plane collide, FULL GJK (CuGjkClosestPoints incl fix-normal aabb branch +
   CuShapeAabb box/sphere/triangle), FULL EPA (btGjkEpa2 port: Cu2GJK/Cu2EPA/
   Cu2Penetration/Cu2Distance/CuEpaPenDepth; pointer->index adaptation with initStore
   for the 4 GJK hull verts preserving the 128-vertex capacity semantics).
   *** MATHTEST gjkBoxTri PASS: 500k random box-vs-triangle configs (separated ->
   deeply penetrating) BIT-EXACT vs real btGjkPairDetector+btGjkEpaPenetrationDepthSolver
   (host ref in cuda_math_test host_ref.cpp links rocketsim_work). NOTE: EPA expand()
   recursion needs cudaDeviceSetLimit(cudaLimitStackSize, 64KB) - set in cuda_math_test;
   ADD TO CudaStep init for production kernels! ***
   sphereTri standalone fuzz PASS (500k, vs real SphereTriangleDetector, identity tri
   transform). btBoxBoxDetector FULLY PORTED (CuDBoxBox2/CuBoxBoxClosestPoints -
   verbatim ODE scalar code; cullPoints2 atan2f = double-rounded stand-in, measure
   later). sphere-box/convex-plane/box-box device ports not yet fuzz-validated
   standalone (manifold-compare will cover);
   BVH SEMANTICS READ: processAllTriangles -> reportAabbOverlappingNodex ->
   quantizeWithClamp(query aabb: clamp to bvh bounds via setMax/setMin, v = (p-min)*
   m_bvhQuantization, max: (ushort)(v+1)|1 per axis, min: (ushort)v & 0xfffe) ->
   walkStacklessQuantizedTreeCacheFriendly = iterate m_SubtreeHeaders (READ ITS LOOP
   NEXT - ~25 ln, headers have quantized aabb + rootNodeIndex + subtreeSize), per
   overlapping subtree -> walkStacklessQuantizedTree(start,end): LINEAR ARRAY WALK
   (read in full): test quantized AABB (pure ushort compares), leaf+overlap ->
   processNode(partId, triangleIndex) in array order; nonoverlap+internal -> skip by
   escapeIndex. DEVICE NEEDS per mesh: quantized node array (btQuantizedBvhNode:
   ushort[3]x2 + int escapeOrTriIdx), subtree headers, m_bvhAabbMin/Max,
   m_bvhQuantization (extract host-side from mesh->getOptimizedBvh()).
   PORTED (2026-06-10 cont.): CuBvhQuantizeClamp/CuTestQuantAabb/CuBvhWalk (subtree
   loop + linear walk, exact CPU triangle order), CuTestTriangleAabb (Z,X,Y),
   CuConvexVsMesh (concave per-pair: AABB in tri space + mesh-margin expand, BVH walk,
   per-tri early-out [support WITH margin via basis.INVERSE, both plane sides,
   threshold = manifold breaking], dispatch sphere->CuSphereTriangleCollide (identity-
   transform canon trees, swap=false) / box->CuGjkClosestPoints(maxDistSq=(margins+
   breaking)^2), shared manifold accumulation with onContactAdded callback hook per
   inserted point, CuManifoldRefresh at end). Builds clean. WORK COPY CHANGE:
   btQuantizedBvh.h got public getBvhAabbMin/Max/getBvhQuantization accessors (gates
   still to re-verify after; no behavior change). Host extraction still TODO: BVH
   node/subtree arrays via getQuantizedNodeArray/getSubtreeInfoArray + the new
   accessors; mesh margin per mesh (concaveShape->getMargin()).
   btAdjustInternalEdgeContacts READ (struct): trimesh triangleInfoMap lookup by
   btGetHash(partId,index) (READ btGetHash + btTriangleInfo layout: m_flags +
   m_edgeV0V1Angle/V1V2/V2V0); flags arg = 0 (frontFacing=1, no double-sided);
   localContactNormalOnB = basisT * normalWorld, normalize() [rsqrt]; bestedge by
   min length to edges with |angle| < m_maxEdgeAngleThreshold (via
   btNearestPointInLineSegment - READ); per-edge block (3x, verbatim quirks incl.
   STALE `nearest` reuse in edge-0 block): len < m_edgeDistanceThreshold && bestedge
   match -> angle==0 ? concaveHit++ : [swapFactor by TRI_INFO_*_CONVEX flag, nB =
   swapFactor*quatRotate(btQuaternion(edge, angle), triNormal) (*-1 if SWAP_NORMALB),
   backFacing if NdotA<convexEps && NdotB<convexEps -> concave++ else convex++ +
   btClampNormal(edge, swapFactor*triNormal, localN, angle, clamped) (READ) ->
   if clamped.dot(triNormal) > 0: cp.m_normalWorldOnB = basis*clamped,
   positionWorldOnB REPROJECTED = posA - normal*distance1, localPointB = invXform].
   INTERNAL EDGE FULLY PORTED (CuAdjustInternalEdgeContacts + CuClampNormal +
   CuNearestPointInLineSegment + CuQuatRotate [quat*vec SSE tree decoded:
   x=(w*vx+y*vz)-z*vy etc.] + CuGetAngleIE [atan2 double-rounded stand-in]); CPU
   quirks preserved (stale nearest in V0V1 block, shadowed un-normalized localN in
   V1V2/V2V0 blocks); tail: isNearEdge && numConcaveEdgeHits>0 -> normal := triNormal
   (skip if dot<0), reproject posB + localPointB. Thresholds: convexEps 0,
   edgeDist 0.1, maxEdgeAngle 2pi. btGetHash = (partId<<21)|triIdx (partId 0 here).
   ALL COLLISION NARROWPHASE NOW IN DEVICE CODE (CudaCollision.cuh ~2400 ln).
   DONE since: (1) Host extraction+upload of mesh BVHs/margins/triInfos
   (ExtractMeshBvhs in bench_work + rscuda::UploadMeshBvhs; GpuTriInfo moved to
   GpuTypes.h; triInfoMap find key = triIdx since partId=0; absent entry ->
   hasInfo=0 = CPU early-return). (2) rscap manifold+proxy capture at
   AFTER_COLLISION (ManifoldSnap/ContactSnap/ProxySnap) + bench --manifold-dump N
   (verified live: car-vs-plane manifolds match expectations).
   COLLISION ORCHESTRATION SPEC (decoded 2026-06-10 from RS bullet source, verified
   by uid probe + manifold dump; THE reference for the device port):
   - updateAabbs: m_forceUpdateAllAabbs=TRUE (Arena never clears) -> ALL objects
     each tick, but statics recompute identical aabbs -> setAabb `!=` guard no-ops
     (device: skip statics, proxy aabbs are upload-time constants). Dyn aabb =
     shape getAabb(worldTransform) +-gContactBreakingThreshold(0.02) UNION (because
     dispatchInfo.m_useContinuous=true) interp-transform aabb +-0.02.
     btCompoundShape::getAabb uses CONST m_localAabbMin/Max + margin: center=T*mid,
     extent=dot3(halfExt+margin, |basis rows|). Sphere: origin +- radius.
     btStaticPlaneShape::getAabb (RS mod): slab on single axis: [const-0.2,const+0.2],
     back side extended to +-LARGE (m_singleAxisBackwards). Plane proxy aabbs are
     THIN -> cars at suspension height (zmin 0.35) do NOT pair with floor (slab
     zmax 0.2). Ball at rest DOES pair (zmin -0.062).
   - setAabb (dyn): if aabb bitwise-changed: update aabb; newCell=GetCellIdx(aabbMin);
     if cell changed AND numDynProxies>1: remove from old 3x3x3 around (i,j,k),
     append to new 3x3x3 (clamped). Cell dynHandles order = push_back order =
     ascending per-proxy "re-registration seq" (each move removes from ALL old cells
     and appends; device model: per-proxy regSeq counter, cell list = members sorted
     by regSeq; initial regSeq = creation order ball,car0,car1...).
   - Proxy uid = handle index + 2 (set in btRSBroadphase ctor; NOT garbage).
     Handle order: statics first (16 meshes then 4 planes), ball, cars. internalAddPair
     swaps so pair.proxy0 = lower uid -> static always proxy0 of dyn-static pairs;
     ball proxy0 of ball-car; lower car proxy0 of car-car.
   - calculateOverlappingPairs: removes ALL activePairs (releases algos+manifolds ->
     NO persistence, per-tick manifolds only), then per dyn handle in order:
     cell=cells[cellIdx]; for s in cell.staticHandles (order = smear insertion):
     TestAabbAgainstAabb2 -> addOverlappingPair (group/mask filter BOTH ways inside,
     findPair dedup); then if numDynProxies>1 && cell.dynHandles.size()>1: same for
     dyns (skip self). Pair ARRAY order = add order = dispatch order.
   - STATICS ARE ISLAND_SLEEPING (btDiscreteDynamicsWorld::addRigidBody) ->
     isActive()=false. needsCollision(a,b)=!(!a.active && !b.active) -> dyn-static
     pair dispatches ONLY if the dyn is active. Sleeping ball next to floor =>
     pair exists, NO algorithm, NO manifold (verified: kickoff ball, 0 manifolds).
   - defaultNearCallback: needsCollision -> findAlgorithm (per-tick fresh since pairs
     rebuilt) -> processCollision. Algorithm matrix (RS: USE_BUGGY_SPHERE_BOX
     undefined, no initializePolyhedralFeatures, convex-convex/plane perturbation
     DEAD [ccCF: numPerturb=0; planeCF: minPointsThresh=0]):
     ball-mesh: swappedConvexConcave -> manifold(ball,mesh) AT pair-algo ctor.
     ball-plane: planeConvex(swapped) -> manifold(ball,plane) AT ctor.
     car-mesh: swappedCompound(no dbvt: Car compound=btCompoundShape(false,1)) ->
       preallocate child algo at ctor: convexConcave(carObj,mesh) -> manifold(car,mesh)
       AT ctor. processCollision: ProcessChildShape(child0): child world aabb
       (orgTrans*childTrans) + extend 0 vs MESH shape aabb (const, identity) ->
       TestAabbAgainstAabb2 -> set result wrap+shape ids (partId -1, index 0 on the
       compound side) -> child convexConcave processCollision (box GJK per tri).
     car-plane: swappedCompound -> child convexPlane -> manifold(car,plane) AT ctor.
     ball-car: swappedCompound(car) -> child algo convexConvex(carbox, ball) at ctor
       but manifold LAZY: created (car,ball) at FIRST processCollision = only when
       child-vs-ball aabb test passes. Then plain GJK (box margin 0.04+sphere r,
       maxDistSq=(mA+mB+manifoldBreaking)^2) + EPA fallback.
     car-car: compoundCompound -> NO dbvt trees -> falls back to base compound
       processCollision. Ctor chain at pair creation: outer preallocate ->
       findAlgorithm(car1box, car2compound) -> swappedCompound nested -> its
       preallocate -> findAlgorithm(car2box, car1box) -> BOXBOX ctor -> manifold
       (car2, car1) AT ctor (needsCollision-gated) -- body0 = HIGHER car!
       Runtime: outer ProcessChildShape(car1 child): aabb car1box vs car2 COMPOUND
       -> sets result body0Wrap=car1box ids A(-1,0) -> nested: ProcessChildShape
       (car2 child): aabb car2box vs car1box -> sets body1Wrap=car2box ids B(-1,0)
       -> boxbox processCollision(body0arg=car2box!, body1arg=car1box): detector
       (box0=car2box at car2 childWT, box1=car1box); manifold(car2,car1) vs result
       (body0=car1box) -> addContactPoint SWAPPED insertion path.
   - Manifold m_manifoldsPtr order = getNewManifold call order = pair dispatch order
     (+ lazy convex-convex inserts at first contact-pass). SOLVER consumes this order.
   - getNewManifold: contactBreakingThreshold = MIN over both shapes of
     shape->getContactBreakingThreshold(0.02) = getAngularMotionDisc()*0.02 (RELATIVE
     flag default ON) -> e.g. car-plane admits d up to ~0.07 (verified dump d=0.0305);
     contactProcessingThreshold = min(both objs) = BT_LARGE_FLOAT default.
   - Contact-added callback (gContactAddedCallback=Arena::_BulletContactAddedCallback,
     fires from btManifoldResult::addContactPoint since ball+cars have
     CF_CUSTOM_MATERIAL_CALLBACK; args = RESULT wrapper order): skip if either body
     !hasContactResponse; shouldSwap so bodyA=lower userIndex (car=BT_USERINFO_TYPE_CAR
     < ball) / the one WITH userIndex; car-ball -> friction=CARBALL_COLLISION_FRICTION,
     restitution=CARBALL_COLLISION_RESTITUTION + Ball::_OnHit(car, relPosOnBall=
     localPointA-or-B(ballIsBodyA)*BT_TO_UU): ballHitInfo writes EVERY add; extra
     impulse guarded (tickCount > lastApplied+1 || lastApplied > tickCount) ->
     _velocityImpulseCache += (Vec trees, UU, curves); car-car -> friction/rest :=
     CARCAR consts + bump logic (2-way loop, cooldown, vel-toward checks Vec trees,
     bumper localPoint.x check, demo mode, BUMP curves -> victim _velocityImpulseCache,
     carContact id+cooldown writes); car-world -> worldContact.hasContact=true +
     normal + friction:=carWorldFriction, restitution:=carWorldRestitution;
     ball-world -> Ball::_OnWorldCollision (soccar: no-op) + m_isSpecial=true (non-
     snowday). THEN btAdjustInternalEdgeContacts(cp, trimeshWrapSide, otherSide,
     partId, index) ALWAYS (early-out if shape != trimesh).
   IMPLEMENTED (2026-06-10, commits after ee5e371):
   a. DONE Host extraction (bench ExtractCollideData): static table (handle order,
      proxy aabb + RAW shape aabb + group/mask/material/breaking), static CSR cell
      lists from bp->cells, initial dyn proxies, GpuCollideConsts (compound
      localAabb via NEW btCompoundShape accessors getLocalAabbMin/Max, mutator +
      RLConst values, 4 curves).
   b. DONE Device CudaCollideWorld.cuh (CuCollideTick: CuCollideUpdateAabbs +
      CuCollideGenPairs + dispatch per pair type + CuContactAddedCallback +
      internal-edge; manifold create/release order incl. compound post-loop
      swap-remove). Wired as GPU_PHASE_COLLISION in GpuStepArenasKernel (thread 0
      per arena); GpuBatch grew proxy/static/cell/manifold/ballstate/debug buffers;
      APIs UploadCollideData/UploadDynProxies/UploadBallState/SetTickCount/
      DownloadCollideResults/DownloadCollideDebug. cudaDeviceSetLimit(stack 64KB)
      now set in CreateBatch (EPA recursion).
   c. DONE Bench collision compare in --gpu-compare: ball given seed-derived
      velocity + activate at setup (coverage); per tick uploads AFTER_PREDICT
      bodies + host-lockstep proxies (regSeq maintained by host via cell-change
      rule), runs the segment, compares proxies + manifolds (order/bodies/points,
      bitwise) vs rscap AFTER_COLLISION, then advances host proxies from CPU truth.
   BUGS FOUND/FIXED by the compare (all root-caused):
   - btSphereShape::getAabb is RS-MODIFIED: margin = radius + 0.08f (aabb only;
     GJK margins keep radius). Device CuSphereAabbAt fixed.
   - Statics' PROXY aabbs change ONCE at the first updateAabbs (forceUpdateAllAabbs
     pushes them through setAabb WITH +-0.02, which createProxy lacked) and the
     static cell smear is REDONE then (per-cell order stays handle order). Fix:
     bench calls arena->_bulletWorld.updateAabbs() once before extraction so the
     snapshot equals steady state. Production GPU runs must extract the same way.
   - GpuStaticObj must carry BOTH aabbs: proxy aabb (pair-gen TestAabbAgainstAabb2)
     and RAW shape aabb (compound ProcessChildShape + post-loop tests). They differ
     by the +-0.02 expansion; car-plane at grazing height proved it (tick 745).
   - CuConvexVsMesh's trailing refreshContactPoints must use the manifold BODY
     transforms (car body), not the child convex transform: localPointA is in body
     space; the child offset corrupted distance1 and pruned valid points (tick
     3063 car-mesh). Other 4 refresh sites audited: already body transforms.
   STATUS: COLLISION PHASE VALIDATED - 20000-tick gpu-compare PASS on seeds
   1337/42/7777/123456789 (all 5 phases: car-pretick, predict, COLLISION,
   integrate, activation bit-exact), both gates PASS (commit 3280716).
   Two more root-cause fixes during multi-seed (NOT collision):
   - Autoflip atan2f stand-in: replaced with VERBATIM glibc 2.4x CORE-MATH
     atan2f/asinf device ports (CudaVehicle.cuh CuAtan2f/CuAsinf, sources
     fetched from glibc-2.42 e_atan2f.c/e_asinf.c/s_asincosf_data.c; correctly
     rounded, doubles/fma only -> bit-identical). cuda_math_test now fuzzes
     both vs host glibc: 16M values PASS each ("atan2f"/"asinf" ops).
   - Autoflip timer: CPU `CAR_AUTOFLIP_TIME * (absRoll / M_PI)` promotes to
     DOUBLE (M_PI) with one float rounding at the store; device had float pi +
     two roundings. Audited remaining M_PI uses in per-tick paths: none others.
   TODO: side-effect compare (ballHitInfo/velocityImpulseCache/worldContact/
   carContact) - only manifolds+proxies compared so far; remove TEMP debug[100+]
   diagnostics in CudaCollideWorld.cuh/bench when convenient.
   Then solver (btSequentialImpulse - READ NEXT after collision validated), ball
   pre/finish + pads + arena tail, monolithic assembly, flip --gpu-step.
   (f) compound wrapper (single box child: childWorldTrans compose, LocalInfoAdder
       partId/index semantics);
   (g) manifold assembly: per-pair btPersistentManifold (4-pt cache, sortCachedPoints
       (read tail!), RS no-replacement), btManifoldResult combine rules + Arena
       contact callbacks (decoded) + refreshContactPoints (read tail!) which prunes
       points after processCollision per-algo;
   (h) manifold-compare harness: rscap captures CPU manifolds at AFTER_COLLISION
       (pair body ids in dispatcher manifold order, per-point localA/B, worldB,
       normalWorldOnB, distance, partId0/1, index0/1, combinedFriction/Restitution,
       m_isSpecial, lateralFrictionDirs); GPU runs collision from AFTER_PREDICT
       uploaded state; bitwise compare.
2. Solver port (btSequentialImpulseConstraintSolver setup/iterations/finish + islands
   + RS m_isSpecial handling - READ PENDING) with solver-compare (AFTER_SOLVE vels).
3. Ball pre/finish (incl rsqrt-normalized vel clamps), pads (BoostPadGrid::CheckCollision
   exact), Arena tail; assemble monolithic kernel; flip --gpu-step; gates with new
   deterministic GPU hashes if FP-identical impossible end-to-end (target: identical).
2. Port btVehicleRL updateVehicleFirst/Second + resolveSingle* + Car game logic
   (_UpdateWheels/jump/flip/boost/airtorque/autoflip/autoroll - Car.cpp state machines,
   RLConst curves via LinearPieceCurve) with per-sub-phase capture points in
   Car::_PreTickUpdate; extend GpuCarGameState wheel state; validate via --gpu-compare.
3. Then collision (35%), solver+islands, ball/pads/game tail; flip --gpu-step.

### Open Questions / Blockers
- None currently.

---