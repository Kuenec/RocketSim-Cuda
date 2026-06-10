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

3. **Validation gates (deterministic), You have to do this yourself to start:**
   - Stable gate:
   - Hard gate:
   - Both gates must pass after every change.
   - FP drift from GPU execution is acceptable IF it is minimized and the simulation behavior is negligibly different. New deterministic hashes for the GPU path are acceptable if the physics behavior is correct.

4. **Full CUDA — no hybrids accepted as final state.**
   - The end state is 100% GPU execution of all physics and game logic.
   - CPU-GPU hybrids (e.g. GPU raycasts + CPU solver) are only acceptable as intermediate stepping stones, never as the final deliverable.
   - Every Bullet phase must be ported: broadphase overlap detection, narrowphase contact generation, constraint solving (PGS iterations), integration (transform updates), damping, island management, activation state, everything.
   - Every RocketSim phase must be ported: car pre-tick updates, wheel raycasts, vehicle dynamics, boost pad collision/updates, ball updates, post-tick updates.
   - "CUDA ownership" means the GPU produces the result that the simulation uses. Shadow/dry-run GPU kernels that run alongside CPU do not count.

5. **Build with Ninja.**
   - CMake + Ninja + MSVC x64 + CUDA.
   - Build script: `build.ps1 -Target <bench|reference|both> [-Clean]`
   - If you cannot use ninja or the others on linux, find the best alternative that is good and optimized for speed like ninja.

---

## Workspace Layout

```
reference/RocketSim-ORIGINAL/    # pristine upstream (NEVER modify)
work/RocketSim-cuda/             # working copy (modify here)
  src/Profiling.h                # phase timing counters (active)
  src/Cuda/                      # ALL GPU kernel code goes here
  libsrc/bullet3-3.24/           # Bullet physics (compiled by MSVC as .cpp, NOT by NVCC)
tools/rocketsim_bench/           # workspace benchmark (--profile flag, --gpu-step flag)
tools/rocketsim_reference_bench/ # reference benchmark (never modify)
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
- **Author**: All commits under the user's git config — NO co-authoring, NO `Co-Authored-By` lines
- **Commit after every kept change** that passes both gates. Don't batch up multiple changes.
- **Push to GitHub after every commit.** The remote must always reflect the latest working state.
- **Commits**: All commits Must be short and Consise about what you changed

---