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
1. PORT-WHILE-READING (fresh window, source side-by-side, faithful trees) the remaining
   heavy narrowphase into src/Cuda/CudaCollision.cuh slices:
   (a) GJK decoded above - port directly; still read: triangle support case +
       btGjkEpaPenetrationDepthSolver/btGjkEpa2 (EPA polytope ~1000 ln) + shape
       getAabb trees (box btTransformAabb, sphere +-(radius+0.08+margin?)) for the
       fix-normal branch;
   (b) btBoxBoxDetector (767 ln ODE SAT, car-car);
   (c) sphere-triangle + sphere-box (decoded above, small);
   (d) convex-plane (decoded above, small);
   (e) btConvexConcave FULLY READ (btConvexConcaveCollisionAlgorithm.cpp): per pair:
       collisionMarginTriangle = MESH getMargin; convex AABB in tri space (transform
       inverse compose) + extra margin (triMargin + 0); processAllTriangles over that
       AABB -> per triangle: TestTriangleAgainstAabb2 cull; EARLY-OUT both sides:
       support WITH margin along +-world tri normal (normalize=RSQRT; dir via
       basis.INVERSE() (cofactor CuMat3Inverse!) * normal), dist = projTr - projPt >
       breakingThreshold -> skip; then btTriangleShape(margin=triMargin) + dispatch:
       sphere->sphereTriangle special, box->convexConvex GJK; SHARED per-pair manifold
       accumulates across triangles in BVH WALK ORDER (replicate
       btQuantizedBvh::walkStacklessQuantizedTree + quantization params; upload BVH
       node array host-side from mesh getOptimizedBvh); partId/triangleIndex set into
       result for tri side (feeds internal-edge + callbacks); refreshContactPoints at
       end. ALSO btInternalEdgeUtility adjust in contact callback (triangleInfoMap
       extract host-side);
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
   REMAINING IMPLEMENTATION (in order):
   a. Host extraction: static object table in handle order (kind mesh/plane, proxy
      aabbMin/Max consts, group/mask, shape breaking threshold), static CSR cell
      lists from bp->cells (staticHandles per cell), dyn proxy initial state
      (aabbMin/Max/cellIdx/ijk from CPU proxies after setup), per-body collide
      consts (compound localAabbMin/Max+margin, box halfExt+margin, sphere radius,
      angular motion discs, processing thresholds), game consts (CARBALL_*/CARCAR_*/
      carWorld fric+rest, BUMP curves already in GpuCarConsts? bump curves =
      BUMP_VEL_AMOUNT_GROUND/AIR/UPWARD curves - extract), userIndex mapping.
   b. Device CudaCollideWorld.cuh: CuCollidePhase(arena): dyn aabb update + cell
      maintenance (regSeq model), pair gen (ordered, filtered, deduped), dispatch
      per pair type exactly as spec above, manifolds into per-arena buffer in
      creation order (incl. EMPTY manifolds for ordering!), contact callbacks
      mutating GpuCarGameState/ball state + _velocityImpulseCache accumulators,
      internal-edge adjust for mesh contacts. Kernel phase GPU_PHASE_COLLIDE,
      single thread per arena (like other phases).
   c. Bench --gpu-compare extension: upload AFTER_PREDICT state + proxy snaps,
      run collide segment, download manifolds + proxy state, compare vs rscap
      AFTER_COLLISION (bitwise: aabbs, cells, manifold order/bodies/points all
      fields). NOTE ball asleep at kickoff -> need ball-motion coverage: give ball
      initial velocity in compare setup (bench-side state injection) + multi-seed.
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