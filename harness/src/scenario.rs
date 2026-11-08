use std::sync::Arc;

use rocketsim::{
    Arena, ArenaConfig, ArenaEvent, ArenaState, BallState, BoostPadState, CarBodyConfig,
    CarControls, CarCore, DemoMode, GameMode, Mat3A, MutatorConfig, PhysState, Team,
    TileDamageState, TileStates, Vec3A,
};
use serde::Serialize;

use crate::ffi;
use crate::flatten::{first_difference, flatten};
use crate::mirror;

pub const GAME_MODES: [GameMode; 6] = [
    GameMode::Soccar,
    GameMode::Hoops,
    GameMode::Heatseeker,
    GameMode::Snowday,
    GameMode::Dropshot,
    GameMode::TheVoid,
];

pub fn mode_code(mode: GameMode) -> u8 {
    GAME_MODES.iter().position(|&m| m == mode).unwrap() as u8
}

const BODIES: [CarBodyConfig; 7] = [
    CarBodyConfig::OCTANE,
    CarBodyConfig::DOMINUS,
    CarBodyConfig::PLANK,
    CarBodyConfig::BREAKOUT,
    CarBodyConfig::HYBRID,
    CarBodyConfig::MERC,
    CarBodyConfig::PSYCLOPS,
];

pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    fn axis(&mut self) -> f32 {
        match self.below(6) {
            0 => -1.0,
            1 => 1.0,
            2 => 0.0,
            _ => self.range(-1.0, 1.0),
        }
    }
}

#[derive(Clone)]
pub enum ControlSource {
    Idle,
    Random,
    Recording(Arc<Vec<Vec<CarControls>>>),
}

#[derive(Clone, Copy)]
pub enum MutatorChange {
    Gravity,
    CarMass,
    BallGeometry,
    BoostAndPads,
    Mixed,
}

#[derive(Clone, Copy)]
pub enum Action {
    KickoffSeeded,
    KickoffUnseeded,
    RandomBall,
    RandomCars,
    TransplantCars,
    CloneArena,
    Mutators(MutatorChange),
    PadStates,
    Tiles,
    BallIntoGoal,
    KickoffMaskedSeeded,
    KickoffMaskedRng,
    CarsCollide,
    SnapshotRoundTrip,
    CrossBatchCopy,
    CrossBackendCopy,
    PermuteCopy,
}

#[derive(Clone)]
pub struct Scenario {
    pub name: String,
    pub game_mode: GameMode,
    pub cars: Vec<(Team, usize)>,
    pub demo_mode: DemoMode,
    pub arenas: usize,
    pub ticks: u32,
    pub controls: ControlSource,
    pub compare_every: u32,
    pub ticks_per_step: u32,
    pub actions: Vec<(u32, Action)>,
    pub seed: u64,
}

#[derive(Clone, Copy)]
pub enum Backend {
    Cpu,
    Cuda,
    Hybrid,
}

impl Backend {
    pub fn code(&self) -> u8 {
        match self {
            Backend::Cpu => 0,
            Backend::Cuda => 1,
            Backend::Hybrid => 2,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Hybrid => "hybrid",
        }
    }

    pub fn has_device_io(&self) -> bool {
        match self {
            Backend::Cpu => false,
            Backend::Cuda | Backend::Hybrid => true,
        }
    }
}

#[derive(Serialize)]
struct UpstreamSnapshot<'a> {
    state: ArenaState,
    rng: u64,
    events: &'a [ArenaEvent],
    mutators: MutatorConfig,
    ball_scored: bool,
}

fn upstream_flat(arena: &Arena) -> crate::flatten::Flat {
    flatten(&UpstreamSnapshot {
        state: arena.get_arena_state(),
        rng: arena.rng.get_seed(),
        events: arena.get_last_step_events(),
        mutators: *arena.mutator_config(),
        ball_scored: arena.is_ball_scored(),
    })
}

fn to_ffi_vec(v: Vec3A) -> ffi::Vec3 {
    ffi::Vec3 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

fn to_ffi_mat(m: Mat3A) -> ffi::RotMat {
    ffi::RotMat {
        forward: to_ffi_vec(m.x_axis),
        right: to_ffi_vec(m.y_axis),
        up: to_ffi_vec(m.z_axis),
    }
}

fn to_ffi_phys(p: &PhysState) -> ffi::PhysState {
    ffi::PhysState {
        pos: to_ffi_vec(p.pos),
        rot_mat: to_ffi_mat(p.rot_mat),
        vel: to_ffi_vec(p.vel),
        ang_vel: to_ffi_vec(p.ang_vel),
    }
}

pub fn to_ffi_controls(c: &CarControls) -> ffi::CarControls {
    ffi::CarControls {
        throttle: c.throttle,
        steer: c.steer,
        pitch: c.pitch,
        yaw: c.yaw,
        roll: c.roll,
        jump: c.jump,
        boost: c.boost,
        handbrake: c.handbrake,
    }
}

pub fn to_ffi_mutators(m: &MutatorConfig) -> ffi::MutatorConfig {
    ffi::MutatorConfig {
        gravity: to_ffi_vec(m.gravity),
        car_mass: m.car_mass,
        ball_mass: m.ball_mass,
        ball_max_speed: m.ball_max_speed,
        ball_drag: m.ball_drag,
        jump_accel: m.jump_accel,
        jump_immediate_force: m.jump_immediate_force,
        boost_accel_ground: m.boost_accel_ground,
        boost_accel_air: m.boost_accel_air,
        boost_used_per_second: m.boost_used_per_second,
        respawn_delay: m.respawn_delay,
        bump_cooldown_time: m.bump_cooldown_time,
        car_max_boost_amount: m.car_max_boost_amount,
        car_spawn_boost_amount: m.car_spawn_boost_amount,
        boost_pad_amount_small: m.boost_pad_amount_small,
        boost_pad_amount_big: m.boost_pad_amount_big,
        boost_pad_cooldown_big: m.boost_pad_cooldown_big,
        boost_pad_cooldown_small: m.boost_pad_cooldown_small,
        ball_hit_extra_force_scale: m.ball_hit_extra_force_scale,
        bump_force_scale: m.bump_force_scale,
        ball_radius: m.ball_radius,
        unlimited_flips: m.unlimited_flips,
        unlimited_double_jumps: m.unlimited_double_jumps,
        recharge_boost_enabled: m.recharge_boost_enabled,
        recharge_boost_per_second: m.recharge_boost_per_second,
        recharge_boost_delay: m.recharge_boost_delay,
        demo_mode: match m.demo_mode {
            DemoMode::Normal => 0,
            DemoMode::OnContact => 1,
            DemoMode::Disabled => 2,
        },
        enable_team_demos: m.enable_team_demos,
        goal_base_threshold_y: m.goal_base_threshold_y,
        boost_pads_enabled: m.boost_pads_enabled,
    }
}

fn random_rot(rng: &mut SplitMix) -> Mat3A {
    let q = glam::Quat::from_xyzw(
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    )
    .normalize();
    Mat3A::from_quat(q)
}

fn random_vec(rng: &mut SplitMix, lo: Vec3A, hi: Vec3A) -> Vec3A {
    Vec3A::new(
        rng.range(lo.x, hi.x),
        rng.range(lo.y, hi.y),
        rng.range(lo.z, hi.z),
    )
}

fn random_phys(rng: &mut SplitMix, mode: GameMode) -> PhysState {
    let half = match mode {
        GameMode::Hoops => Vec3A::new(2700.0, 3300.0, 1700.0),
        GameMode::Dropshot => Vec3A::new(4500.0, 4000.0, 1900.0),
        _ => Vec3A::new(3900.0, 4900.0, 1900.0),
    };
    PhysState {
        pos: random_vec(rng, Vec3A::new(-half.x, -half.y, 120.0), half),
        rot_mat: random_rot(rng),
        vel: random_vec(rng, Vec3A::splat(-2000.0), Vec3A::splat(2000.0)),
        ang_vel: random_vec(rng, Vec3A::splat(-5.0), Vec3A::splat(5.0)),
    }
}

struct ControlState {
    current: CarControls,
    hold: u32,
}

#[derive(Default)]
pub struct Coverage {
    pub goals: u64,
    pub ball_touches: u64,
    pub bumps: u64,
    pub demos: u64,
    pub pickups: u64,
    pub landings: u64,
}

impl Coverage {
    fn add(&mut self, c: &ffi::EventCounts) {
        self.goals += c.goal_scored as u64;
        for car in &c.cars {
            self.ball_touches += car.hit_ball as u64;
            self.bumps += car.bumps_dealt as u64;
            self.demos += car.demos_dealt as u64;
            self.pickups += car.boost_pickups as u64;
            self.landings += car.landed as u64;
        }
    }
}

pub struct Outcome {
    pub ticks: u64,
    pub comparisons: u64,
    pub coverage: Coverage,
}

fn batch_desc(
    scenario: &Scenario,
    backend: &Backend,
    seeds: &[u64],
) -> Result<ffi::BatchDesc, String> {
    let mut mutators = MutatorConfig::new(scenario.game_mode);
    mutators.demo_mode = scenario.demo_mode;
    let mut cars = Vec::new();
    for &(team, body) in &scenario.cars {
        cars.push(ffi::CarSpec {
            team: team as u8,
            config: ffi::car_body_config(body as u8)?,
        });
    }
    Ok(ffi::BatchDesc {
        backend: backend.code(),
        game_mode: mode_code(scenario.game_mode),
        mutators: to_ffi_mutators(&mutators),
        cars,
        rng_seeds: seeds.to_vec(),
        cpu_threads: 0,
        hybrid_cpu_arenas: match backend {
            Backend::Hybrid => seeds.len().div_ceil(2) as u32,
            Backend::Cpu | Backend::Cuda => 0,
        },
    })
}

pub fn run(
    scenario: &Scenario,
    backend: &Backend,
    meshes: &ffi::MeshSet,
) -> Result<Outcome, String> {
    let mode = scenario.game_mode;
    let mut mutators = MutatorConfig::new(mode);
    mutators.demo_mode = scenario.demo_mode;
    let seeds: Vec<u64> = (0..scenario.arenas as u64)
        .map(|i| scenario.seed.wrapping_mul(0x1000_0000_01b3).wrapping_add(i))
        .collect();

    let mut upstream: Vec<Arena> = seeds
        .iter()
        .map(|&seed| {
            let config = ArenaConfig::new(mode)
                .with_mutators(mutators)
                .with_rng_seed(seed);
            let mut arena = Arena::new_with_config(config);
            for &(team, body) in &scenario.cars {
                arena.add_car(team, BODIES[body]);
            }
            arena
        })
        .collect();

    let mut ours = ffi::Batch::new(meshes, &batch_desc(scenario, backend, &seeds)?)?;

    let num_cars = scenario.cars.len();
    let num_pads = upstream[0].num_boost_pads();
    let has_tiles = mode == GameMode::Dropshot;
    let mut exported = ffi::Exported::new(scenario.arenas, num_cars, num_pads, has_tiles);
    let device_export = if backend.has_device_io() {
        Some(ffi::DeviceExport::new(scenario.arenas, num_cars, num_pads)?)
    } else {
        None
    };
    let mut device_controls = if backend.has_device_io() {
        Some(ffi::DeviceBuffer::new(
            scenario.arenas * num_cars.max(1) * std::mem::size_of::<ffi::CarControls>(),
        )?)
    } else {
        None
    };
    let mut check_exports = |upstream: &[Arena],
                             ours: &mut ffi::Batch,
                             tick: u32,
                             counts: Option<&[ffi::EventCounts]>|
     -> Result<(), String> {
        ours.export_host(&mut exported)?;
        check_export(upstream, &exported, num_cars, num_pads, counts)
            .map_err(|e| format!("host export after tick {tick}: {e}"))?;
        if let Some(device) = &device_export {
            ours.export_device(device, &mut exported)?;
            check_export(upstream, &exported, num_cars, num_pads, counts)
                .map_err(|e| format!("device export after tick {tick}: {e}"))?;
        }
        Ok(())
    };
    let mut rng = SplitMix::new(scenario.seed ^ 0x5eed_0fc0_ffee);
    let mut control_states: Vec<ControlState> = (0..scenario.arenas * num_cars)
        .map(|_| ControlState {
            current: CarControls::DEFAULT,
            hold: 0,
        })
        .collect();
    let mut comparisons = 0u64;

    let compare = |upstream: &[Arena],
                   ours: &mut ffi::Batch,
                   tick: u32,
                   label: &str,
                   counts: Option<&[ffi::EventCounts]>|
     -> Result<u64, String> {
        for (i, arena) in upstream.iter().enumerate() {
            if let Some(counts) = counts {
                check_counts(&counts[i], &ours.event_counts(i as u32)?).map_err(|e| {
                    format!("{label}: arena {i} after tick {tick}: event counts {e}")
                })?;
            }
            let expected = upstream_flat(arena);
            let snap = ours.snapshot(i as u32)?;
            if snap.scalars.error_flags != 0 {
                return Err(format!(
                    "{label}: arena {i} tick {tick}: capacity error flags 0x{:x}",
                    snap.scalars.error_flags
                ));
            }
            check_getters(ours, i as u32, &snap).map_err(|e| {
                format!("{label}: arena {i} tick {tick}: single getter mismatch {e}")
            })?;
            let actual = flatten(&mirror::snapshot(mode_code(mode), &snap));
            if let Some(diff) = first_difference(&expected, &actual) {
                return Err(format!(
                    "{label}: arena {i} after tick {tick}: first difference {diff}"
                ));
            }
        }
        Ok(upstream.len() as u64)
    };

    for (i, arena) in upstream.iter_mut().enumerate() {
        arena.reset_to_random_kickoff(Some(seeds[i]));
        ours.reset_kickoff(i as u32, Some(seeds[i]))?;
    }
    comparisons += compare(&upstream, &mut ours, 0, "initial kickoff", None)?;
    check_exports(&upstream, &mut ours, 0, None)?;

    let mut coverage = Coverage::default();
    let mut step_compares = 0u64;
    let mut tick = 0u32;
    while tick < scenario.ticks {
        for &(at, action) in &scenario.actions {
            if at == tick {
                apply_action(
                    action,
                    scenario,
                    &mut Ctx {
                        upstream: &mut upstream,
                        ours: &mut ours,
                        rng: &mut rng,
                        seeds: &seeds,
                        meshes,
                        backend,
                    },
                    tick,
                )?;
                comparisons += compare(&upstream, &mut ours, tick, "after action", None)?;
                check_exports(&upstream, &mut ours, tick, None)?;
            }
        }
        let bulk_controls = scenario.ticks_per_step > 1;
        let mut all_controls = Vec::with_capacity(scenario.arenas * num_cars);
        for a in 0..scenario.arenas {
            for c in 0..num_cars {
                let controls = match &scenario.controls {
                    ControlSource::Idle => CarControls::DEFAULT,
                    ControlSource::Random => {
                        let st = &mut control_states[a * num_cars + c];
                        if st.hold == 0 {
                            st.current = CarControls {
                                throttle: rng.axis(),
                                steer: rng.axis(),
                                pitch: rng.axis(),
                                yaw: rng.axis(),
                                roll: rng.axis(),
                                jump: rng.below(5) == 0,
                                boost: rng.below(3) == 0,
                                handbrake: rng.below(8) == 0,
                            };
                            st.hold = 1 + rng.below(40) as u32;
                        }
                        st.hold -= 1;
                        st.current
                    }
                    ControlSource::Recording(rec) => {
                        let row = &rec[(tick as usize + a * 977) % rec.len()];
                        row[c % row.len()]
                    }
                };
                upstream[a].set_car_controls(c, controls);
                if bulk_controls {
                    all_controls.push(to_ffi_controls(&controls));
                } else {
                    ours.set_controls(a as u32, c as u32, &to_ffi_controls(&controls))?;
                }
            }
        }
        if bulk_controls {
            match device_controls.as_mut() {
                Some(buffer) => {
                    buffer.upload(&all_controls)?;
                    ours.set_all_controls_device(buffer)?;
                }
                None => ours.set_all_controls(&all_controls)?,
            }
        }
        let next_action = scenario
            .actions
            .iter()
            .map(|&(at, _)| at)
            .filter(|&at| at > tick)
            .min()
            .unwrap_or(u32::MAX);
        let span = scenario
            .ticks_per_step
            .min(scenario.ticks - tick)
            .min(next_action - tick);
        let mut expected_counts = vec![ffi::EventCounts::default(); upstream.len()];
        for (arena, counts) in upstream.iter_mut().zip(expected_counts.iter_mut()) {
            for _ in 0..span {
                arena.step_tick();
                accumulate_counts(arena, counts);
            }
        }
        if scenario.ticks_per_step > 1 {
            ours.step_async(span)?;
            if backend.has_device_io() || (tick / scenario.ticks_per_step).is_multiple_of(2) {
                ours.sync()?;
            }
        } else {
            ours.step(span)?;
        }
        for counts in &expected_counts {
            coverage.add(counts);
        }
        let before = tick;
        tick += span;
        if tick / scenario.compare_every != before / scenario.compare_every
            || tick == scenario.ticks
        {
            comparisons += compare(&upstream, &mut ours, tick, "step", Some(&expected_counts))?;
            step_compares += 1;
            if step_compares.is_multiple_of(4) || tick == scenario.ticks {
                check_exports(&upstream, &mut ours, tick, Some(&expected_counts))?;
            }
        }
    }
    Ok(Outcome {
        ticks: scenario.ticks as u64 * scenario.arenas as u64,
        comparisons,
        coverage,
    })
}

fn check_export(
    upstream: &[Arena],
    ex: &ffi::Exported,
    num_cars: usize,
    num_pads: usize,
    counts: Option<&[ffi::EventCounts]>,
) -> Result<(), String> {
    for (i, arena) in upstream.iter().enumerate() {
        let info = &ex.infos[i];
        if info.tick_count != arena.tick_count()
            || info.rng_state != arena.rng.get_seed()
            || info.error_flags != 0
            || info.ball_scored != arena.is_ball_scored()
            || info.num_events as usize != arena.get_last_step_events().len()
        {
            return Err(format!("arena {i} info {info:?}"));
        }
        if flatten(arena.get_ball_state()) != mirror::ball_flat(&ex.balls[i]) {
            return Err(format!("arena {i} ball state"));
        }
        for c in 0..num_cars {
            if flatten(arena.get_car_state(c)) != mirror::car_flat(&ex.cars[i * num_cars + c]) {
                return Err(format!("arena {i} car {c} state"));
            }
        }
        for p in 0..num_pads {
            let up = arena.get_boost_pad_state(p);
            let pad = &ex.pads[i * num_pads + p];
            let touching: Vec<usize> = pad.touching[..pad.num_touching as usize]
                .iter()
                .map(|&t| t as usize)
                .collect();
            if up.cooldown.to_bits() != pad.cooldown.to_bits() || up.touching_car_idxs != touching {
                return Err(format!("arena {i} pad {p} state"));
            }
        }
        if !ex.tiles.is_empty() {
            let up = arena.get_tile_states();
            for team in 0..2 {
                for t in 0..70 {
                    if up.states[team][t] as u8 != ex.tiles[i].states[team][t] {
                        return Err(format!("arena {i} tile {team}/{t}"));
                    }
                }
            }
        }
        if let Some(counts) = counts {
            check_counts(&counts[i], &ex.counts[i])
                .map_err(|e| format!("arena {i} event counts {e}"))?;
        }
    }
    Ok(())
}

fn check_getters(
    ours: &mut ffi::Batch,
    arena: u32,
    snap: &ffi::ArenaSnapshot,
) -> Result<(), String> {
    if mirror::ball_flat(&ours.ball_state(arena)?) != mirror::ball_flat(&snap.ball) {
        return Err("ball state".into());
    }
    for (c, car) in snap.cars.iter().enumerate() {
        if mirror::car_flat(&ours.car_state(arena, c as u32)?) != mirror::car_flat(car) {
            return Err(format!("car {c} state"));
        }
    }
    let spot = if snap.pads.is_empty() {
        None
    } else {
        Some((snap.scalars.tick_count as usize + arena as usize) % snap.pads.len())
    };
    for (p, pad) in snap
        .pads
        .iter()
        .enumerate()
        .filter(|&(p, _)| Some(p) == spot)
    {
        let (cooldown, touching) = ours.pad_state(arena, p as u32)?;
        if cooldown.to_bits() != pad.cooldown.to_bits()
            || touching[..] != pad.touching[..pad.num_touching as usize]
        {
            return Err(format!("pad {p} state"));
        }
    }
    if snap.scalars.has_tiles && ours.tile_states(arena)?.states != snap.tiles.states {
        return Err("tile states".into());
    }
    Ok(())
}

fn accumulate_counts(arena: &Arena, expected: &mut ffi::EventCounts) {
    for event in arena.get_last_step_events() {
        match event {
            ArenaEvent::BallHitWorld(_) => expected.ball_hit_world += 1,
            ArenaEvent::CarHitBall(e) => {
                let car = &mut expected.cars[e.car_idx];
                car.hit_ball += 1;
                car.last_hit_ball_tick = arena.tick_count() - 1;
            }
            ArenaEvent::CarHitCar(e) => {
                if e.is_demo {
                    expected.cars[e.bumper_car_idx].demos_dealt += 1;
                    expected.cars[e.victim_car_idx].demos_received += 1;
                } else {
                    expected.cars[e.bumper_car_idx].bumps_dealt += 1;
                    expected.cars[e.victim_car_idx].bumps_received += 1;
                }
            }
            ArenaEvent::CarHitWorld(e) => expected.cars[e.car_idx].hit_world += 1,
            ArenaEvent::CarPickupBoost(e) => expected.cars[e.car_idx].boost_pickups += 1,
            ArenaEvent::CarLanded(e) => expected.cars[e.car_idx].landed += 1,
        }
    }
    if !expected.goal_scored && arena.is_ball_scored() {
        expected.goal_scored = true;
        expected.goal_tick = arena.tick_count();
    }
}

fn check_counts(expected: &ffi::EventCounts, got: &ffi::EventCounts) -> Result<(), String> {
    let same_cars = (0..8).all(|c| {
        let (a, b) = (&expected.cars[c], &got.cars[c]);
        a.hit_ball == b.hit_ball
            && a.hit_world == b.hit_world
            && a.bumps_dealt == b.bumps_dealt
            && a.demos_dealt == b.demos_dealt
            && a.bumps_received == b.bumps_received
            && a.demos_received == b.demos_received
            && a.boost_pickups == b.boost_pickups
            && a.landed == b.landed
            && a.last_hit_ball_tick == b.last_hit_ball_tick
    });
    let goal_ok = expected.goal_scored == got.goal_scored
        && (!expected.goal_scored || expected.goal_tick == got.goal_tick);
    if expected.ball_hit_world != got.ball_hit_world || !same_cars || !goal_ok {
        return Err(format!("expected {expected:?}, got {got:?}"));
    }
    Ok(())
}

fn ffi_core(core: &CarCore) -> ffi::CarCore {
    ffi::CarCore {
        phys: to_ffi_phys(&core.phys),
        controls: to_ffi_controls(&core.controls),
        has_jumped: core.has_jumped,
        has_double_jumped: core.has_double_jumped,
        has_flipped: core.has_flipped,
        flip_rel_torque: to_ffi_vec(core.flip_rel_torque),
        jump_ticks: core.jump_ticks,
        flip_time: core.flip_time,
        is_flipping: core.is_flipping,
        is_jumping: core.is_jumping,
        jump_hold_broken: core.jump_hold_broken,
        air_time_since_jump: core.air_time_since_jump,
        boost: core.boost,
        is_auto_flipping: core.is_auto_flipping,
        auto_flip_timer: core.auto_flip_timer,
        auto_flip_torque_scale: core.auto_flip_torque_scale,
        bump_cooldown_timer: core.bump_cooldown_timer,
        has_bump_cooldown_victim: core.bump_cooldown_victim.is_some(),
        bump_cooldown_victim: core.bump_cooldown_victim.unwrap_or(0) as u32,
        has_last_extra_hit_tick: core.last_extra_hit_tick.is_some(),
        last_extra_hit_tick: core.last_extra_hit_tick.unwrap_or(0),
        is_demoed: core.is_demoed,
        demo_respawn_timer: core.demo_respawn_timer,
    }
}

struct Ctx<'a> {
    upstream: &'a mut [Arena],
    ours: &'a mut ffi::Batch,
    rng: &'a mut SplitMix,
    seeds: &'a [u64],
    meshes: &'a ffi::MeshSet,
    backend: &'a Backend,
}

fn apply_action(
    action: Action,
    scenario: &Scenario,
    ctx: &mut Ctx,
    tick: u32,
) -> Result<(), String> {
    let mode = scenario.game_mode;
    let upstream = &mut *ctx.upstream;
    let ours = &mut *ctx.ours;
    let rng = &mut *ctx.rng;
    let seeds = ctx.seeds;
    let n = upstream.len();
    match action {
        Action::KickoffSeeded => {
            for i in 0..n {
                let seed = seeds[i] ^ tick as u64;
                upstream[i].reset_to_random_kickoff(Some(seed));
                ours.reset_kickoff(i as u32, Some(seed))?;
            }
        }
        Action::KickoffUnseeded => {
            for (i, arena) in upstream.iter_mut().enumerate() {
                arena.reset_to_random_kickoff(None);
                ours.reset_kickoff(i as u32, None)?;
            }
        }
        Action::RandomBall => {
            for (i, arena) in upstream.iter_mut().enumerate() {
                let mut ball = BallState::DEFAULT;
                ball.phys = random_phys(rng, mode);
                arena.set_ball_state(ball);
                let mut ours_ball = ours.snapshot(i as u32)?.ball;
                ours_ball.phys = to_ffi_phys(&ball.phys);
                ours_ball.hs_info = ffi::HeatseekerInfo {
                    y_target_dir: ball.hs_info.y_target_dir,
                    cur_target_speed: ball.hs_info.cur_target_speed,
                    time_since_hit: ball.hs_info.time_since_hit,
                };
                ours_ball.ds_info = ffi::DropshotInfo {
                    charge_level: ball.ds_info.charge_level,
                    accumulated_hit_force: ball.ds_info.accumulated_hit_force,
                    y_target_dir: ball.ds_info.y_target_dir,
                    has_last_damage_tick: ball.ds_info.last_damage_tick.is_some(),
                    last_damage_tick: ball.ds_info.last_damage_tick.unwrap_or(0),
                };
                ours_ball.tick_count_since_kickoff = ball.tick_count_since_kickoff;
                ours.set_ball_state(i as u32, &ours_ball)?;
            }
        }
        Action::RandomCars => {
            for (i, arena) in upstream.iter_mut().enumerate() {
                for c in 0..scenario.cars.len() {
                    let phys = random_phys(rng, mode);
                    let boost = rng.range(0.0, 100.0);
                    let core = CarCore {
                        phys,
                        boost,
                        ..CarCore::DEFAULT
                    };
                    arena.set_car_state(c, core);
                    ours.set_car_state(i as u32, c as u32, &ffi_core(&core))?;
                }
            }
        }
        Action::TransplantCars => {
            let up_states: Vec<Vec<rocketsim::CarState>> = upstream
                .iter()
                .map(|a| {
                    (0..scenario.cars.len())
                        .map(|c| *a.get_car_state(c))
                        .collect()
                })
                .collect();
            let mut our_states = Vec::new();
            for i in 0..n {
                our_states.push(ours.snapshot(i as u32)?.cars);
            }
            for (i, arena) in upstream.iter_mut().enumerate() {
                let src = (i + 1) % n;
                for c in 0..scenario.cars.len() {
                    arena.set_car_state_full(c, up_states[src][c]);
                    ours.set_car_state_full(i as u32, c as u32, &our_states[src][c])?;
                }
            }
        }
        Action::CloneArena => {
            if n >= 2 {
                for i in (0..n).step_by(2) {
                    let dst = (i + 1) % n;
                    upstream[dst] = upstream[i].clone();
                    ours.copy_from(None, &[i as u32], &[dst as u32])?;
                }
            }
        }
        Action::Mutators(change) => {
            for (i, arena) in upstream.iter_mut().enumerate() {
                let mut m = *arena.mutator_config();
                match change {
                    MutatorChange::Gravity => {
                        m.gravity =
                            Vec3A::new(rng.range(-200.0, 200.0), 0.0, rng.range(-1300.0, -200.0))
                    }
                    MutatorChange::CarMass => m.car_mass = rng.range(120.0, 260.0),
                    MutatorChange::BallGeometry => {
                        m.ball_radius = rng.range(80.0, 160.0);
                        m.ball_mass = rng.range(20.0, 60.0);
                        m.ball_drag = rng.range(0.0, 0.1);
                    }
                    MutatorChange::BoostAndPads => {
                        m.boost_pad_amount_small = rng.range(5.0, 40.0);
                        m.boost_pad_cooldown_big = rng.range(2.0, 12.0);
                        m.boost_used_per_second = rng.range(10.0, 60.0);
                        m.boost_accel_air = rng.range(500.0, 1500.0);
                    }
                    MutatorChange::Mixed => {
                        m.jump_accel = rng.range(1000.0, 2000.0);
                        m.bump_force_scale = rng.range(0.5, 3.0);
                        m.ball_hit_extra_force_scale = rng.range(0.5, 2.0);
                        m.unlimited_flips = rng.below(2) == 0;
                        m.unlimited_double_jumps = rng.below(2) == 0;
                        m.recharge_boost_enabled = rng.below(2) == 0;
                        m.respawn_delay = rng.range(0.5, 4.0);
                    }
                }
                arena.set_mutator_config(m);
                ours.set_mutators(i as u32, &to_ffi_mutators(&m))?;
            }
        }
        Action::PadStates => {
            for (i, arena) in upstream.iter_mut().enumerate() {
                let num_pads = arena.num_boost_pads();
                for p in 0..num_pads {
                    let cooldown = if rng.below(3) == 0 {
                        0.0
                    } else {
                        rng.range(0.0, 10.0)
                    };
                    let touching: Vec<usize> = (0..scenario.cars.len())
                        .filter(|_| rng.below(6) == 0)
                        .collect();
                    arena.set_boost_pad_state(
                        p,
                        BoostPadState {
                            touching_car_idxs: touching.clone(),
                            cooldown,
                        },
                    );
                    let t32: Vec<u32> = touching.iter().map(|&t| t as u32).collect();
                    ours.set_pad_state(i as u32, p as u32, cooldown, &t32)?;
                }
            }
        }
        Action::CarsCollide => {
            let num_cars = scenario.cars.len();
            let half = num_cars.div_ceil(2);
            for (i, arena) in upstream.iter_mut().enumerate() {
                for c in 0..num_cars {
                    let (pair, side) = if c < half { (c, 0) } else { (c - half, 1) };
                    let dir = if side == 0 { 1.0 } else { -1.0 };
                    let yaw = if side == 0 {
                        std::f32::consts::FRAC_PI_2
                    } else {
                        -std::f32::consts::FRAC_PI_2
                    };
                    let airborne = (i + pair) % 3 == 2;
                    let phys = PhysState {
                        pos: Vec3A::new(
                            -1600.0 + pair as f32 * 800.0 + rng.range(-40.0, 40.0),
                            -dir * rng.range(250.0, 500.0),
                            if airborne { 300.0 } else { 17.0 },
                        ),
                        rot_mat: Mat3A::from_rotation_z(yaw),
                        vel: Vec3A::new(0.0, dir * rng.range(1400.0, 2290.0), 0.0),
                        ang_vel: Vec3A::ZERO,
                    };
                    let core = CarCore {
                        phys,
                        boost: 100.0,
                        controls: CarControls {
                            throttle: 1.0,
                            boost: true,
                            ..CarControls::DEFAULT
                        },
                        ..CarCore::DEFAULT
                    };
                    arena.set_car_state(c, core);
                    ours.set_car_state(i as u32, c as u32, &ffi_core(&core))?;
                }
            }
        }
        Action::KickoffMaskedSeeded | Action::KickoffMaskedRng => {
            let mask: Vec<u8> = (0..n)
                .map(|i| ((i + tick as usize) % 3 != 1) as u8)
                .collect();
            let reset_seeds: Vec<u64> = (0..n)
                .map(|i| seeds[i] ^ (tick as u64).wrapping_mul(7919))
                .collect();
            let seeded = matches!(action, Action::KickoffMaskedSeeded);
            for i in 0..n {
                if mask[i] != 0 {
                    upstream[i].reset_to_random_kickoff(seeded.then_some(reset_seeds[i]));
                }
            }
            let use_device = match ctx.backend {
                Backend::Cpu => false,
                Backend::Cuda => true,
                Backend::Hybrid => seeded,
            };
            if use_device {
                let mut mask_buffer = ffi::DeviceBuffer::new(n)?;
                mask_buffer.upload(&mask)?;
                let mut seed_buffer = ffi::DeviceBuffer::new(n * 8)?;
                seed_buffer.upload(&reset_seeds)?;
                ours.reset_kickoff_masked_device(&mask_buffer, seeded.then_some(&seed_buffer))?;
            } else {
                ours.reset_kickoff_masked_host(&mask, seeded.then_some(&reset_seeds[..]))?;
            }
        }
        Action::SnapshotRoundTrip => {
            if n >= 2 {
                let snapshots = (0..n)
                    .map(|i| ours.save(i as u32))
                    .collect::<Result<Vec<_>, _>>()?;
                let clones: Vec<Arena> = upstream.to_vec();
                for (i, arena) in upstream.iter_mut().enumerate() {
                    let src = (i + n - 1) % n;
                    *arena = clones[src].clone();
                    ours.load(i as u32, &snapshots[src])?;
                }
            }
        }
        Action::CrossBatchCopy | Action::CrossBackendCopy => {
            if n >= 2 {
                let other = match (action, ctx.backend) {
                    (Action::CrossBatchCopy, b) => *b,
                    (_, Backend::Cpu) => Backend::Cuda,
                    (_, Backend::Cuda) => Backend::Cpu,
                    (_, Backend::Hybrid) => Backend::Cuda,
                };
                let mut temp =
                    ffi::Batch::new(ctx.meshes, &batch_desc(scenario, &other, &seeds[..2])?)?;
                let clone = upstream[0].clone();
                temp.copy_from(Some(ours), &[0], &[1])?;
                ours.copy_from(Some(&temp), &[1], &[n as u32 - 1])?;
                upstream[n - 1] = clone;
            }
        }
        Action::PermuteCopy => {
            let mut srcs: Vec<u32> = (0..n as u32).rev().collect();
            let mut dsts: Vec<u32> = (0..n as u32).collect();
            srcs.push(0);
            dsts.push(0);
            let snapshot: Vec<_> = srcs.iter().map(|&s| upstream[s as usize].clone()).collect();
            for (k, &d) in dsts.iter().enumerate() {
                upstream[d as usize] = snapshot[k].clone();
            }
            ours.copy_from(None, &srcs, &dsts)?;
        }
        Action::BallIntoGoal => {
            for (i, arena) in upstream.iter_mut().enumerate() {
                let side = if (i + tick as usize).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                let mut ball = BallState::DEFAULT;
                ball.phys.pos = match mode {
                    GameMode::Hoops => Vec3A::new(rng.range(-150.0, 150.0), side * 2900.0, 650.0),
                    _ => Vec3A::new(
                        rng.range(-600.0, 600.0),
                        side * 4300.0,
                        rng.range(100.0, 500.0),
                    ),
                };
                ball.phys.vel = match mode {
                    GameMode::Hoops => Vec3A::new(0.0, side * 150.0, -600.0),
                    _ => Vec3A::new(
                        rng.range(-200.0, 200.0),
                        side * rng.range(1500.0, 3000.0),
                        0.0,
                    ),
                };
                ball.phys.ang_vel = random_vec(rng, Vec3A::splat(-3.0), Vec3A::splat(3.0));
                arena.set_ball_state(ball);
                let mut ours_ball = ours.snapshot(i as u32)?.ball;
                ours_ball.phys = to_ffi_phys(&ball.phys);
                ours_ball.hs_info = ffi::HeatseekerInfo {
                    y_target_dir: ball.hs_info.y_target_dir,
                    cur_target_speed: ball.hs_info.cur_target_speed,
                    time_since_hit: ball.hs_info.time_since_hit,
                };
                ours_ball.ds_info = ffi::DropshotInfo {
                    charge_level: ball.ds_info.charge_level,
                    accumulated_hit_force: ball.ds_info.accumulated_hit_force,
                    y_target_dir: ball.ds_info.y_target_dir,
                    has_last_damage_tick: ball.ds_info.last_damage_tick.is_some(),
                    last_damage_tick: ball.ds_info.last_damage_tick.unwrap_or(0),
                };
                ours_ball.tick_count_since_kickoff = ball.tick_count_since_kickoff;
                ours.set_ball_state(i as u32, &ours_ball)?;
            }
        }
        Action::Tiles => {
            for (i, arena) in upstream.iter_mut().enumerate() {
                let mut tiles = TileStates::DEFAULT;
                let mut ffi_tiles = ffi::TileStates::default();
                for team in 0..2 {
                    for t in 0..70 {
                        let s = rng.below(3);
                        tiles.states[team][t] = match s {
                            0 => TileDamageState::Full,
                            1 => TileDamageState::Damaged,
                            _ => TileDamageState::Broken,
                        };
                        ffi_tiles.states[team][t] = s as u8;
                    }
                }
                arena.set_tile_states(tiles);
                ours.set_tiles(i as u32, &ffi_tiles)?;
            }
        }
    }
    Ok(())
}
