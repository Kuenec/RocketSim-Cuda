use serde::Serialize;

use crate::ffi;

const USER_INFO_NAMES: [&str; 4] = ["None", "Car", "Ball", "DropshotTile"];
const TEAM_NAMES: [&str; 2] = ["Blue", "Orange"];
const GAME_MODE_NAMES: [&str; 6] = [
    "Soccar",
    "Hoops",
    "Heatseeker",
    "Snowday",
    "Dropshot",
    "TheVoid",
];
const TILE_NAMES: [&str; 3] = ["Full", "Damaged", "Broken"];
const DEMO_MODE_NAMES: [&str; 3] = ["Normal", "OnContact", "Disabled"];

struct Variant(&'static str);

impl Serialize for Variant {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_unit_variant("", 0, self.0)
    }
}

fn v3(v: ffi::Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

fn mat(m: ffi::RotMat) -> [f32; 9] {
    [
        m.forward.x,
        m.forward.y,
        m.forward.z,
        m.right.x,
        m.right.y,
        m.right.z,
        m.up.x,
        m.up.y,
        m.up.z,
    ]
}

#[derive(Serialize)]
struct PhysState {
    pos: [f32; 3],
    rot_mat: [f32; 9],
    vel: [f32; 3],
    ang_vel: [f32; 3],
}

fn phys(p: &ffi::PhysState) -> PhysState {
    PhysState {
        pos: v3(p.pos),
        rot_mat: mat(p.rot_mat),
        vel: v3(p.vel),
        ang_vel: v3(p.ang_vel),
    }
}

#[derive(Serialize)]
struct CarControls {
    throttle: f32,
    steer: f32,
    pitch: f32,
    yaw: f32,
    roll: f32,
    jump: bool,
    boost: bool,
    handbrake: bool,
}

fn controls(c: &ffi::CarControls) -> CarControls {
    CarControls {
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

#[derive(Serialize)]
struct RaycastHitInfo {
    hit_point: [f32; 3],
    hit_normal: [f32; 3],
    hit_fraction: f32,
    user_info: Variant,
}

fn hit(h: &ffi::RaycastHitInfo) -> RaycastHitInfo {
    RaycastHitInfo {
        hit_point: v3(h.hit_point),
        hit_normal: v3(h.hit_normal),
        hit_fraction: h.hit_fraction,
        user_info: Variant(USER_INFO_NAMES[h.user_info as usize]),
    }
}

#[derive(Serialize)]
struct CarCache {
    is_on_ground: bool,
    wheels_with_contact: Vec<Option<RaycastHitInfo>>,
    world_contact_normal: Option<[f32; 3]>,
    prev_controls: CarControls,
    world_wheels_prev: u8,
    demo_tick: u64,
    demoer_idx: Option<u64>,
}

#[derive(Serialize)]
struct CarState {
    phys: PhysState,
    controls: CarControls,
    has_jumped: bool,
    has_double_jumped: bool,
    has_flipped: bool,
    flip_rel_torque: [f32; 3],
    jump_ticks: u32,
    flip_time: f32,
    is_flipping: bool,
    is_jumping: bool,
    jump_hold_broken: bool,
    air_time: f32,
    air_time_since_jump: f32,
    boost: f32,
    handbrake_val: f32,
    is_supersonic: bool,
    supersonic_grace_timer: f32,
    time_since_boosted: f32,
    boosting_time: f32,
    is_boosting: bool,
    is_auto_flipping: bool,
    auto_flip_timer: f32,
    auto_flip_torque_scale: f32,
    bump_cooldown_timer: f32,
    bump_cooldown_victim: Option<u64>,
    last_extra_hit_tick: Option<u64>,
    is_demoed: bool,
    demo_respawn_timer: f32,
    cache: CarCache,
}

fn car_state(s: &ffi::CarState) -> CarState {
    let c = &s.cache;
    CarState {
        phys: phys(&s.phys),
        controls: controls(&s.controls),
        has_jumped: s.has_jumped,
        has_double_jumped: s.has_double_jumped,
        has_flipped: s.has_flipped,
        flip_rel_torque: v3(s.flip_rel_torque),
        jump_ticks: s.jump_ticks,
        flip_time: s.flip_time,
        is_flipping: s.is_flipping,
        is_jumping: s.is_jumping,
        jump_hold_broken: s.jump_hold_broken,
        air_time: s.air_time,
        air_time_since_jump: s.air_time_since_jump,
        boost: s.boost,
        handbrake_val: s.handbrake_val,
        is_supersonic: s.is_supersonic,
        supersonic_grace_timer: s.supersonic_grace_timer,
        time_since_boosted: s.time_since_boosted,
        boosting_time: s.boosting_time,
        is_boosting: s.is_boosting,
        is_auto_flipping: s.is_auto_flipping,
        auto_flip_timer: s.auto_flip_timer,
        auto_flip_torque_scale: s.auto_flip_torque_scale,
        bump_cooldown_timer: s.bump_cooldown_timer,
        bump_cooldown_victim: s
            .has_bump_cooldown_victim
            .then_some(s.bump_cooldown_victim as u64),
        last_extra_hit_tick: s.has_last_extra_hit_tick.then_some(s.last_extra_hit_tick),
        is_demoed: s.is_demoed,
        demo_respawn_timer: s.demo_respawn_timer,
        cache: CarCache {
            is_on_ground: c.is_on_ground,
            wheels_with_contact: (0..4)
                .map(|i| c.has_wheel_contact[i].then(|| hit(&c.wheels_with_contact[i])))
                .collect(),
            world_contact_normal: c
                .has_world_contact_normal
                .then(|| v3(c.world_contact_normal)),
            prev_controls: controls(&c.prev_controls),
            world_wheels_prev: c.world_wheels_prev,
            demo_tick: c.demo_tick,
            demoer_idx: c.has_demoer.then_some(c.demoer_idx as u64),
        },
    }
}

#[derive(Serialize)]
struct WheelPairConfig {
    wheel_radius: f32,
    suspension_rest_length: f32,
    connection_point_offset: [f32; 3],
}

fn wheels(w: &ffi::WheelPairConfig) -> WheelPairConfig {
    WheelPairConfig {
        wheel_radius: w.wheel_radius,
        suspension_rest_length: w.suspension_rest_length,
        connection_point_offset: v3(w.connection_point_offset),
    }
}

#[derive(Serialize)]
struct CarBodyConfig {
    hitbox_size: [f32; 3],
    hitbox_pos_offset: [f32; 3],
    front_wheels: WheelPairConfig,
    back_wheels: WheelPairConfig,
    three_wheels: bool,
    dodge_deadzone: f32,
    steer_angle_curve: Vec<(f32, f32)>,
    powerslide_steer_angle_curve: Vec<(f32, f32)>,
}

#[derive(Serialize)]
struct CarInfo {
    idx: u64,
    team: Variant,
    config: CarBodyConfig,
}

fn car_info(i: &ffi::CarInfo) -> CarInfo {
    let c = &i.config;
    CarInfo {
        idx: i.idx as u64,
        team: Variant(TEAM_NAMES[i.team as usize]),
        config: CarBodyConfig {
            hitbox_size: v3(c.hitbox_size),
            hitbox_pos_offset: v3(c.hitbox_pos_offset),
            front_wheels: wheels(&c.front_wheels),
            back_wheels: wheels(&c.back_wheels),
            three_wheels: c.three_wheels,
            dodge_deadzone: c.dodge_deadzone,
            steer_angle_curve: c.steer_angle_curve.iter().map(|p| (p.x, p.y)).collect(),
            powerslide_steer_angle_curve: c
                .powerslide_steer_angle_curve
                .iter()
                .map(|p| (p.x, p.y))
                .collect(),
        },
    }
}

#[derive(Serialize)]
struct HeatseekerInfo {
    y_target_dir: i8,
    cur_target_speed: f32,
    time_since_hit: f32,
}

#[derive(Serialize)]
struct DropshotInfo {
    charge_level: u8,
    accumulated_hit_force: f32,
    y_target_dir: i8,
    last_damage_tick: Option<u64>,
}

#[derive(Serialize)]
struct BallState {
    phys: PhysState,
    hs_info: HeatseekerInfo,
    ds_info: DropshotInfo,
    tick_count_since_kickoff: u64,
}

fn ball_state(b: &ffi::BallState) -> BallState {
    BallState {
        phys: phys(&b.phys),
        hs_info: HeatseekerInfo {
            y_target_dir: b.hs_info.y_target_dir,
            cur_target_speed: b.hs_info.cur_target_speed,
            time_since_hit: b.hs_info.time_since_hit,
        },
        ds_info: DropshotInfo {
            charge_level: b.ds_info.charge_level,
            accumulated_hit_force: b.ds_info.accumulated_hit_force,
            y_target_dir: b.ds_info.y_target_dir,
            last_damage_tick: b
                .ds_info
                .has_last_damage_tick
                .then_some(b.ds_info.last_damage_tick),
        },
        tick_count_since_kickoff: b.tick_count_since_kickoff,
    }
}

#[derive(Serialize)]
struct BoostPadConfig {
    pos: [f32; 3],
    is_big: bool,
}

#[derive(Serialize)]
struct BoostPadState {
    touching_car_idxs: Vec<u64>,
    cooldown: f32,
}

#[derive(Serialize)]
struct TileStates {
    states: Vec<Vec<Variant>>,
}

#[derive(Serialize)]
struct ArenaState {
    game_mode: Variant,
    tick_count: u64,
    cars: Vec<(CarInfo, CarState)>,
    ball: BallState,
    boost_pads: Vec<(BoostPadConfig, BoostPadState)>,
    tile_states: Option<TileStates>,
}

#[derive(Serialize)]
struct BallHitWorldEvent {
    contact_point: [f32; 3],
    contact_normal: [f32; 3],
}

#[derive(Serialize)]
struct CarHitBallEvent {
    car_idx: u64,
    contact_point: [f32; 3],
    extra_hit_vel: [f32; 3],
}

#[derive(Serialize)]
struct CarHitCarEvent {
    bumper_car_idx: u64,
    victim_car_idx: u64,
    contact_point: [f32; 3],
    is_demo: bool,
}

#[derive(Serialize)]
struct CarHitWorldEvent {
    car_idx: u64,
    contact_point: [f32; 3],
    contact_normal: [f32; 3],
}

#[derive(Serialize)]
struct CarPickupBoostEvent {
    car_idx: u64,
    boost_pad_idx: u64,
}

#[derive(Serialize)]
struct CarLandedEvent {
    car_idx: u64,
    wheels: Vec<Option<RaycastHitInfo>>,
}

#[derive(Serialize)]
enum ArenaEvent {
    BallHitWorld(BallHitWorldEvent),
    CarHitBall(CarHitBallEvent),
    CarHitCar(CarHitCarEvent),
    CarHitWorld(CarHitWorldEvent),
    CarPickupBoost(CarPickupBoostEvent),
    CarLanded(CarLandedEvent),
}

fn event(e: &ffi::ArenaEvent) -> ArenaEvent {
    match e.kind {
        0 => ArenaEvent::BallHitWorld(BallHitWorldEvent {
            contact_point: v3(e.contact_point),
            contact_normal: v3(e.contact_normal),
        }),
        1 => ArenaEvent::CarHitBall(CarHitBallEvent {
            car_idx: e.car_idx as u64,
            contact_point: v3(e.contact_point),
            extra_hit_vel: v3(e.extra_hit_vel),
        }),
        2 => ArenaEvent::CarHitCar(CarHitCarEvent {
            bumper_car_idx: e.car_idx as u64,
            victim_car_idx: e.victim_car_idx as u64,
            contact_point: v3(e.contact_point),
            is_demo: e.is_demo,
        }),
        3 => ArenaEvent::CarHitWorld(CarHitWorldEvent {
            car_idx: e.car_idx as u64,
            contact_point: v3(e.contact_point),
            contact_normal: v3(e.contact_normal),
        }),
        4 => ArenaEvent::CarPickupBoost(CarPickupBoostEvent {
            car_idx: e.car_idx as u64,
            boost_pad_idx: e.boost_pad_idx as u64,
        }),
        _ => ArenaEvent::CarLanded(CarLandedEvent {
            car_idx: e.car_idx as u64,
            wheels: (0..4)
                .map(|i| e.has_wheel_contact[i].then(|| hit(&e.wheels[i])))
                .collect(),
        }),
    }
}

#[derive(Serialize)]
struct MutatorConfig {
    gravity: [f32; 3],
    car_mass: f32,
    ball_mass: f32,
    ball_max_speed: f32,
    ball_drag: f32,
    jump_accel: f32,
    jump_immediate_force: f32,
    boost_accel_ground: f32,
    boost_accel_air: f32,
    boost_used_per_second: f32,
    respawn_delay: f32,
    bump_cooldown_time: f32,
    car_max_boost_amount: f32,
    car_spawn_boost_amount: f32,
    boost_pad_amount_small: f32,
    boost_pad_amount_big: f32,
    boost_pad_cooldown_big: f32,
    boost_pad_cooldown_small: f32,
    ball_hit_extra_force_scale: f32,
    bump_force_scale: f32,
    ball_radius: f32,
    unlimited_flips: bool,
    unlimited_double_jumps: bool,
    recharge_boost_enabled: bool,
    recharge_boost_per_second: f32,
    recharge_boost_delay: f32,
    demo_mode: Variant,
    enable_team_demos: bool,
    goal_base_threshold_y: f32,
    boost_pads_enabled: bool,
}

fn mutators(m: &ffi::MutatorConfig) -> MutatorConfig {
    MutatorConfig {
        gravity: v3(m.gravity),
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
        demo_mode: Variant(DEMO_MODE_NAMES[m.demo_mode as usize]),
        enable_team_demos: m.enable_team_demos,
        goal_base_threshold_y: m.goal_base_threshold_y,
        boost_pads_enabled: m.boost_pads_enabled,
    }
}

#[derive(Serialize)]
pub struct Snapshot {
    state: ArenaState,
    rng: u64,
    events: Vec<ArenaEvent>,
    mutators: MutatorConfig,
    ball_scored: bool,
}

pub fn car_flat(c: &ffi::CarState) -> crate::flatten::Flat {
    crate::flatten::flatten(&car_state(c))
}

pub fn ball_flat(b: &ffi::BallState) -> crate::flatten::Flat {
    crate::flatten::flatten(&ball_state(b))
}

pub fn snapshot(game_mode: u8, s: &ffi::ArenaSnapshot) -> Snapshot {
    let state = ArenaState {
        game_mode: Variant(GAME_MODE_NAMES[game_mode as usize]),
        tick_count: s.scalars.tick_count,
        cars: s
            .car_infos
            .iter()
            .zip(s.cars.iter())
            .map(|(i, c)| (car_info(i), car_state(c)))
            .collect(),
        ball: ball_state(&s.ball),
        boost_pads: s
            .pads
            .iter()
            .map(|p| {
                (
                    BoostPadConfig {
                        pos: v3(p.config.pos),
                        is_big: p.config.is_big,
                    },
                    BoostPadState {
                        touching_car_idxs: p.touching[..p.num_touching as usize]
                            .iter()
                            .map(|&i| i as u64)
                            .collect(),
                        cooldown: p.cooldown,
                    },
                )
            })
            .collect(),
        tile_states: s.scalars.has_tiles.then(|| TileStates {
            states: s
                .tiles
                .states
                .iter()
                .map(|team| {
                    team.iter()
                        .map(|&t| Variant(TILE_NAMES[t as usize]))
                        .collect()
                })
                .collect(),
        }),
    };
    Snapshot {
        state,
        rng: s.scalars.rng_state,
        events: s.events.iter().map(event).collect(),
        mutators: mutators(&s.mutators),
        ball_scored: s.scalars.ball_scored,
    }
}
