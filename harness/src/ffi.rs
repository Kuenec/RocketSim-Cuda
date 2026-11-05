use std::ffi::{CStr, CString, c_char};

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct RotMat {
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct PhysState {
    pub pos: Vec3,
    pub rot_mat: RotMat,
    pub vel: Vec3,
    pub ang_vel: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CarControls {
    pub throttle: f32,
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub jump: bool,
    pub boost: bool,
    pub handbrake: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct RaycastHitInfo {
    pub hit_point: Vec3,
    pub hit_normal: Vec3,
    pub hit_fraction: f32,
    pub user_info: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CarCache {
    pub is_on_ground: bool,
    pub has_wheel_contact: [bool; 4],
    pub wheels_with_contact: [RaycastHitInfo; 4],
    pub has_world_contact_normal: bool,
    pub world_contact_normal: Vec3,
    pub prev_controls: CarControls,
    pub world_wheels_prev: u8,
    pub demo_tick: u64,
    pub has_demoer: bool,
    pub demoer_idx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CarState {
    pub phys: PhysState,
    pub controls: CarControls,
    pub has_jumped: bool,
    pub has_double_jumped: bool,
    pub has_flipped: bool,
    pub flip_rel_torque: Vec3,
    pub jump_ticks: u32,
    pub flip_time: f32,
    pub is_flipping: bool,
    pub is_jumping: bool,
    pub jump_hold_broken: bool,
    pub air_time: f32,
    pub air_time_since_jump: f32,
    pub boost: f32,
    pub handbrake_val: f32,
    pub is_supersonic: bool,
    pub supersonic_grace_timer: f32,
    pub time_since_boosted: f32,
    pub boosting_time: f32,
    pub is_boosting: bool,
    pub is_auto_flipping: bool,
    pub auto_flip_timer: f32,
    pub auto_flip_torque_scale: f32,
    pub bump_cooldown_timer: f32,
    pub has_bump_cooldown_victim: bool,
    pub bump_cooldown_victim: u32,
    pub has_last_extra_hit_tick: bool,
    pub last_extra_hit_tick: u64,
    pub is_demoed: bool,
    pub demo_respawn_timer: f32,
    pub cache: CarCache,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CarCore {
    pub phys: PhysState,
    pub controls: CarControls,
    pub has_jumped: bool,
    pub has_double_jumped: bool,
    pub has_flipped: bool,
    pub flip_rel_torque: Vec3,
    pub jump_ticks: u32,
    pub flip_time: f32,
    pub is_flipping: bool,
    pub is_jumping: bool,
    pub jump_hold_broken: bool,
    pub air_time_since_jump: f32,
    pub boost: f32,
    pub is_auto_flipping: bool,
    pub auto_flip_timer: f32,
    pub auto_flip_torque_scale: f32,
    pub bump_cooldown_timer: f32,
    pub has_bump_cooldown_victim: bool,
    pub bump_cooldown_victim: u32,
    pub has_last_extra_hit_tick: bool,
    pub last_extra_hit_tick: u64,
    pub is_demoed: bool,
    pub demo_respawn_timer: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CurvePoint {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct WheelPairConfig {
    pub wheel_radius: f32,
    pub suspension_rest_length: f32,
    pub connection_point_offset: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CarBodyConfig {
    pub hitbox_size: Vec3,
    pub hitbox_pos_offset: Vec3,
    pub front_wheels: WheelPairConfig,
    pub back_wheels: WheelPairConfig,
    pub three_wheels: bool,
    pub dodge_deadzone: f32,
    pub steer_angle_curve: [CurvePoint; 6],
    pub powerslide_steer_angle_curve: [CurvePoint; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CarInfo {
    pub idx: u32,
    pub team: u8,
    pub config: CarBodyConfig,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct HeatseekerInfo {
    pub y_target_dir: i8,
    pub cur_target_speed: f32,
    pub time_since_hit: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct DropshotInfo {
    pub charge_level: u8,
    pub accumulated_hit_force: f32,
    pub y_target_dir: i8,
    pub has_last_damage_tick: bool,
    pub last_damage_tick: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct BallState {
    pub phys: PhysState,
    pub hs_info: HeatseekerInfo,
    pub ds_info: DropshotInfo,
    pub tick_count_since_kickoff: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct BoostPadConfig {
    pub pos: Vec3,
    pub is_big: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TileStates {
    pub states: [[u8; 70]; 2],
}

impl Default for TileStates {
    fn default() -> Self {
        Self {
            states: [[0; 70]; 2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MutatorConfig {
    pub gravity: Vec3,
    pub car_mass: f32,
    pub ball_mass: f32,
    pub ball_max_speed: f32,
    pub ball_drag: f32,
    pub jump_accel: f32,
    pub jump_immediate_force: f32,
    pub boost_accel_ground: f32,
    pub boost_accel_air: f32,
    pub boost_used_per_second: f32,
    pub respawn_delay: f32,
    pub bump_cooldown_time: f32,
    pub car_max_boost_amount: f32,
    pub car_spawn_boost_amount: f32,
    pub boost_pad_amount_small: f32,
    pub boost_pad_amount_big: f32,
    pub boost_pad_cooldown_big: f32,
    pub boost_pad_cooldown_small: f32,
    pub ball_hit_extra_force_scale: f32,
    pub bump_force_scale: f32,
    pub ball_radius: f32,
    pub unlimited_flips: bool,
    pub unlimited_double_jumps: bool,
    pub recharge_boost_enabled: bool,
    pub recharge_boost_per_second: f32,
    pub recharge_boost_delay: f32,
    pub demo_mode: u8,
    pub enable_team_demos: bool,
    pub goal_base_threshold_y: f32,
    pub boost_pads_enabled: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ArenaEvent {
    pub kind: u8,
    pub car_idx: u32,
    pub victim_car_idx: u32,
    pub boost_pad_idx: u32,
    pub is_demo: bool,
    pub contact_point: Vec3,
    pub contact_normal: Vec3,
    pub extra_hit_vel: Vec3,
    pub has_wheel_contact: [bool; 4],
    pub wheels: [RaycastHitInfo; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CarEventCounts {
    pub hit_ball: u32,
    pub hit_world: u32,
    pub bumps_dealt: u32,
    pub demos_dealt: u32,
    pub bumps_received: u32,
    pub demos_received: u32,
    pub boost_pickups: u32,
    pub landed: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct EventCounts {
    pub ball_hit_world: u32,
    pub goal_scored: bool,
    pub goal_tick: u64,
    pub cars: [CarEventCounts; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CarSpec {
    pub team: u8,
    pub config: CarBodyConfig,
}

#[repr(C)]
pub struct RscBatchDesc {
    pub backend: u8,
    pub game_mode: u8,
    pub mem_weight_mode: u8,
    pub no_ball_rot: bool,
    pub min_pos: Vec3,
    pub max_pos: Vec3,
    pub max_aabb_len: f32,
    pub mutators: MutatorConfig,
    pub cars: *const CarSpec,
    pub num_cars: u32,
    pub rng_seeds: *const u64,
    pub num_arenas: u32,
    pub custom_pads: *const BoostPadConfig,
    pub num_custom_pads: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct RscArenaScalars {
    pub tick_count: u64,
    pub rng_state: u64,
    pub error_flags: u32,
    pub ball_scored: bool,
    pub has_tiles: bool,
    pub num_cars: u32,
    pub num_pads: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct RscPadState {
    pub config: BoostPadConfig,
    pub cooldown: f32,
    pub num_touching: u32,
    pub touching: [u32; 8],
}

#[repr(C)]
pub struct RscMeshSet {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RscBatch {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn rsc_last_error() -> *const c_char;
    fn rsc_abi_sizes(out: *mut u32, count: u32) -> i32;
    fn rsc_meshes_load(path: *const c_char) -> *mut RscMeshSet;
    fn rsc_meshes_free(meshes: *mut RscMeshSet);
    fn rsc_car_body_config(body: u8, out: *mut CarBodyConfig) -> i32;
    fn rsc_batch_create(meshes: *const RscMeshSet, desc: *const RscBatchDesc) -> *mut RscBatch;
    fn rsc_batch_free(batch: *mut RscBatch);
    fn rsc_batch_step(batch: *mut RscBatch, ticks: u32) -> i32;
    fn rsc_batch_step_async(batch: *mut RscBatch, ticks: u32) -> i32;
    fn rsc_batch_sync(batch: *mut RscBatch) -> i32;
    fn rsc_batch_get_car_state(
        batch: *mut RscBatch,
        arena: u32,
        car: u32,
        out: *mut CarState,
    ) -> i32;
    fn rsc_batch_get_ball_state(batch: *mut RscBatch, arena: u32, out: *mut BallState) -> i32;
    fn rsc_batch_get_pad_state(
        batch: *mut RscBatch,
        arena: u32,
        pad: u32,
        cooldown: *mut f32,
        touching: *mut u32,
        num_touching: *mut u32,
    ) -> i32;
    fn rsc_batch_get_tile_states(batch: *mut RscBatch, arena: u32, out: *mut TileStates) -> i32;
    fn rsc_batch_save(
        batch: *mut RscBatch,
        arena: u32,
        out: *mut u8,
        capacity: usize,
        size: *mut usize,
    ) -> i32;
    fn rsc_batch_load(batch: *mut RscBatch, arena: u32, data: *const u8, size: usize) -> i32;
    fn rsc_batch_set_controls(
        batch: *mut RscBatch,
        arena: u32,
        car: u32,
        controls: *const CarControls,
    ) -> i32;
    fn rsc_batch_set_all_controls(
        batch: *mut RscBatch,
        controls: *const CarControls,
        count: u32,
    ) -> i32;
    fn rsc_batch_get_state(
        batch: *mut RscBatch,
        arena: u32,
        scalars: *mut RscArenaScalars,
        ball: *mut BallState,
        tiles: *mut TileStates,
        car_infos: *mut CarInfo,
        cars: *mut CarState,
        car_capacity: u32,
        pads: *mut RscPadState,
        pad_capacity: u32,
    ) -> i32;
    fn rsc_batch_get_events(
        batch: *mut RscBatch,
        arena: u32,
        out: *mut ArenaEvent,
        capacity: u32,
        count: *mut u32,
    ) -> i32;
    fn rsc_batch_get_event_counts(batch: *mut RscBatch, arena: u32, out: *mut EventCounts) -> i32;
    fn rsc_batch_get_mutators(batch: *mut RscBatch, arena: u32, out: *mut MutatorConfig) -> i32;
    fn rsc_batch_set_car_state(
        batch: *mut RscBatch,
        arena: u32,
        car: u32,
        core: *const CarCore,
    ) -> i32;
    fn rsc_batch_set_car_state_full(
        batch: *mut RscBatch,
        arena: u32,
        car: u32,
        state: *const CarState,
    ) -> i32;
    fn rsc_batch_set_ball_state(batch: *mut RscBatch, arena: u32, state: *const BallState) -> i32;
    fn rsc_batch_set_pad_state(
        batch: *mut RscBatch,
        arena: u32,
        pad: u32,
        cooldown: f32,
        touching: *const u32,
        num_touching: u32,
    ) -> i32;
    fn rsc_batch_set_tiles(batch: *mut RscBatch, arena: u32, states: *const TileStates) -> i32;
    fn rsc_batch_set_mutators(
        batch: *mut RscBatch,
        arena: u32,
        mutators: *const MutatorConfig,
    ) -> i32;
    fn rsc_batch_reset_kickoff(batch: *mut RscBatch, arena: u32, has_seed: bool, seed: u64) -> i32;
    fn rsc_batch_copy(
        dst: *mut RscBatch,
        src: *const RscBatch,
        src_arenas: *const u32,
        dst_arenas: *const u32,
        count: u32,
    ) -> i32;
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(rsc_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn check(code: i32, what: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{what}: {}", last_error()))
    }
}

pub fn check_abi() -> Result<(), String> {
    use std::mem::size_of;
    let expected = [
        size_of::<Vec3>(),
        size_of::<PhysState>(),
        size_of::<CarControls>(),
        size_of::<RaycastHitInfo>(),
        size_of::<CarCache>(),
        size_of::<CarState>(),
        size_of::<CarCore>(),
        size_of::<CarBodyConfig>(),
        size_of::<CarInfo>(),
        size_of::<BallState>(),
        size_of::<BoostPadConfig>(),
        size_of::<TileStates>(),
        size_of::<MutatorConfig>(),
        size_of::<ArenaEvent>(),
        size_of::<EventCounts>(),
        size_of::<CarSpec>(),
        size_of::<RscBatchDesc>(),
        size_of::<RscPadState>(),
        size_of::<RscArenaScalars>(),
    ];
    let mut actual = [0u32; 19];
    check(
        unsafe { rsc_abi_sizes(actual.as_mut_ptr(), actual.len() as u32) },
        "rsc_abi_sizes",
    )?;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a as usize != e {
            return Err(format!("ABI size mismatch at entry {i}: C++ {a}, Rust {e}"));
        }
    }
    Ok(())
}

pub struct MeshSet(*mut RscMeshSet);

impl MeshSet {
    pub fn load(path: &str) -> Result<Self, String> {
        let c = CString::new(path).map_err(|e| e.to_string())?;
        let ptr = unsafe { rsc_meshes_load(c.as_ptr()) };
        if ptr.is_null() {
            Err(format!("rsc_meshes_load: {}", last_error()))
        } else {
            Ok(Self(ptr))
        }
    }
}

impl Drop for MeshSet {
    fn drop(&mut self) {
        unsafe { rsc_meshes_free(self.0) };
    }
}

pub fn car_body_config(body: u8) -> Result<CarBodyConfig, String> {
    let mut out = CarBodyConfig::default();
    check(
        unsafe { rsc_car_body_config(body, &mut out) },
        "rsc_car_body_config",
    )?;
    Ok(out)
}

pub struct BatchDesc {
    pub backend: u8,
    pub game_mode: u8,
    pub mutators: MutatorConfig,
    pub cars: Vec<CarSpec>,
    pub rng_seeds: Vec<u64>,
}

pub struct ArenaSnapshot {
    pub scalars: RscArenaScalars,
    pub ball: BallState,
    pub tiles: TileStates,
    pub car_infos: Vec<CarInfo>,
    pub cars: Vec<CarState>,
    pub pads: Vec<RscPadState>,
    pub events: Vec<ArenaEvent>,
    pub mutators: MutatorConfig,
}

pub struct Batch(*mut RscBatch);

impl Batch {
    pub fn new(meshes: &MeshSet, desc: &BatchDesc) -> Result<Self, String> {
        let raw = RscBatchDesc {
            backend: desc.backend,
            game_mode: desc.game_mode,
            mem_weight_mode: 1,
            no_ball_rot: false,
            min_pos: Vec3 {
                x: -5600.0,
                y: -6000.0,
                z: 0.0,
            },
            max_pos: Vec3 {
                x: 5600.0,
                y: 6000.0,
                z: 2200.0,
            },
            max_aabb_len: 370.0,
            mutators: desc.mutators,
            cars: desc.cars.as_ptr(),
            num_cars: desc.cars.len() as u32,
            rng_seeds: desc.rng_seeds.as_ptr(),
            num_arenas: desc.rng_seeds.len() as u32,
            custom_pads: std::ptr::null(),
            num_custom_pads: -1,
        };
        let ptr = unsafe { rsc_batch_create(meshes.0, &raw) };
        if ptr.is_null() {
            Err(format!("rsc_batch_create: {}", last_error()))
        } else {
            Ok(Self(ptr))
        }
    }

    pub fn step(&mut self, ticks: u32) -> Result<(), String> {
        check(unsafe { rsc_batch_step(self.0, ticks) }, "rsc_batch_step")
    }

    pub fn step_async(&mut self, ticks: u32) -> Result<(), String> {
        check(
            unsafe { rsc_batch_step_async(self.0, ticks) },
            "rsc_batch_step_async",
        )
    }

    pub fn sync(&mut self) -> Result<(), String> {
        check(unsafe { rsc_batch_sync(self.0) }, "rsc_batch_sync")
    }

    pub fn car_state(&mut self, arena: u32, car: u32) -> Result<CarState, String> {
        let mut out = CarState::default();
        check(
            unsafe { rsc_batch_get_car_state(self.0, arena, car, &mut out) },
            "rsc_batch_get_car_state",
        )?;
        Ok(out)
    }

    pub fn ball_state(&mut self, arena: u32) -> Result<BallState, String> {
        let mut out = BallState::default();
        check(
            unsafe { rsc_batch_get_ball_state(self.0, arena, &mut out) },
            "rsc_batch_get_ball_state",
        )?;
        Ok(out)
    }

    pub fn pad_state(&mut self, arena: u32, pad: u32) -> Result<(f32, Vec<u32>), String> {
        let mut cooldown = 0.0f32;
        let mut touching = [0u32; 8];
        let mut count = 0u32;
        check(
            unsafe {
                rsc_batch_get_pad_state(
                    self.0,
                    arena,
                    pad,
                    &mut cooldown,
                    touching.as_mut_ptr(),
                    &mut count,
                )
            },
            "rsc_batch_get_pad_state",
        )?;
        Ok((cooldown, touching[..count as usize].to_vec()))
    }

    pub fn tile_states(&mut self, arena: u32) -> Result<TileStates, String> {
        let mut out = TileStates::default();
        check(
            unsafe { rsc_batch_get_tile_states(self.0, arena, &mut out) },
            "rsc_batch_get_tile_states",
        )?;
        Ok(out)
    }

    pub fn save(&mut self, arena: u32) -> Result<Vec<u8>, String> {
        let mut size = 0usize;
        let mut buf = vec![0u8; 1 << 20];
        check(
            unsafe { rsc_batch_save(self.0, arena, buf.as_mut_ptr(), buf.len(), &mut size) },
            "rsc_batch_save",
        )?;
        buf.truncate(size);
        Ok(buf)
    }

    pub fn load(&mut self, arena: u32, data: &[u8]) -> Result<(), String> {
        check(
            unsafe { rsc_batch_load(self.0, arena, data.as_ptr(), data.len()) },
            "rsc_batch_load",
        )
    }

    pub fn set_controls(&mut self, arena: u32, car: u32, c: &CarControls) -> Result<(), String> {
        check(
            unsafe { rsc_batch_set_controls(self.0, arena, car, c) },
            "rsc_batch_set_controls",
        )
    }

    pub fn set_all_controls(&mut self, controls: &[CarControls]) -> Result<(), String> {
        check(
            unsafe { rsc_batch_set_all_controls(self.0, controls.as_ptr(), controls.len() as u32) },
            "rsc_batch_set_all_controls",
        )
    }

    pub fn snapshot(&mut self, arena: u32) -> Result<ArenaSnapshot, String> {
        let mut scalars = RscArenaScalars::default();
        let mut ball = BallState::default();
        let mut tiles = TileStates::default();
        let mut car_infos = vec![CarInfo::default(); 8];
        let mut cars = vec![CarState::default(); 8];
        let mut pads = vec![RscPadState::default(); 48];
        check(
            unsafe {
                rsc_batch_get_state(
                    self.0,
                    arena,
                    &mut scalars,
                    &mut ball,
                    &mut tiles,
                    car_infos.as_mut_ptr(),
                    cars.as_mut_ptr(),
                    8,
                    pads.as_mut_ptr(),
                    48,
                )
            },
            "rsc_batch_get_state",
        )?;
        car_infos.truncate(scalars.num_cars as usize);
        cars.truncate(scalars.num_cars as usize);
        pads.truncate(scalars.num_pads as usize);
        let mut events = vec![ArenaEvent::default(); 256];
        let mut count = 0u32;
        check(
            unsafe {
                rsc_batch_get_events(
                    self.0,
                    arena,
                    events.as_mut_ptr(),
                    events.len() as u32,
                    &mut count,
                )
            },
            "rsc_batch_get_events",
        )?;
        events.truncate(count as usize);
        let mut mutators = MutatorConfig::default();
        check(
            unsafe { rsc_batch_get_mutators(self.0, arena, &mut mutators) },
            "rsc_batch_get_mutators",
        )?;
        Ok(ArenaSnapshot {
            scalars,
            ball,
            tiles,
            car_infos,
            cars,
            pads,
            events,
            mutators,
        })
    }

    pub fn event_counts(&mut self, arena: u32) -> Result<EventCounts, String> {
        let mut out = EventCounts::default();
        check(
            unsafe { rsc_batch_get_event_counts(self.0, arena, &mut out) },
            "rsc_batch_get_event_counts",
        )?;
        Ok(out)
    }

    pub fn set_car_state(&mut self, arena: u32, car: u32, core: &CarCore) -> Result<(), String> {
        check(
            unsafe { rsc_batch_set_car_state(self.0, arena, car, core) },
            "rsc_batch_set_car_state",
        )
    }

    pub fn set_car_state_full(
        &mut self,
        arena: u32,
        car: u32,
        state: &CarState,
    ) -> Result<(), String> {
        check(
            unsafe { rsc_batch_set_car_state_full(self.0, arena, car, state) },
            "rsc_batch_set_car_state_full",
        )
    }

    pub fn set_ball_state(&mut self, arena: u32, state: &BallState) -> Result<(), String> {
        check(
            unsafe { rsc_batch_set_ball_state(self.0, arena, state) },
            "rsc_batch_set_ball_state",
        )
    }

    pub fn set_pad_state(
        &mut self,
        arena: u32,
        pad: u32,
        cooldown: f32,
        touching: &[u32],
    ) -> Result<(), String> {
        check(
            unsafe {
                rsc_batch_set_pad_state(
                    self.0,
                    arena,
                    pad,
                    cooldown,
                    touching.as_ptr(),
                    touching.len() as u32,
                )
            },
            "rsc_batch_set_pad_state",
        )
    }

    pub fn set_tiles(&mut self, arena: u32, states: &TileStates) -> Result<(), String> {
        check(
            unsafe { rsc_batch_set_tiles(self.0, arena, states) },
            "rsc_batch_set_tiles",
        )
    }

    pub fn set_mutators(&mut self, arena: u32, mutators: &MutatorConfig) -> Result<(), String> {
        check(
            unsafe { rsc_batch_set_mutators(self.0, arena, mutators) },
            "rsc_batch_set_mutators",
        )
    }

    pub fn reset_kickoff(&mut self, arena: u32, seed: Option<u64>) -> Result<(), String> {
        check(
            unsafe { rsc_batch_reset_kickoff(self.0, arena, seed.is_some(), seed.unwrap_or(0)) },
            "rsc_batch_reset_kickoff",
        )
    }

    pub fn copy_from(
        &mut self,
        src: Option<&Batch>,
        src_arenas: &[u32],
        dst_arenas: &[u32],
    ) -> Result<(), String> {
        let src_ptr = src.map_or(self.0 as *const RscBatch, |b| b.0 as *const RscBatch);
        check(
            unsafe {
                rsc_batch_copy(
                    self.0,
                    src_ptr,
                    src_arenas.as_ptr(),
                    dst_arenas.as_ptr(),
                    src_arenas.len() as u32,
                )
            },
            "rsc_batch_copy",
        )
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { rsc_batch_free(self.0) };
    }
}
