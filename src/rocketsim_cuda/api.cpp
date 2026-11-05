#include "rocketsim_cuda/rocketsim_cuda.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "rocketsim_cuda/cuda_batch.hpp"
#include "rsc/host/builder.hpp"

namespace rocketsim_cuda {

namespace {

constexpr uint64_t k_snapshot_magic = 0x5253435633415245ULL;

rsc::Vec3A to_internal(Vec3 v) { return {v.x, v.y, v.z}; }

Vec3 to_public(rsc::Vec3A v) { return {v.x, v.y, v.z}; }

rsc::Mat3A to_internal(const RotMat &m) {
  return rsc::Mat3A::from_cols(to_internal(m.forward), to_internal(m.right),
                               to_internal(m.up));
}

RotMat to_public(const rsc::Mat3A &m) {
  return {to_public(m.x_axis), to_public(m.y_axis), to_public(m.z_axis)};
}

rsc::GameMode to_internal(GameMode m) {
  return static_cast<rsc::GameMode>(static_cast<uint8_t>(m));
}

rsc::Team to_internal(Team t) {
  return t == Team::Blue ? rsc::Team::Blue : rsc::Team::Orange;
}

Team to_public(rsc::Team t) {
  return t == rsc::Team::Blue ? Team::Blue : Team::Orange;
}

rsc::PhysState to_internal(const PhysState &p) {
  return {to_internal(p.pos), to_internal(p.rot_mat), to_internal(p.vel),
          to_internal(p.ang_vel)};
}

PhysState to_public(const rsc::PhysState &p) {
  return {to_public(p.pos), to_public(p.rot_mat), to_public(p.vel),
          to_public(p.ang_vel)};
}

rsc::CarControls to_internal(const CarControls &c) {
  return {c.throttle, c.steer, c.pitch, c.yaw,
          c.roll,     c.jump,  c.boost, c.handbrake};
}

CarControls to_public(const rsc::CarControls &c) {
  return {c.throttle, c.steer, c.pitch, c.yaw,
          c.roll,     c.jump,  c.boost, c.handbrake};
}

rsc::RaycastHitInfo to_internal(const RaycastHitInfo &h) {
  return {to_internal(h.hit_point), to_internal(h.hit_normal), h.hit_fraction,
          static_cast<rsc::UserInfoType>(static_cast<uint8_t>(h.user_info))};
}

RaycastHitInfo to_public(const rsc::RaycastHitInfo &h) {
  return {to_public(h.hit_point), to_public(h.hit_normal), h.hit_fraction,
          static_cast<UserInfoType>(static_cast<uint8_t>(h.user_info))};
}

int32_t optional_index(bool has, uint32_t idx) {
  return has ? static_cast<int32_t>(idx) : -1;
}

rsc::CarState to_internal(const CarState &s) {
  rsc::CarState r;
  r.phys = to_internal(s.phys);
  r.controls = to_internal(s.controls);
  r.has_jumped = s.has_jumped;
  r.has_double_jumped = s.has_double_jumped;
  r.has_flipped = s.has_flipped;
  r.flip_rel_torque = to_internal(s.flip_rel_torque);
  r.jump_ticks = s.jump_ticks;
  r.flip_time = s.flip_time;
  r.is_flipping = s.is_flipping;
  r.is_jumping = s.is_jumping;
  r.jump_hold_broken = s.jump_hold_broken;
  r.air_time = s.air_time;
  r.air_time_since_jump = s.air_time_since_jump;
  r.boost = s.boost;
  r.handbrake_val = s.handbrake_val;
  r.is_supersonic = s.is_supersonic;
  r.supersonic_grace_timer = s.supersonic_grace_timer;
  r.time_since_boosted = s.time_since_boosted;
  r.boosting_time = s.boosting_time;
  r.is_boosting = s.is_boosting;
  r.is_auto_flipping = s.is_auto_flipping;
  r.auto_flip_timer = s.auto_flip_timer;
  r.auto_flip_torque_scale = s.auto_flip_torque_scale;
  r.bump_cooldown_timer = s.bump_cooldown_timer;
  r.bump_cooldown_victim =
      optional_index(s.has_bump_cooldown_victim, s.bump_cooldown_victim);
  r.has_last_extra_hit_tick = s.has_last_extra_hit_tick;
  r.last_extra_hit_tick = s.last_extra_hit_tick;
  r.is_demoed = s.is_demoed;
  r.demo_respawn_timer = s.demo_respawn_timer;
  r.cache.is_on_ground = s.cache.is_on_ground;
  for (int i = 0; i < 4; ++i) {
    r.cache.wheel_has_contact[i] = s.cache.has_wheel_contact[i];
    r.cache.wheels_with_contact[i] =
        to_internal(s.cache.wheels_with_contact[i]);
  }
  r.cache.has_world_contact_normal = s.cache.has_world_contact_normal;
  r.cache.world_contact_normal = to_internal(s.cache.world_contact_normal);
  r.cache.prev_controls = to_internal(s.cache.prev_controls);
  r.cache.world_wheels_prev = s.cache.world_wheels_prev;
  r.cache.demo_tick = s.cache.demo_tick;
  r.cache.demoer_idx = optional_index(s.cache.has_demoer, s.cache.demoer_idx);
  return r;
}

CarState to_public(const rsc::CarState &r) {
  CarState s{};
  s.phys = to_public(r.phys);
  s.controls = to_public(r.controls);
  s.has_jumped = r.has_jumped;
  s.has_double_jumped = r.has_double_jumped;
  s.has_flipped = r.has_flipped;
  s.flip_rel_torque = to_public(r.flip_rel_torque);
  s.jump_ticks = r.jump_ticks;
  s.flip_time = r.flip_time;
  s.is_flipping = r.is_flipping;
  s.is_jumping = r.is_jumping;
  s.jump_hold_broken = r.jump_hold_broken;
  s.air_time = r.air_time;
  s.air_time_since_jump = r.air_time_since_jump;
  s.boost = r.boost;
  s.handbrake_val = r.handbrake_val;
  s.is_supersonic = r.is_supersonic;
  s.supersonic_grace_timer = r.supersonic_grace_timer;
  s.time_since_boosted = r.time_since_boosted;
  s.boosting_time = r.boosting_time;
  s.is_boosting = r.is_boosting;
  s.is_auto_flipping = r.is_auto_flipping;
  s.auto_flip_timer = r.auto_flip_timer;
  s.auto_flip_torque_scale = r.auto_flip_torque_scale;
  s.bump_cooldown_timer = r.bump_cooldown_timer;
  s.has_bump_cooldown_victim = r.bump_cooldown_victim >= 0;
  s.bump_cooldown_victim = r.bump_cooldown_victim >= 0
                               ? static_cast<uint32_t>(r.bump_cooldown_victim)
                               : 0;
  s.has_last_extra_hit_tick = r.has_last_extra_hit_tick;
  s.last_extra_hit_tick = r.last_extra_hit_tick;
  s.is_demoed = r.is_demoed;
  s.demo_respawn_timer = r.demo_respawn_timer;
  s.cache.is_on_ground = r.cache.is_on_ground;
  for (int i = 0; i < 4; ++i) {
    s.cache.has_wheel_contact[i] = r.cache.wheel_has_contact[i];
    s.cache.wheels_with_contact[i] = to_public(r.cache.wheels_with_contact[i]);
  }
  s.cache.has_world_contact_normal = r.cache.has_world_contact_normal;
  s.cache.world_contact_normal = to_public(r.cache.world_contact_normal);
  s.cache.prev_controls = to_public(r.cache.prev_controls);
  s.cache.world_wheels_prev = r.cache.world_wheels_prev;
  s.cache.demo_tick = r.cache.demo_tick;
  s.cache.has_demoer = r.cache.demoer_idx >= 0;
  s.cache.demoer_idx =
      r.cache.demoer_idx >= 0 ? static_cast<uint32_t>(r.cache.demoer_idx) : 0;
  return s;
}

rsc::CarState core_to_internal(const CarCore &c) {
  rsc::CarState r = rsc::car_state_default();
  r.phys = to_internal(c.phys);
  r.controls = to_internal(c.controls);
  r.has_jumped = c.has_jumped;
  r.has_double_jumped = c.has_double_jumped;
  r.has_flipped = c.has_flipped;
  r.flip_rel_torque = to_internal(c.flip_rel_torque);
  r.jump_ticks = c.jump_ticks;
  r.flip_time = c.flip_time;
  r.is_flipping = c.is_flipping;
  r.is_jumping = c.is_jumping;
  r.jump_hold_broken = c.jump_hold_broken;
  r.air_time_since_jump = c.air_time_since_jump;
  r.boost = c.boost;
  r.is_auto_flipping = c.is_auto_flipping;
  r.auto_flip_timer = c.auto_flip_timer;
  r.auto_flip_torque_scale = c.auto_flip_torque_scale;
  r.bump_cooldown_timer = c.bump_cooldown_timer;
  r.bump_cooldown_victim =
      optional_index(c.has_bump_cooldown_victim, c.bump_cooldown_victim);
  r.has_last_extra_hit_tick = c.has_last_extra_hit_tick;
  r.last_extra_hit_tick = c.last_extra_hit_tick;
  r.is_demoed = c.is_demoed;
  r.demo_respawn_timer = c.demo_respawn_timer;
  return r;
}

rsc::BallState to_internal(const BallState &b) {
  rsc::BallState r;
  r.phys = to_internal(b.phys);
  r.hs_info = {b.hs_info.y_target_dir, b.hs_info.cur_target_speed,
               b.hs_info.time_since_hit};
  r.ds_info = {b.ds_info.charge_level, b.ds_info.accumulated_hit_force,
               b.ds_info.y_target_dir, b.ds_info.has_last_damage_tick,
               b.ds_info.last_damage_tick};
  r.tick_count_since_kickoff = b.tick_count_since_kickoff;
  return r;
}

BallState to_public(const rsc::BallState &r) {
  BallState b{};
  b.phys = to_public(r.phys);
  b.hs_info = {r.hs_info.y_target_dir, r.hs_info.cur_target_speed,
               r.hs_info.time_since_hit};
  b.ds_info = {r.ds_info.charge_level, r.ds_info.accumulated_hit_force,
               r.ds_info.y_target_dir, r.ds_info.has_last_damage_tick,
               r.ds_info.last_damage_tick};
  b.tick_count_since_kickoff = r.tick_count_since_kickoff;
  return b;
}

rsc::MutatorConfig to_internal(const MutatorConfig &m) {
  rsc::MutatorConfig r;
  r.gravity = to_internal(m.gravity);
  r.car_mass = m.car_mass;
  r.ball_mass = m.ball_mass;
  r.ball_max_speed = m.ball_max_speed;
  r.ball_drag = m.ball_drag;
  r.jump_accel = m.jump_accel;
  r.jump_immediate_force = m.jump_immediate_force;
  r.boost_accel_ground = m.boost_accel_ground;
  r.boost_accel_air = m.boost_accel_air;
  r.boost_used_per_second = m.boost_used_per_second;
  r.respawn_delay = m.respawn_delay;
  r.bump_cooldown_time = m.bump_cooldown_time;
  r.car_max_boost_amount = m.car_max_boost_amount;
  r.car_spawn_boost_amount = m.car_spawn_boost_amount;
  r.boost_pad_amount_small = m.boost_pad_amount_small;
  r.boost_pad_amount_big = m.boost_pad_amount_big;
  r.boost_pad_cooldown_big = m.boost_pad_cooldown_big;
  r.boost_pad_cooldown_small = m.boost_pad_cooldown_small;
  r.ball_hit_extra_force_scale = m.ball_hit_extra_force_scale;
  r.bump_force_scale = m.bump_force_scale;
  r.ball_radius = m.ball_radius;
  r.unlimited_flips = m.unlimited_flips;
  r.unlimited_double_jumps = m.unlimited_double_jumps;
  r.recharge_boost_enabled = m.recharge_boost_enabled;
  r.recharge_boost_per_second = m.recharge_boost_per_second;
  r.recharge_boost_delay = m.recharge_boost_delay;
  r.demo_mode = static_cast<rsc::DemoMode>(static_cast<uint8_t>(m.demo_mode));
  r.enable_team_demos = m.enable_team_demos;
  r.goal_base_threshold_y = m.goal_base_threshold_y;
  r.boost_pads_enabled = m.boost_pads_enabled;
  return r;
}

MutatorConfig to_public(const rsc::MutatorConfig &r) {
  MutatorConfig m{};
  m.gravity = to_public(r.gravity);
  m.car_mass = r.car_mass;
  m.ball_mass = r.ball_mass;
  m.ball_max_speed = r.ball_max_speed;
  m.ball_drag = r.ball_drag;
  m.jump_accel = r.jump_accel;
  m.jump_immediate_force = r.jump_immediate_force;
  m.boost_accel_ground = r.boost_accel_ground;
  m.boost_accel_air = r.boost_accel_air;
  m.boost_used_per_second = r.boost_used_per_second;
  m.respawn_delay = r.respawn_delay;
  m.bump_cooldown_time = r.bump_cooldown_time;
  m.car_max_boost_amount = r.car_max_boost_amount;
  m.car_spawn_boost_amount = r.car_spawn_boost_amount;
  m.boost_pad_amount_small = r.boost_pad_amount_small;
  m.boost_pad_amount_big = r.boost_pad_amount_big;
  m.boost_pad_cooldown_big = r.boost_pad_cooldown_big;
  m.boost_pad_cooldown_small = r.boost_pad_cooldown_small;
  m.ball_hit_extra_force_scale = r.ball_hit_extra_force_scale;
  m.bump_force_scale = r.bump_force_scale;
  m.ball_radius = r.ball_radius;
  m.unlimited_flips = r.unlimited_flips;
  m.unlimited_double_jumps = r.unlimited_double_jumps;
  m.recharge_boost_enabled = r.recharge_boost_enabled;
  m.recharge_boost_per_second = r.recharge_boost_per_second;
  m.recharge_boost_delay = r.recharge_boost_delay;
  m.demo_mode = static_cast<DemoMode>(static_cast<uint8_t>(r.demo_mode));
  m.enable_team_demos = r.enable_team_demos;
  m.goal_base_threshold_y = r.goal_base_threshold_y;
  m.boost_pads_enabled = r.boost_pads_enabled;
  return m;
}

template <int N>
rsc::LinearPieceCurve<N> curve_to_internal(const CurvePoint (&points)[N]) {
  float raw[N][2];
  for (int i = 0; i < N; ++i) {
    raw[i][0] = points[i].x;
    raw[i][1] = points[i].y;
  }
  return rsc::LinearPieceCurve<N>::make(raw);
}

template <int N>
void curve_to_public(const rsc::LinearPieceCurve<N> &c, CurvePoint (&out)[N]) {
  for (int i = 0; i < N; ++i) {
    out[i] = {c.curve[i].max_x, c.curve[i].max_y};
  }
}

rsc::WheelPairConfig to_internal(const WheelPairConfig &w) {
  return {w.wheel_radius, w.suspension_rest_length,
          to_internal(w.connection_point_offset)};
}

WheelPairConfig to_public(const rsc::WheelPairConfig &w) {
  return {w.wheel_radius, w.suspension_rest_length,
          to_public(w.connection_point_offset)};
}

rsc::CarBodyConfig to_internal(const CarBodyConfig &c) {
  rsc::CarBodyConfig r;
  r.hitbox_size = to_internal(c.hitbox_size);
  r.hitbox_pos_offset = to_internal(c.hitbox_pos_offset);
  r.front_wheels = to_internal(c.front_wheels);
  r.back_wheels = to_internal(c.back_wheels);
  r.three_wheels = c.three_wheels;
  r.dodge_deadzone = c.dodge_deadzone;
  r.steer_angle_curve = curve_to_internal(c.steer_angle_curve);
  r.powerslide_steer_angle_curve =
      curve_to_internal(c.powerslide_steer_angle_curve);
  return r;
}

CarBodyConfig to_public(const rsc::CarBodyConfig &r) {
  CarBodyConfig c{};
  c.hitbox_size = to_public(r.hitbox_size);
  c.hitbox_pos_offset = to_public(r.hitbox_pos_offset);
  c.front_wheels = to_public(r.front_wheels);
  c.back_wheels = to_public(r.back_wheels);
  c.three_wheels = r.three_wheels;
  c.dodge_deadzone = r.dodge_deadzone;
  curve_to_public(r.steer_angle_curve, c.steer_angle_curve);
  curve_to_public(r.powerslide_steer_angle_curve,
                  c.powerslide_steer_angle_curve);
  return c;
}

TileStates to_public(const rsc::TileStates &t) {
  TileStates r{};
  for (int team = 0; team < 2; ++team) {
    for (uint32_t i = 0; i < k_tiles_per_team; ++i) {
      r.states[team][i] = static_cast<TileDamageState>(t.states[team][i]);
    }
  }
  return r;
}

rsc::TileStates to_internal(const TileStates &t) {
  rsc::TileStates r{};
  for (int team = 0; team < 2; ++team) {
    for (uint32_t i = 0; i < k_tiles_per_team; ++i) {
      r.states[team][i] = static_cast<uint8_t>(t.states[team][i]);
    }
  }
  return r;
}

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

}

MutatorConfig default_mutators(GameMode mode) {
  return to_public(rsc::default_mutators(to_internal(mode)));
}

CarBodyConfig car_body_config(CarBody body) {
  return to_public(rsc::car_body_preset(static_cast<int>(body)));
}

CarCore car_core_default() {
  rsc::CarState d = rsc::car_state_default();
  CarState s = to_public(d);
  CarCore c{};
  c.phys = s.phys;
  c.controls = s.controls;
  c.flip_rel_torque = s.flip_rel_torque;
  c.boost = s.boost;
  return c;
}

struct MeshSet::Impl {
  rsc::MeshLibrary library;
};

MeshSet::MeshSet(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

MeshSet::~MeshSet() = default;

std::shared_ptr<const MeshSet>
MeshSet::load_directory(const std::string &collision_meshes_folder) {
  auto impl = std::make_unique<Impl>(
      Impl{rsc::MeshLibrary::load_directory(collision_meshes_folder)});
  return std::shared_ptr<const MeshSet>(new MeshSet(std::move(impl)));
}

bool MeshSet::has_game_mode(GameMode mode) const {
  if (mode == GameMode::TheVoid) {
    return true;
  }
  return impl_->library.has_mode(to_internal(mode));
}

struct ArenaBatch::Impl {
  Backend backend;
  GameMode mode;
  std::shared_ptr<const MeshSet> meshes;
  std::unique_ptr<rsc::HostStatic> st;
  uint32_t num_arenas;
  uint32_t num_cars;
  std::vector<rsc::Arena> host;
  std::vector<uint8_t> valid;
  std::vector<uint8_t> dirty;
  std::unique_ptr<rsc::StepScratch> scratch;
  std::vector<rsc::CarControls> staged_controls;
  std::vector<uint8_t> staged_mask;
  bool any_staged = false;
#ifdef RSC_HAS_CUDA
  std::unique_ptr<rsc::CudaBatch> cuda;
  bool in_flight = false;
#endif

  bool on_device() const {
#ifdef RSC_HAS_CUDA
    return cuda != nullptr;
#else
    return false;
#endif
  }

  void check_arena(uint32_t arena) const {
    if (arena >= num_arenas) {
      throw std::out_of_range("arena index out of range");
    }
  }

  void check_car(uint32_t car) const {
    if (car >= num_cars) {
      throw std::out_of_range("car index out of range");
    }
  }

  void wait_idle() {
#ifdef RSC_HAS_CUDA
    if (in_flight) {
      cuda->sync();
      in_flight = false;
    }
#endif
  }

  void flush_device() {
#ifdef RSC_HAS_CUDA
    wait_idle();
    if (!cuda) {
      return;
    }
    for (uint32_t i = 0; i < num_arenas; ++i) {
      if (dirty[i]) {
        cuda->upload(i, host[i]);
        dirty[i] = 0;
      }
    }
    if (any_staged) {
      cuda->apply_controls(staged_controls.data(), staged_mask.data(),
                           num_cars);
      std::fill(staged_mask.begin(), staged_mask.end(), 0);
      any_staged = false;
    }
#endif
  }

  rsc::Arena &read(uint32_t arena) {
    check_arena(arena);
    wait_idle();
#ifdef RSC_HAS_CUDA
    if (cuda && !valid[arena]) {
      flush_device();
      cuda->download(arena, host[arena]);
      valid[arena] = 1;
    }
#endif
    return host[arena];
  }

  rsc::Arena &write(uint32_t arena) {
    rsc::Arena &a = read(arena);
    if (on_device()) {
      dirty[arena] = 1;
    }
    return a;
  }

  void drop_pending_controls(uint32_t arena, uint32_t car) {
    if (any_staged) {
      staged_mask[static_cast<size_t>(arena) * num_cars + car] = 0;
    }
  }

  void set_controls(uint32_t arena, uint32_t car, const CarControls &c) {
    check_arena(arena);
    check_car(car);
    wait_idle();
    if (!on_device() || valid[arena]) {
      host[arena].cars[car].state.controls = to_internal(c);
    }
    if (on_device()) {
      size_t slot = static_cast<size_t>(arena) * num_cars + car;
      staged_controls[slot] = to_internal(c);
      staged_mask[slot] = 1;
      any_staged = true;
    }
  }

  void launch_step(uint32_t ticks) {
    wait_idle();
#ifdef RSC_HAS_CUDA
    if (cuda) {
      flush_device();
      cuda->step(ticks);
      std::fill(valid.begin(), valid.end(), 0);
      in_flight = true;
      return;
    }
#endif
    const rsc::ArenaStatic &view = st->view();
    for (uint32_t i = 0; i < num_arenas; ++i) {
      rsc::Arena &a = host[i];
      a.counters = rsc::ArenaEventCounters{};
      for (uint32_t t = 0; t < ticks; ++t) {
        rsc::arena_step_tick(view, a, *scratch);
      }
    }
  }

#ifdef RSC_HAS_CUDA
  void copy_on_device(Impl &src, std::span<const uint32_t> source_arenas,
                      std::span<const uint32_t> dest_arenas) {
    src.flush_device();
    flush_device();
    cuda->copy_from(*src.cuda, source_arenas.data(), dest_arenas.data(),
                    static_cast<uint32_t>(source_arenas.size()));
    for (uint32_t d : dest_arenas) {
      valid[d] = 0;
    }
  }
#endif
};

ArenaBatch::ArenaBatch(Backend backend, std::shared_ptr<const MeshSet> meshes,
                       const BatchConfig &config)
    : impl_(std::make_unique<Impl>()) {
  require(meshes != nullptr, "mesh set is null");
  require(config.cars.size() <= k_max_cars, "too many cars per arena");
  require(!config.rng_seeds.empty(), "rng_seeds must list one seed per arena");
  Impl &im = *impl_;
  im.backend = backend;
  im.mode = config.game_mode;
  im.meshes = std::move(meshes);
  im.num_arenas = static_cast<uint32_t>(config.rng_seeds.size());
  im.num_cars = static_cast<uint32_t>(config.cars.size());
  rsc::ArenaSetup setup;
  setup.mode = to_internal(config.game_mode);
  setup.mem_weight_mode = static_cast<rsc::MemWeightMode>(
      static_cast<uint8_t>(config.mem_weight_mode));
  setup.min_pos = to_internal(config.min_pos);
  setup.max_pos = to_internal(config.max_pos);
  setup.max_aabb_len = config.max_aabb_len;
  setup.no_ball_rot = config.no_ball_rot;
  setup.has_custom_pads = config.custom_boost_pads.has_value();
  if (config.custom_boost_pads) {
    for (const BoostPadConfig &p : *config.custom_boost_pads) {
      setup.custom_pads.push_back({to_internal(p.pos), p.is_big});
    }
  }
  im.st = std::make_unique<rsc::HostStatic>(im.meshes->impl().library, setup);
  std::vector<rsc::CarSpec> cars;
  for (const CarSpec &c : config.cars) {
    cars.push_back({to_internal(c.team), to_internal(c.config)});
  }
  rsc::MutatorConfig mutators = to_internal(config.mutators);
  im.host.resize(im.num_arenas);
  for (uint32_t i = 0; i < im.num_arenas; ++i) {
    rsc::create_arena(*im.st, mutators, config.rng_seeds[i], cars, im.host[i]);
  }
  im.valid.assign(im.num_arenas, 1);
  im.dirty.assign(im.num_arenas, 0);
  im.staged_controls.resize(static_cast<size_t>(im.num_arenas) * im.num_cars);
  im.staged_mask.assign(static_cast<size_t>(im.num_arenas) * im.num_cars, 0);
  if (backend == Backend::Cuda) {
#ifdef RSC_HAS_CUDA
    im.cuda = std::make_unique<rsc::CudaBatch>(*im.st, im.num_arenas);
    std::fill(im.dirty.begin(), im.dirty.end(), 1);
    im.flush_device();
#else
    throw std::runtime_error(
        "this build of rocketsim_cuda has no CUDA backend");
#endif
  } else {
    im.scratch = std::make_unique<rsc::StepScratch>();
  }
}

ArenaBatch::~ArenaBatch() = default;

Backend ArenaBatch::backend() const { return impl_->backend; }

GameMode ArenaBatch::game_mode() const { return impl_->mode; }

uint32_t ArenaBatch::num_arenas() const { return impl_->num_arenas; }

uint32_t ArenaBatch::num_cars() const { return impl_->num_cars; }

uint32_t ArenaBatch::num_boost_pads() const {
  return impl_->st->view().pads.num_pads;
}

void ArenaBatch::set_car_controls(uint32_t arena, uint32_t car,
                                  const CarControls &controls) {
  impl_->set_controls(arena, car, controls);
}

void ArenaBatch::set_all_controls(std::span<const CarControls> controls) {
  Impl &im = *impl_;
  require(controls.size() == static_cast<size_t>(im.num_arenas) * im.num_cars,
          "controls must hold num_arenas * num_cars entries");
  for (uint32_t a = 0; a < im.num_arenas; ++a) {
    for (uint32_t c = 0; c < im.num_cars; ++c) {
      im.set_controls(a, c, controls[static_cast<size_t>(a) * im.num_cars + c]);
    }
  }
}

void ArenaBatch::step(uint32_t ticks) {
  impl_->launch_step(ticks);
  impl_->wait_idle();
}

void ArenaBatch::step_async(uint32_t ticks) { impl_->launch_step(ticks); }

void ArenaBatch::sync() { impl_->wait_idle(); }

namespace {

BoostPadState pad_state_of(const rsc::ArenaStatic &view, const rsc::Arena &a,
                           uint32_t pad) {
  const rsc::PadState &p = a.pads[pad];
  BoostPadState s;
  s.cooldown = 0.0f;
  if (p.has_gave_tick) {
    float max_cooldown =
        rsc::pad_max_cooldown(view.pads.configs[pad].is_big, a.mutators);
    float since =
        static_cast<float>(static_cast<int64_t>(a.tick_count) - p.gave_tick) *
        rsc::consts::tick_time;
    s.cooldown = rsc::max_f32_c(max_cooldown - since, 0.0f);
  }
  for (uint32_t i = 0; i < p.num_touching; ++i) {
    s.touching_car_idxs.push_back(p.touching[i]);
  }
  return s;
}

}

ArenaState ArenaBatch::get_arena_state(uint32_t arena) {
  Impl &im = *impl_;
  const rsc::Arena &a = im.read(arena);
  const rsc::ArenaStatic &view = im.st->view();
  ArenaState s;
  s.game_mode = im.mode;
  s.tick_count = a.tick_count;
  for (uint32_t c = 0; c < a.num_cars; ++c) {
    s.car_infos.push_back({a.cars[c].info.idx, to_public(a.cars[c].info.team),
                           to_public(a.cars[c].info.config)});
    s.cars.push_back(to_public(a.cars[c].state));
  }
  s.ball = to_public(a.ball.state);
  if (view.has_pads) {
    for (uint32_t p = 0; p < view.pads.num_pads; ++p) {
      s.boost_pad_configs.push_back(
          {to_public(view.pads.configs[p].pos), view.pads.configs[p].is_big});
      s.boost_pads.push_back(pad_state_of(view, a, p));
    }
  }
  if (a.has_tiles) {
    s.tile_states = to_public(a.tiles);
  }
  return s;
}

CarState ArenaBatch::get_car_state(uint32_t arena, uint32_t car) {
  impl_->check_car(car);
  return to_public(impl_->read(arena).cars[car].state);
}

BallState ArenaBatch::get_ball_state(uint32_t arena) {
  return to_public(impl_->read(arena).ball.state);
}

BoostPadState ArenaBatch::get_boost_pad_state(uint32_t arena, uint32_t pad) {
  const rsc::ArenaStatic &view = impl_->st->view();
  if (!view.has_pads || pad >= view.pads.num_pads) {
    throw std::out_of_range("boost pad index out of range");
  }
  return pad_state_of(view, impl_->read(arena), pad);
}

TileStates ArenaBatch::get_tile_states(uint32_t arena) {
  const rsc::Arena &a = impl_->read(arena);
  if (!a.has_tiles) {
    throw std::logic_error("this arena has no dropshot tiles");
  }
  return to_public(a.tiles);
}

MutatorConfig ArenaBatch::get_mutator_config(uint32_t arena) {
  return to_public(impl_->read(arena).mutators);
}

uint64_t ArenaBatch::get_tick_count(uint32_t arena) {
  return impl_->read(arena).tick_count;
}

uint64_t ArenaBatch::get_rng_state(uint32_t arena) {
  return impl_->read(arena).rng.state;
}

bool ArenaBatch::is_ball_scored(uint32_t arena) {
  return rsc::arena_is_ball_scored(impl_->st->view(), impl_->read(arena));
}

std::vector<ArenaEvent> ArenaBatch::get_last_step_events(uint32_t arena) {
  const rsc::Arena &a = impl_->read(arena);
  std::vector<ArenaEvent> out;
  for (uint32_t i = 0; i < a.num_events; ++i) {
    const rsc::ArenaEvent &e = a.events[i];
    ArenaEvent r{};
    r.kind = static_cast<ArenaEventKind>(static_cast<uint8_t>(e.kind));
    switch (e.kind) {
    case rsc::ArenaEventKind::BallHitWorld:
      r.contact_point = to_public(e.point);
      r.contact_normal = to_public(e.vec);
      break;
    case rsc::ArenaEventKind::CarHitBall:
      r.car_idx = e.car_idx;
      r.contact_point = to_public(e.point);
      r.extra_hit_vel = to_public(e.vec);
      break;
    case rsc::ArenaEventKind::CarHitCar:
      r.car_idx = e.car_idx;
      r.victim_car_idx = e.other_idx;
      r.is_demo = e.is_demo;
      r.contact_point = to_public(e.point);
      break;
    case rsc::ArenaEventKind::CarHitWorld:
      r.car_idx = e.car_idx;
      r.contact_point = to_public(e.point);
      r.contact_normal = to_public(e.vec);
      break;
    case rsc::ArenaEventKind::CarPickupBoost:
      r.car_idx = e.car_idx;
      r.boost_pad_idx = e.other_idx;
      break;
    case rsc::ArenaEventKind::CarLanded:
      r.car_idx = e.car_idx;
      for (int w = 0; w < 4; ++w) {
        r.has_wheel_contact[w] = a.landed[e.car_idx].has_contact[w];
        r.wheels[w] = to_public(a.landed[e.car_idx].wheels[w]);
      }
      break;
    }
    out.push_back(r);
  }
  return out;
}

EventCounts ArenaBatch::get_event_counts(uint32_t arena) {
  const rsc::ArenaEventCounters &c = impl_->read(arena).counters;
  EventCounts r{};
  r.ball_hit_world = c.ball_hit_world;
  r.goal_scored = c.goal_scored;
  r.goal_tick = c.goal_tick;
  for (uint32_t i = 0; i < k_max_cars; ++i) {
    const rsc::CarEventCounters &s = c.cars[i];
    r.cars[i] = {s.hit_ball,      s.hit_world,      s.bumps_dealt,
                 s.demos_dealt,   s.bumps_received, s.demos_received,
                 s.boost_pickups, s.landed};
  }
  return r;
}

uint32_t ArenaBatch::get_error_flags(uint32_t arena) {
  return impl_->read(arena).phys.error_flags;
}

void ArenaBatch::set_car_state(uint32_t arena, uint32_t car,
                               const CarCore &core) {
  Impl &im = *impl_;
  im.check_car(car);
  rsc::Arena &a = im.write(arena);
  im.drop_pending_controls(arena, car);
  rsc::arena_set_car_state(im.st->view(), a, car, core_to_internal(core));
}

void ArenaBatch::set_car_state_full(uint32_t arena, uint32_t car,
                                    const CarState &state) {
  Impl &im = *impl_;
  im.check_car(car);
  rsc::Arena &a = im.write(arena);
  im.drop_pending_controls(arena, car);
  rsc::arena_set_car_state_full(im.st->view(), a, car, to_internal(state));
}

void ArenaBatch::set_ball_state(uint32_t arena, const BallState &state) {
  Impl &im = *impl_;
  rsc::Arena &a = im.write(arena);
  rsc::arena_set_ball_state(im.st->view(), a, to_internal(state));
}

void ArenaBatch::set_boost_pad_state(uint32_t arena, uint32_t pad,
                                     const BoostPadState &state) {
  Impl &im = *impl_;
  rsc::Arena &a = im.write(arena);
  rsc::arena_set_boost_pad_state(*im.st, a, pad, state.cooldown,
                                 state.touching_car_idxs);
}

void ArenaBatch::set_tile_states(uint32_t arena, const TileStates &states) {
  Impl &im = *impl_;
  rsc::Arena &a = im.write(arena);
  if (!a.has_tiles) {
    throw std::logic_error("this arena has no dropshot tiles");
  }
  rsc::arena_set_tile_states(a, to_internal(states));
}

void ArenaBatch::set_mutator_config(uint32_t arena,
                                    const MutatorConfig &mutators) {
  Impl &im = *impl_;
  rsc::Arena &a = im.write(arena);
  rsc::arena_set_mutator_config(*im.st, a, to_internal(mutators));
}

void ArenaBatch::reset_to_random_kickoff(uint32_t arena,
                                         std::optional<uint64_t> seed) {
  Impl &im = *impl_;
  rsc::Arena &a = im.write(arena);
  rsc::arena_reset_to_random_kickoff(im.st->view(), a, seed.has_value(),
                                     seed.value_or(0));
}

void ArenaBatch::copy_arenas(const ArenaBatch &source,
                             std::span<const uint32_t> source_arenas,
                             std::span<const uint32_t> dest_arenas) {
  Impl &dst = *impl_;
  Impl &src = *source.impl_;
  require(source_arenas.size() == dest_arenas.size(),
          "source and destination index lists differ in length");
  require(src.meshes == dst.meshes && src.mode == dst.mode &&
              src.num_cars == dst.num_cars &&
              src.st->view().world.num_static_bodies ==
                  dst.st->view().world.num_static_bodies &&
              src.st->view().pads.num_pads == dst.st->view().pads.num_pads,
          "arena batches have different static configurations");
  for (size_t k = 0; k < source_arenas.size(); ++k) {
    src.check_arena(source_arenas[k]);
    dst.check_arena(dest_arenas[k]);
  }
#ifdef RSC_HAS_CUDA
  if (src.on_device() && dst.on_device()) {
    dst.copy_on_device(src, source_arenas, dest_arenas);
    return;
  }
#endif
  std::vector<rsc::Arena> staged(source_arenas.size());
  for (size_t k = 0; k < source_arenas.size(); ++k) {
    staged[k] = src.read(source_arenas[k]);
  }
  for (size_t k = 0; k < dest_arenas.size(); ++k) {
    for (uint32_t c = 0; c < dst.num_cars; ++c) {
      dst.drop_pending_controls(dest_arenas[k], c);
    }
    dst.write(dest_arenas[k]) = staged[k];
  }
}

std::vector<uint8_t> ArenaBatch::save_arena(uint32_t arena) {
  const rsc::Arena &a = impl_->read(arena);
  std::vector<uint8_t> out(sizeof(uint64_t) * 2 + sizeof(rsc::Arena));
  uint64_t header[2] = {k_snapshot_magic, sizeof(rsc::Arena)};
  std::memcpy(out.data(), header, sizeof(header));
  std::memcpy(out.data() + sizeof(header), &a, sizeof(rsc::Arena));
  return out;
}

void ArenaBatch::load_arena(uint32_t arena, std::span<const uint8_t> snapshot) {
  Impl &im = *impl_;
  uint64_t header[2];
  require(snapshot.size() == sizeof(header) + sizeof(rsc::Arena),
          "arena snapshot has the wrong size");
  std::memcpy(header, snapshot.data(), sizeof(header));
  require(header[0] == k_snapshot_magic && header[1] == sizeof(rsc::Arena),
          "arena snapshot header does not match this build");
  rsc::Arena &a = im.write(arena);
  for (uint32_t c = 0; c < im.num_cars; ++c) {
    im.drop_pending_controls(arena, c);
  }
  std::memcpy(&a, snapshot.data() + sizeof(header), sizeof(rsc::Arena));
}

}
