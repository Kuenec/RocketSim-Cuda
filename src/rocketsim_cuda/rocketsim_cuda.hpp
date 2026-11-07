#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocketsim_cuda {

enum class Backend : uint8_t { Cpu, Cuda, Hybrid };

enum class GameMode : uint8_t {
  Soccar,
  Hoops,
  Heatseeker,
  Snowday,
  Dropshot,
  TheVoid
};

enum class Team : uint8_t { Blue, Orange };

enum class DemoMode : uint8_t { Normal, OnContact, Disabled };

enum class MemWeightMode : uint8_t { Heavy, Balanced, Light };

enum class UserInfoType : uint8_t { None, Car, Ball, DropshotTile };

enum class CarBody : uint8_t {
  Octane,
  Dominus,
  Plank,
  Breakout,
  Hybrid,
  Merc,
  Psyclops
};

enum class TileDamageState : uint8_t { Full, Damaged, Broken };

inline constexpr uint32_t k_max_cars = 8;
inline constexpr uint32_t k_max_boost_pads = 48;
inline constexpr uint32_t k_tiles_per_team = 70;

struct Vec3 {
  float x;
  float y;
  float z;
};

struct RotMat {
  Vec3 forward;
  Vec3 right;
  Vec3 up;
};

struct PhysState {
  Vec3 pos;
  RotMat rot_mat;
  Vec3 vel;
  Vec3 ang_vel;
};

struct CarControls {
  float throttle;
  float steer;
  float pitch;
  float yaw;
  float roll;
  bool jump;
  bool boost;
  bool handbrake;
};

struct RaycastHitInfo {
  Vec3 hit_point;
  Vec3 hit_normal;
  float hit_fraction;
  UserInfoType user_info;
};

struct CarCache {
  bool is_on_ground;
  bool has_wheel_contact[4];
  RaycastHitInfo wheels_with_contact[4];
  bool has_world_contact_normal;
  Vec3 world_contact_normal;
  CarControls prev_controls;
  uint8_t world_wheels_prev;
  uint64_t demo_tick;
  bool has_demoer;
  uint32_t demoer_idx;
};

struct CarState {
  PhysState phys;
  CarControls controls;
  bool has_jumped;
  bool has_double_jumped;
  bool has_flipped;
  Vec3 flip_rel_torque;
  uint32_t jump_ticks;
  float flip_time;
  bool is_flipping;
  bool is_jumping;
  bool jump_hold_broken;
  float air_time;
  float air_time_since_jump;
  float boost;
  float handbrake_val;
  bool is_supersonic;
  float supersonic_grace_timer;
  float time_since_boosted;
  float boosting_time;
  bool is_boosting;
  bool is_auto_flipping;
  float auto_flip_timer;
  float auto_flip_torque_scale;
  float bump_cooldown_timer;
  bool has_bump_cooldown_victim;
  uint32_t bump_cooldown_victim;
  bool has_last_extra_hit_tick;
  uint64_t last_extra_hit_tick;
  bool is_demoed;
  float demo_respawn_timer;
  CarCache cache;
};

struct CarCore {
  PhysState phys;
  CarControls controls;
  bool has_jumped;
  bool has_double_jumped;
  bool has_flipped;
  Vec3 flip_rel_torque;
  uint32_t jump_ticks;
  float flip_time;
  bool is_flipping;
  bool is_jumping;
  bool jump_hold_broken;
  float air_time_since_jump;
  float boost;
  bool is_auto_flipping;
  float auto_flip_timer;
  float auto_flip_torque_scale;
  float bump_cooldown_timer;
  bool has_bump_cooldown_victim;
  uint32_t bump_cooldown_victim;
  bool has_last_extra_hit_tick;
  uint64_t last_extra_hit_tick;
  bool is_demoed;
  float demo_respawn_timer;
};

struct CarReset {
  CarCore core;
  bool on_ground;
};

struct CurvePoint {
  float x;
  float y;
};

struct WheelPairConfig {
  float wheel_radius;
  float suspension_rest_length;
  Vec3 connection_point_offset;
};

struct CarBodyConfig {
  Vec3 hitbox_size;
  Vec3 hitbox_pos_offset;
  WheelPairConfig front_wheels;
  WheelPairConfig back_wheels;
  bool three_wheels;
  float dodge_deadzone;
  CurvePoint steer_angle_curve[6];
  CurvePoint powerslide_steer_angle_curve[2];
};

struct CarInfo {
  uint32_t idx;
  Team team;
  CarBodyConfig config;
};

struct HeatseekerInfo {
  int8_t y_target_dir;
  float cur_target_speed;
  float time_since_hit;
};

struct DropshotInfo {
  uint8_t charge_level;
  float accumulated_hit_force;
  int8_t y_target_dir;
  bool has_last_damage_tick;
  uint64_t last_damage_tick;
};

struct BallState {
  PhysState phys;
  HeatseekerInfo hs_info;
  DropshotInfo ds_info;
  uint64_t tick_count_since_kickoff;
};

struct BoostPadConfig {
  Vec3 pos;
  bool is_big;
};

struct BoostPadState {
  float cooldown;
  std::vector<uint32_t> touching_car_idxs;
};

struct TileStates {
  TileDamageState states[2][k_tiles_per_team];
};

struct MutatorConfig {
  Vec3 gravity;
  float car_mass;
  float ball_mass;
  float ball_max_speed;
  float ball_drag;
  float jump_accel;
  float jump_immediate_force;
  float boost_accel_ground;
  float boost_accel_air;
  float boost_used_per_second;
  float respawn_delay;
  float bump_cooldown_time;
  float car_max_boost_amount;
  float car_spawn_boost_amount;
  float boost_pad_amount_small;
  float boost_pad_amount_big;
  float boost_pad_cooldown_big;
  float boost_pad_cooldown_small;
  float ball_hit_extra_force_scale;
  float bump_force_scale;
  float ball_radius;
  bool unlimited_flips;
  bool unlimited_double_jumps;
  bool recharge_boost_enabled;
  float recharge_boost_per_second;
  float recharge_boost_delay;
  DemoMode demo_mode;
  bool enable_team_demos;
  float goal_base_threshold_y;
  bool boost_pads_enabled;
};

struct ArenaState {
  GameMode game_mode;
  uint64_t tick_count;
  std::vector<CarInfo> car_infos;
  std::vector<CarState> cars;
  BallState ball;
  std::vector<BoostPadConfig> boost_pad_configs;
  std::vector<BoostPadState> boost_pads;
  std::optional<TileStates> tile_states;
};

enum class ArenaEventKind : uint8_t {
  BallHitWorld,
  CarHitBall,
  CarHitCar,
  CarHitWorld,
  CarPickupBoost,
  CarLanded
};

struct ArenaEvent {
  ArenaEventKind kind;
  uint32_t car_idx;
  uint32_t victim_car_idx;
  uint32_t boost_pad_idx;
  bool is_demo;
  Vec3 contact_point;
  Vec3 contact_normal;
  Vec3 extra_hit_vel;
  bool has_wheel_contact[4];
  RaycastHitInfo wheels[4];
};

struct CarEventCounts {
  uint32_t hit_ball;
  uint32_t hit_world;
  uint32_t bumps_dealt;
  uint32_t demos_dealt;
  uint32_t bumps_received;
  uint32_t demos_received;
  uint32_t boost_pickups;
  uint32_t landed;
};

struct EventCounts {
  uint32_t ball_hit_world;
  bool goal_scored;
  uint64_t goal_tick;
  CarEventCounts cars[k_max_cars];
};

struct ArenaInfo {
  uint64_t tick_count;
  uint64_t rng_state;
  uint32_t error_flags;
  uint32_t num_events;
  bool ball_scored;
};

struct PadStateView {
  float cooldown;
  uint32_t num_touching;
  uint32_t touching[k_max_cars];
};

enum class MemorySpace : uint8_t { Host, Device };

struct StateBuffers {
  ArenaInfo *infos = nullptr;
  BallState *balls = nullptr;
  CarState *cars = nullptr;
  PadStateView *pads = nullptr;
  TileStates *tiles = nullptr;
  EventCounts *event_counts = nullptr;
};

struct CarSpec {
  Team team;
  CarBodyConfig config;
};

struct BatchConfig {
  GameMode game_mode = GameMode::Soccar;
  MemWeightMode mem_weight_mode = MemWeightMode::Balanced;
  Vec3 min_pos{-5600.0f, -6000.0f, 0.0f};
  Vec3 max_pos{5600.0f, 6000.0f, 2200.0f};
  float max_aabb_len = 370.0f;
  bool no_ball_rot = false;
  std::optional<std::vector<BoostPadConfig>> custom_boost_pads;
  MutatorConfig mutators;
  std::vector<CarSpec> cars;
  std::vector<uint64_t> rng_seeds;
  uint32_t cpu_threads = 0;
  uint32_t hybrid_cpu_arenas = 0;
};

MutatorConfig default_mutators(GameMode mode);
CarBodyConfig car_body_config(CarBody body);
CarCore car_core_default();
BallState ball_state_default();

class MeshSet {
public:
  static std::shared_ptr<const MeshSet>
  load_directory(const std::string &collision_meshes_folder);
  ~MeshSet();
  MeshSet(const MeshSet &) = delete;
  MeshSet &operator=(const MeshSet &) = delete;

  bool has_game_mode(GameMode mode) const;

  struct Impl;
  const Impl &impl() const { return *impl_; }

private:
  explicit MeshSet(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class ArenaBatch {
public:
  ArenaBatch(Backend backend, std::shared_ptr<const MeshSet> meshes,
             const BatchConfig &config);
  ~ArenaBatch();
  ArenaBatch(const ArenaBatch &) = delete;
  ArenaBatch &operator=(const ArenaBatch &) = delete;

  Backend backend() const;
  GameMode game_mode() const;
  uint32_t num_arenas() const;
  uint32_t num_cars() const;
  uint32_t num_boost_pads() const;

  void set_car_controls(uint32_t arena, uint32_t car,
                        const CarControls &controls);
  void set_all_controls(std::span<const CarControls> controls);
  void set_all_controls_device(const CarControls *device_controls);
  void reset_to_random_kickoff_masked(const uint8_t *mask,
                                      const uint64_t *seeds, MemorySpace space);
  void set_states_masked(const uint8_t *mask, const BallState *balls,
                         const CarReset *cars, MemorySpace space);
  void export_state(const StateBuffers &out, MemorySpace space);
  void *cuda_stream() const;

  void step(uint32_t ticks);
  void step_async(uint32_t ticks);
  void sync();

  ArenaState get_arena_state(uint32_t arena);
  ArenaInfo get_arena_info(uint32_t arena);
  CarState get_car_state(uint32_t arena, uint32_t car);
  BallState get_ball_state(uint32_t arena);
  BoostPadState get_boost_pad_state(uint32_t arena, uint32_t pad);
  TileStates get_tile_states(uint32_t arena);
  MutatorConfig get_mutator_config(uint32_t arena);
  std::vector<ArenaEvent> get_last_step_events(uint32_t arena);
  EventCounts get_event_counts(uint32_t arena);

  void set_car_state(uint32_t arena, uint32_t car, const CarCore &core);
  void set_car_state_full(uint32_t arena, uint32_t car, const CarState &state);
  void set_ball_state(uint32_t arena, const BallState &state);
  void set_boost_pad_state(uint32_t arena, uint32_t pad,
                           const BoostPadState &state);
  void set_tile_states(uint32_t arena, const TileStates &states);
  void set_mutator_config(uint32_t arena, const MutatorConfig &mutators);
  void reset_to_random_kickoff(uint32_t arena, std::optional<uint64_t> seed);

  void copy_arenas(const ArenaBatch &source,
                   std::span<const uint32_t> source_arenas,
                   std::span<const uint32_t> dest_arenas);
  std::vector<uint8_t> save_arena(uint32_t arena);
  void load_arena(uint32_t arena, std::span<const uint8_t> snapshot);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}
