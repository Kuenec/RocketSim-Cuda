#pragma once

#include <cstdint>

#include "rsc/bullet/dynamics.hpp"
#include "rsc/bullet/mesh_algorithms.hpp"
#include "rsc/bullet/vehicle.hpp"
#include "rsc/math/rng.hpp"
#include "rsc/sim/ball.hpp"
#include "rsc/sim/boost_pads.hpp"
#include "rsc/sim/bump_demo.hpp"
#include "rsc/sim/car.hpp"
#include "rsc/sim/consts.hpp"
#include "rsc/sim/dropshot.hpp"
#include "rsc/sim/quantize.hpp"

namespace rsc {

inline constexpr int k_max_records = 160;
inline constexpr uint32_t k_job_records = 32;
inline constexpr int k_max_ball_world_records = 32;
inline constexpr int k_max_normal_groups = 64;

inline constexpr int k_max_events = 192;

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
  uint8_t car_idx;
  uint8_t other_idx;
  bool is_demo;
  Vec3A point;
  Vec3A vec;
};

struct LandedWheels {
  bool has_contact[4];
  RaycastHitInfo wheels[4];
};

struct CarEventCounters {
  uint32_t hit_ball;
  uint32_t hit_world;
  uint32_t bumps_dealt;
  uint32_t demos_dealt;
  uint32_t bumps_received;
  uint32_t demos_received;
  uint32_t boost_pickups;
  uint32_t landed;
};

struct ArenaEventCounters {
  uint32_t ball_hit_world;
  bool goal_scored;
  uint64_t goal_tick;
  CarEventCounters cars[k_max_cars];
};

struct Arena {
  ArenaPhysics phys;
  Ball ball;
  uint32_t num_cars;
  Car cars[k_max_cars];
  uint64_t tick_count;
  bool ball_only;
  Rng rng;
  MutatorConfig mutators;
  uint32_t num_pads;
  PadState pads[k_max_pads];
  uint64_t touched_pads;
  bool pad_lists_unchecked;
  bool has_tiles;
  TileStates tiles;
  uint32_t num_events;
  ArenaEvent events[k_max_events];
  LandedWheels landed[k_max_cars];
  ArenaEventCounters counters;
};

struct ArenaStatic {
  StaticWorldData world;
  GameMode mode;
  PadLayoutView pads;
  bool has_pads;
};

struct ContactRecord {
  bool is_swap;
  uint32_t a;
  uint32_t b;
  Vec3A pos_world_on_a;
  Vec3A pos_world_on_b;
  Vec3A normal_world_on_b;
};

struct BallWorldRecord {
  uint32_t rb;
  Vec3A point;
  Vec3A normal;
};

struct WorldNormalGroup {
  uint32_t car_idx;
  uint32_t object_idx;
  Vec3A sum;
};

enum class CarPhasePlan : uint8_t { Sequential, Precast, Parallel };

struct StepScratch;

inline constexpr uint32_t k_team_sat_slots = 16;
inline constexpr uint32_t k_collect_frontier = 128;
inline constexpr uint32_t k_parallel_collect_lanes = 4;
inline constexpr uint32_t k_collect_memo_slots = 4;
inline constexpr uint32_t k_collect_memo_leaves = 128;
inline constexpr float k_collect_memo_margin = 0.5f;

struct CollectMemo {
  bool valid;
  uint32_t mesh;
  uint32_t count;
  Aabb outer;
  uint32_t leaves[k_collect_memo_leaves];
};

struct CollectMemoOwner {
  uint32_t next;
  CollectMemo slots[k_collect_memo_slots];
};

struct TeamSatSlot {
  uint32_t group;
  uint32_t item;
  Vec3A half;
  Vec3A q[3];
  SatSweepState state;
};

struct TeamSatScratch {
  TeamSatSlot slots[k_team_sat_slots];
  sat::CandidatePair pairs[k_team_sat_slots][sat::k_sat_candidates];
};

struct SingleLane {
  static constexpr bool k_cooperative = false;
  static constexpr uint32_t lane = 0;
  static constexpr uint32_t size = 1;

  RS_HDI static constexpr bool leader() { return true; }
  RS_HDI static constexpr uint32_t group_ballot(bool pred) {
    return pred ? 1u : 0u;
  }
  RS_HDI static void barrier() {}
};

struct LaneGroup {
  static constexpr bool k_cooperative = true;
  uint32_t lane;
  uint32_t size;
  uint32_t team_mask;
  uint32_t team_lane;
  uint32_t team_size;
  StepScratch *team_scratch;
  TeamSatScratch *team_sat;

  RS_HDI bool leader() const { return lane == 0; }
  RS_HDI uint32_t team_ballot(bool pred) const {
#ifdef __CUDA_ARCH__
    return __ballot_sync(team_mask, pred);
#else
    return pred ? 1u : 0u;
#endif
  }
  RS_HDI uint32_t team_rank(uint32_t ballot) const {
    return popcount32(ballot & ((1u << team_lane) - 1u));
  }
  RS_HDI uint32_t group_mask() const {
    return size >= 32 ? 0xffffffffu : ((1u << size) - 1u) << (team_lane - lane);
  }
  RS_HDI uint32_t group_ballot(bool pred) const {
#ifdef __CUDA_ARCH__
    return __ballot_sync(group_mask(), pred);
#else
    return pred ? 1u : 0u;
#endif
  }
  RS_HDI CarRayLanes car_ray_lanes() const {
    uint32_t base = team_lane - lane % CarRayLanes::size;
    return {0xfu << base, base, lane % CarRayLanes::size};
  }
  RS_HDI void group_sync() const {
#ifdef __CUDA_ARCH__
    __syncwarp(group_mask());
#endif
  }
  RS_HDI void barrier() const {
#ifdef __CUDA_ARCH__
    if (size > 1) {
      __syncwarp(team_mask);
    }
#endif
  }
};

struct JobRecords {
  uint32_t count;
  bool overflow;
  ContactRecord records[k_job_records];
};

struct WheelImpulse {
  bool apply;
  Vec3A lin;
  Vec3A ang;
};

struct StepScratch {
  CarPhasePlan car_plan;
  bool was_on_ground[k_max_cars];
  bool vehicle_run[k_max_cars];
  float real_throttle[k_max_cars];
  WheelImpulse wheel_impulses[k_max_cars][k_num_wheels];
  Aabb body_aabbs[k_max_dyn_bodies];
  VehicleHit rays[k_max_cars][k_num_wheels];
  SolverScratch solver;
  PadUpdateScratch pads;
  int num_records;
  ContactRecord records[k_max_records];
  int num_ball_world;
  BallWorldRecord ball_world[k_max_ball_world_records];
  int num_groups;
  WorldNormalGroup groups[k_max_normal_groups];
  PairPlan pair_plans[k_max_pairs];
  uint32_t num_gate_jobs;
  GateJob gate_jobs[k_max_gate_jobs];
  GateItems gate_items;
  JobRecords job_records[k_max_gate_jobs];
  uint32_t collect_frontier[2][k_collect_frontier];
  uint32_t collect_leaves[k_max_gate_triangles];
  CollectMemoOwner collect_memo[k_max_dyn_bodies];
};

enum class RecordSink : uint8_t { Records, Job };

struct ContactTracker {
  World *w;
  StepScratch *sc;
  bool ball_only;
  RecordSink sink;
  uint32_t job;

  RS_HD PairPlan *pair_plans() const { return sc->pair_plans; }
  RS_HD GateJob *gate_jobs() const { return sc->gate_jobs; }
  RS_HD GateItems &gate_items() const { return sc->gate_items; }

  RS_HD ContactTracker job_tracker(uint32_t j) const {
    sc->job_records[j].count = 0;
    sc->job_records[j].overflow = false;
    return {w, sc, ball_only, RecordSink::Job, j};
  }

  RS_HD bool job_overflowed(uint32_t j) const {
    return sc->job_records[j].overflow;
  }

  RS_HD void replay_job(uint32_t j) {
    const JobRecords &jr = sc->job_records[j];
    for (uint32_t k = 0; k < jr.count; ++k) {
      push_record(jr.records[k]);
    }
  }

  RS_HD static bool should_swap(const RigidBody &a, const RigidBody &b) {
    if (a.user_idx != UserInfoType::None && b.user_idx != UserInfoType::None) {
      return a.user_idx > b.user_idx;
    }
    return b.user_idx != UserInfoType::None;
  }

  RS_HD void push_record(const ContactRecord &record) {
    if (sink == RecordSink::Job) {
      JobRecords &jr = sc->job_records[job];
      if (jr.count >= k_job_records) {
        jr.overflow = true;
        return;
      }
      jr.records[jr.count++] = record;
      return;
    }
    if (sc->num_records >= k_max_records) {
      w->flag_error(k_error_records_overflow);
      return;
    }
    sc->records[sc->num_records++] = record;
  }

  RS_HD void push_record(bool swap, uint32_t a, uint32_t b,
                         const ManifoldPoint &p) {
    push_record(
        {swap, a, b, p.pos_world_on_a, p.pos_world_on_b, p.normal_world_on_b});
  }

  RS_HD void callback(ManifoldPoint &cp, const RigidBody &body_a_in,
                      const RigidBody &body_b_in, int idx) {
    bool swap = should_swap(body_a_in, body_b_in);
    const RigidBody &a = swap ? body_b_in : body_a_in;
    const RigidBody &b = swap ? body_a_in : body_b_in;
    if (a.user_idx == UserInfoType::Car) {
      if (b.user_idx == UserInfoType::Ball) {
        cp.combined_friction = consts::car::hit_ball_coefs.friction;
        cp.combined_restitution = consts::car::hit_ball_coefs.restitution;
      } else if (b.user_idx == UserInfoType::Car) {
        cp.combined_friction = consts::car::hit_car_coefs.friction;
        cp.combined_restitution = consts::car::hit_car_coefs.restitution;
      } else {
        cp.combined_friction = consts::car::hit_world_coefs.friction;
        cp.combined_restitution = consts::car::hit_world_coefs.restitution;
      }
    } else if (a.user_idx == UserInfoType::Ball &&
               b.user_idx == UserInfoType::None && b.is_static_obj()) {
      cp.is_special = true;
    }
    if (ball_only && a.user_idx == UserInfoType::Ball &&
        b.user_idx == UserInfoType::None && b.is_static_obj()) {
      if (sink == RecordSink::Job) {
        sc->job_records[job].overflow = true;
      } else if (sc->num_ball_world >= k_max_ball_world_records) {
        w->flag_error(k_error_records_overflow);
      } else {
        sc->ball_world[sc->num_ball_world++] = {
            a.world_array_idx, cp.pos_world_on_b, cp.normal_world_on_b};
      }
    } else {
      push_record(swap, a.world_array_idx, b.world_array_idx, cp);
    }
    if (idx >= 0) {
      adjust_internal_edge_contacts(*w, cp, b, idx);
    }
  }

  RS_HD void contact_processed(const ManifoldPoint &point,
                               const RigidBody &body_a_in,
                               const RigidBody &body_b_in) {
    bool swap = should_swap(body_a_in, body_b_in);
    const RigidBody &a = swap ? body_b_in : body_a_in;
    const RigidBody &b = swap ? body_a_in : body_b_in;
    if (a.user_idx != UserInfoType::Car || b.user_idx == UserInfoType::Ball ||
        b.user_idx == UserInfoType::Car) {
      return;
    }
    push_record(swap, a.world_array_idx, b.world_array_idx, point);
  }

  RS_HD void manifold_processed(const PersistentManifold &m,
                                const RigidBody &body_a_in,
                                const RigidBody &body_b_in) {
    bool swap = should_swap(body_a_in, body_b_in);
    const RigidBody &a = swap ? body_b_in : body_a_in;
    const RigidBody &b = swap ? body_a_in : body_b_in;
    if (a.user_idx != UserInfoType::Car || b.user_idx == UserInfoType::Ball ||
        b.user_idx == UserInfoType::Car) {
      return;
    }
    for (int i = 0; i < m.num_points; ++i) {
      uint32_t car_idx = a.user_pointer;
      uint32_t object_idx = b.world_array_idx;
      Vec3A normal = m.points[i].normal_world_on_b;
      bool found = false;
      for (int g = 0; g < sc->num_groups; ++g) {
        WorldNormalGroup &group = sc->groups[g];
        if (group.car_idx == car_idx && group.object_idx == object_idx) {
          group.sum += normal;
          found = true;
          break;
        }
      }
      if (!found) {
        if (sc->num_groups >= k_max_normal_groups) {
          w->flag_error(k_error_records_overflow);
        } else {
          sc->groups[sc->num_groups++] = {car_idx, object_idx, normal};
        }
      }
    }
  }
};

RS_HDI World arena_world(const ArenaStatic &st, Arena &a) {
  return {&st.world, &a.phys};
}

RS_HD bool arena_is_ball_scored(const ArenaStatic &st, const Arena &a) {
  using namespace consts;
  const RigidBody &rb =
      a.phys.dyn_bodies[a.ball.rigid_body_idx - st.world.num_static_bodies];
  Vec3A pos = rb.world_trans.translation * bt_to_uu;
  switch (st.mode) {
  case GameMode::Soccar:
  case GameMode::Heatseeker:
  case GameMode::Snowday:
    return abs_f32(pos.y) >
           a.mutators.goal_base_threshold_y + a.mutators.ball_radius;
  case GameMode::Hoops: {
    if (pos.z < goal::hoops_goal_score_threshold_z) {
      float dy = abs_f32(pos.y) * 0.9f - 2770.0f;
      float dist_sq = pos.x * pos.x + dy * dy;
      return dist_sq - 716.0f * 716.0f < 0.0f;
    }
    return false;
  }
  case GameMode::Dropshot:
    return pos.z < -a.mutators.ball_radius * 1.75f;
  case GameMode::TheVoid:
    return false;
  }
  return false;
}

RS_HD void arena_push_event(Arena &a, ArenaEventKind kind, uint32_t car_idx,
                            uint32_t other_idx, bool is_demo, Vec3A point,
                            Vec3A vec) {
  if (a.num_events >= static_cast<uint32_t>(k_max_events)) {
    a.phys.error_flags |= k_error_events_overflow;
    return;
  }
  a.events[a.num_events++] = {kind,
                              static_cast<uint8_t>(car_idx),
                              static_cast<uint8_t>(other_idx),
                              is_demo,
                              point,
                              vec};
}

RS_HD void arena_count_events(Arena &a) {
  ArenaEventCounters &c = a.counters;
  for (uint32_t i = 0; i < a.num_events; ++i) {
    const ArenaEvent &e = a.events[i];
    switch (e.kind) {
    case ArenaEventKind::BallHitWorld:
      c.ball_hit_world += 1;
      break;
    case ArenaEventKind::CarHitBall:
      c.cars[e.car_idx].hit_ball += 1;
      break;
    case ArenaEventKind::CarHitCar:
      if (e.is_demo) {
        c.cars[e.car_idx].demos_dealt += 1;
        c.cars[e.other_idx].demos_received += 1;
      } else {
        c.cars[e.car_idx].bumps_dealt += 1;
        c.cars[e.other_idx].bumps_received += 1;
      }
      break;
    case ArenaEventKind::CarHitWorld:
      c.cars[e.car_idx].hit_world += 1;
      break;
    case ArenaEventKind::CarPickupBoost:
      c.cars[e.car_idx].boost_pickups += 1;
      break;
    case ArenaEventKind::CarLanded:
      c.cars[e.car_idx].landed += 1;
      break;
    }
  }
}

RS_HD void arena_update_tile_flags(Arena &a) {
  for (int team = 0; team < 2; ++team) {
    for (int i = 0; i < consts::dropshot::num_tiles_per_team; ++i) {
      a.phys.tile_broken[team * consts::dropshot::num_tiles_per_team + i] =
          a.tiles.states[team][i] == k_tile_broken ? 1 : 0;
    }
  }
}

RS_HDN void arena_on_car_car(Arena &a, uint32_t car_1, uint32_t car_2,
                             const ContactRecord &record) {
  using namespace consts;
  for (int pass = 0; pass < 2; ++pass) {
    bool is_swapped = pass == 1;
    uint32_t attacker_idx = is_swapped ? car_2 : car_1;
    uint32_t victim_idx = is_swapped ? car_1 : car_2;
    Car &attacker = a.cars[attacker_idx];
    Car &victim = a.cars[victim_idx];
    const CarState &as = attacker.state;
    const CarState &vs = victim.state;
    bool attacker_demoed_by_victim =
        as.cache.demo_tick == a.tick_count &&
        as.cache.demoer_idx == static_cast<int32_t>(victim_idx);
    int gate;
    if (!(as.is_demoed || vs.is_demoed)) {
      gate = 0;
    } else if (vs.is_demoed || !attacker_demoed_by_victim) {
      gate = 1;
    } else {
      gate = 2;
    }
    if (gate == 1) {
      continue;
    }
    Vec3A delta_pos = vs.phys.pos - as.phys.pos;
    if (as.phys.vel.dot(delta_pos) < 0.0f) {
      continue;
    }
    Vec3A vel_dir = as.phys.vel.normalize_or_zero();
    Vec3A dir_to_victim = delta_pos.normalize_or_zero();
    float speed_towards = as.phys.vel.dot(dir_to_victim);
    float other_away = vs.phys.vel.dot(vel_dir);
    if (speed_towards <= other_away) {
      continue;
    }
    bool bumper = bump::is_within_forward_cone(
        as, attacker.info.config, vs, victim.info.config, false,
        car::bump::yaw_limit_deg, car::bump::pitch_limit_deg);
    bool is_demo;
    switch (a.mutators.demo_mode) {
    case DemoMode::OnContact:
      is_demo = true;
      break;
    case DemoMode::Disabled:
      is_demo = false;
      break;
    default:
      is_demo = bump::should_demolish(as, attacker.info.config, vs,
                                      victim.info.config, tick_time);
      break;
    }
    if (is_demo && !a.mutators.enable_team_demos) {
      is_demo = attacker.info.team != victim.info.team;
    }
    Vec3A contact_point =
        (is_swapped ? record.pos_world_on_b : record.pos_world_on_a) * bt_to_uu;
    if (is_demo) {
      victim.state.is_demoed = true;
      victim.state.demo_respawn_timer = a.mutators.respawn_delay;
      victim.state.cache.demo_tick = a.tick_count;
      victim.state.cache.demoer_idx = static_cast<int32_t>(attacker_idx);
      arena_push_event(a, ArenaEventKind::CarHitCar, attacker_idx, victim_idx,
                       true, contact_point, Vec3A::zero());
      continue;
    }
    if (gate != 0) {
      continue;
    }
    if (as.bump_cooldown_timer > 0.0f &&
        (as.bump_cooldown_victim < 0 ||
         as.bump_cooldown_victim == static_cast<int32_t>(victim_idx))) {
      continue;
    }
    if (bumper) {
      bool ground_hit = vs.cache.is_on_ground;
      float attacker_speed = min_f32_c(as.phys.vel.length(), 2200.0f);
      Vec3A impulse;
      if (ground_hit) {
        float base =
            curves::bump_vel_amount_ground().get_output(attacker_speed);
        float upward =
            curves::bump_upward_vel_amount().get_output(attacker_speed);
        Vec3A hit_up = vs.phys.rot_mat.z_axis;
        impulse = (vel_dir * base) + (hit_up * upward);
      } else {
        impulse =
            vel_dir * curves::bump_vel_amount_air().get_output(attacker_speed);
      }
      victim.vel_impulse_cache +=
          impulse * a.mutators.bump_force_scale * uu_to_bt;
    }
    attacker.state.bump_cooldown_timer =
        a.mutators.bump_cooldown_time + tick_time * 0.5f;
    attacker.state.bump_cooldown_victim = static_cast<int32_t>(victim_idx);
    if (bumper) {
      arena_push_event(a, ArenaEventKind::CarHitCar, attacker_idx, victim_idx,
                       false, contact_point, Vec3A::zero());
    }
  }
}

RS_HD void arena_update_world_contact_normal(Arena &a, const StepScratch &sc) {
  for (uint32_t car_idx = 0; car_idx < a.num_cars; ++car_idx) {
    Vec3A total = Vec3A::zero();
    for (int g = 0; g < sc.num_groups; ++g) {
      if (sc.groups[g].car_idx != car_idx) {
        continue;
      }
      Vec3A sum = sc.groups[g].sum;
      float len_sq = sum.length_squared();
      if (len_sq >= 1e-8f) {
        total += sum / sqrt_f32(len_sq);
      }
    }
    float len_sq = total.length_squared();
    if (len_sq >= 1e-8f) {
      a.cars[car_idx].state.cache.has_world_contact_normal = true;
      a.cars[car_idx].state.cache.world_contact_normal =
          total / sqrt_f32(len_sq);
    }
  }
}

RS_HD void arena_tick_front_item(const ArenaStatic &st, Arena &a,
                                 uint32_t item) {
  using namespace consts;
  World w = arena_world(st, a);
  if (item == a.num_cars) {
    RigidBody &ball_rb = w.dyn_body(a.ball.rigid_body_idx);
    ball_rb.limit_vels(a.mutators.ball_max_speed * uu_to_bt,
                       ball::max_ang_speed);
    quantize_body(ball_rb);
    a.ball.state.phys.vel = ball_rb.lin_vel * bt_to_uu;
    a.ball.state.phys.ang_vel = ball_rb.ang_vel;
    return;
  }
  Car &car = a.cars[item];
  RigidBody &rb = w.dyn_body(car.rigid_body_idx);
  rb.limit_vels(car::max_speed * uu_to_bt, car::max_ang_speed);
  quantize_body(rb);
  car.state.phys.vel = rb.lin_vel * bt_to_uu;
  car.state.phys.ang_vel = rb.ang_vel;
  car_apply_flip_zdamp_prepass(car, w);
}

RS_HD void arena_push_car_landed(Arena &a, uint32_t c) {
  for (int i = 0; i < 4; ++i) {
    a.landed[c].has_contact[i] = a.cars[c].state.cache.wheel_has_contact[i];
    a.landed[c].wheels[i] = a.cars[c].state.cache.wheels_with_contact[i];
  }
  arena_push_event(a, ArenaEventKind::CarLanded, c, 0, false, Vec3A::zero(),
                   Vec3A::zero());
}

RS_HD void arena_car_pre_ticks_in_order(const ArenaStatic &st, Arena &a,
                                        const StepScratch *precast) {
  World w = arena_world(st, a);
  for (uint32_t c = 0; c < a.num_cars; ++c) {
    bool was_on_ground = a.cars[c].state.cache.is_on_ground;
    car_pre_tick_update(a.cars[c], w, a.rng, st.mode, a.mutators,
                        precast ? precast->rays[c] : nullptr, nullptr);
    if (!was_on_ground && a.cars[c].state.cache.is_on_ground) {
      arena_push_car_landed(a, c);
    }
  }
}

RS_HD bool arena_rays_precomputable(const ArenaStatic &st, Arena &a) {
  World w = arena_world(st, a);
  for (uint32_t c = 0; c < a.num_cars; ++c) {
    const RigidBody &rb = w.dyn_body(a.cars[c].rigid_body_idx);
    if (a.cars[c].state.is_demoed ||
        (rb.collision_flags & k_flag_no_contact_response) != 0) {
      return false;
    }
  }
  return true;
}

RS_HD bool ray_hit_dynamic(const World &w, const VehicleHit &hit) {
  return hit.valid && hit.body_idx >= w.num_static();
}

RS_HD void arena_apply_wheel_impulses(const StepScratch &sc, RigidBody &chassis,
                                      uint32_t c) {
  for (int j = 0; j < k_num_wheels; ++j) {
    const WheelImpulse &wi = sc.wheel_impulses[c][j];
    if (wi.apply) {
      chassis.apply_impulse_parts(wi.lin, wi.ang, false);
    }
  }
}

template <typename Lanes>
RS_HDI void arena_tick_vehicle_wheels(Arena &a, StepScratch &sc, World &w,
                                      const Lanes &lanes, bool per_wheel) {
  uint32_t c = lanes.lane / k_num_wheels;
  int i = static_cast<int>(lanes.lane % k_num_wheels);
  bool mine = per_wheel && c < a.num_cars && sc.vehicle_run[c];
  if (mine) {
    Car &car = a.cars[c];
    const Affine3A trans = w.body(car.vehicle.chassis_body_idx).world_trans;
    Vec3A source, target;
    car.vehicle.wheels[i].prepare_for_raycast(trans, source, target);
    WheelAxleCache axles;
    car.vehicle.wheel_contact(w, trans, i, sc.rays[c][i], consts::tick_time,
                              car.state.handbrake_val, sc.real_throttle[c],
                              car.info.config.three_wheels, axles);
  }
  lanes.barrier();
  int contacts = 0;
  if (mine) {
    VehicleRL &vehicle = a.cars[c].vehicle;
    const RigidBody &chassis = w.body(vehicle.chassis_body_idx);
    contacts = vehicle.num_wheel_contacts();
    vehicle.scale_engine_force(i, contacts);
    WheelImpulse &wi = sc.wheel_impulses[c][i];
    wi.apply = false;
    if (contacts > 0) {
      vehicle.wheels[i].suspension_impulse(
          chassis.world_trans.translation, consts::tick_time,
          [&](Vec3A impulse, Vec3A rel_pos) {
            chassis.linear_rel_pos_impulse_parts(impulse, rel_pos, true, wi.lin,
                                                 wi.ang);
            wi.apply = true;
          });
    }
  }
  lanes.barrier();
  if (mine && i == 0 && contacts > 0) {
    arena_apply_wheel_impulses(
        sc, w.dyn_body(a.cars[c].vehicle.chassis_body_idx), c);
  }
  lanes.barrier();
  if (mine && contacts > 0) {
    WheelInfo &wheel = a.cars[c].vehicle.wheels[i];
    const RigidBody &chassis = w.body(a.cars[c].vehicle.chassis_body_idx);
    wheel.update_friction_impulse(chassis, consts::tick_time,
                                  chassis.get_mass() / 3.0f);
    WheelImpulse &wi = sc.wheel_impulses[c][i];
    wi.apply = false;
    wheel.friction_impulse(chassis.world_trans, consts::tick_time,
                           [&](Vec3A impulse, Vec3A rel_pos) {
                             chassis.linear_rel_pos_impulse_parts(
                                 impulse, rel_pos, true, wi.lin, wi.ang);
                             wi.apply = true;
                           });
  }
  lanes.barrier();
  if (mine && i == 0) {
    if (contacts > 0) {
      arena_apply_wheel_impulses(
          sc, w.dyn_body(a.cars[c].vehicle.chassis_body_idx), c);
    }
    car_update_contact_state(a.cars[c], w);
  }
  lanes.barrier();
}

template <typename Lanes>
RS_HD void arena_tick_cars(const ArenaStatic &st, Arena &a, StepScratch &sc,
                           const Lanes &lanes) {
  if (lanes.size == 1) {
    arena_car_pre_ticks_in_order(st, a, nullptr);
    return;
  }
  if (lanes.leader()) {
    sc.car_plan = arena_rays_precomputable(st, a) ? CarPhasePlan::Precast
                                                  : CarPhasePlan::Sequential;
  }
  lanes.barrier();
  World w = arena_world(st, a);
  bool dynamic_hit = false;
  if (sc.car_plan != CarPhasePlan::Sequential) {
    if constexpr (Lanes::k_cooperative) {
      if (lanes.size >= k_num_wheels * a.num_cars) {
        for (uint32_t item = lanes.lane; item < k_num_wheels * a.num_cars;
             item += lanes.size) {
          uint32_t c = item / k_num_wheels;
          int wheel = static_cast<int>(item % k_num_wheels);
          VehicleHit hit;
          a.cars[c].vehicle.cast_wheel_ray(w, lanes.car_ray_lanes(), hit);
          sc.rays[c][wheel] = hit;
          dynamic_hit = dynamic_hit || ray_hit_dynamic(w, hit);
        }
      }
    }
    if (lanes.size < k_num_wheels * a.num_cars) {
      for (uint32_t c = lanes.lane; c < a.num_cars; c += lanes.size) {
        const char *car_bytes = reinterpret_cast<const char *>(&a.cars[c]);
        for (uint32_t l = 0; l < (sizeof(Car) + 127u) / 128u; ++l) {
          prefetch_l1(car_bytes + 128u * l);
        }
        VehicleHit hits[k_num_wheels];
        a.cars[c].vehicle.cast_wheel_rays(w, hits);
        for (int i = 0; i < k_num_wheels; ++i) {
          sc.rays[c][i] = hits[i];
          dynamic_hit = dynamic_hit || ray_hit_dynamic(w, hits[i]);
        }
      }
    }
  }
  uint32_t dynamic_hits = lanes.group_ballot(dynamic_hit);
  if (lanes.leader() && sc.car_plan == CarPhasePlan::Precast &&
      dynamic_hits == 0) {
    sc.car_plan = CarPhasePlan::Parallel;
  }
  lanes.barrier();
  bool wheel_lanes = lanes.size >= k_num_wheels * a.num_cars;
  bool per_wheel = sc.car_plan == CarPhasePlan::Parallel && wheel_lanes;
  if (sc.car_plan == CarPhasePlan::Parallel) {
    for (uint32_t c = lanes.lane; c < a.num_cars; c += lanes.size) {
      sc.was_on_ground[c] = a.cars[c].state.cache.is_on_ground;
      sc.vehicle_run[c] = car_pre_tick_update(
          a.cars[c], w, a.rng, st.mode, a.mutators, sc.rays[c],
          per_wheel ? &sc.real_throttle[c] : nullptr);
    }
  } else if (lanes.leader()) {
    arena_car_pre_ticks_in_order(
        st, a, sc.car_plan == CarPhasePlan::Precast ? &sc : nullptr);
  }
  lanes.barrier();
  if (wheel_lanes) {
    arena_tick_vehicle_wheels(a, sc, w, lanes, per_wheel);
  }
  if (!lanes.leader() || sc.car_plan != CarPhasePlan::Parallel) {
    return;
  }
  for (uint32_t c = 0; c < a.num_cars; ++c) {
    if (!sc.was_on_ground[c] && a.cars[c].state.cache.is_on_ground) {
      arena_push_car_landed(a, c);
    }
  }
}

RS_HD void arena_tick_contacts(const ArenaStatic &st, Arena &a,
                               StepScratch &sc) {
  using namespace consts;
  World w = arena_world(st, a);
  if (a.ball_only) {
    for (int i = 0; i < sc.num_ball_world; ++i) {
      BallWorldRecord r = sc.ball_world[i];
      Vec3A contact_point = r.point * bt_to_uu;
      ball_on_world_hit(a.ball, w.dyn_body(r.rb), st.mode, r.normal);
      arena_push_event(a, ArenaEventKind::BallHitWorld, 0, 0, false,
                       contact_point, r.normal);
    }
  } else {
    for (int i = 0; i < sc.num_records; ++i) {
      ContactRecord c = sc.records[i];
      const RigidBody &rb_a = w.body(c.a);
      const RigidBody &rb_b = w.body(c.b);
      UserInfoType ua = rb_a.user_idx;
      UserInfoType ub = rb_b.user_idx;
      uint32_t pa = rb_a.user_pointer;
      uint32_t pb = rb_b.user_pointer;
      if (ua == UserInfoType::Car) {
        if (ub == UserInfoType::Ball) {
          RigidBody &brb = w.dyn_body(a.ball.rigid_body_idx);
          Vec3A vel_before = brb.lin_vel;
          ball_on_hit(a.ball, a.cars[pa], st.mode, a.mutators, a.tick_count,
                      brb, st.world.sin_table);
          Vec3A contact_point =
              (c.is_swap ? c.pos_world_on_a : c.pos_world_on_b) * bt_to_uu;
          Vec3A extra_hit_vel = (brb.lin_vel - vel_before) * bt_to_uu;
          arena_push_event(a, ArenaEventKind::CarHitBall, pa, 0, false,
                           contact_point, extra_hit_vel);
        } else if (ub == UserInfoType::Car) {
          arena_on_car_car(a, pa, pb, c);
        } else {
          arena_push_event(a, ArenaEventKind::CarHitWorld, pa, 0, false,
                           c.pos_world_on_b * bt_to_uu, c.normal_world_on_b);
          a.cars[pa].state.cache.has_world_contact_normal = true;
          a.cars[pa].state.cache.world_contact_normal = c.normal_world_on_b;
        }
      } else if (ua == UserInfoType::Ball) {
        if (ub == UserInfoType::DropshotTile) {
          ball_on_dropshot_tile_collision(a.ball, a.tiles, pb, a.tick_count);
        } else if (ub == UserInfoType::None) {
          Vec3A contact_point = c.pos_world_on_b * bt_to_uu;
          ball_on_world_hit(a.ball, w.dyn_body(c.a), st.mode,
                            c.normal_world_on_b);
          arena_push_event(a, ArenaEventKind::BallHitWorld, 0, 0, false,
                           contact_point, c.normal_world_on_b);
        }
      }
    }
    arena_update_world_contact_normal(a, sc);
  }
}

RS_HD bool arena_pads_active(const ArenaStatic &st, const Arena &a) {
  return !a.ball_only && a.mutators.boost_pads_enabled && st.has_pads;
}

RS_HD void arena_give_pad_boosts(const ArenaStatic &st, Arena &a,
                                 StepScratch &sc) {
  a.pad_lists_unchecked = false;
  if (!pad_any_touching(sc.pads, a.num_cars)) {
    return;
  }
  for (uint32_t c = 0; c < a.num_cars; ++c) {
    int32_t pad = pad_maybe_give_car_boost(
        st.pads, a.pads, sc.pads, a.cars[c].state, a.mutators, a.tick_count, c);
    if (pad >= 0) {
      arena_push_event(a, ArenaEventKind::CarPickupBoost, c,
                       static_cast<uint32_t>(pad), false, Vec3A::zero(),
                       Vec3A::zero());
      pad_refresh_winner(a.pads, sc.pads, static_cast<uint32_t>(pad), a.cars,
                         a.num_cars, a.mutators.car_max_boost_amount);
    }
  }
}

template <typename Lanes>
RS_HD void arena_tick_pads(const ArenaStatic &st, Arena &a, StepScratch &sc,
                           const Lanes &lanes) {
  bool pads = arena_pads_active(st, a);
  if (!Lanes::k_cooperative || lanes.size == 1) {
    if (pads) {
      pad_update_membership(st.pads, a.pads, sc.pads, a.cars, a.num_cars,
                            a.mutators.car_max_boost_amount, a.touched_pads);
      arena_give_pad_boosts(st, a, sc);
    }
    return;
  }
  if (lanes.leader() && pads) {
    pad_begin_update(sc.pads, a.num_cars);
  }
  lanes.barrier();
  if (pads) {
    uint32_t first = a.pad_lists_unchecked ? 0 : lanes.lane;
    uint32_t stride = a.pad_lists_unchecked ? 1 : lanes.size;
    if (!a.pad_lists_unchecked || lanes.leader()) {
      uint32_t rank = 0;
      for (uint64_t m = a.touched_pads; m != 0; m &= m - 1, ++rank) {
        if (rank % stride == first) {
          pad_recheck_touching(st.pads, a.pads, sc.pads, a.cars, a.num_cars,
                               lowest_set_bit(m));
        }
      }
    }
  }
  lanes.barrier();
  if (lanes.leader() && pads) {
    pad_prune_touched(a.pads, a.touched_pads);
    pad_enter_touching(st.pads, a.pads, sc.pads, a.cars, a.num_cars,
                       a.touched_pads);
  }
  lanes.barrier();
  if (pads) {
    uint32_t rank = 0;
    for (uint64_t m = a.touched_pads; m != 0; m &= m - 1, ++rank) {
      if (rank % lanes.size == lanes.lane) {
        pad_refresh_winner(a.pads, sc.pads, lowest_set_bit(m), a.cars,
                           a.num_cars, a.mutators.car_max_boost_amount);
      }
    }
  }
  lanes.barrier();
  if (lanes.leader() && pads) {
    arena_give_pad_boosts(st, a, sc);
  }
}

RS_HD void arena_tick_finish(const ArenaStatic &st, Arena &a) {
  using namespace consts;
  World w = arena_world(st, a);
  ball_finish_physics_tick(a.ball, w.dyn_body(a.ball.rigid_body_idx));
  if (st.mode == GameMode::Dropshot &&
      a.ball.state.ds_info.has_last_damage_tick &&
      a.ball.state.ds_info.last_damage_tick == a.tick_count) {
    arena_update_tile_flags(a);
  }
  a.tick_count += 1;
  arena_count_events(a);
  if (!a.counters.goal_scored && arena_is_ball_scored(st, a)) {
    a.counters.goal_scored = true;
    a.counters.goal_tick = a.tick_count;
  }
}

RS_HD void arena_setup_gate_jobs(const World &w, StepScratch &sc,
                                 const LaneGroup &lanes) {
  uint32_t jobs = 0;
  for (uint32_t base = 0; base < w.a->num_pairs; base += lanes.size) {
    uint32_t i = base + lanes.lane;
    bool gate = i < w.a->num_pairs && sc.pair_plans[i] == PairPlan::Gate;
    uint32_t gates = lanes.group_ballot(gate);
    uint32_t j = jobs + lanes.team_rank(gates);
    if (gate && j < k_max_gate_jobs) {
      gate_job_setup(w, i, sc.gate_jobs[j]);
    }
    jobs += popcount32(gates);
  }
  if (lanes.leader()) {
    sc.num_gate_jobs = jobs < k_max_gate_jobs ? jobs : k_max_gate_jobs;
    sc.gate_items.count = 0;
  }
}

RS_HD bool aabb_contains(const Aabb &outer, const Aabb &inner) {
  return outer.min.cmple(inner.min).all() && inner.max.cmple(outer.max).all();
}

RS_HD CollectMemo *collect_memo_find(StepScratch &sc, const GateJob &job) {
  CollectMemoOwner &owner = sc.collect_memo[job.memo_owner];
  for (CollectMemo &memo : owner.slots) {
    if (memo.valid && memo.mesh == job.memo_mesh &&
        aabb_contains(memo.outer, job.params.local_convex_aabb)) {
      return &memo;
    }
  }
  return nullptr;
}

RS_HD Aabb collect_memo_outer(const Aabb &query) {
  Vec3A margin = Vec3A::splat(k_collect_memo_margin);
  return {query.min - margin, query.max + margin};
}

RS_HD bool arena_collect_job_leaves(const LaneGroup &lanes, StepScratch &sc,
                                    const GateJob &job, const Aabb &query,
                                    uint32_t &found) {
  const BvhView &bvh = *job.bvh;
  found = 0;
  if (!query.intersects(bvh.aabb)) {
    return true;
  }
  const BvhWideNode *nodes = bvh.nodes;
  const Aabb box = query;
  if (lanes.leader()) {
    sc.collect_frontier[0][0] = 0;
  }
  lanes.group_sync();
  uint32_t cur = 0;
  uint32_t size = 1;
  while (size != 0) {
    uint32_t next = 0;
    for (uint32_t base = 0; base < size; base += lanes.size) {
      uint32_t e = base + lanes.lane;
      uint32_t mask = 0;
      uint32_t children[4] = {0, 0, 0, 0};
      if (e < size) {
        const BvhWideNode &node = nodes[sc.collect_frontier[cur][e]];
        mask = node.intersection_mask(box);
        for (int c = 0; c < 4; ++c) {
          children[c] = node.children[c];
        }
      }
      for (int c = 0; c < 4; ++c) {
        bool has = ((mask >> c) & 1u) != 0;
        bool leaf = has && (children[c] & k_bvh_leaf_bit) != 0;
        bool inner = has && !leaf;
        uint32_t leaf_ballot = lanes.group_ballot(leaf);
        uint32_t inner_ballot = lanes.group_ballot(inner);
        if (leaf) {
          uint32_t slot = found + lanes.team_rank(leaf_ballot);
          if (slot < k_max_gate_triangles) {
            sc.collect_leaves[slot] = children[c] & ~k_bvh_leaf_bit;
          }
        }
        if (inner) {
          uint32_t slot = next + lanes.team_rank(inner_ballot);
          if (slot < k_collect_frontier) {
            sc.collect_frontier[cur ^ 1][slot] = children[c];
          }
        }
        found += popcount32(leaf_ballot);
        next += popcount32(inner_ballot);
      }
    }
    lanes.group_sync();
    if (found > k_max_gate_triangles || next > k_collect_frontier) {
      return false;
    }
    cur ^= 1;
    size = next;
  }
  return true;
}

RS_HD void arena_collect_gate_items(StepScratch &sc, const LaneGroup &lanes) {
  GateItems &items = sc.gate_items;
  if (lanes.size < k_parallel_collect_lanes) {
    if (lanes.leader()) {
      for (uint32_t j = 0; j < sc.num_gate_jobs; ++j) {
        if (sc.gate_jobs[j].mesh) {
          gate_collect_job_serial(sc.gate_jobs[j], j, items);
        }
      }
    }
    return;
  }
  for (uint32_t j = 0; j < sc.num_gate_jobs; ++j) {
    GateJob &job = sc.gate_jobs[j];
    if (!job.mesh) {
      continue;
    }
    CollectMemo *memo = collect_memo_find(sc, job);
    if (!memo) {
      Aabb outer = collect_memo_outer(job.params.local_convex_aabb);
      uint32_t wide = 0;
      if (arena_collect_job_leaves(lanes, sc, job, outer, wide) &&
          wide <= k_collect_memo_leaves) {
        CollectMemoOwner &owner = sc.collect_memo[job.memo_owner];
        CollectMemo &slot = owner.slots[owner.next % k_collect_memo_slots];
        for (uint32_t e = lanes.lane; e < wide; e += lanes.size) {
          uint32_t leaf = sc.collect_leaves[e];
          uint32_t rank = 0;
          for (uint32_t k = 0; k < wide; ++k) {
            rank += sc.collect_leaves[k] < leaf ? 1u : 0u;
          }
          slot.leaves[rank] = leaf;
        }
        lanes.group_sync();
        if (lanes.leader()) {
          slot.valid = true;
          slot.mesh = job.memo_mesh;
          slot.count = wide;
          slot.outer = outer;
          owner.next = (owner.next + 1) % k_collect_memo_slots;
        }
        lanes.group_sync();
        memo = &slot;
      }
    }
    if (memo) {
      uint32_t start = items.count;
      uint32_t room = k_max_gate_triangles - start;
      uint32_t hits = 0;
      const BvhWideLeaf *leaves = job.bvh->leaves;
      const uint32_t *memo_leaves = memo->leaves;
      const Aabb box = job.params.local_convex_aabb;
      uint32_t memo_count = memo->count;
      for (uint32_t base = 0; base < memo_count; base += lanes.size) {
        uint32_t e = base + lanes.lane;
        bool hit = false;
        uint32_t tri = 0;
        if (e < memo_count) {
          const BvhWideLeaf &leaf = leaves[memo_leaves[e]];
          hit = leaf.bounds.intersects(box);
          tri = leaf.leaf_idx;
        }
        uint32_t ballot = lanes.group_ballot(hit);
        if (hit) {
          uint32_t rank = hits + lanes.team_rank(ballot);
          if (rank < room) {
            items.job[start + rank] = static_cast<uint8_t>(j);
            items.tri_idx[start + rank] = tri;
          }
        }
        hits += popcount32(ballot);
      }
      lanes.group_sync();
      if (lanes.leader()) {
        uint32_t stored = hits < room ? hits : room;
        job.items_complete = hits <= room;
        job.first_item = start;
        job.num_items = stored;
        items.count = start + stored;
      }
      lanes.group_sync();
      continue;
    }
    uint32_t found = 0;
    bool listed = arena_collect_job_leaves(lanes, sc, job,
                                           job.params.local_convex_aabb, found);
    uint32_t start = items.count;
    if (!listed) {
      if (lanes.leader()) {
        gate_collect_job_serial(job, j, items);
      }
      lanes.group_sync();
      continue;
    }
    uint32_t room = k_max_gate_triangles - start;
    const BvhWideLeaf *leaves = job.bvh->leaves;
    for (uint32_t e = lanes.lane; e < found; e += lanes.size) {
      uint32_t leaf = sc.collect_leaves[e];
      uint32_t rank = 0;
      for (uint32_t k = 0; k < found; ++k) {
        rank += sc.collect_leaves[k] < leaf ? 1u : 0u;
      }
      if (rank < room) {
        items.job[start + rank] = static_cast<uint8_t>(j);
        items.tri_idx[start + rank] = leaves[leaf].leaf_idx;
      }
    }
    lanes.group_sync();
    if (lanes.leader()) {
      uint32_t stored = found < room ? found : room;
      job.items_complete = found <= room;
      job.first_item = start;
      job.num_items = stored;
      items.count = start + stored;
    }
    lanes.group_sync();
  }
}

inline constexpr uint32_t k_max_team_groups = 32;

RS_HD void arena_test_team_triangles(const LaneGroup &lanes) {
  uint32_t groups = lanes.team_size / lanes.size;
  uint32_t prefix[k_max_team_groups + 1];
  prefix[0] = 0;
  for (uint32_t g = 0; g < groups; ++g) {
    prefix[g + 1] = prefix[g] + lanes.team_scratch[g].gate_items.count;
  }
  uint32_t total = prefix[groups];
  TeamSatScratch &ts = *lanes.team_sat;
  uint32_t claimed = 0;
  uint32_t group = 0;
  for (uint32_t base = 0; base < total; base += lanes.team_size) {
    uint32_t k = base + lanes.team_lane;
    bool needs_pairs = false;
    uint32_t item = 0;
    Vec3A q[3];
    SatSweepState state;
    if (k < total) {
      while (k >= prefix[group + 1]) {
        ++group;
      }
      item = k - prefix[group];
      StepScratch &gs = lanes.team_scratch[group];
      needs_pairs =
          gate_test_item_begin(gs.gate_jobs, gs.gate_items, item, q, state);
    }
    uint32_t ballot = lanes.team_ballot(needs_pairs);
    if (needs_pairs) {
      uint32_t slot = claimed + lanes.team_rank(ballot);
      StepScratch &gs = lanes.team_scratch[group];
      if (slot < k_team_sat_slots) {
        Vec3A half = gs.gate_jobs[gs.gate_items.job[item]].params.half;
        ts.slots[slot] = {group, item, half, {q[0], q[1], q[2]}, state};
      } else {
        gate_test_item_serial_finish(gs.gate_jobs, gs.gate_items, item, q,
                                     state);
      }
    }
    claimed += popcount32(ballot);
  }
  uint32_t used = claimed < k_team_sat_slots ? claimed : k_team_sat_slots;
  lanes.barrier();
  for (uint32_t u = lanes.team_lane; u < used * sat::k_sat_candidates;
       u += lanes.team_size) {
    const TeamSatSlot &s = ts.slots[u / sat::k_sat_candidates];
    int p = static_cast<int>(u % sat::k_sat_candidates);
    ts.pairs[u / sat::k_sat_candidates][p] =
        sat::sat_candidate_pair(s.half, s.q, s.state.edges, p);
  }
  lanes.barrier();
  for (uint32_t e = lanes.team_lane; e < used; e += lanes.team_size) {
    const TeamSatSlot &s = ts.slots[e];
    StepScratch &gs = lanes.team_scratch[s.group];
    gate_test_item_finish(gs.gate_jobs, gs.gate_items, s.item, s.q, s.state,
                          ts.pairs[e]);
  }
}

template <typename Lanes>
RS_HD void arena_step_tick(const ArenaStatic &st, Arena &a, StepScratch &sc,
                           const Lanes &lanes) {
  using namespace consts;
  World w = arena_world(st, a);
  if (lanes.leader()) {
    a.num_events = 0;
    for (uint32_t i = 0; i < a.phys.num_dyn_bodies; ++i) {
      a.phys.dyn_bodies[i].clear_accum_vels();
    }
  }
  lanes.barrier();
  for (uint32_t item = lanes.lane; item <= a.num_cars; item += lanes.size) {
    arena_tick_front_item(st, a, item);
  }
  lanes.barrier();
  arena_tick_cars(st, a, sc, lanes);
  lanes.barrier();
  if (lanes.leader()) {
    ball_pre_tick_update(a.ball, w.dyn_body(a.ball.rigid_body_idx), st.mode);
    sc.num_records = 0;
    sc.num_ball_world = 0;
    sc.num_groups = 0;
    a.phys.ball_only = a.ball_only;
  }
  lanes.barrier();
  for (uint32_t i = lanes.lane; i < a.phys.num_dyn_bodies; i += lanes.size) {
    dynamics_predict_body(a.phys, i, tick_time);
    if constexpr (Lanes::k_cooperative) {
      sc.body_aabbs[i] = collision_body_aabb(a.phys.dyn_bodies[i]);
    }
  }
  lanes.barrier();
  ContactTracker tracker{&w, &sc, a.ball_only, RecordSink::Records, 0};
  if (lanes.leader()) {
    if constexpr (Lanes::k_cooperative) {
      collision_apply_aabbs(w, sc.body_aabbs);
    } else {
      collision_update_aabbs(w);
    }
  }
  lanes.barrier();
  PlanSource plans = PlanSource::Inline;
  if (lanes.leader()) {
    sc.num_gate_jobs = 0;
  }
  if constexpr (Lanes::k_cooperative) {
    if (lanes.size > 1) {
      plans = PlanSource::Precomputed;
      dispatch_plan_pairs(w, tracker, lanes.lane, lanes.size);
      lanes.barrier();
      arena_setup_gate_jobs(w, sc, lanes);
      lanes.barrier();
      arena_collect_gate_items(sc, lanes);
      lanes.barrier();
      arena_test_team_triangles(lanes);
      lanes.barrier();
      dispatch_compute_jobs(w, tracker, sc.num_gate_jobs, lanes.lane,
                            lanes.size);
      lanes.barrier();
    }
  }
  if (lanes.size == 2 && lanes.lane == 1) {
    const char *manifolds = reinterpret_cast<const char *>(a.phys.manifolds);
    uint32_t lines = (a.phys.num_manifolds *
                          static_cast<uint32_t>(sizeof(PersistentManifold)) +
                      127u) /
                     128u;
    for (uint32_t l = 0; l < lines; ++l) {
      prefetch_l1(manifolds + 128u * l);
    }
  }
  if (lanes.leader()) {
    dispatch_apply_pairs(w, tracker, sc.num_gate_jobs, plans);
    dynamics_solve(w, sc.solver, tick_time);
    if constexpr (!Lanes::k_cooperative) {
      solver_finish(w, sc.solver, tick_time);
    }
  }
  lanes.barrier();
  for (uint32_t i = lanes.lane; i < a.phys.num_dyn_bodies; i += lanes.size) {
    if constexpr (Lanes::k_cooperative) {
      dynamics_finish_body(a.phys, sc.solver, i, tick_time);
    }
    dynamics_integrate_body(a.phys, i, tick_time);
  }
  lanes.barrier();
  if (lanes.leader()) {
    arena_tick_contacts(st, a, sc);
  }
  lanes.barrier();
  for (uint32_t c = lanes.lane; c < a.num_cars; c += lanes.size) {
    car_post_tick_update(a.cars[c], w);
    car_finish_physics_tick(a.cars[c], w.dyn_body(a.cars[c].rigid_body_idx));
  }
  lanes.barrier();
  arena_tick_pads(st, a, sc, lanes);
  if (lanes.leader()) {
    arena_tick_finish(st, a);
  }
  lanes.barrier();
}

}
