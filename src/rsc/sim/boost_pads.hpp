#pragma once

#include <cstdint>

#include "rsc/math/bvh.hpp"
#include "rsc/sim/car.hpp"
#include "rsc/sim/consts.hpp"

namespace rsc {

inline constexpr int k_max_pads = 48;
static_assert(k_max_pads <= 64);

struct BoostPadConfig {
  Vec3A pos;
  bool is_big;
};

struct PadLayoutView {
  uint32_t num_pads;
  const BoostPadConfig *configs;
  const Aabb *aabbs;
  BvhView bvh;
  float max_pad_z;
};

struct PadState {
  bool has_gave_tick;
  int64_t gave_tick;
  uint32_t num_touching;
  uint8_t touching[k_max_cars];
};

RS_HDI uint64_t pad_bit(uint32_t pad_idx) { return uint64_t{1} << pad_idx; }

RS_HDI float pad_cyl_radius(bool big) {
  return big ? consts::boost_pads::cyl_rad_big
             : consts::boost_pads::cyl_rad_small;
}

RS_HDI float pad_box_radius(bool big) {
  return big ? consts::boost_pads::box_rad_big
             : consts::boost_pads::box_rad_small;
}

RS_HDI float pad_max_cooldown(bool big, const MutatorConfig &mut) {
  return big ? mut.boost_pad_cooldown_big : mut.boost_pad_cooldown_small;
}

RS_HDI float pad_boost_amount(bool big, const MutatorConfig &mut) {
  return big ? mut.boost_pad_amount_big : mut.boost_pad_amount_small;
}

RS_HD bool pad_overlaps_values(Vec3A pad_pos, float radius, Vec3A pos) {
  float dx = pad_pos.x - pos.x;
  float dy = pad_pos.y - pos.y;
  float dist_sq_2d = dx * dx + dy * dy;
  return dist_sq_2d < radius * radius &&
         abs_f32(pos.z - pad_pos.z) <= consts::boost_pads::cyl_height;
}

RS_HD bool pad_can_receive_boost(const Car &car, float max_boost) {
  if (car.state.is_demoed) {
    return false;
  }
  return car.state.boost < max_boost || is_nan_f32(car.state.boost) ||
         is_nan_f32(max_boost);
}

RS_HD bool hitbox_overlaps_pad_box(Vec3A pad_pos, float box_radius,
                                   Vec3A car_center, const Car &car) {
  const Mat3A &rot = car.state.phys.rot_mat;
  const CarBodyConfig &config = car.info.config;
  Vec3A center = car_center + rot * config.hitbox_pos_offset;
  Vec3A car_axes[3] = {rot.x_axis, rot.y_axis, rot.z_axis};
  Vec3A car_half = config.hitbox_size * 0.5f;
  Vec3A pad_half{box_radius, box_radius, consts::boost_pads::box_height};
  Vec3A delta = center - pad_pos;
  const Vec3A world_axes[3] = {
      {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
  for (int k = 0; k < 15; ++k) {
    Vec3A axis;
    if (k < 3) {
      axis = world_axes[k];
    } else if (k < 6) {
      axis = car_axes[k - 3];
    } else {
      int c = (k - 6) / 3;
      int wa = (k - 6) % 3;
      axis = car_axes[c].cross(world_axes[wa]);
    }
    if (axis.length_squared() <= k_f32_epsilon) {
      continue;
    }
    float pad_radius = pad_half.dot({abs_f32(world_axes[0].dot(axis)),
                                     abs_f32(world_axes[1].dot(axis)),
                                     abs_f32(world_axes[2].dot(axis))});
    float car_radius = car_half.dot({abs_f32(car_axes[0].dot(axis)),
                                     abs_f32(car_axes[1].dot(axis)),
                                     abs_f32(car_axes[2].dot(axis))});
    if (abs_f32(delta.dot(axis)) > pad_radius + car_radius) {
      return false;
    }
  }
  return true;
}

struct PadUpdateScratch {
  int32_t touching_pad_by_car[k_max_cars];
  int32_t winners[k_max_pads];
};

RS_HD bool pad_contains(const PadState &p, uint32_t car_idx) {
  for (uint32_t i = 0; i < p.num_touching; ++i) {
    if (p.touching[i] == car_idx) {
      return true;
    }
  }
  return false;
}

RS_HD void pad_refresh_winner(PadState *pads, PadUpdateScratch &sc,
                              uint32_t pad_idx, const Car *cars,
                              uint32_t num_cars, float max_boost) {
  int32_t winner = -1;
  const PadState &p = pads[pad_idx];
  for (uint32_t i = 0; i < p.num_touching; ++i) {
    uint32_t idx = p.touching[i];
    if (idx < num_cars && pad_can_receive_boost(cars[idx], max_boost)) {
      winner = static_cast<int32_t>(idx);
      break;
    }
  }
  sc.winners[pad_idx] = winner;
}

struct PadMembershipProcessor {
  const PadLayoutView *layout;
  PadState *pads;
  PadUpdateScratch *sc;
  uint64_t *touched_pads;
  uint32_t car_idx;
  Vec3A car_pos;

  RS_HD void process_node(uint32_t pad_idx) {
    if (sc->touching_pad_by_car[car_idx] >= 0) {
      return;
    }
    const BoostPadConfig &cfg = layout->configs[pad_idx];
    if (pad_overlaps_values(cfg.pos, pad_cyl_radius(cfg.is_big), car_pos)) {
      PadState &p = pads[pad_idx];
      if (!pad_contains(p, car_idx)) {
        p.touching[p.num_touching++] = static_cast<uint8_t>(car_idx);
      }
      *touched_pads |= pad_bit(pad_idx);
      sc->touching_pad_by_car[car_idx] = static_cast<int32_t>(pad_idx);
    }
  }
};

RS_HD void pad_begin_update(PadUpdateScratch &sc, uint32_t num_cars) {
  for (uint32_t i = 0; i < num_cars; ++i) {
    sc.touching_pad_by_car[i] = -1;
  }
}

RS_HD void pad_recheck_touching(const PadLayoutView &layout, PadState *pads,
                                PadUpdateScratch &sc, const Car *cars,
                                uint32_t num_cars, uint32_t pad_idx) {
  constexpr float half_tick = 0.5f * consts::tick_time;
  PadState &p = pads[pad_idx];
  if (p.num_touching == 0) {
    return;
  }
  const BoostPadConfig &cfg = layout.configs[pad_idx];
  float radius = pad_cyl_radius(cfg.is_big);
  float box_radius = pad_box_radius(cfg.is_big);
  uint32_t kept = 0;
  for (uint32_t i = 0; i < p.num_touching; ++i) {
    uint32_t idx = p.touching[i];
    bool keep = false;
    if (idx < num_cars) {
      const Car &car = cars[idx];
      if (!car.state.is_demoed && sc.touching_pad_by_car[idx] < 0) {
        Vec3A car_pos = car.state.phys.pos - car.state.phys.vel * half_tick;
        if (pad_overlaps_values(cfg.pos, radius, car_pos) ||
            hitbox_overlaps_pad_box(cfg.pos, box_radius, car_pos, car)) {
          sc.touching_pad_by_car[idx] = static_cast<int32_t>(pad_idx);
          keep = true;
        }
      }
    }
    if (keep) {
      p.touching[kept++] = static_cast<uint8_t>(idx);
    }
  }
  p.num_touching = kept;
}

RS_HD void pad_prune_touched(const PadState *pads, uint64_t &touched_pads) {
  for (uint64_t m = touched_pads; m != 0; m &= m - 1) {
    uint32_t pad_idx = lowest_set_bit(m);
    if (pads[pad_idx].num_touching == 0) {
      touched_pads &= ~pad_bit(pad_idx);
    }
  }
}

RS_HD void pad_enter_touching(const PadLayoutView &layout, PadState *pads,
                              PadUpdateScratch &sc, const Car *cars,
                              uint32_t num_cars, uint64_t &touched_pads) {
  constexpr float half_tick = 0.5f * consts::tick_time;
  for (uint32_t car_idx = 0; car_idx < num_cars; ++car_idx) {
    const Car &car = cars[car_idx];
    if (car.state.is_demoed || car.state.phys.pos.z > layout.max_pad_z) {
      continue;
    }
    if (sc.touching_pad_by_car[car_idx] >= 0) {
      continue;
    }
    Vec3A car_pos = car.state.phys.pos - car.state.phys.vel * half_tick;
    Aabb car_aabb{car_pos, car_pos};
    PadMembershipProcessor proc{&layout,       pads,    &sc,
                                &touched_pads, car_idx, car_pos};
    layout.bvh.report_aabb_overlapping_node(proc, car_aabb);
  }
}

RS_HD bool pad_any_touching(const PadUpdateScratch &sc, uint32_t num_cars) {
  for (uint32_t i = 0; i < num_cars; ++i) {
    if (sc.touching_pad_by_car[i] >= 0) {
      return true;
    }
  }
  return false;
}

RS_HDI bool pad_update_membership(const PadLayoutView &layout, PadState *pads,
                                  PadUpdateScratch &sc, const Car *cars,
                                  uint32_t num_cars, float max_boost,
                                  uint64_t &touched_pads) {
  pad_begin_update(sc, num_cars);
  for (uint64_t m = touched_pads; m != 0; m &= m - 1) {
    pad_recheck_touching(layout, pads, sc, cars, num_cars, lowest_set_bit(m));
  }
  pad_prune_touched(pads, touched_pads);
  pad_enter_touching(layout, pads, sc, cars, num_cars, touched_pads);
  for (uint64_t m = touched_pads; m != 0; m &= m - 1) {
    pad_refresh_winner(pads, sc, lowest_set_bit(m), cars, num_cars, max_boost);
  }
  return pad_any_touching(sc, num_cars);
}

RS_HD int32_t pad_maybe_give_car_boost(const PadLayoutView &layout,
                                       PadState *pads, PadUpdateScratch &sc,
                                       CarState &car_state,
                                       const MutatorConfig &mut,
                                       uint64_t tick_count, uint32_t car_idx) {
  if (car_state.is_demoed) {
    return -1;
  }
  if (car_state.boost >= mut.car_max_boost_amount) {
    return -1;
  }
  int32_t pad_idx = sc.touching_pad_by_car[car_idx];
  if (pad_idx < 0) {
    return -1;
  }
  if (sc.winners[pad_idx] != static_cast<int32_t>(car_idx)) {
    return -1;
  }
  PadState &p = pads[pad_idx];
  bool big = layout.configs[pad_idx].is_big;
  if (p.has_gave_tick &&
      static_cast<float>(static_cast<int64_t>(tick_count) - p.gave_tick) *
              consts::tick_time <
          pad_max_cooldown(big, mut)) {
    return -1;
  }
  car_state.boost = min_f32(car_state.boost + pad_boost_amount(big, mut),
                            mut.car_max_boost_amount);
  p.has_gave_tick = true;
  p.gave_tick = static_cast<int64_t>(tick_count);
  return pad_idx;
}

RS_HD float pad_cooldown(const PadState &p, bool big, const MutatorConfig &mut,
                         uint64_t tick_count) {
  if (!p.has_gave_tick) {
    return 0.0f;
  }
  float max_cooldown = pad_max_cooldown(big, mut);
  float time_since =
      static_cast<float>(static_cast<int64_t>(tick_count) - p.gave_tick) *
      consts::tick_time;
  return max_f32_c(max_cooldown - time_since, 0.0f);
}

}
