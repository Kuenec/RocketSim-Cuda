#pragma once

#include <cstdint>

#include "rsc/bullet/dispatch.hpp"
#include "rsc/bullet/solver.hpp"
#include "rsc/bullet/world_types.hpp"

namespace rsc {

RS_HD int solver_fixed_body(SolverScratch &sc) {
  if (sc.fixed_body_id >= 0) {
    return sc.fixed_body_id;
  }
  int id = sc.num_bodies;
  sc.fixed_body_id = id;
  sc.bodies[sc.num_bodies++] = SolverBody::fixed();
  return id;
}

RS_HD int solver_get_or_init_body(World &w, SolverScratch &sc, uint32_t idx) {
  if (idx < w.num_static()) {
    return solver_fixed_body(sc);
  }
  RigidBody &rb = w.dyn_body(idx);
  if (rb.companion_id >= 0) {
    return rb.companion_id;
  }
  if (rb.inv_mass != 0.0f) {
    int id = sc.num_bodies;
    rb.companion_id = id;
    sc.bodies[sc.num_bodies++] = SolverBody::from(rb);
    return id;
  }
  int id = solver_fixed_body(sc);
  rb.companion_id = id;
  return id;
}

RS_HD bool push_contact(World &w, SolverScratch &sc,
                        const SolverConstraint &c) {
  if (sc.num_contacts >= k_max_constraints) {
    w.flag_error(k_error_solver_overflow);
    return false;
  }
  sc.contacts[sc.num_contacts++] = c;
  return true;
}

RS_HD bool push_friction(World &w, SolverScratch &sc,
                         const SolverConstraint &c) {
  if (sc.num_frictions >= k_max_constraints) {
    w.flag_error(k_error_solver_overflow);
    return false;
  }
  sc.frictions[sc.num_frictions++] = c;
  return true;
}

RS_HD void accumulate_special(SolverScratch &sc, uint32_t obj_idx, Vec3A normal,
                              float lever_len, float friction,
                              float restitution) {
  for (int i = 0; i < sc.num_specials; ++i) {
    SpecialAccumulator &acc = sc.specials[i];
    if (acc.obj_idx == obj_idx) {
      acc.total_normal += normal;
      acc.total_lever_len += lever_len;
      acc.total_friction += friction;
      acc.total_restitution += restitution;
      acc.count += 1;
      return;
    }
  }
  sc.specials[sc.num_specials++] = {obj_idx,  normal,      lever_len,
                                    friction, restitution, 1};
}

RS_HDN void solver_setup(World &w, SolverScratch &sc, float time_step) {
  bool skip = w.a->skip_separated_special_rows;
  sc.fixed_body_id = -1;
  sc.num_bodies = 0;
  sc.num_contacts = 0;
  sc.num_frictions = 0;
  sc.num_specials = 0;
  for (uint32_t i = 0; i < w.a->num_dyn_bodies; ++i) {
    RigidBody &rb = w.a->dyn_bodies[i];
    rb.companion_id = -1;
    if (!rb.is_active()) {
      continue;
    }
    int id = sc.num_bodies;
    rb.companion_id = id;
    sc.bodies[sc.num_bodies++] = SolverBody::from(rb);
  }
  for (uint32_t ai = 0; ai < w.a->num_active; ++ai) {
    PersistentManifold &m = w.a->manifolds[w.a->active[ai]];
    int id_a = solver_get_or_init_body(w, sc, m.body0_idx);
    int id_b = solver_get_or_init_body(w, sc, m.body1_idx);
    if (m.body0_idx >= w.num_static()) {
      w.dyn_body(m.body0_idx).companion_id = id_a;
    }
    if (m.body1_idx >= w.num_static()) {
      w.dyn_body(m.body1_idx).companion_id = id_b;
    }
    const RigidBody &body0 = w.body(m.body0_idx);
    const RigidBody &body1 = w.body(m.body1_idx);
    Vec3A trans0 = body0.world_trans.translation;
    Vec3A trans1 = body1.world_trans.translation;
    for (int pi = 0; pi < m.num_points; ++pi) {
      ManifoldPoint &cp = m.points[pi];
      if (cp.distance_1 > m.contact_processing_threshold) {
        continue;
      }
      SolverBody &sa = sc.bodies[id_a];
      SolverBody &sb = sc.bodies[id_b];
      Vec3A rel_pos1 = cp.pos_world_on_a - trans0;
      Vec3A rel_pos2 = cp.pos_world_on_b - trans1;
      const RigidBody *rb0 = sa.original_body >= 0 ? &body0 : nullptr;
      const RigidBody *rb1 = sb.original_body >= 0 ? &body1 : nullptr;
      if (cp.is_special) {
        if (!skip || cp.distance_1 < 0.0f) {
          SolverConstraint c = SolverConstraint::blank(id_a, id_b);
          c.friction_idx = -1;
          c.friction = cp.combined_friction;
          c.lower_limit = 0.0f;
          c.upper_limit = 1e10f;
          c.is_split_only = true;
          c.setup_split(rb0, rb1, rel_pos1, rel_pos2, cp, time_step);
          if (skip) {
            c.make_dynamic_side_a(rb0 == nullptr && rb1 != nullptr);
          }
          push_contact(w, sc, c);
        }
        bool has_side = true;
        uint32_t obj_idx = 0;
        Vec3A lever = Vec3A::zero();
        if (!body0.is_static_obj()) {
          obj_idx = body0.world_array_idx;
          lever = rel_pos1;
        } else if (!body1.is_static_obj()) {
          obj_idx = body1.world_array_idx;
          lever = rel_pos2;
        } else {
          has_side = false;
        }
        if (has_side) {
          float lever_len = lever.length();
          if (skip && sc.num_specials > 0 &&
              sc.specials[0].obj_idx == obj_idx) {
            SpecialAccumulator &acc = sc.specials[0];
            acc.total_normal += cp.normal_world_on_b;
            acc.total_lever_len += lever_len;
            acc.total_friction += cp.combined_friction;
            acc.total_restitution += cp.combined_restitution;
            acc.count += 1;
          } else {
            accumulate_special(sc, obj_idx, cp.normal_world_on_b, lever_len,
                               cp.combined_friction, cp.combined_restitution);
          }
        }
        continue;
      }
      int friction_idx = sc.num_contacts;
      Vec3A lat = calc_lat_friction_dir(cp, sa, sb, rel_pos1, rel_pos2);
      SolverConstraint c = SolverConstraint::blank(id_a, id_b);
      c.friction_idx = friction_idx;
      c.friction = cp.combined_friction;
      c.lower_limit = 0.0f;
      c.upper_limit = 1e10f;
      c.setup_contact(sa, sb, rb0, rb1, rel_pos1, rel_pos2, cp, time_step);
      push_contact(w, sc, c);
      SolverConstraint f = SolverConstraint::blank(id_a, id_b);
      f.friction_idx = friction_idx;
      f.lower_limit = -cp.combined_friction;
      f.upper_limit = cp.combined_friction;
      f.friction = cp.combined_friction;
      f.setup_friction(sa, sb, rb0, rb1, rel_pos1, rel_pos2, lat);
      push_friction(w, sc, f);
    }
  }
  w.a->num_active = 0;
  for (int i = 0; i < sc.num_specials; ++i) {
    SpecialAccumulator acc = sc.specials[i];
    float n = static_cast<float>(acc.count);
    Vec3A mean_normal = (acc.total_normal / n).normalize_or_zero();
    float mean_lever_len = acc.total_lever_len / n;
    float mean_friction = acc.total_friction / n;
    float mean_restitution = acc.total_restitution / n;
    const RigidBody &body = w.body(acc.obj_idx);
    int id_a = body.companion_id;
    int id_b = solver_fixed_body(sc);
    SolverBody &sa = sc.bodies[id_a];
    SolverBody &sb = sc.bodies[id_b];
    Vec3A rel_pos1 = mean_normal * -mean_lever_len;
    Vec3A rel_pos2 = Vec3A::zero();
    ManifoldPoint pt = ManifoldPoint::zero();
    pt.normal_world_on_b = mean_normal;
    pt.combined_friction = mean_friction;
    pt.combined_restitution = mean_restitution;
    pt.distance_1 = 0.0f;
    pt.applied_impulse = 0.0f;
    const RigidBody *rb0 = sa.original_body >= 0 ? &body : nullptr;
    int friction_idx = sc.num_contacts;
    SolverConstraint c = SolverConstraint::blank(id_a, id_b);
    c.friction_idx = friction_idx;
    c.friction = pt.combined_friction;
    c.lower_limit = 0.0f;
    c.upper_limit = 1e10f;
    c.setup_contact(sa, sb, rb0, nullptr, rel_pos1, rel_pos2, pt, time_step);
    push_contact(w, sc, c);
    Vec3A lat = calc_lat_friction_dir(pt, sa, sb, rel_pos1, rel_pos2);
    SolverConstraint f = SolverConstraint::blank(id_a, id_b);
    f.friction_idx = friction_idx;
    f.lower_limit = -pt.combined_friction;
    f.upper_limit = pt.combined_friction;
    f.friction = pt.combined_friction;
    f.setup_friction(sa, sb, rb0, nullptr, rel_pos1, rel_pos2, lat);
    push_friction(w, sc, f);
  }
}

RS_HDN void solver_split_iterations(World &w, SolverScratch &sc) {
  bool skip = w.a->skip_separated_special_rows;
  int row_count = sc.num_contacts;
  for (int i = 0; i < row_count; ++i) {
    sc.split_should_run[i] = true;
  }
  int remaining = row_count;
  for (int it = 0; it < k_solver_iterations; ++it) {
    if (remaining == 0) {
      break;
    }
    for (int i = 0; i < row_count; ++i) {
      if (!sc.split_should_run[i]) {
        continue;
      }
      SolverConstraint &c = sc.contacts[i];
      float residual =
          skip ? c.resolve_split_one_dynamic(sc.bodies[c.solver_body_id_a])
               : c.resolve_split_two(sc.bodies[c.solver_body_id_a],
                                     sc.bodies[c.solver_body_id_b]);
      if (residual * residual == 0.0f) {
        sc.split_should_run[i] = false;
        --remaining;
      }
    }
  }
}

RS_HDN float solver_single_iteration(World &w, SolverScratch &sc) {
  if (w.a->skip_separated_special_rows) {
    if (sc.num_contacts == 0) {
      return 0.0f;
    }
    SolverConstraint &c = sc.contacts[sc.num_contacts - 1];
    float residual =
        c.resolve_lower_limit_one_dynamic(sc.bodies[c.solver_body_id_a]);
    float lsr = residual * residual;
    for (int i = 0; i < sc.num_frictions; ++i) {
      SolverConstraint &f = sc.frictions[i];
      float total = sc.contacts[f.friction_idx].applied_impulse;
      if (total <= 0.0f) {
        continue;
      }
      float limit = f.friction * total;
      f.lower_limit = -limit;
      f.upper_limit = limit;
      float r = f.resolve_generic_one_dynamic(sc.bodies[f.solver_body_id_a]);
      lsr = max_f32(r * r, lsr);
    }
    return lsr;
  }
  float lsr = 0.0f;
  for (int i = 0; i < sc.num_contacts; ++i) {
    SolverConstraint &c = sc.contacts[i];
    if (c.is_split_only) {
      continue;
    }
    float r = c.resolve_two(sc.bodies[c.solver_body_id_a],
                            sc.bodies[c.solver_body_id_b], false);
    lsr = max_f32(r * r, lsr);
  }
  for (int i = 0; i < sc.num_frictions; ++i) {
    SolverConstraint &f = sc.frictions[i];
    float total = sc.contacts[f.friction_idx].applied_impulse;
    if (total <= 0.0f) {
      continue;
    }
    float limit = f.friction * total;
    f.lower_limit = -limit;
    f.upper_limit = limit;
    float r = f.resolve_two(sc.bodies[f.solver_body_id_a],
                            sc.bodies[f.solver_body_id_b], true);
    lsr = max_f32(r * r, lsr);
  }
  return lsr;
}

RS_HDN void solver_finish(World &w, SolverScratch &sc, float time_step) {
  for (int i = 0; i < sc.num_bodies; ++i) {
    SolverBody &s = sc.bodies[i];
    if (s.original_body < 0) {
      continue;
    }
    RigidBody &body = w.dyn_body(static_cast<uint32_t>(s.original_body));
    s.lin_vel += s.delta_lin_vel;
    s.ang_vel += s.delta_ang_vel;
    if (s.push_vel.length_squared() != 0.0f ||
        s.turn_vel.length_squared() != 0.0f) {
      Affine3A world_trans = body.world_trans;
      if (body.collision_flags & k_flag_no_angular_motion) {
        integrate_trans_no_rot(world_trans.translation, s.push_vel, time_step);
      } else {
        Quat world_rot = body.world_quat;
        integrate_trans(world_trans, world_rot, s.push_vel,
                        s.turn_vel * k_split_impulse_turn_erp, time_step);
      }
      body.set_world_trans(world_trans);
    }
    Vec3A lin = s.lin_vel + s.external_force_impulse;
    Vec3A ang = s.ang_vel + s.external_torque_impulse;
    bool was_sleeping = body.activation == ActivationState::Sleeping;
    body.lin_vel = lin;
    body.ang_vel = ang;
    if (was_sleeping && (lin != Vec3A::zero() || ang != Vec3A::zero())) {
      body.set_activation_state(ActivationState::WantsDeactivation);
      body.deactivation_time = 0.0f;
    }
  }
}

RS_HD void solve_group(World &w, SolverScratch &sc, float time_step) {
  solver_setup(w, sc, time_step);
  solver_split_iterations(w, sc);
  for (int it = 0; it < k_solver_iterations; ++it) {
    if (solver_single_iteration(w, sc) == 0.0f) {
      break;
    }
  }
  solver_finish(w, sc, time_step);
}

RS_HD uint32_t island_find_recursive(uint32_t *parents, uint32_t index) {
  uint32_t root = index;
  while (parents[root] != root) {
    root = parents[root];
  }
  uint32_t cur = index;
  while (parents[cur] != root) {
    uint32_t next = parents[cur];
    parents[cur] = root;
    cur = next;
  }
  return root;
}

RS_HD uint32_t island_find(uint32_t *parents, uint32_t index) {
  uint32_t root = index;
  while (parents[root] != root) {
    root = parents[root];
  }
  uint32_t cur = index;
  while (parents[cur] != cur) {
    uint32_t parent = parents[cur];
    parents[cur] = root;
    cur = parent;
  }
  return root;
}

RS_HD void hoare_sort(uint16_t *indices, const int32_t *keys, int lo, int hi) {
  int range_lo[k_max_manifolds];
  int range_hi[k_max_manifolds];
  int num_ranges = 0;
  if (lo < hi) {
    range_lo[0] = lo;
    range_hi[0] = hi;
    num_ranges = 1;
  }
  while (num_ranges > 0) {
    --num_ranges;
    int first = range_lo[num_ranges];
    int last = range_hi[num_ranges];
    int i = first;
    int j = last;
    int32_t pivot = keys[indices[(first + last) / 2]];
    for (;;) {
      while (keys[indices[i]] < pivot) {
        ++i;
      }
      while (pivot < keys[indices[j]]) {
        --j;
      }
      if (i > j) {
        break;
      }
      uint16_t t = indices[i];
      indices[i] = indices[j];
      indices[j] = t;
      ++i;
      if (j == 0) {
        break;
      }
      --j;
      if (i > j) {
        break;
      }
    }
    if (first < j) {
      range_lo[num_ranges] = first;
      range_hi[num_ranges] = j;
      ++num_ranges;
    }
    if (i < last) {
      range_lo[num_ranges] = i;
      range_hi[num_ranges] = last;
      ++num_ranges;
    }
  }
}

RS_HDN void island_sort_manifolds(World &w) {
  ArenaPhysics &a = *w.a;
  uint32_t ns = w.num_static();
  uint32_t nd = a.num_dyn_bodies;
  uint32_t parents[k_max_dyn_bodies];
  int32_t tags[k_max_dyn_bodies];
  for (uint32_t i = 0; i < nd; ++i) {
    parents[i] = i;
  }
  for (uint32_t p = 0; p < a.num_pairs; ++p) {
    const BroadphaseProxy &p0 = w.proxy(a.pairs[p].proxy0);
    const BroadphaseProxy &p1 = w.proxy(a.pairs[p].proxy1);
    if ((p0.collision_filter_group & k_group_static) != 0 ||
        (p1.collision_filter_group & k_group_static) != 0) {
      continue;
    }
    uint32_t b0 = p0.client_obj_idx;
    uint32_t b1 = p1.client_obj_idx;
    if (b0 < ns || b1 < ns || !w.has_contact_response(b0) ||
        !w.has_contact_response(b1)) {
      continue;
    }
    uint32_t r0 = island_find_recursive(parents, b0 - ns);
    uint32_t r1 = island_find_recursive(parents, b1 - ns);
    if (r0 != r1) {
      parents[r0] = r1;
    }
  }
  for (uint32_t i = 0; i < nd; ++i) {
    tags[i] = static_cast<int32_t>(island_find(parents, i));
  }
  bool island_active[k_max_dyn_bodies];
  for (uint32_t i = 0; i < nd; ++i) {
    island_active[i] = false;
  }
  for (uint32_t i = 0; i < nd; ++i) {
    if (a.dyn_bodies[i].activation == ActivationState::Active) {
      island_active[tags[i]] = true;
    }
  }
  for (uint32_t i = 0; i < nd; ++i) {
    if (!island_active[tags[i]] &&
        a.dyn_bodies[i].activation == ActivationState::WantsDeactivation) {
      a.dyn_bodies[i].sleep();
    }
  }
  if (a.num_active <= 1) {
    return;
  }
  int32_t keys[k_max_manifolds];
  for (uint32_t i = 0; i < a.num_active; ++i) {
    const PersistentManifold &m = a.manifolds[a.active[i]];
    int32_t tag0 = m.body0_idx >= ns ? tags[m.body0_idx - ns] : -1;
    int32_t tag1 = m.body1_idx >= ns ? tags[m.body1_idx - ns] : -1;
    keys[a.active[i]] = tag0 >= 0 ? tag0 : tag1;
  }
  for (uint32_t i = 1; i < a.num_active; ++i) {
    uint16_t v = a.active[i];
    uint32_t j = i;
    while (j > 0 && a.active[j - 1] > v) {
      a.active[j] = a.active[j - 1];
      --j;
    }
    a.active[j] = v;
  }
  hoare_sort(a.active, keys, 0, static_cast<int>(a.num_active) - 1);
}

RS_HD void dynamics_predict_body(ArenaPhysics &a, uint32_t i, float time_step) {
  RigidBody &body = a.dyn_bodies[i];
  if (body.is_active()) {
    body.add_linear_impulse(a.gravity * time_step, false, true);
  }
  if (body.activation == ActivationState::Sleeping) {
    body.interp_world_trans = body.world_trans;
    return;
  }
  body.apply_damping();
  body.interp_world_trans = body.predict_integration_trans(time_step);
}

RS_HD void dynamics_solve(World &w, SolverScratch &sc, float time_step) {
  ArenaPhysics &a = *w.a;
  island_sort_manifolds(w);
  a.skip_separated_special_rows =
      a.ball_only && a.num_dyn_bodies == 1 &&
      a.dyn_bodies[0].user_idx == UserInfoType::Ball;
  solve_group(w, sc, time_step);
}

RS_HD void dynamics_integrate_body(ArenaPhysics &a, uint32_t i,
                                   float time_step) {
  RigidBody &body = a.dyn_bodies[i];
  if (body.is_active()) {
    body.set_center_of_mass_trans(body.predict_integration_trans(time_step));
  }
  body.update_activation_state(time_step);
}

}
