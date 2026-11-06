#pragma once

#include <cstdint>

#include "rsc/bullet/algorithms.hpp"
#include "rsc/bullet/box_box.hpp"
#include "rsc/bullet/broadphase.hpp"
#include "rsc/bullet/mesh_algorithms.hpp"
#include "rsc/bullet/world_types.hpp"

namespace rsc {

RS_HD int dispatcher_lookup(const ArenaPhysics &a, uint32_t b0, uint32_t b1) {
  for (uint32_t i = 0; i < a.num_manifolds; ++i) {
    if (a.manifolds[i].matches(b0, b1)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

RS_HD int dispatcher_insert(World &w, const PersistentManifold &m) {
  ArenaPhysics &a = *w.a;
  if (a.num_manifolds >= static_cast<uint32_t>(k_max_manifolds)) {
    w.flag_error(k_error_manifolds_overflow);
    return -1;
  }
  a.manifolds[a.num_manifolds].copy_used_from(m);
  return static_cast<int>(a.num_manifolds++);
}

RS_HD void dispatcher_release_manifold(World &w, uint32_t body0,
                                       uint32_t body1) {
  ArenaPhysics &a = *w.a;
  int index = dispatcher_lookup(a, body0, body1);
  if (index < 0) {
    return;
  }
  uint32_t last = a.num_manifolds - 1;
  uint32_t kept = 0;
  for (uint32_t i = 0; i < a.num_active; ++i) {
    if (a.active[i] != index) {
      a.active[kept++] = a.active[i];
    }
  }
  a.num_active = kept;
  if (static_cast<uint32_t>(index) != last) {
    a.manifolds[index].copy_used_from(a.manifolds[last]);
  }
  a.num_manifolds = last;
  if (static_cast<uint32_t>(index) != last) {
    for (uint32_t i = 0; i < a.num_active; ++i) {
      if (a.active[i] == last) {
        a.active[i] = static_cast<uint16_t>(index);
      }
    }
  }
}

RS_HD void dispatcher_clear(ArenaPhysics &a) {
  a.num_manifolds = 0;
  a.num_active = 0;
}

RS_HD void push_active(World &w, int idx) {
  if (idx < 0) {
    return;
  }
  w.a->active[w.a->num_active++] = static_cast<uint16_t>(idx);
}

RS_HD bool is_mesh_convex_pair(const RigidBody &a, const RigidBody &b) {
  bool a_mesh = a.shape.kind == ShapeKind::TriangleMesh;
  bool b_mesh = b.shape.kind == ShapeKind::TriangleMesh;
  bool a_conv = a.shape.kind == ShapeKind::Sphere ||
                a.shape.kind == ShapeKind::ConvexHull;
  bool b_conv = b.shape.kind == ShapeKind::Sphere ||
                b.shape.kind == ShapeKind::ConvexHull;
  return (a_conv && b_mesh) || (a_mesh && b_conv);
}

template <typename Callback>
RS_HD void process_mesh_collision_into(const World &w, const RigidBody &convex,
                                       const RigidBody &concave,
                                       PersistentManifold &m,
                                       Callback &callback) {
  if (convex.shape.kind == ShapeKind::Sphere) {
    sphere_concave_into(w, convex, convex.shape.sphere, concave, m, callback);
  } else {
    convex_concave_into(w, convex, concave, m, callback);
  }
}

template <typename Callback>
RS_HDN bool process_collision_inner(const World &w, const RigidBody &a,
                                    const RigidBody &b, Callback &callback,
                                    PersistentManifold &out) {
  if (is_mesh_convex_pair(a, b)) {
    const RigidBody &convex = a.shape.kind == ShapeKind::TriangleMesh ? b : a;
    const RigidBody &concave = a.shape.kind == ShapeKind::TriangleMesh ? a : b;
    PersistentManifold m = PersistentManifold::make(convex, concave);
    process_mesh_collision_into(w, convex, concave, m, callback);
    if (m.num_points != 0) {
      out = m;
      return true;
    }
    return false;
  }
  ShapeKind ka = a.shape.kind;
  ShapeKind kb = b.shape.kind;
  switch (ka) {
  case ShapeKind::StaticPlane:
    if (kb == ShapeKind::Sphere || kb == ShapeKind::ConvexHull) {
      BodyWrapper wrap{&b, b.world_trans, nullptr};
      return convex_plane(w, wrap, a, a.shape.plane, callback, out);
    }
    if (kb == ShapeKind::Compound) {
      Affine3A child = b.world_trans * b.shape.compound.child_trans;
      if (!b.shape.compound.child_shape.get_aabb(child).intersects(
              a.shape.get_aabb(a.world_trans))) {
        return false;
      }
      return compound_after_child_gate(w, b, b.shape.compound, child, a,
                                       callback, out);
    }
    return false;
  case ShapeKind::Sphere:
    if (kb == ShapeKind::StaticPlane) {
      BodyWrapper wrap{&a, a.world_trans, nullptr};
      return convex_plane(w, wrap, b, b.shape.plane, callback, out);
    }
    if (kb == ShapeKind::Compound) {
      return sphere_obb(a, a.shape.sphere, b, b.shape.compound, callback, out);
    }
    if (kb == ShapeKind::ConvexHull) {
      BodyWrapper wrap{&a, a.world_trans, nullptr};
      return convex_convex(w, wrap, b, callback, out);
    }
    return false;
  case ShapeKind::TriangleMesh:
    if (kb == ShapeKind::Compound) {
      Affine3A child = b.world_trans * b.shape.compound.child_trans;
      if (!b.shape.compound.child_shape.get_aabb(child).intersects(
              a.shape.get_aabb(a.world_trans))) {
        return false;
      }
      return compound_after_child_gate(w, b, b.shape.compound, child, a,
                                       callback, out);
    }
    return false;
  case ShapeKind::Compound:
    if (kb == ShapeKind::StaticPlane || kb == ShapeKind::TriangleMesh ||
        kb == ShapeKind::ConvexHull) {
      Affine3A child = a.world_trans * a.shape.compound.child_trans;
      if (!a.shape.compound.child_shape.get_aabb(child).intersects(
              b.shape.get_aabb(b.world_trans))) {
        return false;
      }
      return compound_after_child_gate(w, a, a.shape.compound, child, b,
                                       callback, out);
    }
    if (kb == ShapeKind::Sphere) {
      return sphere_obb(b, b.shape.sphere, a, a.shape.compound, callback, out);
    }
    if (kb == ShapeKind::Compound) {
      return obb_obb(a, a.shape.compound, b, b.shape.compound, callback, out);
    }
    return false;
  case ShapeKind::ConvexHull:
    if (kb == ShapeKind::StaticPlane) {
      BodyWrapper wrap{&a, a.world_trans, nullptr};
      return convex_plane(w, wrap, b, b.shape.plane, callback, out);
    }
    if (kb == ShapeKind::Compound) {
      Affine3A child = b.world_trans * b.shape.compound.child_trans;
      if (!b.shape.compound.child_shape.get_aabb(child).intersects(
              a.shape.get_aabb(a.world_trans))) {
        return false;
      }
      return compound_after_child_gate(w, b, b.shape.compound, child, a,
                                       callback, out);
    }
    if (kb == ShapeKind::Sphere) {
      BodyWrapper wrap{&a, a.world_trans, nullptr};
      return convex_convex(w, wrap, b, callback, out);
    }
    return false;
  }
  return false;
}

RS_HD void empty_manifold_bodies(const World &w, uint32_t a_idx, uint32_t b_idx,
                                 uint32_t &body0, uint32_t &body1) {
  ShapeKind ka = w.body(a_idx).shape.kind;
  ShapeKind kb = w.body(b_idx).shape.kind;
  bool swap = false;
  if (ka == ShapeKind::StaticPlane) {
    swap = true;
  } else if (ka == ShapeKind::Sphere && kb == ShapeKind::Compound) {
    swap = true;
  } else if (ka == ShapeKind::Sphere && kb == ShapeKind::TriangleMesh) {
    swap = false;
  } else if (ka == ShapeKind::TriangleMesh &&
             (kb == ShapeKind::Sphere || kb == ShapeKind::ConvexHull)) {
    swap = true;
  } else if (ka == ShapeKind::TriangleMesh && kb == ShapeKind::Compound) {
    swap = true;
  } else if (ka == ShapeKind::ConvexHull && kb == ShapeKind::Compound) {
    swap = true;
  }
  body0 = swap ? b_idx : a_idx;
  body1 = swap ? a_idx : b_idx;
}

enum class PairPlan : uint8_t { Skip, Release, MeshConvex, Gate, Inner };

struct PairSetup {
  PairPlan plan;
  int compound_side;
  Affine3A child_world;
};

inline constexpr uint32_t k_max_gate_jobs = 16;
inline constexpr uint32_t k_max_gate_triangles = 256;

struct GateJob {
  uint32_t pair;
  uint32_t memo_owner;
  uint32_t memo_mesh;
  int compound_side;
  bool computed;
  bool has_fresh;
  bool mesh;
  bool items_complete;
  uint32_t first_item;
  uint32_t num_items;
  const TriangleShape *tris;
  const BvhView *bvh;
  Affine3A mesh_to_box;
  GateMeshParams params;
  PersistentManifold fresh;
};

struct GateItems {
  uint32_t count;
  uint8_t job[k_max_gate_triangles];
  uint32_t tri_idx[k_max_gate_triangles];
  bool hit[k_max_gate_triangles];
  BoxTriangleSatContact contact[k_max_gate_triangles];
};

struct GateLeafSink {
  GateItems *items;
  GateJob *job;
  uint8_t job_idx;

  RS_HD void process_node(uint32_t triangle_idx) {
    if (items->count >= k_max_gate_triangles) {
      job->items_complete = false;
      return;
    }
    items->job[items->count] = job_idx;
    items->tri_idx[items->count] = triangle_idx;
    ++items->count;
    ++job->num_items;
  }
};

RS_HD PairSetup plan_pair(const World &w, const BroadphaseProxy &proxy0,
                          const BroadphaseProxy &proxy1) {
  PairSetup setup{PairPlan::Skip, -1, Affine3A::identity()};
  uint32_t rb0_idx = proxy0.client_obj_idx;
  uint32_t rb1_idx = proxy1.client_obj_idx;
  const RigidBody &rb0 = w.body(rb0_idx);
  const RigidBody &rb1 = w.body(rb1_idx);
  if ((!rb0.is_active() && !rb1.is_active()) ||
      !w.has_contact_response(rb0_idx) || !w.has_contact_response(rb1_idx)) {
    return setup;
  }
  bool c0 = rb0.shape.kind == ShapeKind::Compound;
  bool c1 = rb1.shape.kind == ShapeKind::Compound;
  if (c0 && !c1) {
    setup.compound_side = 0;
  } else if (c1 && !c0) {
    setup.compound_side = 1;
  }
  if (setup.compound_side >= 0) {
    const RigidBody &compound_obj = setup.compound_side == 0 ? rb0 : rb1;
    const RigidBody &other_obj = setup.compound_side == 0 ? rb1 : rb0;
    setup.child_world =
        compound_obj.world_trans * compound_obj.shape.compound.child_trans;
    Aabb child_aabb =
        compound_obj.shape.compound.child_shape.get_aabb(setup.child_world);
    const BroadphaseProxy &other_proxy =
        setup.compound_side == 0 ? proxy1 : proxy0;
    Aabb other_aabb = other_obj.is_static_obj()
                          ? other_proxy.aabb
                          : other_obj.shape.get_aabb(other_obj.world_trans);
    if (!child_aabb.intersects(other_aabb)) {
      setup.plan = PairPlan::Release;
      return setup;
    }
  }
  if (is_mesh_convex_pair(rb0, rb1)) {
    setup.plan = PairPlan::MeshConvex;
    return setup;
  }
  if (setup.compound_side >= 0) {
    ShapeKind other_kind = (setup.compound_side == 0 ? rb1 : rb0).shape.kind;
    if (other_kind == ShapeKind::StaticPlane ||
        other_kind == ShapeKind::TriangleMesh ||
        other_kind == ShapeKind::ConvexHull) {
      setup.plan = PairPlan::Gate;
      return setup;
    }
  }
  setup.plan = PairPlan::Inner;
  return setup;
}

template <typename Callback>
RS_HD bool gate_narrowphase(const World &w, const BroadphaseProxy &proxy0,
                            const BroadphaseProxy &proxy1,
                            const PairSetup &setup, Callback &callback,
                            PersistentManifold &fresh) {
  const RigidBody &rb0 = w.body(proxy0.client_obj_idx);
  const RigidBody &rb1 = w.body(proxy1.client_obj_idx);
  const RigidBody &compound_obj = setup.compound_side == 0 ? rb0 : rb1;
  const RigidBody &other_obj = setup.compound_side == 0 ? rb1 : rb0;
  return compound_after_child_gate(w, compound_obj, compound_obj.shape.compound,
                                   setup.child_world, other_obj, callback,
                                   fresh);
}

template <typename Callback>
RS_HD void mesh_convex_pair(World &w, uint32_t rb0_idx, uint32_t rb1_idx,
                            Callback &callback) {
  const RigidBody &rb0 = w.body(rb0_idx);
  const RigidBody &rb1 = w.body(rb1_idx);
  int idx = dispatcher_lookup(*w.a, rb0_idx, rb1_idx);
  const RigidBody &convex =
      rb0.shape.kind == ShapeKind::TriangleMesh ? rb1 : rb0;
  const RigidBody &concave =
      rb0.shape.kind == ShapeKind::TriangleMesh ? rb0 : rb1;
  if (idx < 0) {
    idx = dispatcher_insert(w, PersistentManifold::make(convex, concave));
    if (idx < 0) {
      return;
    }
  }
  PersistentManifold &m = w.a->manifolds[idx];
  process_mesh_collision_into(w, convex, concave, m, callback);
  callback.manifold_processed(m, w.body(m.body0_idx), w.body(m.body1_idx));
  push_active(w, idx);
}

template <typename Callback>
RS_HD void finish_pair(World &w, uint32_t rb0_idx, uint32_t rb1_idx,
                       bool has_fresh, const PersistentManifold &fresh,
                       Callback &callback) {
  int cached_idx = dispatcher_lookup(*w.a, rb0_idx, rb1_idx);
  int active_idx;
  if (cached_idx >= 0) {
    PersistentManifold &m = w.a->manifolds[cached_idx];
    if (m.num_points != 0) {
      m.refresh_contact_points(w.body(m.body0_idx), w.body(m.body1_idx));
    }
    if (has_fresh) {
      m.merge_contact_points(fresh);
    } else {
      for (int i = 0; i < m.num_points; ++i) {
        ManifoldPoint point = m.points[i];
        callback.contact_processed(point, w.body(m.body0_idx),
                                   w.body(m.body1_idx));
      }
    }
    active_idx = cached_idx;
  } else if (has_fresh) {
    active_idx = dispatcher_insert(w, fresh);
  } else {
    uint32_t b0, b1;
    empty_manifold_bodies(w, rb0_idx, rb1_idx, b0, b1);
    active_idx =
        dispatcher_insert(w, PersistentManifold::make(w.body(b0), w.body(b1)));
  }
  if (active_idx < 0) {
    return;
  }
  const PersistentManifold &m = w.a->manifolds[active_idx];
  callback.manifold_processed(m, w.body(m.body0_idx), w.body(m.body1_idx));
  push_active(w, active_idx);
}

template <typename Callback>
RS_HD void dispatch_plan_pairs(const World &w, Callback &callback,
                               uint32_t first, uint32_t stride) {
  PairPlan *plans = callback.pair_plans();
  for (uint32_t i = first; i < w.a->num_pairs; i += stride) {
    BroadphasePair pair = w.a->pairs[i];
    plans[i] = plan_pair(w, w.proxy(pair.proxy0), w.proxy(pair.proxy1)).plan;
  }
}

RS_HD void gate_job_setup(const World &w, uint32_t pair_idx, GateJob &job) {
  job.pair = pair_idx;
  job.mesh = false;
  BroadphasePair pair = w.a->pairs[pair_idx];
  const BroadphaseProxy &proxy0 = w.proxy(pair.proxy0);
  const BroadphaseProxy &proxy1 = w.proxy(pair.proxy1);
  PairSetup setup = plan_pair(w, proxy0, proxy1);
  const RigidBody &rb0 = w.body(proxy0.client_obj_idx);
  const RigidBody &rb1 = w.body(proxy1.client_obj_idx);
  const RigidBody &compound_obj = setup.compound_side == 0 ? rb0 : rb1;
  const RigidBody &other = setup.compound_side == 0 ? rb1 : rb0;
  if (other.shape.kind == ShapeKind::TriangleMesh) {
    job.mesh = true;
    job.params = gate_mesh_params(compound_obj, compound_obj.shape.compound,
                                  setup.child_world, other);
    job.mesh_to_box = mesh_to_box_transform(job.params.box_trans,
                                            other.world_trans, job.params.flat);
    const MeshView &mesh = w.mesh(other.shape.mesh);
    job.tris = mesh.tris;
    job.bvh = &mesh.bvh;
    job.memo_owner = compound_obj.world_array_idx - w.num_static();
    job.memo_mesh = other.shape.mesh.mesh_idx;
    job.compound_side = setup.compound_side;
  }
}

RS_HD void gate_collect_job_serial(GateJob &job, uint32_t job_idx,
                                   GateItems &items) {
  job.items_complete = true;
  job.first_item = items.count;
  job.num_items = 0;
  GateLeafSink sink{&items, &job, static_cast<uint8_t>(job_idx)};
  job.bvh->report_aabb_overlapping_node(sink, job.params.local_convex_aabb);
}

RS_HD bool gate_test_item_begin(const GateJob *jobs, GateItems &items,
                                uint32_t k, Vec3A *q, SatSweepState &state) {
  const GateJob &job = jobs[items.job[k]];
  const TriangleShape &tri = job.tris[items.tri_idx[k]];
  items.hit[k] = false;
  if (!gate_triangle_overlaps(job.params, tri)) {
    return false;
  }
  GateTriangle t = gate_triangle_in_box(job.mesh_to_box, tri, items.tri_idx[k]);
  const GateMeshParams &p = job.params;
  SatStage stage =
      box_triangle_sat_stage(p.box_trans, p.half, p.margin, t.q,
                             p.maximum_distance, &state, t.contact);
  if (stage == SatStage::NeedsPairs) {
    q[0] = t.q[0];
    q[1] = t.q[1];
    q[2] = t.q[2];
    return true;
  }
  if (stage == SatStage::Hit) {
    items.hit[k] = true;
    items.contact[k] = t.contact;
  }
  return false;
}

RS_HD void gate_test_item_finish(const GateJob *jobs, GateItems &items,
                                 uint32_t k, const Vec3A *q,
                                 const SatSweepState &state,
                                 const sat::CandidatePair *pairs) {
  const GateMeshParams &p = jobs[items.job[k]].params;
  BoxTriangleSatContact contact;
  if (box_triangle_sat_finish(p.box_trans, p.half, p.margin, q,
                              p.maximum_distance, state, pairs, contact)) {
    items.hit[k] = true;
    items.contact[k] = contact;
  }
}

RS_HDN void gate_test_item_serial_finish(const GateJob *jobs, GateItems &items,
                                         uint32_t k, const Vec3A *q,
                                         const SatSweepState &state) {
  sat::CandidatePair pairs[sat::k_sat_candidates];
  sat::enum_aabb_triangle_pairs(jobs[items.job[k]].params.half, q, state.edges,
                                pairs);
  gate_test_item_finish(jobs, items, k, q, state, pairs);
}

template <typename Callback>
RS_HD void dispatch_compute_jobs(const World &w, Callback &callback,
                                 uint32_t num_jobs, uint32_t first,
                                 uint32_t stride) {
  GateJob *jobs = callback.gate_jobs();
  const GateItems &items = callback.gate_items();
  for (uint32_t j = first; j < num_jobs; j += stride) {
    GateJob &job = jobs[j];
    BroadphasePair pair = w.a->pairs[job.pair];
    const BroadphaseProxy &proxy0 = w.proxy(pair.proxy0);
    const BroadphaseProxy &proxy1 = w.proxy(pair.proxy1);
    Callback job_callback = callback.job_tracker(j);
    if (job.mesh && job.items_complete) {
      const RigidBody &rb0 = w.body(proxy0.client_obj_idx);
      const RigidBody &rb1 = w.body(proxy1.client_obj_idx);
      const RigidBody &compound_obj = job.compound_side == 0 ? rb0 : rb1;
      const RigidBody &other = job.compound_side == 0 ? rb1 : rb0;
      GateManifoldBuilder<Callback> builder(&compound_obj, &other, job.tris,
                                            &job_callback);
      for (uint32_t k = job.first_item; k < job.first_item + job.num_items;
           ++k) {
        GateTriangle t;
        t.tri_idx = items.tri_idx[k];
        t.hit = items.hit[k];
        t.contact = items.contact[k];
        builder.add(t);
      }
      job.has_fresh = builder.finish(job.fresh);
    } else {
      job.has_fresh =
          gate_narrowphase(w, proxy0, proxy1, plan_pair(w, proxy0, proxy1),
                           job_callback, job.fresh);
    }
    job.computed = !callback.job_overflowed(j);
  }
}

enum class PlanSource : uint8_t { Precomputed, Inline };

template <typename Callback>
RS_HD void dispatch_apply_pairs(World &w, Callback &callback, uint32_t num_jobs,
                                PlanSource source) {
  const PairPlan *plans = callback.pair_plans();
  const GateJob *jobs = callback.gate_jobs();
  uint32_t next_job = 0;
  for (uint32_t i = 0; i < w.a->num_pairs; ++i) {
    if (source == PlanSource::Precomputed && plans[i] == PairPlan::Skip) {
      continue;
    }
    BroadphasePair pair = w.a->pairs[i];
    const BroadphaseProxy &proxy0 = w.proxy(pair.proxy0);
    const BroadphaseProxy &proxy1 = w.proxy(pair.proxy1);
    uint32_t rb0_idx = proxy0.client_obj_idx;
    uint32_t rb1_idx = proxy1.client_obj_idx;
    PairSetup setup{PairPlan::Skip, -1, Affine3A::identity()};
    if (source == PlanSource::Inline) {
      setup = plan_pair(w, proxy0, proxy1);
    } else {
      setup.plan = plans[i];
    }
    switch (setup.plan) {
    case PairPlan::Skip:
      break;
    case PairPlan::Release:
      dispatcher_release_manifold(w, rb0_idx, rb1_idx);
      break;
    case PairPlan::MeshConvex:
      mesh_convex_pair(w, rb0_idx, rb1_idx, callback);
      break;
    case PairPlan::Gate: {
      if (next_job < num_jobs && jobs[next_job].pair == i) {
        uint32_t j = next_job++;
        if (jobs[j].computed) {
          callback.replay_job(j);
          finish_pair(w, rb0_idx, rb1_idx, jobs[j].has_fresh, jobs[j].fresh,
                      callback);
          break;
        }
      }
      if (source == PlanSource::Precomputed) {
        setup = plan_pair(w, proxy0, proxy1);
      }
      PersistentManifold fresh;
      bool has_fresh =
          gate_narrowphase(w, proxy0, proxy1, setup, callback, fresh);
      finish_pair(w, rb0_idx, rb1_idx, has_fresh, fresh, callback);
      break;
    }
    case PairPlan::Inner: {
      PersistentManifold fresh;
      bool has_fresh = process_collision_inner(
          w, w.body(rb0_idx), w.body(rb1_idx), callback, fresh);
      finish_pair(w, rb0_idx, rb1_idx, has_fresh, fresh, callback);
      break;
    }
    }
  }
}

RS_HD Aabb collision_body_aabb(const RigidBody &obj) {
  const Vec3A cbt = Vec3A::splat(k_contact_breaking_threshold);
  Aabb aabb = obj.shape.get_aabb(obj.world_trans);
  aabb.min -= cbt;
  aabb.max += cbt;
  Aabb aabb2 = obj.shape.get_aabb(obj.interp_world_trans);
  aabb2.min -= cbt;
  aabb2.max += cbt;
  aabb += aabb2;
  return aabb;
}

RS_HDN void collision_apply_aabbs(World &w, const Aabb *aabbs) {
  for (uint32_t i = 0; i < w.a->num_dyn_bodies; ++i) {
    uint16_t handle =
        static_cast<uint16_t>(w.a->dyn_bodies[i].broadphase_handle);
    sweep_set_aabb(w, handle, aabbs[i]);
    grid_update_dynamic(w, handle, aabbs[i]);
  }
}

RS_HD void collision_update_aabbs(World &w) {
  Aabb aabbs[k_max_dyn_bodies];
  for (uint32_t i = 0; i < w.a->num_dyn_bodies; ++i) {
    aabbs[i] = collision_body_aabb(w.a->dyn_bodies[i]);
  }
  collision_apply_aabbs(w, aabbs);
}

}
