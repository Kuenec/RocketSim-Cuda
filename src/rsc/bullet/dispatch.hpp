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
  a.manifolds[a.num_manifolds] = m;
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
  a.manifolds[index] = a.manifolds[last];
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

template <typename Callback>
RS_HD void near_callback(World &w, const BroadphaseProxy &proxy0,
                         const BroadphaseProxy &proxy1, Callback &callback) {
  uint32_t rb0_idx = proxy0.client_obj_idx;
  uint32_t rb1_idx = proxy1.client_obj_idx;
  const RigidBody &rb0 = w.body(rb0_idx);
  const RigidBody &rb1 = w.body(rb1_idx);
  if ((!rb0.is_active() && !rb1.is_active()) ||
      !w.has_contact_response(rb0_idx) || !w.has_contact_response(rb1_idx)) {
    return;
  }
  int cached_idx = dispatcher_lookup(*w.a, rb0_idx, rb1_idx);
  bool c0 = rb0.shape.kind == ShapeKind::Compound;
  bool c1 = rb1.shape.kind == ShapeKind::Compound;
  int compound_side = -1;
  if (c0 && !c1) {
    compound_side = 0;
  } else if (c1 && !c0) {
    compound_side = 1;
  }
  const RigidBody *compound_obj = nullptr;
  const RigidBody *other_obj = nullptr;
  Affine3A child_world = Affine3A::identity();
  if (compound_side >= 0) {
    compound_obj = compound_side == 0 ? &rb0 : &rb1;
    other_obj = compound_side == 0 ? &rb1 : &rb0;
    child_world =
        compound_obj->world_trans * compound_obj->shape.compound.child_trans;
    Aabb child_aabb =
        compound_obj->shape.compound.child_shape.get_aabb(child_world);
    const BroadphaseProxy &other_proxy = compound_side == 0 ? proxy1 : proxy0;
    Aabb other_aabb = other_obj->is_static_obj()
                          ? other_proxy.aabb
                          : other_obj->shape.get_aabb(other_obj->world_trans);
    if (!child_aabb.intersects(other_aabb)) {
      dispatcher_release_manifold(w, rb0_idx, rb1_idx);
      return;
    }
  }
  if (is_mesh_convex_pair(rb0, rb1)) {
    const RigidBody &convex =
        rb0.shape.kind == ShapeKind::TriangleMesh ? rb1 : rb0;
    const RigidBody &concave =
        rb0.shape.kind == ShapeKind::TriangleMesh ? rb0 : rb1;
    int idx = cached_idx;
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
    return;
  }
  PersistentManifold fresh;
  bool has_fresh;
  if (compound_side >= 0 && (other_obj->shape.kind == ShapeKind::StaticPlane ||
                             other_obj->shape.kind == ShapeKind::TriangleMesh ||
                             other_obj->shape.kind == ShapeKind::ConvexHull)) {
    has_fresh = compound_after_child_gate(
        w, *compound_obj, compound_obj->shape.compound, child_world, *other_obj,
        callback, fresh);
  } else {
    has_fresh = process_collision_inner(w, rb0, rb1, callback, fresh);
  }
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
RS_HD void dispatch_all_collision_pairs(World &w, Callback &callback) {
  for (uint32_t i = 0; i < w.a->num_pairs; ++i) {
    BroadphasePair pair = w.a->pairs[i];
    near_callback(w, w.proxy(pair.proxy0), w.proxy(pair.proxy1), callback);
  }
}

RS_HDN void collision_update_aabbs(World &w) {
  const Vec3A cbt = Vec3A::splat(k_contact_breaking_threshold);
  for (uint32_t i = 0; i < w.a->num_dyn_bodies; ++i) {
    const RigidBody &obj = w.a->dyn_bodies[i];
    Aabb aabb = obj.shape.get_aabb(obj.world_trans);
    aabb.min -= cbt;
    aabb.max += cbt;
    Aabb aabb2 = obj.shape.get_aabb(obj.interp_world_trans);
    aabb2.min -= cbt;
    aabb2.max += cbt;
    aabb += aabb2;
    uint16_t handle = static_cast<uint16_t>(obj.broadphase_handle);
    sweep_set_aabb(w, handle, aabb);
    grid_update_dynamic(w, handle, aabb);
  }
}

}
