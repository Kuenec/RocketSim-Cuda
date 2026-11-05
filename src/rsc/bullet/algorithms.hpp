#pragma once

#include <cstdint>

#include "rsc/bullet/box_triangle_sat.hpp"
#include "rsc/bullet/gjk.hpp"
#include "rsc/bullet/manifold.hpp"
#include "rsc/bullet/world_types.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

struct BodyWrapper {
  const RigidBody *obj;
  Affine3A world_trans;
  const BoxShape *child_override;

  RS_HD Aabb get_aabb() const {
    if (child_override) {
      return child_override->get_aabb(world_trans);
    }
    return obj->shape.get_aabb(world_trans);
  }
};

RS_HD Vec3A wrapper_supporting_vertex(const World &w, const BodyWrapper &bw,
                                      Vec3A vec) {
  if (bw.child_override) {
    return bw.child_override->convex().support(vec);
  }
  return w.convex_of(bw.obj->shape).support(vec);
}

template <typename Callback>
RS_HD bool convex_plane_after_gate(const World &w, const BodyWrapper &convex,
                                   const RigidBody &plane_obj,
                                   const StaticPlaneShape &plane,
                                   Callback &callback,
                                   PersistentManifold &out) {
  Vec3A plane_normal = plane.plane_normal;
  const Affine3A &plane_trans = plane_obj.world_trans;
  Mat3A plane_in_convex = convex.world_trans.matrix3.transpose();
  Affine3A convex_in_plane{convex.world_trans.matrix3,
                           convex.world_trans.translation -
                               plane_trans.translation};
  Vec3A vtx =
      wrapper_supporting_vertex(w, convex, plane_in_convex * -plane_normal);
  Vec3A vtx_in_plane = convex_in_plane.transform_point3a(vtx);
  float distance = plane_normal.dot(vtx_in_plane);
  PersistentManifold m = PersistentManifold::make(*convex.obj, plane_obj);
  if (distance >= m.contact_breaking_threshold) {
    return false;
  }
  Vec3A projected = vtx_in_plane - distance * plane_normal;
  Vec3A world_pt = projected + plane_trans.translation;
  m.add_contact_point(*convex.obj, plane_obj, plane_normal, world_pt, distance,
                      -1, callback);
  m.refresh_contact_points(*convex.obj, plane_obj);
  out = m;
  return true;
}

template <typename Callback>
RS_HD bool convex_plane(const World &w, const BodyWrapper &convex,
                        const RigidBody &plane_obj,
                        const StaticPlaneShape &plane, Callback &callback,
                        PersistentManifold &out) {
  if (!convex.get_aabb().intersects(plane.aabb_cache)) {
    return false;
  }
  return convex_plane_after_gate(w, convex, plane_obj, plane, callback, out);
}

template <typename Callback> struct ManifoldGjkResult {
  PersistentManifold *manifold;
  const RigidBody *body0;
  const RigidBody *body1;
  Callback *callback;
  int tri_idx;

  RS_HD void add_contact_point(Vec3A normal_on_b, Vec3A point_on_b_world,
                               float depth) {
    manifold->add_contact_point(*body0, *body1, normal_on_b, point_on_b_world,
                                depth, tri_idx, *callback);
  }
};

template <typename Callback>
RS_HDN bool convex_convex(const World &w, const BodyWrapper &a,
                         const RigidBody &b, Callback &callback,
                         PersistentManifold &out) {
  PersistentManifold m = PersistentManifold::make(*a.obj, b);
  float margin_a = a.obj->shape.get_margin();
  float margin_b = b.shape.get_margin();
  float max_dist = margin_a + margin_b + m.contact_breaking_threshold;
  GjkPairDetector detector;
  detector.init(margin_a, margin_b);
  ManifoldGjkResult<Callback> result{&m, a.obj, &b, &callback, -1};
  ConvexShape shape_a = w.convex_of(a.obj->shape);
  ConvexShape shape_b = w.convex_of(b.shape);
  detector.get_closest_points(a.world_trans, b.world_trans, max_dist * max_dist,
                              shape_a, shape_b, result);
  if (m.num_points == 0) {
    return false;
  }
  m.refresh_contact_points(*a.obj, b);
  out = m;
  return true;
}

RS_HD void sphere_penetration(Vec3A box_extents, Vec3A sphere_from_local,
                              float &min_dist, Vec3A &normal) {
  min_dist = box_extents.x - sphere_from_local.x;
  normal = Vec3A{1.0f, 0.0f, 0.0f};
  auto test_face = [&](float face_dist, int axis, float sign) {
    if (face_dist < min_dist) {
      min_dist = face_dist;
      normal = Vec3A::zero();
      normal[axis] = sign;
    }
  };
  test_face(box_extents.x + sphere_from_local.x, 0, -1.0f);
  test_face(box_extents.y - sphere_from_local.y, 1, 1.0f);
  test_face(box_extents.y + sphere_from_local.y, 1, -1.0f);
  test_face(box_extents.z - sphere_from_local.z, 2, 1.0f);
  test_face(box_extents.z + sphere_from_local.z, 2, -1.0f);
}

template <typename Callback>
RS_HDN bool sphere_obb(const RigidBody &sphere_obj, const SphereShape &sphere,
                      const RigidBody &obb_obj, const CompoundShape &obb,
                      Callback &callback, PersistentManifold &out) {
  const Affine3A &sphere_trans = sphere_obj.world_trans;
  Aabb aabb1 = sphere.get_aabb(sphere_trans.translation);
  const Affine3A &org_trans = obb_obj.world_trans;
  Aabb aabb2 = obb.get_aabb(org_trans);
  if (!aabb1.intersects(aabb2)) {
    return false;
  }
  Affine3A child_world = org_trans * obb.child_trans;
  Vec3A box_extents = obb.child_shape.half_extents;
  Vec3A sphere_from_local = child_world.inv_x_form(sphere_trans.translation);
  Vec3A closest = sphere_from_local.clamp(-box_extents, box_extents);
  Vec3A delta = sphere_from_local - closest;
  float dist_sq = delta.length_squared();
  float radius = sphere.radius;
  float box_margin = obb.child_shape.margin;
  PersistentManifold m = PersistentManifold::make(obb_obj, sphere_obj);
  float intersection_dist = radius + box_margin;
  float contact_dist = intersection_dist + m.contact_breaking_threshold;
  if (dist_sq > contact_dist * contact_dist) {
    return false;
  }
  float dist;
  if (dist_sq > k_f32_epsilon) {
    dist = sqrt_f32(dist_sq);
    delta /= dist;
  } else {
    sphere_penetration(box_extents, sphere_from_local, dist, delta);
    dist *= -1.0f;
  }
  Vec3A normal_on_box = child_world.matrix3 * delta;
  float depth = dist - intersection_dist;
  Vec3A normal_on_ball = -normal_on_box;
  Vec3A point_on_ball = sphere_trans.translation - normal_on_box * radius;
  m.add_contact_point(obb_obj, sphere_obj, normal_on_ball, point_on_ball, depth,
                      -1, callback);
  m.refresh_contact_points(obb_obj, sphere_obj);
  if (m.num_points == 0) {
    return false;
  }
  out = m;
  return true;
}

}
