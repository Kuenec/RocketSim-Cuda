#pragma once

#include <cstdint>

#include "rsc/bullet/algorithms.hpp"
#include "rsc/bullet/box_triangle_sat.hpp"
#include "rsc/bullet/gjk.hpp"
#include "rsc/bullet/manifold.hpp"
#include "rsc/bullet/world_types.hpp"

namespace rsc {

RS_HD Affine3A mesh_pair_transform(const Affine3A &mesh_trans, bool identity,
                                   bool no_angular, const Affine3A &child) {
  if (identity) {
    return child;
  }
  if (no_angular) {
    return {child.matrix3, child.translation - mesh_trans.translation};
  }
  Affine3A mesh_to_local = mesh_trans.ext_transpose();
  return {mesh_to_local.matrix3 * child.matrix3,
          mesh_to_local.transform_point3a(child.translation)};
}

RS_HD Affine3A mesh_to_box_transform(const Affine3A &box_trans,
                                     const Affine3A &mesh_trans, bool flat) {
  Affine3A box_to_local = box_trans.ext_transpose();
  Mat3A m =
      flat ? box_to_local.matrix3 : box_to_local.matrix3 * mesh_trans.matrix3;
  return {m, box_to_local.transform_point3a(mesh_trans.translation)};
}

struct GateMeshParams {
  Affine3A box_trans;
  Vec3A half;
  float margin;
  float maximum_distance;
  Aabb local_convex_aabb;
  bool flat;
};

struct GateTriangle {
  uint32_t tri_idx;
  Vec3A q[3];
  bool hit;
  BoxTriangleSatContact contact;
};

RS_HD GateMeshParams gate_mesh_params(const RigidBody &compound_obj,
                                      const CompoundShape &compound,
                                      const Affine3A &child_world,
                                      const RigidBody &other) {
  const BoxShape &box = compound.child_shape;
  bool identity = (other.collision_flags & k_flag_identity_transform) != 0;
  bool no_angular = (other.collision_flags & k_flag_no_angular_motion) != 0;
  Affine3A convex_in_tri =
      mesh_pair_transform(other.world_trans, identity, no_angular, child_world);
  float breaking = min_f32(compound_obj.contact_breaking_threshold,
                           other.contact_breaking_threshold);
  return {child_world,
          box.half_extents,
          box.margin,
          box.margin + breaking,
          box.get_aabb(convex_in_tri),
          no_angular || identity};
}

RS_HD void gate_triangle_sat(const GateMeshParams &p, GateTriangle &t) {
  t.hit = box_triangle_sat(p.box_trans, p.half, p.margin, t.q,
                           p.maximum_distance, t.contact);
}

RS_HD bool gate_triangle_overlaps(const GateMeshParams &p,
                                  const TriangleShape &tri) {
  Vec3A tri_min = tri.points[0].min(tri.points[1]).min(tri.points[2]);
  Vec3A tri_max = tri.points[0].max(tri.points[1]).max(tri.points[2]);
  return tri_min.cmple(p.local_convex_aabb.max).all() &&
         tri_max.cmpge(p.local_convex_aabb.min).all();
}

RS_HD GateTriangle gate_triangle_in_box(const Affine3A &mesh_to_box,
                                        const TriangleShape &tri,
                                        uint32_t triangle_idx) {
  GateTriangle t;
  t.tri_idx = triangle_idx;
  t.q[0] = mesh_to_box.transform_point3a(tri.points[0]);
  t.q[1] = mesh_to_box.transform_point3a(tri.points[1]);
  t.q[2] = mesh_to_box.transform_point3a(tri.points[2]);
  return t;
}

template <typename Sink> struct GateTriangleCollector {
  const GateMeshParams *params;
  const Affine3A *tri_trans;
  const TriangleShape *tris;
  bool have_mesh_to_box;
  Affine3A mesh_to_box;
  Sink *sink;

  RS_HD void process_node(uint32_t triangle_idx) {
    const TriangleShape &tri = tris[triangle_idx];
    if (!gate_triangle_overlaps(*params, tri)) {
      return;
    }
    if (!have_mesh_to_box) {
      mesh_to_box =
          mesh_to_box_transform(params->box_trans, *tri_trans, params->flat);
      have_mesh_to_box = true;
    }
    sink->push(gate_triangle_in_box(mesh_to_box, tri, triangle_idx));
  }
};

template <typename Callback> struct GateManifoldBuilder {
  bool has_manifold;
  PersistentManifold manifold;
  const RigidBody *convex_obj;
  const RigidBody *tri_obj;
  const TriangleShape *tris;
  Callback *callback;

  RS_HD GateManifoldBuilder(const RigidBody *convex, const RigidBody *tri,
                            const TriangleShape *triangles, Callback *cb)
      : has_manifold(false), convex_obj(convex), tri_obj(tri), tris(triangles),
        callback(cb) {}

  RS_HD void add(const GateTriangle &t) {
    if (!t.hit) {
      return;
    }
    Vec3A emit_normal = tri_obj->world_trans.matrix3 * tris[t.tri_idx].normal;
    if (!has_manifold) {
      manifold.reset(*convex_obj, *tri_obj);
      has_manifold = true;
    }
    manifold.add_contact_point(*convex_obj, *tri_obj, emit_normal,
                               t.contact.point_on_b_world, t.contact.distance,
                               static_cast<int>(t.tri_idx), *callback);
  }

  RS_HD bool finish(PersistentManifold &out) {
    if (!has_manifold) {
      return false;
    }
    manifold.refresh_contact_points(*convex_obj, *tri_obj);
    if (manifold.num_points == 0) {
      return false;
    }
    out.copy_used_from(manifold);
    return true;
  }
};

template <typename Callback> struct GateSerialSink {
  const GateMeshParams *params;
  GateManifoldBuilder<Callback> *builder;

  RS_HD void push(GateTriangle t) {
    gate_triangle_sat(*params, t);
    builder->add(t);
  }
};

template <typename Callback>
RS_HD bool
compound_after_child_gate(const World &w, const RigidBody &compound_obj,
                          const CompoundShape &compound,
                          const Affine3A &child_world, const RigidBody &other,
                          Callback &callback, PersistentManifold &out) {
  const BoxShape &box = compound.child_shape;
  switch (other.shape.kind) {
  case ShapeKind::TriangleMesh: {
    GateMeshParams params =
        gate_mesh_params(compound_obj, compound, child_world, other);
    const MeshView &mesh = w.mesh(other.shape.mesh);
    GateManifoldBuilder<Callback> builder(&compound_obj, &other, mesh.tris,
                                          &callback);
    GateSerialSink<Callback> sink{&params, &builder};
    GateTriangleCollector<GateSerialSink<Callback>> collector{
        &params, &other.world_trans, mesh.tris, false, {}, &sink};
    mesh.bvh.report_aabb_overlapping_node(collector, params.local_convex_aabb);
    return builder.finish(out);
  }
  case ShapeKind::StaticPlane: {
    BodyWrapper wrap{&compound_obj, child_world, &box};
    return convex_plane_after_gate(w, wrap, other, other.shape.plane, callback,
                                   out);
  }
  case ShapeKind::ConvexHull: {
    BodyWrapper wrap{&compound_obj, child_world, &box};
    return convex_convex(w, wrap, other, callback, out);
  }
  default:
    return false;
  }
}

template <typename Callback> struct SphereTriangleCallback {
  PersistentManifold *manifold;
  const RigidBody *sphere_obj;
  const RigidBody *tri_obj;
  Vec3A sphere_center;
  float sphere_radius;
  float radius_with_threshold;
  float radius_with_threshold_sqr;
  const TriangleShape *tris;
  Callback *callback;

  RS_HD void process_node(uint32_t triangle_idx) {
    SphereTriContact info;
    if (!triangle_intersect_sphere_front(tris[triangle_idx], sphere_center,
                                         sphere_radius, radius_with_threshold,
                                         radius_with_threshold_sqr, info)) {
      return;
    }
    Vec3A point_in_world =
        info.contact_point + tri_obj->world_trans.translation;
    manifold->add_contact_point(*sphere_obj, *tri_obj, info.result_normal,
                                point_in_world, info.depth,
                                static_cast<int>(triangle_idx), *callback);
  }
};

template <typename Callback>
RS_HD bool
sphere_concave_into(const World &w, const RigidBody &convex_obj,
                    const SphereShape &sphere, const RigidBody &concave_obj,
                    PersistentManifold &manifold, Callback &callback) {
  Vec3A convex_in_tri =
      convex_obj.world_trans.translation - concave_obj.world_trans.translation;
  float radius_with_threshold =
      sphere.radius + manifold.contact_breaking_threshold;
  const MeshView &mesh = w.mesh(concave_obj.shape.mesh);
  SphereTriangleCallback<Callback> cb{&manifold,
                                      &convex_obj,
                                      &concave_obj,
                                      convex_in_tri,
                                      sphere.radius,
                                      radius_with_threshold,
                                      radius_with_threshold *
                                          radius_with_threshold,
                                      mesh.tris,
                                      &callback};
  Aabb aabb = sphere.get_aabb(convex_in_tri);
  mesh.bvh.report_aabb_overlapping_node(cb, aabb);
  if (manifold.num_points != 0) {
    manifold.refresh_contact_points(convex_obj, concave_obj);
  }
  return manifold.num_points != 0;
}

template <typename Callback> struct ConvexTriangleCallback {
  PersistentManifold *manifold;
  const RigidBody *convex_obj;
  const RigidBody *tri_obj;
  ConvexShape convex;
  const TriangleShape *tris;
  Callback *callback;
  int current_triangle_index;

  RS_HD bool is_triangle_separated(Vec3A triangle_normal,
                                   Vec3A triangle_point) const {
    const Affine3A &tri_trans = tri_obj->world_trans;
    Vec3A world_normal = tri_trans.matrix3 * triangle_normal;
    Vec3A world_point = tri_trans.transform_point3a(triangle_point);
    const Affine3A &convex_trans = convex_obj->world_trans;
    Vec3A local_pt =
        convex.support(convex_trans.matrix3.transpose_mul_vec3a(world_normal));
    Vec3A world_pt = convex_trans.transform_point3a(local_pt);
    float dist = world_normal.dot(world_point) - world_normal.dot(world_pt);
    return dist > manifold->contact_breaking_threshold;
  }

  RS_HD void add_contact_point(Vec3A normal_on_b, Vec3A point_on_b_world,
                               float depth) {
    manifold->add_contact_point(*convex_obj, *tri_obj, normal_on_b,
                                point_on_b_world, depth, current_triangle_index,
                                *callback);
  }

  RS_HD void process_node(uint32_t triangle_idx) {
    const TriangleShape &tri = tris[triangle_idx];
    if (is_triangle_separated(tri.normal, tri.points[0]) ||
        is_triangle_separated(-tri.normal, tri.points[0])) {
      return;
    }
    current_triangle_index = static_cast<int>(triangle_idx);
    float margin_a = convex.margin;
    float max_dist = margin_a + manifold->contact_breaking_threshold;
    GjkPairDetector detector;
    detector.init(margin_a, 0.0f);
    ConvexShape tri_shape =
        ConvexShape::triangle(tri.points[0], tri.points[1], tri.points[2]);
    detector.get_closest_points(convex_obj->world_trans, tri_obj->world_trans,
                                max_dist * max_dist, convex, tri_shape, *this);
  }
};

template <typename Callback>
RS_HDN bool convex_concave_into(const World &w, const RigidBody &convex_obj,
                                const RigidBody &concave_obj,
                                PersistentManifold &manifold,
                                Callback &callback) {
  const MeshView &mesh = w.mesh(concave_obj.shape.mesh);
  ConvexTriangleCallback<Callback> cb{&manifold,
                                      &convex_obj,
                                      &concave_obj,
                                      w.convex_of(convex_obj.shape),
                                      mesh.tris,
                                      &callback,
                                      0};
  Affine3A world_to_local = concave_obj.world_trans.ext_transpose();
  Aabb convex_aabb_world = convex_obj.shape.get_aabb(convex_obj.world_trans);
  Aabb aabb = convex_aabb_world.transform(world_to_local, 0.0f);
  mesh.bvh.report_aabb_overlapping_node(cb, aabb);
  if (manifold.num_points != 0) {
    manifold.refresh_contact_points(convex_obj, concave_obj);
  }
  return manifold.num_points != 0;
}

namespace edge {

inline constexpr float k_planar_epsilon = 0.0001f;
inline constexpr float k_equal_vertex_threshold = 0.0001f * 0.0001f;
inline constexpr float k_edge_distance_threshold = 0.1f;
inline constexpr float k_max_edge_angle_threshold = k_tau;
inline constexpr float k_convex_epsilon = 0.0f;

RS_HDI float get_angle(Vec3A edge_a, Vec3A normal_a, Vec3A normal_b) {
  return atan2f_glibc(normal_b.dot(edge_a), normal_b.dot(normal_a));
}

RS_HD Vec3A nearest_point_in_line_segment(Vec3A point, Vec3A line0,
                                          Vec3A line1) {
  Vec3A delta = line1 - line0;
  if (delta.length_squared() < k_f32_epsilon * k_f32_epsilon) {
    return line0;
  }
  float t = (point - line0).dot(delta) / delta.dot(delta);
  return line0 +
         delta * clamp_f32(t, u32_to_f32(0x3727c5acu), u32_to_f32(0x3f7fff58u));
}

RS_HD bool clamp_normal(Vec3A edge, Vec3A tri_normal_org,
                        Vec3A local_contact_normal_on_b,
                        float corrected_edge_angle, Vec3A &out) {
  Vec3A edge_cross = edge.cross(tri_normal_org).normalize();
  float cur_angle =
      get_angle(edge_cross, tri_normal_org, local_contact_normal_on_b);
  if ((corrected_edge_angle < 0.0f && cur_angle < corrected_edge_angle) ||
      (corrected_edge_angle >= 0.0f && cur_angle > corrected_edge_angle)) {
    float diff_angle = corrected_edge_angle - cur_angle;
    Quat rotation = quat_from_axis_angle_simd(edge.normalize(), diff_angle);
    out = quat_mul_vec3a(rotation, local_contact_normal_on_b);
    return true;
  }
  return false;
}

}

RS_HDN void adjust_internal_edge_contacts(const World &w, ManifoldPoint &cp,
                                          const RigidBody &tri_obj, int idx) {
  using namespace edge;
  const MeshView &mesh = w.mesh(tri_obj.shape.mesh);
  const TriangleInfo &info = mesh.infos[idx];
  const TriangleShape &tri = mesh.tris[idx];
  const Mat3A &m3 = tri_obj.world_trans.matrix3;
  Vec3A nearest = nearest_point_in_line_segment(cp.local_point_b, tri.points[0],
                                                tri.points[1]);
  Vec3A contact = cp.local_point_b;
  Vec3A local_contact_normal_on_b =
      m3.transpose_mul_vec3a(cp.normal_world_on_b);
  int best_edge = -1;
  float dists[3] = {k_f32_max, k_f32_max, k_f32_max};
  float dist_to_best = k_f32_max;
  if (abs_f32(info.edge_v0_v1_angle) < k_max_edge_angle_threshold) {
    dists[0] = (contact - nearest).length();
    if (dists[0] < dist_to_best) {
      best_edge = 0;
      dist_to_best = dists[0];
    }
  }
  if (abs_f32(info.edge_v1_v2_angle) < k_max_edge_angle_threshold) {
    Vec3A n1 = nearest_point_in_line_segment(cp.local_point_b, tri.points[1],
                                             tri.points[2]);
    dists[1] = (contact - n1).length();
    if (dists[1] < dist_to_best ||
        (dists[1] == dist_to_best && info.edge_v1_v2_angle != 0.0f &&
         (info.flags & k_v1v2_convex) == 0)) {
      best_edge = 1;
      dist_to_best = dists[1];
    }
  }
  if (abs_f32(info.edge_v2_v0_angle) < k_max_edge_angle_threshold) {
    Vec3A n2 = nearest_point_in_line_segment(cp.local_point_b, tri.points[2],
                                             tri.points[0]);
    dists[2] = (contact - n2).length();
    if (dists[2] < dist_to_best ||
        (dists[2] == dist_to_best && info.edge_v2_v0_angle != 0.0f)) {
      best_edge = 2;
    }
  }
  bool processed_edge = false;
  bool concave_edge_hit = false;
  const float angles[3] = {info.edge_v0_v1_angle, info.edge_v1_v2_angle,
                           info.edge_v2_v0_angle};
  const uint8_t convex_flags[3] = {k_v0v1_convex, k_v1v2_convex, k_v2v0_convex};
  const uint8_t swap_flags[3] = {k_v0v1_swap_normal_b, k_v1v2_swap_normal_b,
                                 k_v2v0_swap_normal_b};
  for (int e = 0; e < 3; ++e) {
    float angle = angles[e];
    if (!(abs_f32(angle) < k_max_edge_angle_threshold &&
          dists[e] < k_edge_distance_threshold &&
          (best_edge == e || (dists[e] < k_f32_epsilon && angle != 0.0f)))) {
      continue;
    }
    processed_edge = true;
    if (angle == 0.0f) {
      concave_edge_hit = true;
      continue;
    }
    bool is_edge_convex = (info.flags & convex_flags[e]) != 0;
    float swap_factor = (is_edge_convex ? 1.0f : 0.0f) * 2.0f - 1.0f;
    Vec3A edge_vec = -tri.edge(e);
    Vec3A n_a = swap_factor * tri.normal;
    Quat orn = quat_from_axis_angle_simd(edge_vec.normalize(), angle);
    Vec3A computed_normal_b = quat_mul_vec3a(orn, tri.normal);
    if (info.flags & swap_flags[e]) {
      computed_normal_b *= -1.0f;
    }
    Vec3A n_b = swap_factor * computed_normal_b;
    float n_dot_a = local_contact_normal_on_b.dot(n_a);
    float n_dot_b = local_contact_normal_on_b.dot(n_b);
    bool back_facing = n_dot_a < k_convex_epsilon && n_dot_b < k_convex_epsilon;
    if (back_facing) {
      concave_edge_hit = true;
      continue;
    }
    Vec3A lcn = e == 0 ? local_contact_normal_on_b
                       : m3.transpose_mul_vec3a(cp.normal_world_on_b);
    Vec3A clamped;
    if (clamp_normal(edge_vec, swap_factor * tri.normal, lcn, angle, clamped) &&
        clamped.dot(tri.normal) > 0.0f) {
      Vec3A new_normal = m3 * clamped;
      cp.normal_world_on_b = new_normal;
      cp.pos_world_on_b = cp.pos_world_on_a - new_normal * cp.distance_1;
      cp.local_point_b = tri_obj.world_trans.inv_x_form(cp.pos_world_on_b);
    }
  }
  if (!processed_edge || !concave_edge_hit) {
    return;
  }
  float d = tri.normal.dot(local_contact_normal_on_b);
  if (d < 0.0f) {
    return;
  }
  cp.normal_world_on_b = m3 * tri.normal;
  cp.pos_world_on_b = cp.pos_world_on_a - cp.normal_world_on_b * cp.distance_1;
  cp.local_point_b = tri_obj.world_trans.inv_x_form(cp.pos_world_on_b);
}

}
