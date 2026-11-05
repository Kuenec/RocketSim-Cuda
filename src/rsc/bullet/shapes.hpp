#pragma once

#include <cstdint>

#include "rsc/bullet/convex.hpp"
#include "rsc/math/bvh.hpp"
#include "rsc/math/shared.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

inline constexpr float k_sphere_radius_margin = 0.08f;

struct BoxShape {
  Vec3A half_extents;
  float margin;

  RS_HD static BoxShape make(Vec3A box_half_extents) {
    float safe_margin = 0.1f * box_half_extents.min_element();
    float margin = min_f32_c(safe_margin, k_convex_distance_margin);
    return {box_half_extents - margin, margin};
  }

  RS_HD Aabb get_aabb(const Affine3A &t) const {
    return Aabb::from_half_extents_transform(half_extents, margin, t);
  }

  RS_HD Vec3A calculate_local_inertia(float mass) const {
    Vec3A l = 2.0f * (half_extents + margin);
    Vec3A yxx{l.y, l.x, l.x};
    Vec3A zzy{l.z, l.z, l.y};
    return mass / 12.0f * (yxx * yxx + zzy * zzy);
  }

  RS_HD ConvexShape convex() const {
    return ConvexShape::box(half_extents, margin);
  }
};

struct SphereShape {
  float radius;

  RS_HD Aabb get_aabb(Vec3A center) const {
    Vec3A extent = Vec3A::splat(radius + k_sphere_radius_margin);
    return {center - extent, center + extent};
  }

  RS_HD Vec3A calculate_local_inertia(float mass) const {
    return Vec3A::splat(0.4f * mass * radius * radius);
  }

  RS_HD bool closest_point(Vec3A point, float &dist, Vec3A &normal) const {
    float len_sq = point.length_squared();
    float len = sqrt_f32(len_sq);
    if (!(len_sq > k_f32_epsilon * k_f32_epsilon)) {
      return false;
    }
    float rlen = 1.0f / len;
    dist = 1.0f / rlen - radius;
    normal = point * rlen;
    return true;
  }
};

struct CompoundShape {
  BoxShape child_shape;
  Affine3A child_trans;
  Aabb local_aabb;
  Vec3A local_center;
  Vec3A local_half_extents;

  RS_HD static CompoundShape make(BoxShape child, const Affine3A &trans) {
    CompoundShape s;
    s.child_shape = child;
    s.child_trans = trans;
    s.local_aabb = child.get_aabb(trans);
    s.local_half_extents = 0.5f * (s.local_aabb.max - s.local_aabb.min);
    s.local_center = 0.5f * (s.local_aabb.max + s.local_aabb.min);
    return s;
  }

  RS_HD Aabb get_aabb(const Affine3A &trans) const {
    Mat3A abs_b = trans.matrix3.abs();
    Vec3A center = trans.transform_point3a(local_center);
    Vec3A extent = abs_b * local_half_extents;
    return {center - extent, center + extent};
  }
};

struct StaticPlaneShape {
  Vec3A plane_normal;
  bool is_single_axis;
  int single_axis_idx;
  bool single_axis_backwards;
  Aabb aabb_ident_cache;
  Aabb aabb_cache;

  RS_HD Aabb get_aabb(const Affine3A &t) const {
    Vec3A min = Vec3A::splat(k_f32_min);
    Vec3A max = Vec3A::splat(k_f32_max);
    if (is_single_axis) {
      constexpr float offset = 0.2f;
      min[single_axis_idx] = t.translation[single_axis_idx] - offset;
      max[single_axis_idx] = t.translation[single_axis_idx] + offset;
      if (single_axis_backwards) {
        max[single_axis_idx] = k_f32_max;
      } else {
        min[single_axis_idx] = k_f32_min;
      }
    }
    return {min, max};
  }

  RS_HD static StaticPlaneShape make(const Affine3A &world_trans,
                                     Vec3A normal) {
    BVec3 sig = normal.abs().cmpge(Vec3A::splat(k_f32_epsilon));
    int count = static_cast<int>(sig.x) + static_cast<int>(sig.y) +
                static_cast<int>(sig.z);
    StaticPlaneShape s;
    s.plane_normal = normal;
    if (count == 1) {
      int axis = normal.abs().max_position();
      s.is_single_axis = true;
      s.single_axis_idx = axis;
      s.single_axis_backwards = is_sign_negative_f32(normal[axis]);
    } else {
      s.is_single_axis = false;
      s.single_axis_idx = 0;
      s.single_axis_backwards = false;
    }
    s.aabb_ident_cache = Aabb::zero();
    s.aabb_cache = Aabb::zero();
    s.aabb_ident_cache = s.get_aabb(Affine3A::identity());
    s.aabb_cache = s.get_aabb(world_trans);
    return s;
  }
};

enum class HullSource : uint8_t { Static, Arena };

struct ConvexHullShape {
  HullSource source;
  uint32_t offset;
  uint32_t count;
  Aabb local_aabb;
  float margin;

  RS_HD static Aabb calc_local_aabb(HullView view, float margin) {
    const Vec3A dirs[6] = {{1.0f, 0.0f, 0.0f},  {0.0f, 1.0f, 0.0f},
                           {0.0f, 0.0f, 1.0f},  {-1.0f, 0.0f, 0.0f},
                           {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}};
    Vec3A supporting[6];
    for (int i = 0; i < 6; ++i) {
      supporting[i] = view.max_dot(dirs[i]);
    }
    Vec3A max = Vec3A::zero();
    Vec3A min = Vec3A::zero();
    for (int i = 0; i < 3; ++i) {
      max[i] = supporting[i][i];
      min[i] = supporting[i + 3][i];
    }
    return {min - margin, max + margin};
  }

  RS_HD static ConvexHullShape make(HullSource source, uint32_t offset,
                                    HullView view) {
    ConvexHullShape s;
    s.source = source;
    s.offset = offset;
    s.count = view.count;
    s.margin = k_convex_distance_margin;
    s.local_aabb = calc_local_aabb(view, s.margin);
    return s;
  }

  RS_HD Aabb get_aabb(const Affine3A &trans) const {
    return local_aabb.transform(trans, margin);
  }

  RS_HD static Vec3A calculate_local_inertia(HullView view, float margin,
                                             float mass) {
    ConvexShape cs = ConvexShape::hull_shape(view);
    Aabb aabb = Aabb::zero();
    for (int i = 0; i < 3; ++i) {
      Vec3A vec = Vec3A::zero();
      vec[i] = 1.0f;
      Vec3A sv = cs.support(vec);
      aabb.max[i] = sv[i] + margin;
      vec[i] = -1.0f;
      sv = cs.support(vec);
      aabb.min[i] = sv[i] - margin;
    }
    Vec3A half_extents = (aabb.max - aabb.min) * 0.5f;
    Vec3A l = 2.0f * (half_extents + margin);
    Vec3A l2 = l * l;
    float scaled_mass = mass * 0.08333333f;
    return scaled_mass * Vec3A{l2.y + l2.z, l2.x + l2.z, l2.x + l2.y};
  }
};

struct TriangleShape {
  Vec3A points[3];
  Vec3A normal;

  RS_HD static TriangleShape make(Vec3A p0, Vec3A p1, Vec3A p2) {
    Vec3A e0 = p1 - p0;
    Vec3A e2 = p0 - p2;
    return {{p0, p1, p2}, e0.cross(-e2).normalize()};
  }

  RS_HD Vec3A edge(int index) const {
    if (index == 0) {
      return points[1] - points[0];
    }
    if (index == 1) {
      return points[2] - points[1];
    }
    return points[0] - points[2];
  }

  RS_HD Aabb aabb() const {
    return {points[0].min(points[1]).min(points[2]),
            points[0].max(points[1]).max(points[2])};
  }

  RS_HD float normal_length() const {
    Vec3A e0 = points[1] - points[0];
    Vec3A e2 = points[0] - points[2];
    return e0.cross(-e2).length();
  }
};

struct SphereTriContact {
  Vec3A result_normal;
  Vec3A contact_point;
  float depth;
};

RS_HD float segment_sqr_distance(Vec3A from, Vec3A to, Vec3A p,
                                 Vec3A &nearest) {
  Vec3A diff = p - from;
  Vec3A v = to - from;
  float t = v.dot(diff);
  if (t > 0.0f) {
    float dot_vv = v.dot(v);
    if (t < dot_vv) {
      t /= dot_vv;
      diff -= t * v;
    } else {
      t = 1.0f;
      diff -= v;
    }
  } else {
    t = 0.0f;
  }
  nearest = from + t * v;
  return diff.dot(diff);
}

RS_HD bool triangle_face_contains(const TriangleShape &tri, Vec3A n,
                                  const Vec3A *obj_to_points) {
  Vec3A c0 = tri.edge(0).cross(obj_to_points[0]);
  if (c0.dot(n) < 0.0f) {
    return false;
  }
  Vec3A c1 = tri.edge(1).cross(obj_to_points[1]);
  if (c1.dot(n) < 0.0f) {
    return false;
  }
  Vec3A c2 = tri.edge(2).cross(obj_to_points[2]);
  return c2.dot(n) >= 0.0f;
}

RS_HD bool triangle_intersect_sphere_front(const TriangleShape &tri,
                                           Vec3A obj_center, float radius,
                                           float radius_with_threshold,
                                           float radius_with_threshold_sqr,
                                           SphereTriContact &out) {
  Vec3A obj_to_center = obj_center - tri.points[0];
  float distance_from_plane = obj_to_center.dot(tri.normal);
  if (distance_from_plane < 0.0f) {
    return false;
  }
  Vec3A triangle_normal = tri.normal;
  if (distance_from_plane >= radius_with_threshold) {
    return false;
  }
  Vec3A obj_to_points[3] = {obj_to_center, obj_center - tri.points[1],
                            obj_center - tri.points[2]};
  Vec3A contact_point;
  if (triangle_face_contains(tri, triangle_normal, obj_to_points)) {
    contact_point = obj_center - triangle_normal * distance_from_plane;
  } else {
    float contact_capsule_radius_sqr = radius_with_threshold_sqr;
    float min_distance_sqr = contact_capsule_radius_sqr;
    Vec3A best = Vec3A::zero();
    for (int edge_idx = 0; edge_idx < 3; ++edge_idx) {
      Vec3A from = tri.points[edge_idx];
      Vec3A to = tri.points[(edge_idx + 1) % 3];
      Vec3A nearest_on_edge = Vec3A::zero();
      float distance_sqr =
          segment_sqr_distance(from, to, obj_center, nearest_on_edge);
      if (distance_sqr < min_distance_sqr) {
        min_distance_sqr = distance_sqr;
        best = nearest_on_edge;
      }
    }
    if (!(min_distance_sqr < contact_capsule_radius_sqr)) {
      return false;
    }
    contact_point = best;
  }
  Vec3A contact_to_center = obj_center - contact_point;
  float distance_sqr = contact_to_center.length_squared();
  if (distance_sqr >= radius_with_threshold_sqr) {
    return false;
  }
  if (distance_sqr > k_f32_epsilon) {
    float distance = sqrt_f32(distance_sqr);
    float inverse_distance = 1.0f / distance;
    out.result_normal = contact_to_center * inverse_distance;
    out.depth = -(radius - distance);
  } else {
    out.result_normal = triangle_normal;
    out.depth = -radius;
  }
  out.contact_point = contact_point;
  return true;
}

enum TriInfoFlag : uint8_t {
  k_v0v1_convex = 1,
  k_v1v2_convex = 1 << 1,
  k_v2v0_convex = 1 << 2,
  k_v0v1_swap_normal_b = 1 << 3,
  k_v1v2_swap_normal_b = 1 << 4,
  k_v2v0_swap_normal_b = 1 << 5,
};

struct TriangleInfo {
  uint8_t flags;
  float edge_v0_v1_angle;
  float edge_v1_v2_angle;
  float edge_v2_v0_angle;
};

struct MeshView {
  BvhView bvh;
  const TriangleShape *tris;
  const TriangleInfo *infos;
  uint32_t num_tris;
  Aabb aabb_ident_cache;
};

struct TriMeshShape {
  uint32_t mesh_idx;
  Aabb aabb_ident_cache;
};

enum class ShapeKind : uint8_t {
  Compound,
  Sphere,
  ConvexHull,
  StaticPlane,
  TriangleMesh
};

struct Shape {
  ShapeKind kind;
  union {
    CompoundShape compound;
    SphereShape sphere;
    ConvexHullShape hull;
    StaticPlaneShape plane;
    TriMeshShape mesh;
  };

  RS_HD Aabb get_aabb(const Affine3A &t) const {
    switch (kind) {
    case ShapeKind::Sphere:
      return sphere.get_aabb(t.translation);
    case ShapeKind::Compound:
      return compound.get_aabb(t);
    case ShapeKind::ConvexHull:
      return hull.get_aabb(t);
    case ShapeKind::StaticPlane:
      return plane.aabb_cache;
    case ShapeKind::TriangleMesh:
      return {mesh.aabb_ident_cache.min + t.translation,
              mesh.aabb_ident_cache.max + t.translation};
    }
    return Aabb::zero();
  }

  RS_HD float get_margin() const {
    switch (kind) {
    case ShapeKind::Sphere:
      return sphere.radius;
    case ShapeKind::Compound:
      return compound.child_shape.margin;
    case ShapeKind::ConvexHull:
      return hull.margin;
    case ShapeKind::StaticPlane:
    case ShapeKind::TriangleMesh:
      return 0.0f;
    }
    return 0.0f;
  }

  RS_HD float get_angular_motion_disc() const {
    Aabb aabb = Aabb::zero();
    switch (kind) {
    case ShapeKind::Sphere:
      return (sphere.radius + k_sphere_radius_margin) + Vec3A::zero().length();
    case ShapeKind::Compound:
      aabb = compound.local_aabb;
      break;
    case ShapeKind::StaticPlane:
      aabb = plane.aabb_ident_cache;
      break;
    case ShapeKind::TriangleMesh:
      aabb = mesh.aabb_ident_cache;
      break;
    case ShapeKind::ConvexHull:
      aabb = hull.local_aabb;
      break;
    }
    Vec3A center = (aabb.min + aabb.max) * 0.5f;
    float radius = (aabb.max - aabb.min).length() * 0.5f;
    return radius + center.length();
  }

  RS_HD float get_contact_breaking_threshold(float default_threshold) const {
    return get_angular_motion_disc() * default_threshold;
  }

  RS_HD static Shape make_compound(const CompoundShape &c) {
    Shape s;
    s.kind = ShapeKind::Compound;
    s.compound = c;
    return s;
  }
  RS_HD static Shape make_sphere(float radius) {
    Shape s;
    s.kind = ShapeKind::Sphere;
    s.sphere = {radius};
    return s;
  }
  RS_HD static Shape make_hull(const ConvexHullShape &h) {
    Shape s;
    s.kind = ShapeKind::ConvexHull;
    s.hull = h;
    return s;
  }
  RS_HD static Shape make_plane(const StaticPlaneShape &p) {
    Shape s;
    s.kind = ShapeKind::StaticPlane;
    s.plane = p;
    return s;
  }
  RS_HD static Shape make_mesh(uint32_t idx, Aabb ident) {
    Shape s;
    s.kind = ShapeKind::TriangleMesh;
    s.mesh = {idx, ident};
    return s;
  }
};

}
