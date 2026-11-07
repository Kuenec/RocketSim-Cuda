#pragma once

#include <cstdint>

#include "rsc/bullet/rigid_body.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

inline constexpr int k_manifold_cache_size = 4;

struct ManifoldPoint {
  Vec3A local_point_a;
  Vec3A local_point_b;
  Vec3A pos_world_on_b;
  Vec3A pos_world_on_a;
  Vec3A normal_world_on_b;
  float distance_1;
  float combined_friction;
  float combined_restitution;
  float applied_impulse;
  bool is_special;

  RS_HD static ManifoldPoint make(Vec3A a, Vec3A b, Vec3A normal,
                                  float distance) {
    ManifoldPoint p;
    p.local_point_a = a;
    p.local_point_b = b;
    p.pos_world_on_a = Vec3A::zero();
    p.pos_world_on_b = Vec3A::zero();
    p.normal_world_on_b = normal;
    p.distance_1 = distance;
    p.combined_friction = 0.0f;
    p.combined_restitution = 0.0f;
    p.applied_impulse = 0.0f;
    p.is_special = false;
    return p;
  }

  RS_HD static ManifoldPoint zero() {
    return make(Vec3A::zero(), Vec3A::zero(), Vec3A::zero(), 0.0f);
  }
};

RS_HD Vec3A body_point_to_local(const RigidBody &body, Vec3A point) {
  if (body.collision_flags & k_flag_identity_transform) {
    return point;
  }
  if (body.collision_flags & k_flag_no_angular_motion) {
    return point - body.world_trans.translation;
  }
  return body.world_trans.inv_x_form(point);
}

RS_HD Vec3A body_point_to_world(const RigidBody &body, Vec3A point) {
  if (body.collision_flags & k_flag_identity_transform) {
    return point;
  }
  if (body.collision_flags & k_flag_no_angular_motion) {
    return point + body.world_trans.translation;
  }
  return body.world_trans.transform_point3a(point);
}

struct PersistentManifold {
  ManifoldPoint points[k_manifold_cache_size];
  int num_points;
  uint32_t body0_idx;
  uint32_t body1_idx;
  float contact_breaking_threshold;
  float contact_processing_threshold;

  RS_HD static PersistentManifold make(const RigidBody &body0,
                                       const RigidBody &body1) {
    PersistentManifold m;
    m.reset(body0, body1);
    return m;
  }

  RS_HD void reset(const RigidBody &body0, const RigidBody &body1) {
    num_points = 0;
    body0_idx = body0.world_array_idx;
    body1_idx = body1.world_array_idx;
    contact_breaking_threshold = min_f32(body0.contact_breaking_threshold,
                                         body1.contact_breaking_threshold);
    contact_processing_threshold = min_f32(body0.contact_processing_threshold,
                                           body1.contact_processing_threshold);
  }

  RS_HD void copy_used_from(const PersistentManifold &o) {
    for (int i = 0; i < o.num_points; ++i) {
      points[i] = o.points[i];
    }
    num_points = o.num_points;
    body0_idx = o.body0_idx;
    body1_idx = o.body1_idx;
    contact_breaking_threshold = o.contact_breaking_threshold;
    contact_processing_threshold = o.contact_processing_threshold;
  }

  RS_HD static float combined_friction(const RigidBody &b0,
                                       const RigidBody &b1) {
    if (b0.is_static_obj() || b1.is_static_obj()) {
      return min_f32(b0.friction, b1.friction);
    }
    return b0.friction * b1.friction;
  }

  RS_HD static float combined_restitution(const RigidBody &b0,
                                          const RigidBody &b1) {
    if (b0.is_static_obj() || b1.is_static_obj()) {
      return max_f32(b0.restitution, b1.restitution);
    }
    return b0.restitution * b1.restitution;
  }

  RS_HD static float get_res(Vec3A new_local, Vec3A p1, Vec3A p2, Vec3A p3) {
    return (new_local - p1).cross(p2 - p3).length_squared();
  }

  RS_HD int sort_cached_points(const ManifoldPoint &new_contact) const {
    Vec3A nl = new_contact.local_point_a;
    int max_penetration_idx = k_manifold_cache_size;
    float max_penetration = new_contact.distance_1;
    for (int i = 0; i < num_points; ++i) {
      if (points[i].distance_1 < max_penetration) {
        max_penetration_idx = i;
        max_penetration = points[i].distance_1;
      }
    }
    Vec3A p0 = points[0].local_point_a;
    Vec3A p1 = points[1].local_point_a;
    Vec3A p2 = points[2].local_point_a;
    Vec3A p3 = points[3].local_point_a;
    float r0 = max_penetration_idx == 0 ? 0.0f : get_res(nl, p1, p3, p2);
    float r1 = max_penetration_idx == 1 ? 0.0f : get_res(nl, p0, p3, p2);
    float r2 = max_penetration_idx == 2 ? 0.0f : get_res(nl, p0, p3, p1);
    float r3 = max_penetration_idx == 3 ? 0.0f : get_res(nl, p0, p2, p1);
    int biggest = 0;
    float max_area = r0;
    if (r1 > max_area) {
      biggest = 1;
      max_area = r1;
    }
    if (r2 > max_area) {
      biggest = 2;
      max_area = r2;
    }
    if (r3 > max_area) {
      biggest = 3;
    }
    return biggest;
  }

  RS_HD int get_cache_entry(const ManifoldPoint &new_contact) const {
    float threshold_sq =
        contact_breaking_threshold * contact_breaking_threshold;
    float shortest = threshold_sq;
    int nearest = -1;
    for (int i = 0; i < num_points; ++i) {
      float d2 = (points[i].local_point_a - new_contact.local_point_a)
                     .length_squared();
      if (d2 < shortest) {
        shortest = d2;
        nearest = i;
      }
    }
    return nearest;
  }

  RS_HD int add_manifold_point(const ManifoldPoint &contact) {
    if (num_points == k_manifold_cache_size) {
      int idx = sort_cached_points(contact);
      points[idx] = contact;
      return idx;
    }
    points[num_points] = contact;
    return num_points++;
  }

  RS_HD int add_contact_without_callback(ManifoldPoint contact) {
    int idx = get_cache_entry(contact);
    if (idx >= 0) {
      contact.applied_impulse = points[idx].applied_impulse;
      points[idx] = contact;
      return idx;
    }
    return add_manifold_point(contact);
  }

  RS_HD void merge_contact_points(const PersistentManifold &other) {
    if (num_points == 0) {
      for (int i = 0; i < other.num_points; ++i) {
        points[num_points++] = other.points[i];
      }
      return;
    }
    for (int i = 0; i < other.num_points; ++i) {
      add_contact_without_callback(other.points[i]);
    }
  }

  RS_HD void swap_remove(int i) {
    --num_points;
    points[i] = points[num_points];
  }

  RS_HD void refresh_contact_points(const RigidBody &body0,
                                    const RigidBody &body1) {
    if (num_points == 0) {
      return;
    }
    for (int i = 0; i < num_points; ++i) {
      ManifoldPoint &p = points[i];
      p.pos_world_on_a = body_point_to_world(body0, p.local_point_a);
      p.pos_world_on_b = body_point_to_world(body1, p.local_point_b);
      p.distance_1 =
          (p.pos_world_on_a - p.pos_world_on_b).dot(p.normal_world_on_b);
    }
    float threshold_sq =
        contact_breaking_threshold * contact_breaking_threshold;
    for (int i = num_points - 1; i >= 0; --i) {
      ManifoldPoint p = points[i];
      if (p.distance_1 > contact_breaking_threshold) {
        swap_remove(i);
        continue;
      }
      Vec3A projected = p.pos_world_on_a - p.normal_world_on_b * p.distance_1;
      Vec3A diff = p.pos_world_on_b - projected;
      if (diff.dot(diff) > threshold_sq) {
        swap_remove(i);
      }
    }
  }

  template <typename Callback>
  RS_HD void add_contact_point(const RigidBody &body0, const RigidBody &body1,
                               Vec3A normal_on_b_in_world, Vec3A point_in_world,
                               float depth, int tri_idx, Callback &callback) {
    if (depth > contact_breaking_threshold) {
      return;
    }
    Vec3A point_a = point_in_world + normal_on_b_in_world * depth;
    Vec3A local_a = body_point_to_local(body0, point_a);
    Vec3A local_b = body_point_to_local(body1, point_in_world);
    ManifoldPoint np =
        ManifoldPoint::make(local_a, local_b, normal_on_b_in_world, depth);
    np.pos_world_on_a = point_a;
    np.pos_world_on_b = point_in_world;
    np.combined_friction = combined_friction(body0, body1);
    np.combined_restitution = combined_restitution(body0, body1);
    int insert_idx = add_contact_without_callback(np);
    if ((body0.collision_flags & k_flag_custom_material_callback) ||
        (body1.collision_flags & k_flag_custom_material_callback)) {
      callback.callback(points[insert_idx], body0, body1, tri_idx);
    }
  }
};

}
