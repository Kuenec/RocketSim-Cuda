#pragma once

#include <cstdint>

#include "rsc/bullet/manifold.hpp"
#include "rsc/bullet/rigid_body.hpp"
#include "rsc/bullet/shapes.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

inline constexpr int k_max_cars = 8;
inline constexpr int k_max_dyn_bodies = 1 + k_max_cars;
inline constexpr int k_max_static_bodies = 160;
inline constexpr int k_max_handles = k_max_static_bodies + k_max_dyn_bodies + 1;
inline constexpr int k_cell_mask_words = (k_max_static_bodies + 31) / 32;
inline constexpr int k_max_edges = 2 * k_max_handles + 2;
inline constexpr int k_max_pairs = 256;
inline constexpr int k_max_manifolds = 128;
inline constexpr int k_max_hull_points = 40;

enum ErrorFlag : uint32_t {
  k_error_pairs_overflow = 1u << 0,
  k_error_manifolds_overflow = 1u << 1,
  k_error_records_overflow = 1u << 2,
  k_error_solver_overflow = 1u << 3,
  k_error_handles_overflow = 1u << 4,
  k_error_invalid_state = 1u << 5,
  k_error_events_overflow = 1u << 6,
};

enum CollisionFilterGroup : uint8_t {
  k_group_default = 1,
  k_group_static = 1 << 1,
  k_group_hoops_net = 1 << 2,
  k_group_dropshot_tile = 1 << 3,
  k_group_dropshot_floor = 1 << 4,
  k_group_all = 0x1f,
};

struct BroadphaseProxy {
  uint32_t client_obj_idx;
  uint8_t collision_filter_group;
  uint8_t collision_filter_mask;
  Aabb aabb;
  uint32_t cell_idx;
  uint32_t indices[3];
  uint64_t stamp;
};

struct SweepEdge {
  uint16_t pos;
  uint16_t handle;
};

struct HandleEdges {
  uint16_t min[3];
  uint16_t max[3];
};

struct BroadphasePair {
  uint16_t proxy0;
  uint16_t proxy1;
};

struct StaticWorldData {
  uint32_t num_static_bodies;
  const RigidBody *static_bodies;
  const BroadphaseProxy *static_proxies;
  const MeshView *meshes;
  const Vec3A *hull_points;
  Vec3A grid_min;
  Vec3A grid_max;
  float cell_size;
  float cell_size_sq;
  uint32_t num_cells[3];
  uint32_t cell_mask_words;
  const uint32_t *cell_static_masks;
  const float *sin_table;
};

struct ArenaPhysics {
  uint32_t num_dyn_bodies;
  RigidBody dyn_bodies[k_max_dyn_bodies];
  BroadphaseProxy dyn_proxies[k_max_dyn_bodies];
  uint64_t next_stamp;
  uint16_t num_handles;
  uint16_t next_new_handle;
  uint16_t free_count;
  uint16_t free_list[k_max_dyn_bodies];
  uint16_t num_edges;
  SweepEdge edges[3][k_max_edges];
  HandleEdges handle_edges[k_max_handles];
  uint32_t num_pairs;
  BroadphasePair pairs[k_max_pairs];
  uint32_t num_manifolds;
  PersistentManifold manifolds[k_max_manifolds];
  uint32_t num_active;
  uint16_t active[k_max_manifolds];
  Vec3A gravity;
  bool ball_only;
  bool skip_separated_special_rows;
  uint32_t error_flags;
  uint32_t puck_hull_count;
  Vec3A puck_hull_points[k_max_hull_points];
  uint8_t tile_broken[140];
};

struct World {
  const StaticWorldData *s;
  ArenaPhysics *a;

  RS_HDI uint32_t num_static() const { return s->num_static_bodies; }

  RS_HDI uint32_t num_bodies() const {
    return s->num_static_bodies + a->num_dyn_bodies;
  }

  RS_HDI const RigidBody &body(uint32_t i) const {
    return i < s->num_static_bodies ? s->static_bodies[i]
                                    : a->dyn_bodies[i - s->num_static_bodies];
  }

  RS_HDI RigidBody &dyn_body(uint32_t i) {
    return a->dyn_bodies[i - s->num_static_bodies];
  }

  RS_HDI bool has_contact_response(uint32_t i) const {
    const RigidBody &b = body(i);
    if (b.user_idx == UserInfoType::DropshotTile) {
      return a->tile_broken[b.user_pointer] == 0;
    }
    return b.has_contact_response();
  }

  RS_HDI const BroadphaseProxy &proxy(uint32_t handle) const {
    return handle <= s->num_static_bodies
               ? s->static_proxies[handle - 1]
               : a->dyn_proxies[handle - s->num_static_bodies - 1];
  }

  RS_HDI BroadphaseProxy &dyn_proxy(uint32_t handle) {
    return a->dyn_proxies[handle - s->num_static_bodies - 1];
  }

  RS_HDI HullView hull_view(const ConvexHullShape &h) const {
    if (h.source == HullSource::Arena) {
      return {a->puck_hull_points + h.offset, h.count};
    }
    return {s->hull_points + h.offset, h.count};
  }

  RS_HDI const MeshView &mesh(const TriMeshShape &m) const {
    return s->meshes[m.mesh_idx];
  }

  RS_HD ConvexShape convex_of(const Shape &shape) const {
    switch (shape.kind) {
    case ShapeKind::Sphere:
      return ConvexShape::sphere(shape.sphere.radius);
    case ShapeKind::Compound:
      return shape.compound.child_shape.convex();
    case ShapeKind::ConvexHull:
      return ConvexShape::hull_shape(hull_view(shape.hull));
    default:
      return ConvexShape::sphere(0.0f);
    }
  }

  RS_HDI void flag_error(uint32_t flag) { a->error_flags |= flag; }
};

}
