#pragma once

#include <cstdint>

#include "rsc/bullet/broadphase.hpp"
#include "rsc/bullet/gjk.hpp"
#include "rsc/bullet/world_types.hpp"
#include "rsc/math/shared.hpp"

namespace rsc {

struct QuadRayResult {
  Vec4 closest_hit_fraction;
  int collision_obj_idx[4];
  int ignore_obj_world_idx;
  uint8_t collision_filter_group;
  uint8_t collision_filter_mask;
  uint8_t active_rays;
  float min_hit_dist_sq;
  const Vec3A *ray_from_world;
  const Vec3A *ray_to_world;
  Vec3A hit_normal_world[4];
  Vec3A hit_point_world[4];

  RS_HD static QuadRayResult make(const Vec3A *from, const Vec3A *to,
                                  int ignore_idx) {
    QuadRayResult r;
    r.closest_hit_fraction = Vec4::splat(1.0f);
    for (int i = 0; i < 4; ++i) {
      r.collision_obj_idx[i] = -1;
      r.hit_normal_world[i] = Vec3A::zero();
      r.hit_point_world[i] = Vec3A::zero();
    }
    r.ignore_obj_world_idx = ignore_idx;
    r.collision_filter_group = k_group_default;
    r.collision_filter_mask = k_group_all;
    r.active_rays = 0xf;
    r.min_hit_dist_sq = 0.0f;
    r.ray_from_world = from;
    r.ray_to_world = to;
    return r;
  }

  RS_HD bool needs_collision(const BroadphaseProxy &p) const {
    if (ignore_obj_world_idx >= 0 &&
        static_cast<uint32_t>(ignore_obj_world_idx) == p.client_obj_idx) {
      return false;
    }
    return (p.collision_filter_group & collision_filter_mask) != 0 &&
           (collision_filter_group & p.collision_filter_mask) != 0;
  }

  RS_HD void add_single_result(int obj_idx, Vec3A normal_world,
                               float hit_fraction, int ray_idx) {
    if (hit_fraction >= closest_hit_fraction[ray_idx]) {
      return;
    }
    float s = 1.0f - hit_fraction;
    Vec3A candidate =
        s * ray_from_world[ray_idx] + hit_fraction * ray_to_world[ray_idx];
    if (min_hit_dist_sq > 0.0f &&
        (candidate - ray_from_world[ray_idx]).length_squared() <
            min_hit_dist_sq) {
      return;
    }
    closest_hit_fraction[ray_idx] = hit_fraction;
    hit_normal_world[ray_idx] = normal_world;
    collision_obj_idx[ray_idx] = obj_idx;
    hit_point_world[ray_idx] = candidate;
  }
};

struct BridgeTriQuadRay {
  const Vec3A *to;
  const Vec3A *from;
  Vec4 hit_fraction;
  const RigidBody *obj;
  int obj_idx;
  QuadRayResult *result;
  const TriangleShape *tris;

  RS_HD void internal_report_hit(Vec3A normal_local, float fraction,
                                 int ray_idx) {
    Vec3A normal_world = obj->world_trans.matrix3 * normal_local;
    result->add_single_result(obj_idx, normal_world, fraction, ray_idx);
  }

  RS_HD void report_hit(Vec3A normal_local, float fraction, int ray_idx) {
    if (fraction >= hit_fraction[ray_idx]) {
      return;
    }
    internal_report_hit(normal_local, fraction, ray_idx);
  }

  RS_HD void process_triangle(const TriangleShape &tri, float &lambda_max,
                              int ray_idx) {
    constexpr float edge_tolerance = -0.0001f;
    float dist = tri.points[0].dot(tri.normal);
    float dist_a = tri.normal.dot(from[ray_idx]) - dist;
    float dist_b = tri.normal.dot(to[ray_idx]) - dist;
    if (dist_a * dist_b >= 0.0f) {
      return;
    }
    float proj_length = dist_a - dist_b;
    float distance = dist_a / proj_length;
    float hf = hit_fraction[ray_idx];
    if (distance >= hf) {
      lambda_max = min_f32(lambda_max, hf);
      return;
    }
    Vec3A point = from[ray_idx].lerp(to[ray_idx], distance);
    Vec3A v0p = tri.points[0] - point;
    Vec3A v1p = tri.points[1] - point;
    Vec3A cp0 = v0p.cross(v1p);
    if (cp0.dot(tri.normal) < edge_tolerance) {
      return;
    }
    Vec3A v2p = tri.points[2] - point;
    Vec3A cp1 = v1p.cross(v2p);
    if (cp1.dot(tri.normal) < edge_tolerance) {
      return;
    }
    Vec3A cp2 = v2p.cross(v0p);
    if (cp2.dot(tri.normal) < edge_tolerance) {
      return;
    }
    float prev = result->closest_hit_fraction[ray_idx];
    if (dist_a <= 0.0f) {
      internal_report_hit(-tri.normal, distance, ray_idx);
    } else {
      internal_report_hit(tri.normal, distance, ray_idx);
    }
    if (result->closest_hit_fraction[ray_idx] < prev) {
      lambda_max = result->closest_hit_fraction[ray_idx];
    }
  }

  RS_HD void process_node(uint32_t tri_idx, uint32_t mask, Vec4 &lambda_max) {
    uint32_t active = mask & result->active_rays;
    for (int i = 0; i < 4; ++i) {
      if (active & (1u << i)) {
        process_triangle(tris[tri_idx], lambda_max[i], i);
      }
    }
  }
};

RS_HD bool solve_rounded_box(Vec3A source_l, Vec3A dir_l, float dist,
                             Vec3A inner_half, float margin, float &t_out,
                             Vec3A &n_out) {
  float points[8] = {0.0f, dist, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int count = 2;
  for (int axis = 0; axis < 3; ++axis) {
    float d = dir_l[axis];
    if (d == 0.0f) {
      continue;
    }
    float s = source_l[axis];
    float h = inner_half[axis];
    const float bounds[2] = {-h, h};
    for (int bi = 0; bi < 2; ++bi) {
      float t = (bounds[bi] - s) / d;
      if (t > 0.0f && t < dist && count < 8) {
        points[count++] = t;
      }
    }
  }
  for (int i = 1; i < count; ++i) {
    float key = points[i];
    int j = i;
    while (j > 0 && points[j - 1] > key) {
      points[j] = points[j - 1];
      --j;
    }
    points[j] = key;
  }
  float margin_sq = margin * margin;
  for (int i = 0; i < count - 1; ++i) {
    float t0 = points[i];
    float t1 = points[i + 1];
    if (t1 <= t0) {
      continue;
    }
    float mid = t0 + (t1 - t0) * 0.5f;
    Vec3A p_mid = source_l + dir_l * mid;
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    uint32_t outside = 0;
    for (int axis = 0; axis < 3; ++axis) {
      float pm = p_mid[axis];
      float h = inner_half[axis];
      float bound;
      if (pm < -h) {
        bound = -h;
      } else if (pm > h) {
        bound = h;
      } else {
        continue;
      }
      ++outside;
      float o = source_l[axis] - bound;
      float d = dir_l[axis];
      a += d * d;
      b += 2.0f * d * o;
      c += o * o;
    }
    if (outside == 0) {
      continue;
    }
    float cc = c - margin_sq;
    float disc = b * b - 4.0f * a * cc;
    if (disc < 0.0f) {
      float scale = b * b + abs_f32(4.0f * a * cc);
      if (disc > -k_f32_epsilon * scale) {
        disc = 0.0f;
      } else {
        continue;
      }
    }
    float sqrt_d = sqrt_f32(disc);
    float t_hit;
    if (disc == 0.0f) {
      t_hit = -b / (2.0f * a);
    } else {
      float q = -0.5f * (b >= 0.0f ? b + sqrt_d : b - sqrt_d);
      if (q == 0.0f) {
        t_hit = -b / (2.0f * a);
      } else {
        float r0 = q / a;
        float r1 = cc / q;
        t_hit = r0 <= r1 ? r0 : r1;
      }
    }
    if (t_hit <= 0.0f || t_hit > dist || t_hit < t0 || t_hit > t1) {
      continue;
    }
    Vec3A hit = source_l + dir_l * t_hit;
    Vec3A clamped = hit.clamp(-inner_half, inner_half);
    Vec3A n = hit - clamped;
    float len_sq = n.length_squared();
    if (len_sq <= k_f32_epsilon * k_f32_epsilon) {
      continue;
    }
    Vec3A n_norm = n / sqrt_f32(len_sq);
    if (n_norm.dot(dir_l) > 0.0f) {
      continue;
    }
    t_out = t_hit;
    n_out = n_norm;
    return true;
  }
  return false;
}

RS_HD void compound_ray(const CompoundShape &cs, BridgeTriQuadRay &cb,
                        Vec3A ray_source, Vec3A ray_target, int ray_idx) {
  Vec3A delta = ray_target - ray_source;
  float dist = delta.length();
  Vec3A dir = delta / dist;
  float tenter = 0.0f;
  float texit = dist;
  int hit_axis = 0;
  Vec3A inv = 1.0f / dir;
  Vec3A t1 = (cs.local_aabb.min - ray_source) * inv;
  Vec3A t2 = (cs.local_aabb.max - ray_source) * inv;
  Vec3A tmin = t1.min(t2);
  Vec3A tmax = t1.max(t2);
  uint32_t is_neg = tmax.is_negative_bitmask();
  BVec3 fin = inv.is_finite_mask();
  const bool is_finite[3] = {fin.x, fin.y, fin.z};
  for (int axis = 0; axis < 3; ++axis) {
    if (!is_finite[axis]) {
      float origin = ray_source[axis];
      if (cs.local_aabb.min[axis] > origin ||
          origin > cs.local_aabb.max[axis]) {
        return;
      }
      continue;
    }
    if (is_neg & (1u << axis)) {
      return;
    }
    texit = min_f32(texit, tmax[axis]);
    if (tmin[axis] > tenter) {
      tenter = tmin[axis];
      hit_axis = axis;
    }
    if (tenter > texit) {
      return;
    }
  }
  if (0.0f >= tenter || tenter > dist) {
    return;
  }
  Vec3A inner_half = cs.child_shape.half_extents;
  float margin = cs.child_shape.margin;
  Vec3A entry = ray_source + dir * tenter;
  Vec3A local = entry - cs.child_trans.translation;
  bool inside_face = true;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis == hit_axis) {
      continue;
    }
    if (abs_f32(local[axis]) > inner_half[axis]) {
      inside_face = false;
      break;
    }
  }
  if (inside_face) {
    Vec3A hit_normal = Vec3A::zero();
    hit_normal[hit_axis] = -signum_f32(dir[hit_axis]);
    if (hit_normal.length_squared() > 0.0f) {
      float hit_fraction = max_f32_c(tenter, 0.0f) / dist;
      cb.report_hit(hit_normal, hit_fraction, ray_idx);
    }
    return;
  }
  Vec3A source_l = ray_source - cs.child_trans.translation;
  float t_hit;
  Vec3A normal_c;
  if (!solve_rounded_box(source_l, dir, dist, inner_half, margin, t_hit,
                         normal_c)) {
    return;
  }
  normal_c = normal_c / normal_c.length();
  if (normal_c.dot(dir) > 0.0f) {
    return;
  }
  if (0.0f >= t_hit || t_hit > dist) {
    return;
  }
  cb.report_hit(normal_c, t_hit / dist, ray_idx);
}

struct SphereClosest {
  const SphereShape *sphere;
  RS_HD bool operator()(Vec3A p, float &dist, Vec3A &normal) const {
    return sphere->closest_point(p, dist, normal);
  }
};

inline constexpr float k_slab_relative_slack = 1e-5f;
inline constexpr float k_slab_absolute_slack = 1e-30f;

RS_HD Aabb quad_ray_node_cull(const QuadRayInfo &ray_info,
                              uint8_t active_rays) {
  if (active_rays == 0 || (active_rays & (active_rays - 1)) != 0) {
    return ray_info.aabb;
  }
  int ray = 0;
  while ((active_rays & (1u << ray)) == 0) {
    ++ray;
  }
  Vec3A source = ray_info.ray_sources[ray];
  Vec3A target = ray_info.ray_targets[ray];
  if (!source.is_finite() || !target.is_finite()) {
    return ray_info.aabb;
  }
  Vec3A slack = (target - source).abs() * k_slab_relative_slack +
                Vec3A::splat(k_slab_absolute_slack);
  Vec3A lo = source.min(target) - slack;
  Vec3A hi = source.max(target) + slack;
  return {lo.max(ray_info.aabb.min), hi.min(ray_info.aabb.max)};
}

RS_HD void shape_quad_raycast(const World &w, const Shape &shape,
                              BridgeTriQuadRay &cb, QuadRayInfo &ray_info) {
  switch (shape.kind) {
  case ShapeKind::Compound: {
    const CompoundShape &cs = shape.compound;
    if (!ray_info.aabb.intersects(cs.local_aabb)) {
      return;
    }
    QuadRayPack pack = ray_info.calc_pos_dir();
    uint32_t mask =
        QuadRayInfo::intersect_quad_ray_aabb(pack.origins, pack.inv_dirs,
                                             cs.local_aabb, cb.hit_fraction) &
        cb.result->active_rays;
    for (int i = 0; i < 4; ++i) {
      if (mask & (1u << i)) {
        compound_ray(cs, cb, ray_info.ray_sources[i], ray_info.ray_targets[i],
                     i);
      }
    }
    return;
  }
  case ShapeKind::Sphere: {
    const Vec3A *s = ray_info.ray_sources;
    const Vec3A *t = ray_info.ray_targets;
    Vec4 sx{s[0].x, s[1].x, s[2].x, s[3].x};
    Vec4 sy{s[0].y, s[1].y, s[2].y, s[3].y};
    Vec4 sz{s[0].z, s[1].z, s[2].z, s[3].z};
    Vec4 tx{t[0].x, t[1].x, t[2].x, t[3].x};
    Vec4 ty{t[0].y, t[1].y, t[2].y, t[3].y};
    Vec4 tz{t[0].z, t[1].z, t[2].z, t[3].z};
    float radius = shape.sphere.radius;
    Vec4 radius_sq = Vec4::splat(radius * radius);
    Vec4 zero = Vec4::splat(0.0f);
    Vec4 cx = zero.clamp(sx.min(tx), sx.max(tx));
    Vec4 cy = zero.clamp(sy.min(ty), sy.max(ty));
    Vec4 cz = zero.clamp(sz.min(tz), sz.max(tz));
    Vec4 len_sq = cx * cx + cy * cy + cz * cz;
    BVec4 within = len_sq.cmple(radius_sq);
    if (!within.any()) {
      return;
    }
    const bool w4[4] = {within.x, within.y, within.z, within.w};
    SphereClosest closest{&shape.sphere};
    for (int i = 0; i < 4; ++i) {
      if (!w4[i] || !(cb.result->active_rays & (1u << i))) {
        continue;
      }
      ConvexCastResult r;
      if (!calc_point_time_of_impact(s[i], t[i], closest, r)) {
        continue;
      }
      cb.report_hit(r.normal, r.fraction, i);
    }
    return;
  }
  case ShapeKind::StaticPlane: {
    const StaticPlaneShape &plane = shape.plane;
    if (!ray_info.aabb.intersects(plane.aabb_ident_cache)) {
      return;
    }
    const Vec3A *s = ray_info.ray_sources;
    const Vec3A *t = ray_info.ray_targets;
    Vec4 sx{s[0].x, s[1].x, s[2].x, s[3].x};
    Vec4 sy{s[0].y, s[1].y, s[2].y, s[3].y};
    Vec4 sz{s[0].z, s[1].z, s[2].z, s[3].z};
    Vec4 tx{t[0].x, t[1].x, t[2].x, t[3].x};
    Vec4 ty{t[0].y, t[1].y, t[2].y, t[3].y};
    Vec4 tz{t[0].z, t[1].z, t[2].z, t[3].z};
    Vec4 dx = tx - sx;
    Vec4 dy = ty - sy;
    Vec4 dz = tz - sz;
    Vec4 one = Vec4::splat(1.0f);
    Vec4 origins[3] = {sx, sy, sz};
    Vec4 invs[3] = {one / dx, one / dy, one / dz};
    uint32_t ray_mask = QuadRayInfo::intersect_quad_ray_aabb(
        origins, invs, plane.aabb_ident_cache, cb.hit_fraction);
    if (ray_mask == 0) {
      return;
    }
    Vec4 dist = (dx * dx + dy * dy + dz * dz).sqrt();
    Vec4 inv_dist = one / dist;
    Vec4 dirx = dx * inv_dist;
    Vec4 diry = dy * inv_dist;
    Vec4 dirz = dz * inv_dist;
    Vec3A n = plane.plane_normal;
    Vec4 dir_align = dirx * n.x + diry * n.y + dirz * n.z;
    BVec4 align_mask = dir_align.abs().cmpge(Vec4::splat(k_f32_epsilon));
    if (!align_mask.any()) {
      return;
    }
    Vec4 normal_start = sx * n.x + sy * n.y + sz * n.z;
    Vec4 tt = -normal_start / dir_align;
    BVec4 t_mask = tt.cmpge(Vec4::splat(0.0f)) & tt.cmplt(dist);
    uint32_t hit_mask =
        ray_mask & (align_mask & t_mask).bitmask() & cb.result->active_rays;
    if (hit_mask == 0) {
      return;
    }
    Vec4 hf = tt / dist;
    for (int i = 0; i < 4; ++i) {
      if (hit_mask & (1u << i)) {
        cb.report_hit(plane.plane_normal, hf[i], i);
      }
    }
    return;
  }
  case ShapeKind::TriangleMesh: {
    const MeshView &mesh = w.mesh(shape.mesh);
    cb.tris = mesh.tris;
    mesh.bvh.report_quad_ray_overlapping_node(
        cb, ray_info, quad_ray_node_cull(ray_info, cb.result->active_rays));
    return;
  }
  case ShapeKind::ConvexHull: {
    const ConvexHullShape &hull = shape.hull;
    if (!ray_info.aabb.intersects(hull.local_aabb)) {
      return;
    }
    QuadRayPack pack = ray_info.calc_pos_dir();
    uint32_t mask =
        QuadRayInfo::intersect_quad_ray_aabb(pack.origins, pack.inv_dirs,
                                             hull.local_aabb, cb.hit_fraction) &
        cb.result->active_rays;
    ConvexShape cs = ConvexShape::hull_shape(w.hull_view(hull));
    for (int i = 0; i < 4; ++i) {
      if (!(mask & (1u << i))) {
        continue;
      }
      ConvexCastResult r;
      if (calc_time_of_impact(cs, ray_info.ray_sources[i],
                              ray_info.ray_targets[i], r)) {
        cb.report_hit(r.normal, r.fraction, i);
      }
    }
    return;
  }
  }
}

RS_HD void quad_ray_test_body(const World &w, const Vec3A *ray_from,
                              const Vec3A *ray_to, const RigidBody &co,
                              int obj_idx, QuadRayResult &result) {
  Vec3A from_local[4];
  Vec3A to_local[4];
  if (co.collision_flags & k_flag_identity_transform) {
    for (int i = 0; i < 4; ++i) {
      from_local[i] = ray_from[i];
      to_local[i] = ray_to[i];
    }
  } else if (co.collision_flags & k_flag_no_angular_motion) {
    Vec3A t = co.world_trans.translation;
    for (int i = 0; i < 4; ++i) {
      from_local[i] = ray_from[i] - t;
      to_local[i] = ray_to[i] - t;
    }
  } else {
    Affine3A world_to_co = co.world_trans.ext_transpose();
    for (int i = 0; i < 4; ++i) {
      from_local[i] = world_to_co.transform_point3a(ray_from[i]);
      to_local[i] = world_to_co.transform_point3a(ray_to[i]);
    }
  }
  BridgeTriQuadRay cb{to_local, from_local, result.closest_hit_fraction,
                      &co,      obj_idx,    &result,
                      nullptr};
  QuadRayInfo ray_info = QuadRayInfo::make(from_local, to_local);
  ray_info.lambda_max = cb.hit_fraction;
  shape_quad_raycast(w, co.shape, cb, ray_info);
}

RS_HD void grid_ray_test(const World &w, const Vec3A *ray_from,
                         const Vec3A *ray_to, QuadRayResult &result) {
  const StaticWorldData &s = *w.s;
  Aabb ray_aabb{ray_from[0].min(ray_to[0]), ray_from[0].max(ray_to[0])};
  for (int i = 1; i < 4; ++i) {
    ray_aabb = ray_aabb.combine(
        Aabb{ray_from[i].min(ray_to[i]), ray_from[i].max(ray_to[i])});
  }
  uint32_t cell_idxs[4];
  uint32_t cell_coords[4][3];
  int num_cells = 0;
  for (int r = 0; r < 4; ++r) {
    uint32_t idx3[3];
    grid_cell_indices(s, ray_from[r], idx3);
    uint32_t idx = grid_cell_index(s, idx3);
    bool found = false;
    for (int c = 0; c < num_cells; ++c) {
      if (cell_idxs[c] == idx) {
        found = true;
        break;
      }
    }
    if (!found) {
      cell_idxs[num_cells] = idx;
      cell_coords[num_cells][0] = idx3[0];
      cell_coords[num_cells][1] = idx3[1];
      cell_coords[num_cells][2] = idx3[2];
      ++num_cells;
    }
  }
  uint32_t dyn_order[k_max_dyn_bodies];
  uint32_t num_dyn = 0;
  for (uint32_t d = 0; d < w.a->num_dyn_bodies; ++d) {
    uint32_t handle = w.a->dyn_bodies[d].broadphase_handle;
    uint32_t pos = num_dyn;
    uint64_t stamp = w.proxy(handle).stamp;
    while (pos > 0 && w.proxy(dyn_order[pos - 1]).stamp > stamp) {
      dyn_order[pos] = dyn_order[pos - 1];
      --pos;
    }
    dyn_order[pos] = handle;
    ++num_dyn;
  }
  uint32_t seen[k_max_handles];
  uint32_t num_seen = 0;
  uint32_t bodies[k_max_handles];
  uint32_t num_bodies = 0;
  for (int c = 0; c < num_cells; ++c) {
    uint32_t cell = cell_idxs[c];
    uint32_t begin = s.cell_static_offsets[cell];
    uint32_t end = s.cell_static_offsets[cell + 1];
    uint32_t total = (end - begin) + num_dyn;
    for (uint32_t k = 0; k < total; ++k) {
      uint32_t handle;
      if (k < end - begin) {
        handle = s.cell_static_handles[begin + k];
      } else {
        handle = dyn_order[k - (end - begin)];
        if (!grid_dyn_in_cell(w.proxy(handle), cell_coords[c])) {
          continue;
        }
      }
      if (num_cells > 1) {
        bool dup = false;
        for (uint32_t q = 0; q < num_seen; ++q) {
          if (seen[q] == handle) {
            dup = true;
            break;
          }
        }
        if (dup) {
          continue;
        }
        seen[num_seen++] = handle;
      }
      const BroadphaseProxy &p = w.proxy(handle);
      if (ray_aabb.intersects(p.aabb) && result.needs_collision(p)) {
        bodies[num_bodies++] = p.client_obj_idx;
      }
    }
  }
  for (uint32_t k = 0; k < num_bodies; ++k) {
    quad_ray_test_body(w, ray_from, ray_to, w.body(bodies[k]),
                       static_cast<int>(bodies[k]), result);
  }
}

}
