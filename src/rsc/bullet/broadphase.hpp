#pragma once

#include <cstdint>

#include "rsc/bullet/world_types.hpp"
#include "rsc/math/shared.hpp"

namespace rsc {

inline constexpr uint16_t k_sweep_handle_mask = 0xfffe;
inline constexpr uint16_t k_sweep_sentinel = 0xffff;

RS_HDI float sweep_world_half_extent() { return u32_to_f32(0x4623d6f6u); }

RS_HDI int32_t x86_cvtt_f32_to_i32(float x) {
  if (!(x > -2147483904.0f && x < 2147483648.0f)) {
    return INT32_MIN;
  }
  return static_cast<int32_t>(x);
}

RS_HD void sweep_quantize_point(Vec3A point, uint16_t is_max, uint16_t *out) {
  float half = sweep_world_half_extent();
  Vec3A world_min = Vec3A::splat(-half);
  float quant = static_cast<float>(k_sweep_sentinel) / (half - -half);
  Vec3A scaled = (point - world_min) * Vec3A::splat(quant);
  Vec3A clamped = Vec3A::select(
      scaled.cmpge(Vec3A::zero()),
      scaled.min(Vec3A::splat(static_cast<float>(k_sweep_sentinel))),
      Vec3A::zero());
  for (int i = 0; i < 3; ++i) {
    uint16_t q = static_cast<uint16_t>(x86_cvtt_f32_to_i32(clamped[i]));
    out[i] = static_cast<uint16_t>((q & k_sweep_handle_mask) | is_max);
  }
}

RS_HDI bool proxies_need_collision(const BroadphaseProxy &p0,
                                   const BroadphaseProxy &p1) {
  return (p0.collision_filter_group & p1.collision_filter_mask) != 0 &&
         (p1.collision_filter_group & p0.collision_filter_mask) != 0;
}

RS_HD int find_pair(const ArenaPhysics &a, uint16_t lo, uint16_t hi) {
  for (uint32_t i = 0; i < a.num_pairs; ++i) {
    if (a.pairs[i].proxy0 == lo && a.pairs[i].proxy1 == hi) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

RS_HD void pair_cache_add(World &w, uint16_t handle_a, uint16_t handle_b) {
  if (handle_a <= w.num_static() && handle_b <= w.num_static()) {
    return;
  }
  if (!proxies_need_collision(w.proxy(handle_a), w.proxy(handle_b))) {
    return;
  }
  uint16_t lo = handle_a <= handle_b ? handle_a : handle_b;
  uint16_t hi = handle_a <= handle_b ? handle_b : handle_a;
  if (find_pair(*w.a, lo, hi) >= 0) {
    return;
  }
  if (w.a->num_pairs >= static_cast<uint32_t>(k_max_pairs)) {
    w.flag_error(k_error_pairs_overflow);
    return;
  }
  w.a->pairs[w.a->num_pairs++] = {lo, hi};
}

RS_HD void dispatcher_release_manifold(World &w, uint32_t body0,
                                       uint32_t body1);

RS_HD void pair_cache_remove(World &w, uint16_t handle_a, uint16_t handle_b) {
  uint16_t lo = handle_a < handle_b ? handle_a : handle_b;
  uint16_t hi = handle_a < handle_b ? handle_b : handle_a;
  int index = find_pair(*w.a, lo, hi);
  if (index < 0) {
    return;
  }
  ArenaPhysics &a = *w.a;
  BroadphasePair removed = a.pairs[index];
  uint32_t last = a.num_pairs - 1;
  if (static_cast<uint32_t>(index) != last) {
    BroadphasePair tmp = a.pairs[index];
    a.pairs[index] = a.pairs[last];
    a.pairs[last] = tmp;
  }
  a.num_pairs -= 1;
  dispatcher_release_manifold(w, w.proxy(removed.proxy0).client_obj_idx,
                              w.proxy(removed.proxy1).client_obj_idx);
}

RS_HD bool sweep_test_overlap_2d(const ArenaPhysics &a, uint16_t ha,
                                 uint16_t hb, int axis) {
  int axis1 = (1 << axis) & 3;
  int axis2 = (1 << axis1) & 3;
  const HandleEdges &ea = a.handle_edges[ha];
  const HandleEdges &eb = a.handle_edges[hb];
  return !(ea.max[axis1] < eb.min[axis1] || eb.max[axis1] < ea.min[axis1] ||
           ea.max[axis2] < eb.min[axis2] || eb.max[axis2] < ea.min[axis2]);
}

RS_HDI bool edge_is_max(SweepEdge e) { return (e.pos & 1) != 0; }

RS_HD void sweep_sort_min_down(World &w, int axis, uint16_t edge,
                               bool update_overlaps) {
  ArenaPhysics &a = *w.a;
  int index = edge;
  uint16_t moving = a.edges[axis][index].handle;
  while (index > 0 && a.edges[axis][index].pos < a.edges[axis][index - 1].pos) {
    SweepEdge prev = a.edges[axis][index - 1];
    if (edge_is_max(prev)) {
      if (update_overlaps &&
          sweep_test_overlap_2d(a, moving, prev.handle, axis)) {
        pair_cache_add(w, moving, prev.handle);
      }
      a.handle_edges[prev.handle].max[axis] =
          static_cast<uint16_t>(a.handle_edges[prev.handle].max[axis] + 1);
    } else {
      a.handle_edges[prev.handle].min[axis] =
          static_cast<uint16_t>(a.handle_edges[prev.handle].min[axis] + 1);
    }
    a.handle_edges[moving].min[axis] =
        static_cast<uint16_t>(a.handle_edges[moving].min[axis] - 1);
    SweepEdge tmp = a.edges[axis][index];
    a.edges[axis][index] = a.edges[axis][index - 1];
    a.edges[axis][index - 1] = tmp;
    --index;
  }
}

RS_HD void sweep_sort_max_up(World &w, int axis, uint16_t edge,
                             bool update_overlaps) {
  ArenaPhysics &a = *w.a;
  int index = edge;
  uint16_t moving = a.edges[axis][index].handle;
  int len = a.num_edges;
  while (index + 1 < len && a.edges[axis][index + 1].handle != 0 &&
         a.edges[axis][index].pos >= a.edges[axis][index + 1].pos) {
    SweepEdge next = a.edges[axis][index + 1];
    if (!edge_is_max(next)) {
      if (update_overlaps &&
          sweep_test_overlap_2d(a, moving, next.handle, axis)) {
        pair_cache_add(w, moving, next.handle);
      }
      a.handle_edges[next.handle].min[axis] =
          static_cast<uint16_t>(a.handle_edges[next.handle].min[axis] - 1);
    } else {
      a.handle_edges[next.handle].max[axis] =
          static_cast<uint16_t>(a.handle_edges[next.handle].max[axis] - 1);
    }
    a.handle_edges[moving].max[axis] =
        static_cast<uint16_t>(a.handle_edges[moving].max[axis] + 1);
    SweepEdge tmp = a.edges[axis][index];
    a.edges[axis][index] = a.edges[axis][index + 1];
    a.edges[axis][index + 1] = tmp;
    ++index;
  }
}

RS_HD void sweep_sort_min_up(World &w, int axis, uint16_t edge,
                             bool update_overlaps) {
  ArenaPhysics &a = *w.a;
  int index = edge;
  uint16_t moving = a.edges[axis][index].handle;
  int len = a.num_edges;
  while (index + 1 < len && a.edges[axis][index + 1].handle != 0 &&
         a.edges[axis][index].pos >= a.edges[axis][index + 1].pos) {
    SweepEdge next = a.edges[axis][index + 1];
    if (edge_is_max(next)) {
      if (update_overlaps &&
          sweep_test_overlap_2d(a, moving, next.handle, axis)) {
        pair_cache_remove(w, moving, next.handle);
      }
      a.handle_edges[next.handle].max[axis] =
          static_cast<uint16_t>(a.handle_edges[next.handle].max[axis] - 1);
    } else {
      a.handle_edges[next.handle].min[axis] =
          static_cast<uint16_t>(a.handle_edges[next.handle].min[axis] - 1);
    }
    a.handle_edges[moving].min[axis] =
        static_cast<uint16_t>(a.handle_edges[moving].min[axis] + 1);
    SweepEdge tmp = a.edges[axis][index];
    a.edges[axis][index] = a.edges[axis][index + 1];
    a.edges[axis][index + 1] = tmp;
    ++index;
  }
}

RS_HD void sweep_sort_max_down(World &w, int axis, uint16_t edge,
                               bool update_overlaps) {
  ArenaPhysics &a = *w.a;
  int index = edge;
  uint16_t moving = a.edges[axis][index].handle;
  while (index > 0 && a.edges[axis][index].pos < a.edges[axis][index - 1].pos) {
    SweepEdge prev = a.edges[axis][index - 1];
    if (!edge_is_max(prev)) {
      if (update_overlaps &&
          sweep_test_overlap_2d(a, moving, prev.handle, axis)) {
        pair_cache_remove(w, moving, prev.handle);
      }
      a.handle_edges[prev.handle].min[axis] =
          static_cast<uint16_t>(a.handle_edges[prev.handle].min[axis] + 1);
    } else {
      a.handle_edges[prev.handle].max[axis] =
          static_cast<uint16_t>(a.handle_edges[prev.handle].max[axis] + 1);
    }
    a.handle_edges[moving].max[axis] =
        static_cast<uint16_t>(a.handle_edges[moving].max[axis] - 1);
    SweepEdge tmp = a.edges[axis][index];
    a.edges[axis][index] = a.edges[axis][index - 1];
    a.edges[axis][index - 1] = tmp;
    --index;
  }
}

RS_HD void sweep_update_axis(World &w, int axis, uint16_t handle, uint16_t qmin,
                             uint16_t qmax) {
  ArenaPhysics &a = *w.a;
  uint16_t min_edge = a.handle_edges[handle].min[axis];
  uint16_t max_edge = a.handle_edges[handle].max[axis];
  int32_t dmin = static_cast<int32_t>(qmin) -
                 static_cast<int32_t>(a.edges[axis][min_edge].pos);
  int32_t dmax = static_cast<int32_t>(qmax) -
                 static_cast<int32_t>(a.edges[axis][max_edge].pos);
  if (dmin == 0 && dmax == 0) {
    return;
  }
  a.edges[axis][min_edge].pos = qmin;
  a.edges[axis][max_edge].pos = qmax;
  if (dmin < 0) {
    sweep_sort_min_down(w, axis, min_edge, true);
  }
  if (dmax > 0) {
    sweep_sort_max_up(w, axis, max_edge, true);
  }
  if (dmin > 0) {
    sweep_sort_min_up(w, axis, min_edge, true);
  }
  if (dmax < 0) {
    sweep_sort_max_down(w, axis, max_edge, true);
  }
}

RS_HD void sweep_set_aabb(World &w, uint16_t handle, Vec3A min, Vec3A max) {
  BroadphaseProxy &proxy = w.dyn_proxy(handle);
  proxy.aabb.min = min;
  proxy.aabb.max = max;
  uint16_t qmin[3];
  uint16_t qmax[3];
  sweep_quantize_point(min, 0, qmin);
  sweep_quantize_point(max, 1, qmax);
  sweep_update_axis(w, 0, handle, qmin[0], qmax[0]);
  sweep_update_axis(w, 1, handle, qmin[1], qmax[1]);
  sweep_update_axis(w, 2, handle, qmin[2], qmax[2]);
}

RS_HD void sweep_add_handle_edges(World &w, uint16_t handle, Aabb aabb) {
  ArenaPhysics &a = *w.a;
  uint16_t qmin[3];
  uint16_t qmax[3];
  sweep_quantize_point(aabb.min, 0, qmin);
  sweep_quantize_point(aabb.max, 1, qmax);
  uint16_t limit = static_cast<uint16_t>(a.num_handles * 2);
  a.handle_edges[handle] = HandleEdges{};
  for (int axis = 0; axis < 3; ++axis) {
    a.handle_edges[0].max[axis] =
        static_cast<uint16_t>(a.handle_edges[0].max[axis] + 2);
    a.edges[axis][limit + 1] = a.edges[axis][limit - 1];
    a.edges[axis][limit - 1] = {qmin[axis], handle};
    a.edges[axis][limit] = {qmax[axis], handle};
    a.handle_edges[handle].min[axis] = static_cast<uint16_t>(limit - 1);
    a.handle_edges[handle].max[axis] = limit;
  }
  a.num_edges = static_cast<uint16_t>(limit + 2);
  for (int axis = 0; axis < 3; ++axis) {
    bool update = axis == 2;
    sweep_sort_min_down(w, axis, a.handle_edges[handle].min[axis], update);
    sweep_sort_max_down(w, axis, a.handle_edges[handle].max[axis], update);
  }
}

RS_HDI uint32_t grid_cell_index(const StaticWorldData &s, const uint32_t *idx) {
  return idx[0] * s.num_cells[1] * s.num_cells[2] + idx[1] * s.num_cells[2] +
         idx[2];
}

RS_HD void grid_cell_indices(const StaticWorldData &s, Vec3A pos,
                             uint32_t *out) {
  Vec3A f = (pos - s.grid_min) / s.cell_size;
  for (int i = 0; i < 3; ++i) {
    int32_t v = x86_cvtt_f32_to_i32(f[i]);
    if (v < 0) {
      v = 0;
    }
    uint32_t u = static_cast<uint32_t>(v);
    uint32_t max = s.num_cells[i] - 1;
    out[i] = u < max ? u : max;
  }
}

RS_HD void grid_update_dynamic(World &w, uint16_t handle, Vec3A min) {
  BroadphaseProxy &p = w.dyn_proxy(handle);
  uint32_t idx[3];
  grid_cell_indices(*w.s, min, idx);
  uint32_t new_idx = grid_cell_index(*w.s, idx);
  if (new_idx != p.cell_idx) {
    p.cell_idx = new_idx;
    p.indices[0] = idx[0];
    p.indices[1] = idx[1];
    p.indices[2] = idx[2];
    p.stamp = w.a->next_stamp++;
  }
}

RS_HD bool grid_dyn_in_cell(const BroadphaseProxy &p, const uint32_t *cell) {
  for (int i = 0; i < 3; ++i) {
    uint32_t lo = p.indices[i] > 0 ? p.indices[i] - 1 : 0;
    uint32_t hi = p.indices[i] + 1;
    if (cell[i] < lo || cell[i] > hi) {
      return false;
    }
  }
  return true;
}

}
