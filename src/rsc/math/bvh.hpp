#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rsc/math/hd.hpp"
#include "rsc/math/shared.hpp"

namespace rsc {

inline constexpr uint32_t k_bvh_leaf_bit = 1u << 31;
inline constexpr int k_bvh_stack_size = 128;

struct BvhWideNode {
  Vec4 min_x, min_y, min_z, max_x, max_y, max_z;
  uint32_t children[4];
  uint32_t child_count;
  uint32_t valid_mask;

  RS_HD uint32_t intersection_mask(const Aabb &aabb) const {
#if defined(__SSE2__) && !defined(__CUDACC__)
    __m128 x = _mm_and_ps(_mm_cmple_ps(to_sse(min_x), _mm_set1_ps(aabb.max.x)),
                          _mm_cmpge_ps(to_sse(max_x), _mm_set1_ps(aabb.min.x)));
    __m128 y = _mm_and_ps(_mm_cmple_ps(to_sse(min_y), _mm_set1_ps(aabb.max.y)),
                          _mm_cmpge_ps(to_sse(max_y), _mm_set1_ps(aabb.min.y)));
    __m128 z = _mm_and_ps(_mm_cmple_ps(to_sse(min_z), _mm_set1_ps(aabb.max.z)),
                          _mm_cmpge_ps(to_sse(max_z), _mm_set1_ps(aabb.min.z)));
    __m128 overlap = _mm_and_ps(_mm_and_ps(x, y), z);
    return static_cast<uint32_t>(_mm_movemask_ps(overlap)) & valid_mask;
#else
    BVec4 overlap = min_x.cmple(Vec4::splat(aabb.max.x)) &
                    max_x.cmpge(Vec4::splat(aabb.min.x)) &
                    min_y.cmple(Vec4::splat(aabb.max.y)) &
                    max_y.cmpge(Vec4::splat(aabb.min.y)) &
                    min_z.cmple(Vec4::splat(aabb.max.z)) &
                    max_z.cmpge(Vec4::splat(aabb.min.z));
    return overlap.bitmask() & valid_mask;
#endif
  }
};

struct BvhWideLeaf {
  Aabb bounds;
  uint32_t leaf_idx;
};

struct BvhView {
  Aabb aabb;
  const BvhWideNode *nodes;
  const BvhWideLeaf *leaves;

  RS_HD bool check_overlap_with(const Aabb &query) const {
    if (!query.intersects(aabb)) {
      return false;
    }
    uint32_t stack[k_bvh_stack_size];
    stack[0] = 0;
    int stack_len = 1;
    while (stack_len != 0) {
      --stack_len;
      uint32_t work = stack[stack_len];
      if (work & k_bvh_leaf_bit) {
        return true;
      }
      const BvhWideNode &node = nodes[work];
      uint32_t mask = node.intersection_mask(query);
      for (int lane = static_cast<int>(node.child_count) - 1; lane >= 0;
           --lane) {
        if (mask & (1u << lane)) {
          stack[stack_len++] = node.children[lane];
        }
      }
    }
    return false;
  }

  template <typename F>
  RS_HD void report_aabb_overlapping_node(F &callback,
                                          const Aabb &query) const {
    if (!query.intersects(aabb)) {
      return;
    }
    uint32_t stack[k_bvh_stack_size];
    stack[0] = 0;
    int stack_len = 1;
    while (stack_len != 0) {
      --stack_len;
      uint32_t work = stack[stack_len];
      if (work & k_bvh_leaf_bit) {
        callback.process_node(leaves[work & ~k_bvh_leaf_bit].leaf_idx);
        continue;
      }
      const BvhWideNode &node = nodes[work];
      uint32_t mask = node.intersection_mask(query);
      for (int lane = static_cast<int>(node.child_count) - 1; lane >= 0;
           --lane) {
        if (mask & (1u << lane)) {
          stack[stack_len++] = node.children[lane];
        }
      }
    }
  }

  template <typename F>
  RS_HD void report_single_ray_overlapping_node(F &callback,
                                                const Aabb &ray_aabb,
                                                Vec3A source, Vec3A target,
                                                float &lambda_max,
                                                const Aabb &node_cull) const {
    if (!ray_aabb.intersects(aabb)) {
      return;
    }
    uint32_t stack[k_bvh_stack_size];
    stack[0] = 0;
    int stack_len = 1;
    bool have_pack = false;
    SingleRayPack pack;
    while (stack_len != 0) {
      --stack_len;
      uint32_t work = stack[stack_len];
      if (work & k_bvh_leaf_bit) {
        const BvhWideLeaf &leaf = leaves[work & ~k_bvh_leaf_bit];
        if (!have_pack) {
          pack = single_ray_pack(source, target);
          have_pack = true;
        }
        if (single_ray_slab(pack.origin, pack.inv_dir, leaf.bounds,
                            lambda_max)) {
          callback.process_node(leaf.leaf_idx, lambda_max);
        }
        continue;
      }
      const BvhWideNode &node = nodes[work];
      uint32_t mask = node.intersection_mask(node_cull);
      for (int lane = static_cast<int>(node.child_count) - 1; lane >= 0;
           --lane) {
        if (mask & (1u << lane)) {
          stack[stack_len++] = node.children[lane];
        }
      }
    }
  }

  template <typename F>
  RS_HD void report_quad_ray_overlapping_node(F &callback,
                                              QuadRayInfo &ray_info,
                                              const Aabb &node_cull) const {
    if (!ray_info.aabb.intersects(aabb)) {
      return;
    }
    uint32_t stack[k_bvh_stack_size];
    stack[0] = 0;
    int stack_len = 1;
    bool have_pack = false;
    QuadRayPack pack;
    while (stack_len != 0) {
      --stack_len;
      uint32_t work = stack[stack_len];
      if (work & k_bvh_leaf_bit) {
        const BvhWideLeaf &leaf = leaves[work & ~k_bvh_leaf_bit];
        if (!have_pack) {
          pack = ray_info.calc_pos_dir();
          have_pack = true;
        }
        uint32_t mask = QuadRayInfo::intersect_quad_ray_aabb(
            pack.origins, pack.inv_dirs, leaf.bounds, ray_info.lambda_max);
        if (mask != 0) {
          callback.process_node(leaf.leaf_idx, mask, ray_info.lambda_max);
        }
        continue;
      }
      const BvhWideNode &node = nodes[work];
      uint32_t mask = node.intersection_mask(node_cull);
      for (int lane = static_cast<int>(node.child_count) - 1; lane >= 0;
           --lane) {
        if (mask & (1u << lane)) {
          stack[stack_len++] = node.children[lane];
        }
      }
    }
  }
};

struct BvhBuildNode {
  Aabb aabb;
  bool is_leaf;
  uint32_t value;
};

struct BvhStorage {
  Aabb aabb;
  std::vector<BvhWideNode> nodes;
  std::vector<BvhWideLeaf> leaves;

  BvhView view() const { return {aabb, nodes.data(), leaves.data()}; }
};

namespace bvh_build {

inline uint32_t calc_sah_split(std::vector<BvhBuildNode> &leaf_nodes,
                               uint32_t start, uint32_t end) {
  constexpr int bins = 4;
  uint32_t count = end - start;
  if (count == 2) {
    return start + 1;
  }
  Vec3A cmin = Vec3A::splat(u32_to_f32(0x7f800000u));
  Vec3A cmax = Vec3A::splat(u32_to_f32(0xff800000u));
  for (uint32_t i = start; i < end; ++i) {
    Vec3A c = leaf_nodes[i].aabb.center();
    cmin = cmin.min(c);
    cmax = cmax.max(c);
  }
  Vec3A extents = cmax - cmin;
  int best_axis = -1;
  float best_cost = u32_to_f32(0x7f800000u);
  int best_bin = 0;
  for (int axis = 0; axis < 3; ++axis) {
    float extent = extents[axis];
    if (extent <= k_f32_epsilon) {
      continue;
    }
    uint64_t bin_counts[bins] = {0, 0, 0, 0};
    Aabb bin_bounds[bins] = {Aabb::zero(), Aabb::zero(), Aabb::zero(),
                             Aabb::zero()};
    float scale = static_cast<float>(bins - 1) / extent;
    for (uint32_t i = start; i < end; ++i) {
      Vec3A c = leaf_nodes[i].aabb.center();
      float f = clamp_f32((c[axis] - cmin[axis]) * scale, 0.0f,
                          static_cast<float>(bins - 1));
      int idx = static_cast<int>(f);
      if (bin_counts[idx] == 0) {
        bin_bounds[idx] = leaf_nodes[i].aabb;
      } else {
        bin_bounds[idx] += leaf_nodes[i].aabb;
      }
      bin_counts[idx] += 1;
    }
    float prefix_area[bins];
    uint64_t prefix_count[bins];
    Aabb running = Aabb::zero();
    uint64_t running_count = 0;
    for (int i = 0; i < bins; ++i) {
      if (bin_counts[i] > 0) {
        running += bin_bounds[i];
        running_count += bin_counts[i];
      }
      prefix_area[i] = running.area();
      prefix_count[i] = running_count;
    }
    float suffix_area[bins];
    uint64_t suffix_count[bins];
    running = Aabb::zero();
    running_count = 0;
    for (int i = bins - 1; i >= 0; --i) {
      if (bin_counts[i] > 0) {
        running += bin_bounds[i];
        running_count += bin_counts[i];
      }
      suffix_area[i] = running.area();
      suffix_count[i] = running_count;
    }
    for (int i = 0; i + 1 < bins; ++i) {
      uint64_t nl = prefix_count[i];
      uint64_t nr = suffix_count[i + 1];
      if (nl == 0 || nr == 0) {
        continue;
      }
      float cost = prefix_area[i] * static_cast<float>(nl) +
                   suffix_area[i + 1] * static_cast<float>(nr);
      if (cost < best_cost) {
        best_cost = cost;
        best_axis = axis;
        best_bin = i;
      }
    }
  }
  if (best_axis < 0) {
    throw std::runtime_error("BVH SAH split found no valid axis");
  }
  float split_value = cmin[best_axis] + extents[best_axis] *
                                            static_cast<float>(best_bin + 1) /
                                            static_cast<float>(bins);
  uint32_t mid = start;
  for (uint32_t i = start; i < end; ++i) {
    Vec3A c = leaf_nodes[i].aabb.center();
    if (c[best_axis] <= split_value) {
      if (i != mid) {
        std::swap(leaf_nodes[mid], leaf_nodes[i]);
      }
      ++mid;
    }
  }
  return mid;
}

inline uint32_t calc_bullet_split(std::vector<BvhBuildNode> &leaf_nodes,
                                  uint32_t start, uint32_t end) {
  uint32_t count = end - start;
  Vec3A mean = Vec3A::zero();
  for (uint32_t i = start; i < end; ++i) {
    mean += leaf_nodes[i].aabb.center();
  }
  mean *= 1.0f / static_cast<float>(count);
  Vec3A variance = Vec3A::zero();
  for (uint32_t i = start; i < end; ++i) {
    Vec3A diff = leaf_nodes[i].aabb.center() - mean;
    variance += diff * diff;
  }
  variance *= 1.0f / (static_cast<float>(count) - 1.0f);
  int axis;
  if (variance.x < variance.y) {
    axis = variance.y < variance.z ? 2 : 1;
  } else if (variance.x < variance.z) {
    axis = 2;
  } else {
    axis = 0;
  }
  float split_value = mean[axis];
  uint32_t split_idx = start;
  for (uint32_t i = start; i < end; ++i) {
    if (leaf_nodes[i].aabb.center()[axis] > split_value) {
      if (i != split_idx) {
        std::swap(leaf_nodes[split_idx], leaf_nodes[i]);
      }
      ++split_idx;
    }
  }
  uint32_t balanced_range = count / 3;
  if (split_idx <= start + balanced_range ||
      split_idx >= end - 1 - balanced_range) {
    split_idx = start + (count >> 1);
  }
  return split_idx;
}

struct BinaryTree {
  Aabb aabb;
  uint32_t cur_node_idx = 0;
  std::vector<BvhBuildNode> nodes;
  std::vector<uint32_t> escape;

  template <typename SplitFn>
  void build_subtree(std::vector<BvhBuildNode> &leaf_nodes, uint32_t start,
                     uint32_t end, SplitFn split_fn) {
    uint32_t num = end - start;
    uint32_t cur_idx = cur_node_idx;
    if (num == 1) {
      nodes[cur_node_idx] = leaf_nodes[start];
      ++cur_node_idx;
      return;
    }
    uint32_t split_idx = split_fn(leaf_nodes, start, end);
    uint32_t internal = cur_node_idx;
    nodes[internal].aabb.min = aabb.max;
    nodes[internal].aabb.max = aabb.min;
    for (uint32_t i = start; i < end; ++i) {
      nodes[internal].aabb += leaf_nodes[i].aabb;
    }
    ++cur_node_idx;
    build_subtree(leaf_nodes, start, split_idx, split_fn);
    build_subtree(leaf_nodes, split_idx, end, split_fn);
    nodes[internal].is_leaf = false;
    nodes[internal].value = cur_node_idx - cur_idx;
  }

  void children(uint32_t node_idx, uint32_t &left, uint32_t &right) const {
    left = node_idx + 1;
    right = nodes[left].is_leaf ? left + 1 : left + nodes[left].value;
  }

  uint32_t build_wide_node(uint32_t binary_root,
                           std::vector<BvhWideNode> &wide_nodes,
                           std::vector<BvhWideLeaf> &leaves) const {
    uint32_t wide_idx = static_cast<uint32_t>(wide_nodes.size());
    wide_nodes.push_back(BvhWideNode{});
    std::vector<uint32_t> frontier{binary_root};
    while (frontier.size() < 4) {
      int best_slot = -1;
      uint32_t best_size = 0;
      for (size_t slot = 0; slot < frontier.size(); ++slot) {
        const BvhBuildNode &n = nodes[frontier[slot]];
        if (n.is_leaf) {
          continue;
        }
        if (best_slot < 0 || n.value >= best_size) {
          best_slot = static_cast<int>(slot);
          best_size = n.value;
        }
      }
      if (best_slot < 0) {
        break;
      }
      uint32_t left, right;
      children(frontier[best_slot], left, right);
      frontier[best_slot] = left;
      frontier.insert(frontier.begin() + best_slot + 1, right);
    }
    BvhWideNode wide{};
    wide.child_count = static_cast<uint32_t>(frontier.size());
    wide.valid_mask = (1u << wide.child_count) - 1;
    uint32_t max_child_depth = 0;
    for (size_t lane = 0; lane < frontier.size(); ++lane) {
      const BvhBuildNode &n = nodes[frontier[lane]];
      wide.min_x[static_cast<int>(lane)] = n.aabb.min.x;
      wide.min_y[static_cast<int>(lane)] = n.aabb.min.y;
      wide.min_z[static_cast<int>(lane)] = n.aabb.min.z;
      wide.max_x[static_cast<int>(lane)] = n.aabb.max.x;
      wide.max_y[static_cast<int>(lane)] = n.aabb.max.y;
      wide.max_z[static_cast<int>(lane)] = n.aabb.max.z;
      if (n.is_leaf) {
        uint32_t storage_idx = static_cast<uint32_t>(leaves.size());
        leaves.push_back({n.aabb, n.value});
        wide.children[lane] = k_bvh_leaf_bit | storage_idx;
      } else {
        uint32_t child_idx = static_cast<uint32_t>(wide_nodes.size());
        uint32_t depth = build_wide_node(frontier[lane], wide_nodes, leaves);
        if (depth > max_child_depth) {
          max_child_depth = depth;
        }
        wide.children[lane] = child_idx;
      }
    }
    wide_nodes[wide_idx] = wide;
    return max_child_depth + 1;
  }
};

template <typename SplitFn>
BvhStorage build(Aabb aabb, std::vector<BvhBuildNode> &leaf_nodes,
                 SplitFn split_fn) {
  if (leaf_nodes.empty()) {
    throw std::runtime_error("BVH build requires at least one leaf");
  }
  BinaryTree tree;
  tree.aabb = aabb;
  tree.nodes.assign(2 * leaf_nodes.size(), BvhBuildNode{Aabb::zero(), true, 0});
  tree.build_subtree(leaf_nodes, 0, static_cast<uint32_t>(leaf_nodes.size()),
                     split_fn);
  BvhStorage storage;
  storage.aabb = tree.aabb;
  storage.leaves.reserve(leaf_nodes.size());
  uint32_t depth = tree.build_wide_node(0, storage.nodes, storage.leaves);
  if (depth * 3 >= static_cast<uint32_t>(k_bvh_stack_size)) {
    throw std::runtime_error("BVH too deep for the traversal stack");
  }
  return storage;
}

}

inline BvhStorage build_bvh_sah(Aabb aabb,
                                std::vector<BvhBuildNode> &leaf_nodes) {
  return bvh_build::build(aabb, leaf_nodes, bvh_build::calc_sah_split);
}

inline BvhStorage build_bvh_bullet(Aabb aabb,
                                   std::vector<BvhBuildNode> &leaf_nodes) {
  return bvh_build::build(aabb, leaf_nodes, bvh_build::calc_bullet_split);
}

}
