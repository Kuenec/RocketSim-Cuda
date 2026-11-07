#pragma once

#include <cstdint>

#include "rsc/math/shared.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

namespace sat {

inline constexpr float k_tie_eps = 1e-4f;

enum class AxisKindTag : uint8_t { None, TriFace, BoxFace, EdgeEdge };

struct AxisKind {
  AxisKindTag tag;
  int box_axis;
  int tri_edge;
};

struct SweepResult {
  bool has_sep;
  float best_gap;
  Vec3A best_sep_axis;
  float best_overlap;
  Vec3A best_pen_axis;
  Vec3A best_pen_raw;
  AxisKind pen_kind;
};

struct CandidatePair {
  Vec3A point_box;
  Vec3A point_tri;
  float distance_sq;
};

RS_HDI float min3(float a, float b, float c) {
  return min_f32(min_f32(a, b), c);
}

RS_HDI float max3(float a, float b, float c) {
  return max_f32(max_f32(a, b), c);
}

RS_HD Vec3A orient_pen_axis(Vec3A a, Vec3A to_box, Vec3A face_axis,
                            bool has_face_axis) {
  float s = a.dot(to_box);
  if (s < 0.0f) {
    return -a;
  }
  if (s > 1e-12f) {
    return a;
  }
  if (has_face_axis && a.dot(face_axis) > 0.0f) {
    return -a;
  }
  return a;
}

RS_HD void pen_axis_exits(Vec3A a, const Vec3A *q, Vec3A half, float &exit_pos,
                          float &exit_neg) {
  float r =
      half.x * abs_f32(a.x) + half.y * abs_f32(a.y) + half.z * abs_f32(a.z);
  float d0 = q[0].dot(a);
  float d1 = q[1].dot(a);
  float d2 = q[2].dot(a);
  exit_pos = max3(d0, d1, d2) + r;
  exit_neg = r - min3(d0, d1, d2);
}

RS_HD void axis_gap_overlap(Vec3A a, const Vec3A *q, Vec3A half, float &gap,
                            float &overlap, float &side) {
  float r =
      half.x * abs_f32(a.x) + half.y * abs_f32(a.y) + half.z * abs_f32(a.z);
  float d0 = q[0].dot(a);
  float d1 = q[1].dot(a);
  float d2 = q[2].dot(a);
  float min_t = min3(d0, d1, d2);
  float max_t = max3(d0, d1, d2);
  if (min_t > r) {
    gap = min_t - r;
    overlap = 0.0f;
    side = 1.0f;
  } else if (max_t < -r) {
    gap = -r - max_t;
    overlap = 0.0f;
    side = -1.0f;
  } else {
    float exit_pos = max_t + r;
    float exit_neg = r - min_t;
    gap = 0.0f;
    overlap = max_f32_c(min_f32(exit_pos, exit_neg), 0.0f);
    side = 0.0f;
  }
}

RS_HD bool consider_axis(SweepResult &out, Vec3A a, AxisKind kind,
                         const Vec3A *q, Vec3A half, float margin,
                         float maximum_distance) {
  float gap, overlap, side;
  axis_gap_overlap(a, q, half, gap, overlap, side);
  if (gap > 0.0f) {
    if (gap - margin >= maximum_distance) {
      return true;
    }
    if (!out.has_sep || gap > out.best_gap) {
      out.has_sep = true;
      out.best_gap = gap;
      out.best_sep_axis = side > 0.0f ? -a : a;
    }
  } else if (!out.has_sep && overlap < out.best_overlap) {
    out.best_overlap = overlap;
    out.pen_kind = kind;
    out.best_pen_raw = a;
  }
  return false;
}

RS_HD bool box_face_axis(SweepResult &out, float tri_min, float tri_max,
                         float half, Vec3A axis, int axis_idx, float margin,
                         float maximum_distance) {
  if (tri_min > half) {
    float gap = tri_min - half;
    if (gap - margin >= maximum_distance) {
      return true;
    }
    if (!out.has_sep || gap > out.best_gap) {
      out.has_sep = true;
      out.best_gap = gap;
      out.best_sep_axis = -axis;
    }
  } else if (tri_max < -half) {
    float gap = -half - tri_max;
    if (gap - margin >= maximum_distance) {
      return true;
    }
    if (!out.has_sep || gap > out.best_gap) {
      out.has_sep = true;
      out.best_gap = gap;
      out.best_sep_axis = axis;
    }
  } else if (!out.has_sep) {
    float overlap = max_f32_c(min_f32(tri_max + half, half - tri_min), 0.0f);
    if (overlap < out.best_overlap) {
      out.best_overlap = overlap;
      out.pen_kind = {AxisKindTag::BoxFace, axis_idx, 0};
      out.best_pen_raw = axis;
    }
  }
  return false;
}

RS_HDN bool sat_sweep(Vec3A half, const Vec3A *q, Vec3A tri_min, Vec3A tri_max,
                      const Vec3A *edges, Vec3A face_axis, bool has_face_axis,
                      Vec3A centroid, float margin, float maximum_distance,
                      SweepResult &out) {
  out.has_sep = false;
  out.best_gap = 0.0f;
  out.best_sep_axis = Vec3A::zero();
  out.best_overlap = k_f32_max;
  out.best_pen_axis = Vec3A::zero();
  out.best_pen_raw = Vec3A::zero();
  out.pen_kind = {AxisKindTag::None, 0, 0};
  Vec3A to_box = -centroid;
  if (box_face_axis(out, tri_min.x, tri_max.x, half.x, {1.0f, 0.0f, 0.0f}, 0,
                    margin, maximum_distance)) {
    return false;
  }
  if (box_face_axis(out, tri_min.y, tri_max.y, half.y, {0.0f, 1.0f, 0.0f}, 1,
                    margin, maximum_distance)) {
    return false;
  }
  if (box_face_axis(out, tri_min.z, tri_max.z, half.z, {0.0f, 0.0f, 1.0f}, 2,
                    margin, maximum_distance)) {
    return false;
  }
  if (has_face_axis &&
      consider_axis(out, face_axis, {AxisKindTag::TriFace, 0, 0}, q, half,
                    margin, maximum_distance)) {
    return false;
  }
  for (int bi = 0; bi < 3; ++bi) {
    Vec3A b_axis = Vec3A::zero();
    b_axis[bi] = 1.0f;
    for (int ei = 0; ei < 3; ++ei) {
      Vec3A c = b_axis.cross(edges[ei]);
      float l2 = c.length_squared();
      if (l2 < 1e-18f) {
        continue;
      }
      if (consider_axis(out, c / sqrt_f32(l2), {AxisKindTag::EdgeEdge, bi, ei},
                        q, half, margin, maximum_distance)) {
        return false;
      }
    }
  }
  if (!out.has_sep && out.pen_kind.tag != AxisKindTag::None) {
    float exit_pos, exit_neg;
    pen_axis_exits(out.best_pen_raw, q, half, exit_pos, exit_neg);
    if (exit_pos < exit_neg) {
      out.best_pen_axis = out.best_pen_raw;
    } else if (exit_neg < exit_pos) {
      out.best_pen_axis = -out.best_pen_raw;
    } else {
      out.best_pen_axis =
          orient_pen_axis(out.best_pen_raw, to_box, face_axis, has_face_axis);
    }
  }
  return true;
}

RS_HD Vec3A tri_support_first_max(const Vec3A *q, Vec3A dir) {
  Vec3A best = q[0];
  float best_d = q[0].dot(dir);
  for (int i = 1; i < 3; ++i) {
    float d = q[i].dot(dir);
    if (d > best_d) {
      best_d = d;
      best = q[i];
    }
  }
  return best;
}

RS_HD int clip_halfspace(const Vec3A *input, int input_len, Vec3A *output,
                         int axis, float bound, bool keep_above) {
  if (input_len == 0) {
    return 0;
  }
  int out = 0;
  Vec3A prev = input[input_len - 1];
  float prev_d = keep_above ? prev[axis] - bound : bound - prev[axis];
  for (int i = 0; i < input_len; ++i) {
    Vec3A v = input[i];
    float d = keep_above ? v[axis] - bound : bound - v[axis];
    if (d >= 0.0f) {
      if (prev_d < 0.0f) {
        float t = prev_d / (prev_d - d);
        output[out++] = prev + (v - prev) * t;
      }
      output[out++] = v;
    } else if (prev_d >= 0.0f) {
      float t = prev_d / (prev_d - d);
      output[out++] = prev + (v - prev) * t;
    }
    prev = v;
    prev_d = d;
  }
  return out;
}

RS_HDN bool clipped_tri_box_centroid(Vec3A half, const Vec3A *q, Vec3A shift,
                                     Vec3A &result) {
  Vec3A poly[9];
  Vec3A scratch[9];
  for (int i = 0; i < 9; ++i) {
    poly[i] = Vec3A::zero();
    scratch[i] = Vec3A::zero();
  }
  poly[0] = q[0];
  poly[1] = q[1];
  poly[2] = q[2];
  int len = 3;
  for (int axis = 0; axis < 3; ++axis) {
    float lo = -half[axis] + shift[axis] - k_tie_eps;
    float hi = half[axis] + shift[axis] + k_tie_eps;
    len = clip_halfspace(poly, len, scratch, axis, lo, true);
    if (len == 0) {
      return false;
    }
    len = clip_halfspace(scratch, len, poly, axis, hi, false);
    if (len == 0) {
      return false;
    }
  }
  Vec3A sum = Vec3A::zero();
  for (int i = 0; i < len; ++i) {
    sum += poly[i];
  }
  result = sum / static_cast<float>(len);
  return true;
}

RS_HD Vec3A clamp_point_to_aabb(Vec3A p, Vec3A half) {
  return {clamp_f32(p.x, -half.x, half.x), clamp_f32(p.y, -half.y, half.y),
          clamp_f32(p.z, -half.z, half.z)};
}

RS_HD Vec3A closest_point_on_segment(Vec3A p, Vec3A a, Vec3A b) {
  Vec3A ab = b - a;
  float denom = ab.dot(ab);
  if (denom < 1e-30f) {
    return a;
  }
  float t = clamp_f32((p - a).dot(ab) / denom, 0.0f, 1.0f);
  return a + ab * t;
}

RS_HD Vec3A closest_point_on_degenerate_triangle(Vec3A p, const Vec3A *q) {
  Vec3A c0 = closest_point_on_segment(p, q[0], q[1]);
  Vec3A c1 = closest_point_on_segment(p, q[1], q[2]);
  Vec3A c2 = closest_point_on_segment(p, q[2], q[0]);
  float d0 = (p - c0).length_squared();
  float d1 = (p - c1).length_squared();
  float d2 = (p - c2).length_squared();
  if (d0 <= d1 && d0 <= d2) {
    return c0;
  }
  if (d1 <= d2) {
    return c1;
  }
  return c2;
}

RS_HDN void segment_segment_closest(Vec3A p1, Vec3A q1, Vec3A p2, Vec3A q2,
                                    Vec3A &out1, Vec3A &out2) {
  Vec3A d1 = q1 - p1;
  Vec3A d2 = q2 - p2;
  Vec3A r = p1 - p2;
  float a = d1.dot(d1);
  float e = d2.dot(d2);
  float f = d2.dot(r);
  constexpr float eps = 1e-30f;
  float s, t;
  if (a <= eps && e <= eps) {
    out1 = p1;
    out2 = p2;
    return;
  }
  if (a <= eps) {
    s = 0.0f;
    t = clamp_f32(f / e, 0.0f, 1.0f);
  } else {
    float c = d1.dot(r);
    if (e <= eps) {
      t = 0.0f;
      s = clamp_f32(-c / a, 0.0f, 1.0f);
    } else {
      float b = d1.dot(d2);
      if (b * b > 0.99900025f * a * e) {
        float t0 = (p2 - p1).dot(d1) / a;
        float t1 = (q2 - p1).dot(d1) / a;
        float lo = max_f32(0.0f, min_f32(t0, t1));
        float hi = min_f32(1.0f, max_f32(t0, t1));
        if (lo <= hi) {
          float sm = (lo + hi) * 0.5f;
          Vec3A c1 = p1 + d1 * sm;
          float tm = clamp_f32((c1 - p2).dot(d2) / e, 0.0f, 1.0f);
          out1 = c1;
          out2 = p2 + d2 * tm;
          return;
        }
      }
      float denom = a * e - b * b;
      s = denom > eps ? clamp_f32((b * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;
      t = (b * s + f) / e;
      if (t < 0.0f) {
        t = 0.0f;
        s = clamp_f32(-c / a, 0.0f, 1.0f);
      } else if (t > 1.0f) {
        t = 1.0f;
        s = clamp_f32((b - c) / a, 0.0f, 1.0f);
      }
    }
  }
  out1 = p1 + d1 * s;
  out2 = p2 + d2 * t;
}

RS_HD Vec3A box_support_corner(Vec3A half, Vec3A dir) {
  return {dir.x < 0.0f ? -half.x : half.x, dir.y < 0.0f ? -half.y : half.y,
          dir.z < 0.0f ? -half.z : half.z};
}

RS_HD bool point_in_triangle_eps(Vec3A p, const Vec3A *q, Vec3A n) {
  constexpr float eps = 1e-6f;
  bool pos = false;
  bool neg = false;
  for (int i = 0; i < 3; ++i) {
    float s = (q[(i + 1) % 3] - q[i]).cross(p - q[i]).dot(n);
    if (s > eps) {
      pos = true;
    } else if (s < -eps) {
      neg = true;
    }
    if (pos && neg) {
      return false;
    }
  }
  return true;
}

RS_HD bool clip_segment_to_box(Vec3A a, Vec3A b, Vec3A half, Vec3A shift,
                               Vec3A &out_a, Vec3A &out_b) {
  float t0 = 0.0f;
  float t1 = 1.0f;
  Vec3A d = b - a;
  for (int axis = 0; axis < 3; ++axis) {
    float h = half[axis];
    for (int k = 0; k < 2; ++k) {
      bool is_min = k == 0;
      float p = is_min ? -d[axis] : d[axis];
      float qv = is_min ? a[axis] + h - shift[axis] + k_tie_eps
                        : h + shift[axis] - a[axis] + k_tie_eps;
      if (abs_f32(p) < k_f32_epsilon) {
        if (qv < 0.0f) {
          return false;
        }
      } else {
        float r = qv / p;
        if (p < 0.0f) {
          if (r > t1) {
            return false;
          }
          if (r > t0) {
            t0 = r;
          }
        } else {
          if (r < t0) {
            return false;
          }
          if (r < t1) {
            t1 = r;
          }
        }
      }
    }
  }
  if (t0 > t1) {
    return false;
  }
  out_a = a + d * t0;
  out_b = a + d * t1;
  return true;
}

RS_HDN bool pen_witness_by_kind(Vec3A half, const Vec3A *q, AxisKind kind,
                                Vec3A n_local, float overlap, Vec3A &out) {
  Vec3A shift = n_local * overlap;
  switch (kind.tag) {
  case AxisKindTag::None:
    return false;
  case AxisKindTag::TriFace: {
    float plane_dist = n_local.dot(q[0]);
    Vec3A corner = box_support_corner(half, -n_local);
    Vec3A p = corner - n_local * (n_local.dot(corner) - plane_dist);
    if (point_in_triangle_eps(p, q, n_local)) {
      out = p;
      return true;
    }
    return false;
  }
  case AxisKindTag::BoxFace: {
    float deepest = u32_to_f32(0xff800000u);
    for (int i = 0; i < 3; ++i) {
      deepest = max_f32(deepest, n_local.dot(q[i]));
    }
    Vec3A tied[3] = {Vec3A::zero(), Vec3A::zero(), Vec3A::zero()};
    int n_tied = 0;
    for (int i = 0; i < 3; ++i) {
      if (abs_f32(n_local.dot(q[i]) - deepest) <= k_tie_eps) {
        tied[n_tied++] = q[i];
      }
    }
    if (n_tied == 1) {
      out = tied[0];
      return true;
    }
    if (n_tied == 2) {
      Vec3A ca, cb;
      if (clip_segment_to_box(tied[0], tied[1], half, shift, ca, cb)) {
        out = (ca + cb) * 0.5f;
        return true;
      }
    }
    return false;
  }
  case AxisKindTag::EdgeEdge: {
    Vec3A s = (-n_local).signum();
    Vec3A start = half * s + shift;
    Vec3A end = half * s + shift;
    start[kind.box_axis] = half[kind.box_axis] + shift[kind.box_axis];
    end[kind.box_axis] = -half[kind.box_axis] + shift[kind.box_axis];
    Vec3A box_pt, tri_pt;
    segment_segment_closest(start, end, q[kind.tri_edge],
                            q[(kind.tri_edge + 1) % 3], box_pt, tri_pt);
    if ((tri_pt - box_pt).length_squared() <= k_tie_eps * k_tie_eps) {
      out = tri_pt;
      return true;
    }
    return false;
  }
  }
  return false;
}

RS_HDN Vec3A closest_point_on_triangle_precomp(Vec3A p, const Vec3A *q,
                                               Vec3A ab, Vec3A ac) {
  Vec3A ap = p - q[0];
  float d1 = ab.dot(ap);
  float d2 = ac.dot(ap);
  if (d1 <= 0.0f && d2 <= 0.0f) {
    return q[0];
  }
  Vec3A bp = p - q[1];
  float d3 = ab.dot(bp);
  float d4 = ac.dot(bp);
  if (d3 >= 0.0f && d4 <= d3) {
    return q[1];
  }
  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    float v = d1 / (d1 - d3);
    return q[0] + ab * v;
  }
  Vec3A cp = p - q[2];
  float d5 = ab.dot(cp);
  float d6 = ac.dot(cp);
  if (d6 >= 0.0f && d5 <= d6) {
    return q[2];
  }
  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    float w = d2 / (d2 - d6);
    return q[0] + ac * w;
  }
  float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return q[1] + (q[2] - q[1]) * w;
  }
  float denom = va + vb + vc;
  if (abs_f32(denom) < 1e-24f) {
    return closest_point_on_degenerate_triangle(p, q);
  }
  float v = vb / denom;
  float w = vc / denom;
  return q[0] + ab * v + ac * w;
}

RS_HDN void tri_edge_vs_box_edge_precomp(Vec3A tri_p, Vec3A tri_d, float tri_a,
                                         Vec3A box_p0, Vec3A box_p1,
                                         Vec3A &point_tri, Vec3A &point_box) {
  Vec3A d2 = box_p1 - box_p0;
  Vec3A r = tri_p - box_p0;
  float e = d2.dot(d2);
  float f = d2.dot(r);
  constexpr float eps = 1e-30f;
  if (tri_a <= eps) {
    if (e <= eps) {
      point_tri = tri_p;
      point_box = box_p0;
      return;
    }
    float t = clamp_f32(f / e, 0.0f, 1.0f);
    point_tri = tri_p;
    point_box = box_p0 + d2 * t;
    return;
  }
  if (e <= eps) {
    float c = tri_d.dot(r);
    float s = clamp_f32(-c / tri_a, 0.0f, 1.0f);
    point_tri = tri_p + tri_d * s;
    point_box = box_p0;
    return;
  }
  float c = tri_d.dot(r);
  float b = tri_d.dot(d2);
  if (b * b > 0.99900025f * tri_a * e) {
    float t0 = (box_p0 - tri_p).dot(tri_d) / tri_a;
    float t1 = (box_p1 - tri_p).dot(tri_d) / tri_a;
    float lo = max_f32(0.0f, min_f32(t0, t1));
    float hi = min_f32(1.0f, max_f32(t0, t1));
    if (lo <= hi) {
      float sm = (lo + hi) * 0.5f;
      Vec3A c1 = tri_p + tri_d * sm;
      float tm = clamp_f32((c1 - box_p0).dot(d2) / e, 0.0f, 1.0f);
      point_tri = c1;
      point_box = box_p0 + d2 * tm;
      return;
    }
  }
  float denom = tri_a * e - b * b;
  float s = denom > eps ? clamp_f32((b * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;
  float t = (b * s + f) / e;
  if (t < 0.0f) {
    t = 0.0f;
    s = clamp_f32(-c / tri_a, 0.0f, 1.0f);
  } else if (t > 1.0f) {
    t = 1.0f;
    s = clamp_f32((b - c) / tri_a, 0.0f, 1.0f);
  }
  point_tri = tri_p + tri_d * s;
  point_box = box_p0 + d2 * t;
}

inline constexpr int k_sat_vertex_pairs = 3;
inline constexpr int k_sat_corner_pairs = 8;
inline constexpr int k_sat_box_edges = 12;
inline constexpr int k_sat_candidates =
    k_sat_vertex_pairs + k_sat_corner_pairs + 3 * k_sat_box_edges;

RS_HD CandidatePair sat_vertex_pair(Vec3A half, const Vec3A *q, int i) {
  Vec3A point_box = clamp_point_to_aabb(q[i], half);
  Vec3A point_tri = q[i];
  return {point_box, point_tri, (point_box - point_tri).length_squared()};
}

RS_HD CandidatePair sat_corner_pair(Vec3A half, const Vec3A *q, Vec3A ab,
                                    Vec3A ac, int corner) {
  const float signs[2] = {-1.0f, 1.0f};
  Vec3A box_corner{signs[corner >> 2] * half.x,
                   signs[(corner >> 1) & 1] * half.y,
                   signs[corner & 1] * half.z};
  Vec3A point_tri = closest_point_on_triangle_precomp(box_corner, q, ab, ac);
  return {box_corner, point_tri, (box_corner - point_tri).length_squared()};
}

RS_HD Vec3A sat_box_edge_vertex(Vec3A half, int k) {
  bool pos_x = k == 1 || k == 2 || k == 5 || k == 6;
  bool pos_y = k == 2 || k == 3 || k == 6 || k == 7;
  bool pos_z = k >= 4;
  return {pos_x ? half.x : -half.x, pos_y ? half.y : -half.y,
          pos_z ? half.z : -half.z};
}

RS_HD void sat_box_edge_ends(int be, int &v0, int &v1) {
  v0 = be < 8 ? be : be - 8;
  v1 = be < 4 ? (be + 1) % 4 : (be < 8 ? 4 + (be - 3) % 4 : be - 4);
}

RS_HD CandidatePair sat_edge_pair(Vec3A tri_p, Vec3A tri_d, float tri_a,
                                  Vec3A box_p0, Vec3A box_p1) {
  Vec3A point_tri, point_box;
  tri_edge_vs_box_edge_precomp(tri_p, tri_d, tri_a, box_p0, box_p1, point_tri,
                               point_box);
  return {point_box, point_tri, (point_box - point_tri).length_squared()};
}

RS_HD CandidatePair sat_candidate_pair(Vec3A half, const Vec3A *q,
                                       const Vec3A *edges, int p) {
  if (p < k_sat_vertex_pairs) {
    return sat_vertex_pair(half, q, p);
  }
  if (p < k_sat_vertex_pairs + k_sat_corner_pairs) {
    return sat_corner_pair(half, q, edges[0], q[2] - q[0],
                           p - k_sat_vertex_pairs);
  }
  int e = p - k_sat_vertex_pairs - k_sat_corner_pairs;
  int ti = e / k_sat_box_edges;
  int v0, v1;
  sat_box_edge_ends(e % k_sat_box_edges, v0, v1);
  Vec3A tri_d = edges[ti];
  return sat_edge_pair(q[ti], tri_d, tri_d.dot(tri_d),
                       sat_box_edge_vertex(half, v0),
                       sat_box_edge_vertex(half, v1));
}

RS_HDN void enum_aabb_triangle_pairs(Vec3A half, const Vec3A *q,
                                     const Vec3A *edges, CandidatePair *pairs) {
  int idx = 0;
  for (int i = 0; i < k_sat_vertex_pairs; ++i) {
    pairs[idx++] = sat_vertex_pair(half, q, i);
  }
  Vec3A ab = edges[0];
  Vec3A ac = q[2] - q[0];
  for (int corner = 0; corner < k_sat_corner_pairs; ++corner) {
    pairs[idx++] = sat_corner_pair(half, q, ab, ac, corner);
  }
  Vec3A c[8];
  for (int k = 0; k < 8; ++k) {
    c[k] = sat_box_edge_vertex(half, k);
  }
  for (int ti = 0; ti < 3; ++ti) {
    Vec3A tri_p = q[ti];
    Vec3A tri_d = edges[ti];
    float tri_a = tri_d.dot(tri_d);
    for (int be = 0; be < k_sat_box_edges; ++be) {
      int v0, v1;
      sat_box_edge_ends(be, v0, v1);
      pairs[idx++] = sat_edge_pair(tri_p, tri_d, tri_a, c[v0], c[v1]);
    }
  }
}

RS_HDN bool flat_tie_segment(Vec3A half, const CandidatePair *pairs,
                             Vec3A &out0, Vec3A &out1) {
  constexpr int n = k_sat_candidates;
  float flat_link = 0.5f * half.max_element();
  if (flat_link <= 0.0f || !is_finite_f32(flat_link)) {
    return false;
  }
  float flat_slope = k_tie_eps / flat_link;
  float best_d2 = k_f32_max;
  RS_NO_UNROLL
  for (int i = 0; i < n; ++i) {
    if (pairs[i].distance_sq < best_d2) {
      best_d2 = pairs[i].distance_sq;
    }
  }
  float best_d = sqrt_f32(best_d2);
  float flat_link2 = flat_link * flat_link;
  float tie_d = best_d + k_tie_eps;
  float tie_d2 = tie_d * tie_d;
  int tied[n];
  int tied_count = 0;
  RS_NO_UNROLL
  for (int i = 0; i < n; ++i) {
    if (pairs[i].distance_sq <= tie_d2) {
      tied[tied_count++] = i;
    }
  }
  if (tied_count == 0) {
    return false;
  }
  Vec3A mean_a = Vec3A::zero();
  Vec3A mean_b = Vec3A::zero();
  RS_NO_UNROLL
  for (int k = 0; k < tied_count; ++k) {
    mean_a += pairs[tied[k]].point_box;
    mean_b += pairs[tied[k]].point_tri;
  }
  float inv = 1.0f / static_cast<float>(tied_count);
  mean_a *= inv;
  mean_b *= inv;
  float d0 = (mean_a - mean_b).length();
  int flat = -1;
  float best_slope = flat_slope;
  RS_NO_UNROLL
  for (int i = 0; i < n; ++i) {
    float sep2 = (pairs[i].point_tri - mean_b).length_squared();
    if (sep2 <= flat_link2) {
      continue;
    }
    float sep = sqrt_f32(sep2);
    float slope = (sqrt_f32(pairs[i].distance_sq) - d0) / sep;
    if (slope <= best_slope) {
      Vec3A mid_a = (mean_a + pairs[i].point_box) * 0.5f;
      Vec3A mid_b = (mean_b + pairs[i].point_tri) * 0.5f;
      if ((mid_a - mid_b).length() <= d0 + k_tie_eps) {
        best_slope = slope;
        flat = i;
      }
    }
  }
  CandidatePair end0 = pairs[tied[0]];
  CandidatePair end1 = pairs[tied[0]];
  float span2 = 0.0f;
  auto consider = [&](const CandidatePair &cand) {
    RS_NO_UNROLL
    for (int k = 0; k <= tied_count; ++k) {
      const CandidatePair *pair = nullptr;
      if (k < tied_count) {
        pair = &pairs[tied[k]];
      } else if (flat >= 0) {
        pair = &pairs[flat];
      } else {
        break;
      }
      float s2 = (cand.point_tri - pair->point_tri).length_squared();
      if (s2 > span2) {
        span2 = s2;
        end0 = cand;
        end1 = *pair;
      }
    }
  };
  RS_NO_UNROLL
  for (int k = 0; k < tied_count; ++k) {
    consider(pairs[tied[k]]);
  }
  if (flat >= 0) {
    consider(pairs[flat]);
  }
  if (span2 <= flat_link2) {
    return false;
  }
  Vec3A mid_a = (end0.point_box + end1.point_box) * 0.5f;
  Vec3A mid_b = (end0.point_tri + end1.point_tri) * 0.5f;
  if ((mid_a - mid_b).length() > best_d + k_tie_eps) {
    return false;
  }
  out0 = end0.point_tri;
  out1 = end1.point_tri;
  return true;
}

}

struct BoxTriangleSatContact {
  Vec3A point_on_b_world;
  float distance;
};

enum class SatStage : uint8_t { Miss, Hit, NeedsPairs };

struct SatSweepState {
  Vec3A edges[3];
  Vec3A centroid;
  sat::SweepResult sweep;
};

RS_HDI bool box_triangle_sat_finish(const Affine3A &box_trans,
                                    Vec3A half_unmargined, float margin,
                                    const Vec3A *q, float maximum_distance,
                                    const SatSweepState &state,
                                    const sat::CandidatePair *pairs,
                                    BoxTriangleSatContact &out) {
  using namespace sat;
  CandidatePair best = pairs[0];
  float best_d2 = best.distance_sq;
  for (int i = 1; i < k_sat_candidates; ++i) {
    if (pairs[i].distance_sq < best_d2) {
      best_d2 = pairs[i].distance_sq;
      best = pairs[i];
    }
  }
  Vec3A pa_first = best.point_box;
  Vec3A pb_first = best.point_tri;
  if ((pa_first - pb_first).length_squared() > 1e-24f) {
    float pair_len = (pa_first - pb_first).length();
    Vec3A pb_local = pb_first;
    bool snapped = false;
    Vec3A te0, te1;
    if (pair_len - margin < 0.0f &&
        flat_tie_segment(half_unmargined, pairs, te0, te1)) {
      pb_local = closest_point_on_segment(state.centroid, te0, te1);
      snapped = true;
    }
    float distance =
        snapped ? (clamp_point_to_aabb(pb_local, half_unmargined) - pb_local)
                          .length() -
                      margin
                : pair_len - margin;
    if (distance >= maximum_distance) {
      return false;
    }
    out = {box_trans.transform_point3a(pb_local), distance};
    return true;
  }
  Vec3A pb_local = tri_support_first_max(q, state.sweep.best_sep_axis);
  out = {box_trans.transform_point3a(pb_local), state.sweep.best_gap - margin};
  return true;
}

RS_HDI SatStage box_triangle_sat_stage(const Affine3A &box_trans,
                                       Vec3A half_unmargined, float margin,
                                       const Vec3A *q, float maximum_distance,
                                       SatSweepState *deferred,
                                       BoxTriangleSatContact &out) {
  using namespace sat;
  SatSweepState local;
  SatSweepState &state = deferred ? *deferred : local;
  Vec3A tri_min = q[0].min(q[1]).min(q[2]);
  Vec3A tri_max = q[0].max(q[1]).max(q[2]);
  for (int axis = 0; axis < 3; ++axis) {
    float h = half_unmargined[axis];
    if (tri_min[axis] > h) {
      if (tri_min[axis] - h - margin >= maximum_distance) {
        return SatStage::Miss;
      }
    } else if (tri_max[axis] < -h &&
               -h - tri_max[axis] - margin >= maximum_distance) {
      return SatStage::Miss;
    }
  }
  Vec3A e0 = q[1] - q[0];
  Vec3A e1 = q[2] - q[1];
  Vec3A e2 = q[0] - q[2];
  Vec3A face_cross = e0.cross(-e2);
  float face_len2 = face_cross.length_squared();
  bool has_face_axis = face_len2 > 1e-20f;
  Vec3A face_axis =
      has_face_axis ? face_cross / sqrt_f32(face_len2) : Vec3A::zero();
  state.centroid = (q[0] + q[1] + q[2]) * (1.0f / 3.0f);
  state.edges[0] = e0;
  state.edges[1] = e1;
  state.edges[2] = e2;
  SweepResult &sweep = state.sweep;
  if (!sat_sweep(half_unmargined, q, tri_min, tri_max, state.edges, face_axis,
                 has_face_axis, state.centroid, margin, maximum_distance,
                 sweep)) {
    return SatStage::Miss;
  }
  if (sweep.has_sep) {
    Vec3A n_world =
        (box_trans.matrix3 * sweep.best_sep_axis).normalize_or_zero();
    if (n_world.length_squared() < 0.5f) {
      return SatStage::Miss;
    }
    if (deferred) {
      return SatStage::NeedsPairs;
    }
    CandidatePair pairs[k_sat_candidates];
    enum_aabb_triangle_pairs(half_unmargined, q, state.edges, pairs);
    return box_triangle_sat_finish(box_trans, half_unmargined, margin, q,
                                   maximum_distance, state, pairs, out)
               ? SatStage::Hit
               : SatStage::Miss;
  }
  if (sweep.best_overlap == k_f32_max) {
    return SatStage::Miss;
  }
  Vec3A n_local;
  if (sweep.best_pen_axis.length_squared() > 0.5f) {
    n_local = sweep.best_pen_axis;
  } else if (has_face_axis) {
    float s = face_axis.dot(-state.centroid);
    n_local = s < 0.0f ? -face_axis : face_axis;
  } else {
    return SatStage::Miss;
  }
  float distance = -sweep.best_overlap - margin;
  if (distance >= maximum_distance) {
    return SatStage::Miss;
  }
  Vec3A pb_local;
  if (!pen_witness_by_kind(half_unmargined, q, sweep.pen_kind, n_local,
                           sweep.best_overlap, pb_local) &&
      !clipped_tri_box_centroid(half_unmargined, q,
                                n_local * sweep.best_overlap, pb_local)) {
    pb_local = tri_support_first_max(q, n_local);
  }
  Vec3A n_world = (box_trans.matrix3 * n_local).normalize_or_zero();
  if (n_world.length_squared() < 0.5f) {
    return SatStage::Miss;
  }
  out = {box_trans.transform_point3a(pb_local), distance};
  return SatStage::Hit;
}

RS_HDN bool box_triangle_sat(const Affine3A &box_trans, Vec3A half_unmargined,
                             float margin, const Vec3A *q,
                             float maximum_distance,
                             BoxTriangleSatContact &out) {
  return box_triangle_sat_stage(box_trans, half_unmargined, margin, q,
                                maximum_distance, nullptr,
                                out) == SatStage::Hit;
}

}
