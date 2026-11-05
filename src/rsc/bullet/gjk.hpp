#pragma once

#include <cstdint>

#include "rsc/bullet/convex.hpp"
#include "rsc/math/shared.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

struct UsageBitfield {
  bool a, b, c, d;
  RS_HDI void reset() { a = b = c = d = false; }
};

struct SubSimplexClosestResult {
  Vec3A closest_point_on_simplex;
  UsageBitfield used;
  float bary[4];
  bool degenerate;

  RS_HDI void set_bary(float a, float b, float c, float d) {
    bary[0] = a;
    bary[1] = b;
    bary[2] = c;
    bary[3] = d;
  }
  RS_HDI void reset() {
    degenerate = false;
    set_bary(0.0f, 0.0f, 0.0f, 0.0f);
    used.reset();
  }
  RS_HDI bool is_valid() const {
    return bary[0] >= 0.0f && bary[1] >= 0.0f && bary[2] >= 0.0f &&
           bary[3] >= 0.0f;
  }
};

struct VoronoiSimplexSolver {
  static constexpr float k_equal_vertex_threshold = 0.0001f;
  static constexpr float k_degenerate_eps = 1.0e-4f;

  int num_vertices;
  Vec3A w[5];
  Vec3A p[5];
  Vec3A q[5];
  Vec3A cached_p1, cached_p2, cached_v, last_w;
  bool cached_valid_closest;
  SubSimplexClosestResult cached_bc;
  bool needs_update;

  RS_HD void init() {
    num_vertices = 0;
    for (int i = 0; i < 5; ++i) {
      w[i] = Vec3A::zero();
      p[i] = Vec3A::zero();
      q[i] = Vec3A::zero();
    }
    cached_p1 = Vec3A::zero();
    cached_p2 = Vec3A::zero();
    cached_v = Vec3A::zero();
    last_w = Vec3A::splat(u32_to_f32(0x7f800000u));
    cached_valid_closest = false;
    cached_bc.closest_point_on_simplex = Vec3A::zero();
    cached_bc.used.reset();
    cached_bc.set_bary(0.0f, 0.0f, 0.0f, 0.0f);
    cached_bc.degenerate = false;
    needs_update = true;
  }

  RS_HD void add_vertex(Vec3A nw, Vec3A np, Vec3A nq) {
    last_w = nw;
    needs_update = true;
    w[num_vertices] = nw;
    p[num_vertices] = np;
    q[num_vertices] = nq;
    ++num_vertices;
  }

  RS_HD bool closest(Vec3A &v) {
    bool success = update_closest_vector_and_points();
    v = cached_v;
    return success;
  }

  RS_HD bool full_simplex() const { return num_vertices == 4; }

  RS_HD bool in_simplex(Vec3A nw) const {
    bool found = false;
    for (int i = 0; i < num_vertices; ++i) {
      if ((w[i] - nw).length_squared() <= k_equal_vertex_threshold) {
        found = true;
        break;
      }
    }
    if (nw == last_w) {
      return true;
    }
    return found;
  }

  RS_HD Vec3A compute_points() {
    update_closest_vector_and_points();
    return cached_p2;
  }

  RS_HD void remove_vertex(int index) {
    --num_vertices;
    w[index] = w[num_vertices];
    p[index] = p[num_vertices];
    q[index] = q[num_vertices];
  }

  RS_HD void reduce_vertices(UsageBitfield used) {
    if (num_vertices >= 4 && !used.d) {
      remove_vertex(3);
    }
    if (num_vertices >= 3 && !used.c) {
      remove_vertex(2);
    }
    if (num_vertices >= 2 && !used.b) {
      remove_vertex(1);
    }
    if (num_vertices >= 1 && !used.a) {
      remove_vertex(0);
    }
  }

  RS_HD bool update_closest_vector_and_points() {
    if (!needs_update) {
      return cached_valid_closest;
    }
    cached_bc.reset();
    needs_update = false;
    switch (num_vertices) {
    case 1:
      cached_p1 = p[0];
      cached_p2 = q[0];
      cached_v = cached_p1 - cached_p2;
      cached_bc.reset();
      cached_bc.set_bary(1.0f, 0.0f, 0.0f, 0.0f);
      cached_valid_closest = cached_bc.is_valid();
      break;
    case 2: {
      Vec3A from = w[0];
      Vec3A to = w[1];
      Vec3A diff = Vec3A::zero() - from;
      Vec3A v = to - from;
      float t = v.dot(diff);
      if (t > 0.0f) {
        float dot_vv = v.dot(v);
        if (t < dot_vv) {
          t /= dot_vv;
          diff -= t * v;
          cached_bc.used.a = true;
        } else {
          t = 1.0f;
          diff -= v;
        }
        cached_bc.used.b = true;
      } else {
        t = 0.0f;
        cached_bc.used.a = true;
      }
      cached_bc.set_bary(1.0f - t, t, 0.0f, 0.0f);
      cached_p1 = p[0] + t * (p[1] - p[0]);
      cached_p2 = q[0] + t * (q[1] - q[0]);
      cached_v = cached_p1 - cached_p2;
      reduce_vertices(cached_bc.used);
      cached_valid_closest = cached_bc.is_valid();
      break;
    }
    case 3: {
      closest_pt_point_triangle(Vec3A::zero(), w[0], w[1], w[2], cached_bc);
      cached_p1 = p[0] * cached_bc.bary[0] + p[1] * cached_bc.bary[1] +
                  p[2] * cached_bc.bary[2];
      cached_p2 = q[0] * cached_bc.bary[0] + q[1] * cached_bc.bary[1] +
                  q[2] * cached_bc.bary[2];
      cached_v = cached_p1 - cached_p2;
      reduce_vertices(cached_bc.used);
      cached_valid_closest = cached_bc.is_valid();
      break;
    }
    case 4: {
      bool has_separation = closest_pt_point_tetrahedron(
          Vec3A::zero(), w[0], w[1], w[2], w[3], cached_bc);
      if (!has_separation) {
        if (cached_bc.degenerate) {
          cached_valid_closest = false;
        } else {
          cached_valid_closest = true;
          cached_v = Vec3A::zero();
        }
        return cached_valid_closest;
      }
      cached_p1 = p[0] * cached_bc.bary[0] + p[1] * cached_bc.bary[1] +
                  p[2] * cached_bc.bary[2] + p[3] * cached_bc.bary[3];
      cached_p2 = q[0] * cached_bc.bary[0] + q[1] * cached_bc.bary[1] +
                  q[2] * cached_bc.bary[2] + q[3] * cached_bc.bary[3];
      cached_v = cached_p1 - cached_p2;
      reduce_vertices(cached_bc.used);
      cached_valid_closest = cached_bc.is_valid();
      break;
    }
    default:
      cached_valid_closest = false;
      break;
    }
    return cached_valid_closest;
  }

  RS_HD static bool closest_pt_point_triangle(Vec3A pt, Vec3A a, Vec3A b,
                                              Vec3A c,
                                              SubSimplexClosestResult &result) {
    result.used.reset();
    Vec3A ab = b - a;
    Vec3A ac = c - a;
    Vec3A ap = pt - a;
    float d1 = ab.dot(ap);
    float d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
      result.closest_point_on_simplex = a;
      result.used.a = true;
      result.set_bary(1.0f, 0.0f, 0.0f, 0.0f);
      return true;
    }
    Vec3A bp = pt - b;
    float d3 = ab.dot(bp);
    float d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
      result.closest_point_on_simplex = b;
      result.used.b = true;
      result.set_bary(0.0f, 1.0f, 0.0f, 0.0f);
      return true;
    }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
      float v = d1 / (d1 - d3);
      result.closest_point_on_simplex = a + v * ab;
      result.used.a = true;
      result.used.b = true;
      result.set_bary(1.0f - v, v, 0.0f, 0.0f);
      return true;
    }
    Vec3A cp = pt - c;
    float d5 = ab.dot(cp);
    float d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
      result.closest_point_on_simplex = c;
      result.used.c = true;
      result.set_bary(0.0f, 0.0f, 1.0f, 0.0f);
      return true;
    }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
      float ww = d2 / (d2 - d6);
      result.closest_point_on_simplex = a + ww * ac;
      result.used.a = true;
      result.used.c = true;
      result.set_bary(1.0f - ww, 0.0f, ww, 0.0f);
      return true;
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
      float ww = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      result.closest_point_on_simplex = b + ww * (c - b);
      result.used.b = true;
      result.used.c = true;
      result.set_bary(0.0f, 1.0f - ww, ww, 0.0f);
      return true;
    }
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float ww = vc * denom;
    result.closest_point_on_simplex = a + ab * v + ac * ww;
    result.used.a = true;
    result.used.b = true;
    result.used.c = true;
    result.set_bary(1.0f - v - ww, v, ww, 0.0f);
    return true;
  }

  RS_HD static int point_outside_of_plane(Vec3A pt, Vec3A a, Vec3A b, Vec3A c,
                                          Vec3A d) {
    Vec3A normal = (b - a).cross(c - a);
    float signp = (pt - a).dot(normal);
    float signd = (d - a).dot(normal);
    if (signd * signd < k_degenerate_eps * k_degenerate_eps) {
      return -1;
    }
    return signp * signd < 0.0f ? 1 : 0;
  }

  RS_HD static bool
  closest_pt_point_tetrahedron(Vec3A pt, Vec3A a, Vec3A b, Vec3A c, Vec3A d,
                               SubSimplexClosestResult &final_result) {
    final_result.closest_point_on_simplex = pt;
    final_result.used.a = true;
    final_result.used.b = true;
    final_result.used.c = true;
    final_result.used.d = true;
    int abc = point_outside_of_plane(pt, a, b, c, d);
    if (abc < 0) {
      final_result.degenerate = true;
      return false;
    }
    int acd = point_outside_of_plane(pt, a, c, d, b);
    if (acd < 0) {
      final_result.degenerate = true;
      return false;
    }
    int adb = point_outside_of_plane(pt, a, d, b, c);
    if (adb < 0) {
      final_result.degenerate = true;
      return false;
    }
    int bdc = point_outside_of_plane(pt, b, d, c, a);
    if (bdc < 0) {
      final_result.degenerate = true;
      return false;
    }
    final_result.degenerate = false;
    if (!abc && !acd && !adb && !bdc) {
      return false;
    }
    float best_sq_dist = k_f32_max;
    SubSimplexClosestResult temp{};
    if (abc) {
      closest_pt_point_triangle(pt, a, b, c, temp);
      Vec3A qq = temp.closest_point_on_simplex;
      float sq_dist = (qq - pt).length_squared();
      if (sq_dist < best_sq_dist) {
        best_sq_dist = sq_dist;
        final_result.closest_point_on_simplex = qq;
        final_result.used.reset();
        final_result.used.a = temp.used.a;
        final_result.used.b = temp.used.b;
        final_result.used.c = temp.used.c;
        final_result.set_bary(temp.bary[0], temp.bary[1], temp.bary[2], 0.0f);
      }
    }
    if (acd) {
      closest_pt_point_triangle(pt, a, c, d, temp);
      Vec3A qq = temp.closest_point_on_simplex;
      float sq_dist = (qq - pt).length_squared();
      if (sq_dist < best_sq_dist) {
        best_sq_dist = sq_dist;
        final_result.closest_point_on_simplex = qq;
        final_result.used.reset();
        final_result.used.a = temp.used.a;
        final_result.used.c = temp.used.b;
        final_result.used.d = temp.used.c;
        final_result.set_bary(temp.bary[0], 0.0f, temp.bary[1], temp.bary[2]);
      }
    }
    if (adb) {
      closest_pt_point_triangle(pt, a, d, b, temp);
      Vec3A qq = temp.closest_point_on_simplex;
      float sq_dist = (qq - pt).length_squared();
      if (sq_dist < best_sq_dist) {
        best_sq_dist = sq_dist;
        final_result.closest_point_on_simplex = qq;
        final_result.used.reset();
        final_result.used.a = temp.used.a;
        final_result.used.b = temp.used.c;
        final_result.used.d = temp.used.b;
        final_result.set_bary(temp.bary[0], temp.bary[2], 0.0f, temp.bary[1]);
      }
    }
    if (bdc) {
      closest_pt_point_triangle(pt, b, d, c, temp);
      Vec3A qq = temp.closest_point_on_simplex;
      float sq_dist = (qq - pt).length_squared();
      if (sq_dist < best_sq_dist) {
        final_result.closest_point_on_simplex = qq;
        final_result.used.reset();
        final_result.used.b = temp.used.a;
        final_result.used.c = temp.used.c;
        final_result.used.d = temp.used.b;
        final_result.set_bary(0.0f, temp.bary[0], temp.bary[2], temp.bary[1]);
      }
    }
    return true;
  }
};

struct MinkowskiDiff {
  const ConvexShape *shape_a;
  const ConvexShape *shape_b;
  Mat3A toshape1;
  Affine3A toshape0;

  RS_HD static MinkowskiDiff make(const ConvexShape &a, const Affine3A &trans_a,
                                  const ConvexShape &b,
                                  const Affine3A &trans_b) {
    Mat3A a_t = trans_a.matrix3.transpose();
    MinkowskiDiff m;
    m.shape_a = &a;
    m.shape_b = &b;
    m.toshape1 = trans_b.matrix3.transpose() * trans_a.matrix3;
    m.toshape0 = {a_t * trans_b.matrix3,
                  a_t * (trans_b.translation - trans_a.translation)};
    return m;
  }

  template <bool Margin>
  RS_HD static Vec3A local_support(const ConvexShape &shape, Vec3A d) {
    if (!Margin) {
      return shape.support_without_margin(d);
    }
    Vec3A vecnorm = d;
    if (vecnorm.length_squared() < k_f32_epsilon * k_f32_epsilon) {
      vecnorm = Vec3A::splat(-1.0f);
    }
    vecnorm = vecnorm * (1.0f / vecnorm.length());
    return shape.support_without_margin(vecnorm) + shape.margin * vecnorm;
  }

  template <bool Margin> RS_HD Vec3A support0(Vec3A d) const {
    return local_support<Margin>(*shape_a, d);
  }

  template <bool Margin> RS_HD Vec3A support1(Vec3A d) const {
    Vec3A d_b = toshape1 * d;
    Vec3A local = local_support<Margin>(*shape_b, d_b);
    return toshape0.transform_point3a(local);
  }

  template <bool Margin> RS_HD Vec3A support(Vec3A d) const {
    return support0<Margin>(d) - support1<Margin>(-d);
  }
};

struct GjkSimplex {
  int rank;
  int c[4];
  float p[4];
};

enum class GjkStatus : uint8_t { Valid, Inside, Failed };

struct GjkSupport {
  Vec3A d, w;
};

struct Gjk {
  static constexpr int k_max_iterations = 128;
  static constexpr float k_accuracy = 1.0e-4f;
  static constexpr float k_min_distance = 1.0e-4f;
  static constexpr float k_duplicated_eps = 1.0e-4f;

  const MinkowskiDiff *shape;
  Vec3A ray;
  float distance;
  GjkSimplex simplices[2];
  GjkSupport store[4];
  int free_list[4];
  int free_count;
  int current;
  int simplex_idx;

  RS_HD void init(const MinkowskiDiff *s) {
    shape = s;
    ray = Vec3A::zero();
    distance = 0.0f;
    for (int i = 0; i < 2; ++i) {
      simplices[i].rank = 0;
      for (int j = 0; j < 4; ++j) {
        simplices[i].c[j] = 0;
        simplices[i].p[j] = 0.0f;
      }
    }
    for (int i = 0; i < 4; ++i) {
      store[i].d = Vec3A::zero();
      store[i].w = Vec3A::zero();
    }
    free_count = 0;
    current = 0;
    simplex_idx = 0;
  }

  RS_HDI void push_free(int v) { free_list[free_count++] = v; }

  RS_HDI int pop_free() { return free_list[--free_count]; }

  RS_HDI const GjkSimplex &simplex() const { return simplices[simplex_idx]; }

  template <bool Margin> RS_HD void getsupport(Vec3A d, int sv_index) {
    Vec3A dir = d * (1.0f / d.length());
    store[sv_index].d = dir;
    store[sv_index].w = shape->support<Margin>(dir);
  }

  RS_HD void remove_vertex(int si) {
    GjkSimplex &s = simplices[si];
    s.rank -= 1;
    push_free(s.c[s.rank]);
  }

  template <bool Margin> RS_HD void append_vertex(int si, Vec3A v) {
    int rank = simplices[si].rank;
    simplices[si].p[rank] = 0.0f;
    int sv_index = pop_free();
    getsupport<Margin>(v, sv_index);
    simplices[si].c[rank] = sv_index;
    simplices[si].rank = rank + 1;
  }

  RS_HDI static float det(Vec3A a, Vec3A b, Vec3A c) {
    return a.dot(b.cross(c));
  }

  RS_HD static float project_origin_2(Vec3A a, Vec3A b, float *w, uint32_t &m) {
    Vec3A d = b - a;
    float l = d.length_squared();
    if (l > 0.0f) {
      float t = l > 0.0f ? -a.dot(d) / l : 0.0f;
      if (t >= 1.0f) {
        w[0] = 0.0f;
        w[1] = 1.0f;
        m = 2;
        return b.length_squared();
      }
      if (t <= 0.0f) {
        w[0] = 1.0f;
        w[1] = 0.0f;
        m = 1;
        return a.length_squared();
      }
      w[1] = t;
      w[0] = 1.0f - t;
      m = 3;
      return (a + d * t).length_squared();
    }
    return -1.0f;
  }

  RS_HD static float project_origin_3(Vec3A a, Vec3A b, Vec3A c, float *w,
                                      uint32_t &m) {
    const int imd3[3] = {1, 2, 0};
    Vec3A vt[3] = {a, b, c};
    Vec3A dl[3] = {a - b, b - c, c - a};
    Vec3A n = dl[0].cross(dl[1]);
    float l = n.length_squared();
    if (l > 0.0f) {
      float mindist = -1.0f;
      float subw[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      uint32_t subm = 0;
      for (int i = 0; i < 3; ++i) {
        if (vt[i].dot(dl[i].cross(n)) > 0.0f) {
          int j = imd3[i];
          float subd = project_origin_2(vt[i], vt[j], subw, subm);
          if (mindist < 0.0f || subd < mindist) {
            mindist = subd;
            m = ((subm & 1) ? (1u << i) : 0u) + ((subm & 2) ? (1u << j) : 0u);
            w[i] = subw[0];
            w[j] = subw[1];
            w[imd3[j]] = 0.0f;
          }
        }
      }
      if (mindist < 0.0f) {
        float d = a.dot(n);
        float s = sqrt_f32(l);
        Vec3A pp = n * (d / l);
        mindist = pp.length_squared();
        m = 7;
        w[0] = dl[1].cross(b - pp).length() / s;
        w[1] = dl[2].cross(c - pp).length() / s;
        w[2] = 1.0f - (w[0] + w[1]);
      }
      return mindist;
    }
    return -1.0f;
  }

  RS_HD static float project_origin_4(Vec3A a, Vec3A b, Vec3A c, Vec3A d,
                                      float *w, uint32_t &m) {
    const int imd3[3] = {1, 2, 0};
    Vec3A vt[4] = {a, b, c, d};
    Vec3A dl[3] = {a - d, b - d, c - d};
    float vl = det(dl[0], dl[1], dl[2]);
    bool ng = (vl * a.dot((b - c).cross(a - b))) <= 0.0f;
    if (ng && abs_f32(vl) > 0.0f) {
      float mindist = -1.0f;
      float subw[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      uint32_t subm = 0;
      for (int i = 0; i < 3; ++i) {
        int j = imd3[i];
        float s = vl * d.dot(dl[i].cross(dl[j]));
        if (s > 0.0f) {
          float subd = project_origin_3(vt[i], vt[j], d, subw, subm);
          if (mindist < 0.0f || subd < mindist) {
            mindist = subd;
            m = ((subm & 1) ? (1u << i) : 0u) + ((subm & 2) ? (1u << j) : 0u) +
                ((subm & 4) ? 8u : 0u);
            w[i] = subw[0];
            w[j] = subw[1];
            w[imd3[j]] = 0.0f;
            w[3] = subw[2];
          }
        }
      }
      if (mindist < 0.0f) {
        mindist = 0.0f;
        m = 15;
        w[0] = det(c, b, d) / vl;
        w[1] = det(a, c, d) / vl;
        w[2] = det(b, a, d) / vl;
        w[3] = 1.0f - (w[0] + w[1] + w[2]);
      }
      return mindist;
    }
    return -1.0f;
  }

  template <bool Margin> RS_HD GjkStatus evaluate(Vec3A guess) {
    GjkStatus status = GjkStatus::Valid;
    int iterations = 0;
    float alpha = 0.0f;
    int clastw = 0;
    push_free(0);
    push_free(1);
    push_free(2);
    push_free(3);
    ray = guess;
    float sqdist = ray.length_squared();
    Vec3A dir = sqdist > 0.0f ? -ray : Vec3A{1.0f, 0.0f, 0.0f};
    append_vertex<Margin>(0, dir);
    simplices[0].p[0] = 1.0f;
    ray = store[simplices[0].c[0]].w;
    Vec3A lastw[4] = {ray, ray, ray, ray};
    while (status == GjkStatus::Valid) {
      float rl = ray.length();
      if (rl < k_min_distance) {
        status = GjkStatus::Inside;
        break;
      }
      int cur = current;
      int next = 1 - cur;
      append_vertex<Margin>(cur, -ray);
      GjkSimplex cs = simplices[cur];
      Vec3A wv = store[cs.c[cs.rank - 1]].w;
      bool found = false;
      for (int i = 0; i < 4; ++i) {
        if ((wv - lastw[i]).length_squared() < k_duplicated_eps) {
          found = true;
          break;
        }
      }
      if (found) {
        remove_vertex(cur);
        break;
      }
      clastw = (clastw + 1) & 3;
      lastw[clastw] = wv;
      float omega = ray.dot(wv) / rl;
      if (omega > alpha) {
        alpha = omega;
      }
      if (((rl - alpha) - (k_accuracy * rl)) <= 0.0f) {
        remove_vertex(cur);
        break;
      }
      float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      uint32_t mask = 0;
      switch (cs.rank) {
      case 2:
        sqdist =
            project_origin_2(store[cs.c[0]].w, store[cs.c[1]].w, weights, mask);
        break;
      case 3:
        sqdist = project_origin_3(store[cs.c[0]].w, store[cs.c[1]].w,
                                  store[cs.c[2]].w, weights, mask);
        break;
      case 4:
        sqdist =
            project_origin_4(store[cs.c[0]].w, store[cs.c[1]].w,
                             store[cs.c[2]].w, store[cs.c[3]].w, weights, mask);
        break;
      default:
        sqdist = -1.0f;
        break;
      }
      if (sqdist >= 0.0f) {
        GjkSimplex ns{};
        ray = Vec3A::zero();
        for (int i = 0; i < cs.rank; ++i) {
          if (mask & (1u << i)) {
            ns.c[ns.rank] = cs.c[i];
            ns.p[ns.rank] = weights[i];
            ray += store[cs.c[i]].w * weights[i];
            ns.rank += 1;
          } else {
            push_free(cs.c[i]);
          }
        }
        if (mask == 15) {
          status = GjkStatus::Inside;
        }
        simplices[next] = ns;
        current = next;
      } else {
        remove_vertex(cur);
        break;
      }
      ++iterations;
      if (iterations >= k_max_iterations) {
        status = GjkStatus::Failed;
      }
    }
    simplex_idx = current;
    if (status == GjkStatus::Valid) {
      distance = ray.length();
    } else if (status == GjkStatus::Inside) {
      distance = 0.0f;
    }
    return status;
  }

  template <bool Margin> RS_HD bool enclose_origin() {
    int si = simplex_idx;
    switch (simplices[si].rank) {
    case 1:
      for (int i = 0; i < 3; ++i) {
        Vec3A axis = Vec3A::zero();
        axis[i] = 1.0f;
        append_vertex<Margin>(si, axis);
        if (enclose_origin<Margin>()) {
          return true;
        }
        remove_vertex(si);
        append_vertex<Margin>(si, -axis);
        if (enclose_origin<Margin>()) {
          return true;
        }
        remove_vertex(si);
      }
      break;
    case 2: {
      Vec3A d = store[simplices[si].c[1]].w - store[simplices[si].c[0]].w;
      for (int i = 0; i < 3; ++i) {
        Vec3A axis = Vec3A::zero();
        axis[i] = 1.0f;
        Vec3A pp = d.cross(axis);
        if (pp.length_squared() > 0.0f) {
          append_vertex<Margin>(si, pp);
          if (enclose_origin<Margin>()) {
            return true;
          }
          remove_vertex(si);
          append_vertex<Margin>(si, -pp);
          if (enclose_origin<Margin>()) {
            return true;
          }
          remove_vertex(si);
        }
      }
      break;
    }
    case 3: {
      Vec3A w0 = store[simplices[si].c[0]].w;
      Vec3A n = (store[simplices[si].c[1]].w - w0)
                    .cross(store[simplices[si].c[2]].w - w0);
      if (n.length_squared() > 0.0f) {
        append_vertex<Margin>(si, n);
        if (enclose_origin<Margin>()) {
          return true;
        }
        remove_vertex(si);
        append_vertex<Margin>(si, -n);
        if (enclose_origin<Margin>()) {
          return true;
        }
        remove_vertex(si);
      }
      break;
    }
    case 4: {
      Vec3A w3 = store[simplices[si].c[3]].w;
      float d = det(store[simplices[si].c[0]].w - w3,
                    store[simplices[si].c[1]].w - w3,
                    store[simplices[si].c[2]].w - w3);
      if (abs_f32(d) > 0.0f) {
        return true;
      }
      break;
    }
    default:
      break;
    }
    return false;
  }
};

enum class EpaStatus : uint8_t {
  Valid,
  Degenerated,
  NonConvex,
  InvalidHull,
  OutOfFaces,
  OutOfVertices,
  AccuracyReached,
  FallBack,
  Failed
};

inline constexpr int16_t k_epa_none = -1;

struct EpaVertex {
  Vec3A w, d;
};

struct EpaFace {
  Vec3A n;
  float d;
  int16_t indices[3];
  int16_t adj[3];
  uint8_t edge[3];
  uint8_t pass;
  int16_t prev;
  int16_t next;
};

struct EpaList {
  int16_t root;
  int16_t count;
};

struct EpaHorizon {
  int16_t cf;
  int16_t ff;
  int nf;
};

struct Epa2 {
  static constexpr int k_max_vertices = 64;
  static constexpr int k_max_iterations = 255;
  static constexpr float k_accuracy = 1.0e-4f;
  static constexpr float k_plane_eps = 1.0e-5f;
  static constexpr int k_tetra_vertices = 4;
  static constexpr int k_store_vertices = k_max_vertices + k_tetra_vertices;
  static constexpr int k_max_faces = k_max_vertices * 2;

  GjkSimplex result;
  Vec3A normal;
  float depth;
  EpaVertex sv_store[k_store_vertices];
  EpaFace fc_store[k_max_faces];
  int next_sv;
  EpaList hull;
  EpaList stock;

  RS_HD void init() {
    result = GjkSimplex{};
    normal = Vec3A::zero();
    depth = 0.0f;
    for (int i = 0; i < k_store_vertices; ++i) {
      sv_store[i].w = Vec3A::zero();
      sv_store[i].d = Vec3A::zero();
    }
    for (int i = 0; i < k_max_faces; ++i) {
      EpaFace &f = fc_store[i];
      f.n = Vec3A::zero();
      f.d = 0.0f;
      f.indices[0] = f.indices[1] = f.indices[2] = 0;
      f.adj[0] = f.adj[1] = f.adj[2] = k_epa_none;
      f.edge[0] = f.edge[1] = f.edge[2] = 0;
      f.pass = 0;
      f.prev = k_epa_none;
      f.next = k_epa_none;
    }
    next_sv = 0;
    hull = {k_epa_none, 0};
    stock = {k_epa_none, 0};
    for (int i = k_max_faces - 1; i >= 0; --i) {
      append(stock, static_cast<int16_t>(i));
    }
  }

  RS_HD void append(EpaList &list, int16_t face_idx) {
    EpaFace &face = fc_store[face_idx];
    face.prev = k_epa_none;
    face.next = list.root;
    if (list.root != k_epa_none) {
      fc_store[list.root].prev = face_idx;
    }
    list.root = face_idx;
    list.count += 1;
  }

  RS_HD void remove(EpaList &list, int16_t face_idx) {
    int16_t prev = fc_store[face_idx].prev;
    int16_t next = fc_store[face_idx].next;
    if (next != k_epa_none) {
      fc_store[next].prev = prev;
    }
    if (prev != k_epa_none) {
      fc_store[prev].next = next;
    }
    if (list.root == face_idx) {
      list.root = next;
    }
    fc_store[face_idx].prev = k_epa_none;
    fc_store[face_idx].next = k_epa_none;
    list.count -= 1;
  }

  RS_HD void bind(int16_t fa, int ea, int16_t fb, int eb) {
    fc_store[fa].edge[ea] = static_cast<uint8_t>(eb);
    fc_store[fa].adj[ea] = fb;
    fc_store[fb].edge[eb] = static_cast<uint8_t>(ea);
    fc_store[fb].adj[eb] = fa;
  }

  RS_HD bool get_edge_dist(Vec3A face_n, int a_idx, int b_idx,
                           float &dist) const {
    Vec3A aw = sv_store[a_idx].w;
    Vec3A bw = sv_store[b_idx].w;
    Vec3A ba = bw - aw;
    Vec3A n_ab = ba.cross(face_n);
    float a_dot_nab = aw.dot(n_ab);
    if (a_dot_nab < 0.0f) {
      float ba_l2 = ba.length_squared();
      float a_dot_ba = aw.dot(ba);
      float b_dot_ba = bw.dot(ba);
      if (a_dot_ba > 0.0f) {
        dist = aw.length();
      } else if (b_dot_ba < 0.0f) {
        dist = bw.length();
      } else {
        float a_dot_b = aw.dot(bw);
        float value =
            (aw.length_squared() * bw.length_squared() - a_dot_b * a_dot_b) /
            ba_l2;
        dist = sqrt_f32(max_f32_c(value, 0.0f));
      }
      return true;
    }
    return false;
  }

  RS_HD int16_t new_face(int a_idx, int b_idx, int c_idx, EpaStatus &status,
                         bool forced) {
    int16_t face_idx = stock.root;
    if (face_idx == k_epa_none) {
      if (!forced) {
        status = EpaStatus::OutOfFaces;
      }
      return k_epa_none;
    }
    remove(stock, face_idx);
    append(hull, face_idx);
    Vec3A aw = sv_store[a_idx].w;
    Vec3A bw = sv_store[b_idx].w;
    Vec3A cw = sv_store[c_idx].w;
    Vec3A n = (bw - aw).cross(cw - aw);
    float l = n.length();
    if (l > k_accuracy) {
      float dist = 0.0f;
      if (!(get_edge_dist(n, a_idx, b_idx, dist) ||
            get_edge_dist(n, b_idx, c_idx, dist) ||
            get_edge_dist(n, c_idx, a_idx, dist))) {
        dist = aw.dot(n) / l;
      }
      EpaFace &face = fc_store[face_idx];
      face.pass = 0;
      face.indices[0] = static_cast<int16_t>(a_idx);
      face.indices[1] = static_cast<int16_t>(b_idx);
      face.indices[2] = static_cast<int16_t>(c_idx);
      face.adj[0] = face.adj[1] = face.adj[2] = k_epa_none;
      face.edge[0] = face.edge[1] = face.edge[2] = 0;
      face.n = n * (1.0f / l);
      face.d = dist;
      if (forced || face.d >= -k_plane_eps) {
        return face_idx;
      }
      status = EpaStatus::NonConvex;
    } else if (!forced) {
      status = EpaStatus::Degenerated;
    }
    remove(hull, face_idx);
    append(stock, face_idx);
    return k_epa_none;
  }

  RS_HD int16_t find_best() const {
    int16_t minf = hull.root;
    float mind = fc_store[minf].d * fc_store[minf].d;
    int16_t cur = fc_store[minf].next;
    while (cur != k_epa_none) {
      float sqd = fc_store[cur].d * fc_store[cur].d;
      if (sqd < mind) {
        minf = cur;
        mind = sqd;
      }
      cur = fc_store[cur].next;
    }
    return minf;
  }

  struct ExpandFrame {
    int16_t face_idx;
    int16_t adj2;
    int8_t edge2;
    bool second;
  };

  RS_HD bool expand_enter(uint8_t pass, int w_index, int16_t face_idx, int edge,
                          int depth_level, EpaHorizon &horizon,
                          EpaStatus &status, ExpandFrame &frame, int16_t &child,
                          int &child_edge, bool &result) {
    const int i1m3[3] = {1, 2, 0};
    const int i2m3[3] = {2, 0, 1};
    result = false;
    if (depth_level >= k_max_faces) {
      return false;
    }
    EpaFace face = fc_store[face_idx];
    if (face.pass == pass) {
      return false;
    }
    int e1 = i1m3[edge];
    Vec3A wv = sv_store[w_index].w;
    if (face.n.dot(wv) - face.d < -k_plane_eps) {
      int16_t nf = new_face(face.indices[e1], face.indices[edge], w_index,
                            status, false);
      if (nf != k_epa_none) {
        bind(nf, 0, face_idx, edge);
        if (horizon.cf != k_epa_none) {
          bind(horizon.cf, 1, nf, 2);
        } else {
          horizon.ff = nf;
        }
        horizon.cf = nf;
        horizon.nf += 1;
        result = true;
      }
      return false;
    }
    int e2 = i2m3[edge];
    fc_store[face_idx].pass = pass;
    int16_t adj1 = fc_store[face_idx].adj[e1];
    int16_t adj2 = fc_store[face_idx].adj[e2];
    if (adj1 == k_epa_none || adj2 == k_epa_none) {
      status = EpaStatus::InvalidHull;
      return false;
    }
    frame = {face_idx, adj2, static_cast<int8_t>(fc_store[face_idx].edge[e2]),
             false};
    child = adj1;
    child_edge = fc_store[face_idx].edge[e1];
    return true;
  }

  RS_HD bool expand(uint8_t pass, int w_index, int16_t face_idx, int edge,
                    EpaHorizon &horizon, EpaStatus &status, int depth_level) {
    ExpandFrame frames[k_max_faces];
    int num_frames = 0;
    int16_t child = face_idx;
    int child_edge = edge;
    bool result = false;
    for (;;) {
      ExpandFrame frame;
      if (expand_enter(pass, w_index, child, child_edge,
                       depth_level + num_frames, horizon, status, frame, child,
                       child_edge, result)) {
        frames[num_frames++] = frame;
        continue;
      }
      for (;;) {
        if (num_frames == 0) {
          return result;
        }
        ExpandFrame &top = frames[num_frames - 1];
        if (!top.second && result) {
          top.second = true;
          child = top.adj2;
          child_edge = top.edge2;
          break;
        }
        if (top.second && result) {
          remove(hull, top.face_idx);
          append(stock, top.face_idx);
        }
        --num_frames;
      }
    }
  }

  RS_HD EpaStatus fallback(int simplex_c0, Vec3A guess) {
    normal = -guess;
    float nl = normal.length();
    if (nl > 0.0f) {
      normal *= 1.0f / nl;
    } else {
      normal = Vec3A{1.0f, 0.0f, 0.0f};
    }
    depth = 0.0f;
    result.rank = 1;
    result.c[0] = simplex_c0;
    result.p[0] = 1.0f;
    return EpaStatus::FallBack;
  }

  template <bool Margin> RS_HD EpaStatus evaluate(Gjk &gjk, Vec3A guess) {
    GjkSimplex simplex_before = gjk.simplex();
    if (simplex_before.rank <= 1 || !gjk.enclose_origin<Margin>()) {
      sv_store[0].w = gjk.store[simplex_before.c[0]].w;
      sv_store[0].d = gjk.store[simplex_before.c[0]].d;
      return fallback(0, guess);
    }
    GjkSimplex simplex = gjk.simplex();
    if (simplex.rank != 4) {
      sv_store[0].w = gjk.store[simplex.c[0]].w;
      sv_store[0].d = gjk.store[simplex.c[0]].d;
      return fallback(0, guess);
    }
    while (hull.root != k_epa_none) {
      int16_t face = hull.root;
      remove(hull, face);
      append(stock, face);
    }
    next_sv = 0;
    int sv_indices[4] = {0, 0, 0, 0};
    for (int i = 0; i < simplex.rank; ++i) {
      int sv_index = next_sv++;
      sv_store[sv_index].w = gjk.store[simplex.c[i]].w;
      sv_store[sv_index].d = gjk.store[simplex.c[i]].d;
      sv_indices[i] = sv_index;
    }
    next_sv = k_tetra_vertices;
    Vec3A w3 = sv_store[sv_indices[3]].w;
    Vec3A w0 = sv_store[sv_indices[0]].w - w3;
    Vec3A w1 = sv_store[sv_indices[1]].w - w3;
    Vec3A w2 = sv_store[sv_indices[2]].w - w3;
    if (Gjk::det(w0, w1, w2) < 0.0f) {
      int tmp = sv_indices[0];
      sv_indices[0] = sv_indices[1];
      sv_indices[1] = tmp;
    }
    const int tetra_idx[4][3] = {{sv_indices[0], sv_indices[1], sv_indices[2]},
                                 {sv_indices[1], sv_indices[0], sv_indices[3]},
                                 {sv_indices[2], sv_indices[1], sv_indices[3]},
                                 {sv_indices[0], sv_indices[2], sv_indices[3]}};
    int16_t tetra[4] = {0, 0, 0, 0};
    bool all_faces = true;
    EpaStatus unused = EpaStatus::Valid;
    for (int k = 0; k < 4; ++k) {
      int16_t f = new_face(tetra_idx[k][0], tetra_idx[k][1], tetra_idx[k][2],
                           unused, true);
      if (f != k_epa_none) {
        tetra[k] = f;
      } else {
        all_faces = false;
      }
    }
    if (!all_faces || hull.count != 4) {
      return fallback(sv_indices[0], guess);
    }
    bind(tetra[0], 0, tetra[1], 0);
    bind(tetra[0], 1, tetra[2], 0);
    bind(tetra[0], 2, tetra[3], 0);
    bind(tetra[1], 1, tetra[3], 2);
    bind(tetra[1], 2, tetra[2], 1);
    bind(tetra[2], 2, tetra[3], 1);
    EpaStatus status = EpaStatus::Valid;
    int16_t best = find_best();
    EpaFace outer = fc_store[best];
    uint8_t pass = 0;
    for (int iter = 0; iter < k_max_iterations; ++iter) {
      if (next_sv >= k_store_vertices) {
        status = EpaStatus::OutOfVertices;
        break;
      }
      int w_index = next_sv++;
      pass = static_cast<uint8_t>(pass + 1);
      fc_store[best].pass = pass;
      Vec3A n = fc_store[best].n;
      Vec3A dir = n * (1.0f / n.length());
      Vec3A wv = gjk.shape->support<Margin>(dir);
      sv_store[w_index].w = wv;
      sv_store[w_index].d = dir;
      float wdist = n.dot(wv) - fc_store[best].d;
      if (!(wdist > k_accuracy)) {
        status = EpaStatus::AccuracyReached;
        break;
      }
      EpaHorizon horizon{k_epa_none, k_epa_none, 0};
      bool valid = true;
      for (int j = 0; j < 3; ++j) {
        int16_t adj = fc_store[best].adj[j];
        int edge = fc_store[best].edge[j];
        if (!expand(pass, w_index, adj, edge, horizon, status, 0)) {
          valid = false;
          break;
        }
      }
      if (valid && horizon.nf >= 3) {
        bind(horizon.cf, 1, horizon.ff, 2);
        remove(hull, best);
        append(stock, best);
        best = find_best();
        outer = fc_store[best];
      } else {
        status = EpaStatus::InvalidHull;
        break;
      }
    }
    Vec3A projection = outer.n * outer.d;
    normal = outer.n;
    depth = outer.d;
    result.rank = 3;
    int a_idx = outer.indices[0];
    int b_idx = outer.indices[1];
    int c_idx = outer.indices[2];
    result.c[0] = a_idx;
    result.c[1] = b_idx;
    result.c[2] = c_idx;
    result.p[0] = (sv_store[b_idx].w - projection)
                      .cross(sv_store[c_idx].w - projection)
                      .length();
    result.p[1] = (sv_store[c_idx].w - projection)
                      .cross(sv_store[a_idx].w - projection)
                      .length();
    result.p[2] = (sv_store[a_idx].w - projection)
                      .cross(sv_store[b_idx].w - projection)
                      .length();
    float sum = result.p[0] + result.p[1] + result.p[2];
    result.p[0] /= sum;
    result.p[1] /= sum;
    result.p[2] /= sum;
    return status;
  }
};

struct GjkEpa2Result {
  Vec3A witnesses[2];
  Vec3A normal;
  bool penetrating;
};

RS_HDN bool gjk_penetration(const ConvexShape &shape_a, const Affine3A &trans_a,
                            const ConvexShape &shape_b, const Affine3A &trans_b,
                            Vec3A guess, GjkEpa2Result &out) {
  MinkowskiDiff shape = MinkowskiDiff::make(shape_a, trans_a, shape_b, trans_b);
  Gjk gjk;
  gjk.init(&shape);
  if (gjk.evaluate<true>(-guess) != GjkStatus::Inside) {
    return false;
  }
  Epa2 epa;
  epa.init();
  EpaStatus status = epa.evaluate<true>(gjk, -guess);
  if (status == EpaStatus::Failed) {
    return false;
  }
  float margin_a = shape_a.margin;
  Vec3A w0 = Vec3A::zero();
  for (int i = 0; i < epa.result.rank; ++i) {
    const EpaVertex &vertex = epa.sv_store[epa.result.c[i]];
    Vec3A dir = vertex.d;
    if (dir.length_squared() < k_f32_epsilon * k_f32_epsilon) {
      dir = Vec3A::splat(-1.0f);
    }
    dir = dir * (1.0f / dir.length());
    Vec3A support0 = shape.support0<false>(dir) + margin_a * dir;
    w0 += support0 * epa.result.p[i];
  }
  out.witnesses[0] = trans_a.transform_point3a(w0);
  out.witnesses[1] = trans_a.transform_point3a(w0 - epa.normal * epa.depth);
  out.normal = trans_a.matrix3 * -epa.normal;
  out.penetrating = true;
  return true;
}

RS_HD bool gjk_distance(const ConvexShape &shape_a, const Affine3A &trans_a,
                        const ConvexShape &shape_b, const Affine3A &trans_b,
                        Vec3A guess, GjkEpa2Result &out) {
  MinkowskiDiff shape = MinkowskiDiff::make(shape_a, trans_a, shape_b, trans_b);
  Gjk gjk;
  gjk.init(&shape);
  if (gjk.evaluate<false>(guess) != GjkStatus::Valid) {
    return false;
  }
  const GjkSimplex &simplex = gjk.simplex();
  Vec3A w0 = Vec3A::zero();
  Vec3A w1 = Vec3A::zero();
  for (int i = 0; i < simplex.rank; ++i) {
    float weight = simplex.p[i];
    Vec3A sv_d = gjk.store[simplex.c[i]].d;
    w0 += shape.support0<false>(sv_d) * weight;
    w1 += shape.support1<false>(-sv_d) * weight;
  }
  out.witnesses[0] = trans_a.transform_point3a(w0);
  out.witnesses[1] = trans_a.transform_point3a(w1);
  Vec3A normal_local = w0 - w1;
  float distance = normal_local.length();
  float normal_scale = 1.0f / (distance > 1.0e-4f ? distance : 1.0f);
  normal_local = normal_local * normal_scale;
  out.normal = trans_a.matrix3 * normal_local;
  out.penetrating = false;
  return true;
}

RS_HD bool calc_pen_depth(const ConvexShape &shape_a,
                          const ConvexShape &shape_b, const Affine3A &trans_a,
                          const Affine3A &trans_b, GjkEpa2Result &out) {
  Vec3A delta = trans_b.translation - trans_a.translation;
  float len2 = delta.length_squared();
  Vec3A guess =
      len2 > 1.0e-20f ? delta * (1.0f / sqrt_f32(len2)) : Vec3A::zero();
  if (gjk_penetration(shape_a, trans_a, shape_b, trans_b, guess, out)) {
    return true;
  }
  return gjk_distance(shape_a, trans_a, shape_b, trans_b, guess, out);
}

struct GjkPairDetector {
  static constexpr int k_max_iterations = 1000;
  static constexpr float k_rel_error2 = 1.0e-6f;

  float margin;
  float margin_b;
  Vec3A separating_axis;
  float separating_distance;
  VoronoiSimplexSolver simplex_solver;

  RS_HD void init(float margin_a, float mb) {
    margin = margin_a + mb;
    margin_b = mb;
    separating_axis = Vec3A{0.0f, 1.0f, 0.0f};
    separating_distance = 0.0f;
    simplex_solver.init();
  }

  template <typename Output>
  RS_HD void
  get_closest_points(const Affine3A &transform_a, const Affine3A &transform_b,
                     float maximum_distance_squared, const ConvexShape &shape_a,
                     const ConvexShape &shape_b, Output &output) {
    Affine3A local_a = transform_a;
    Affine3A local_b = transform_b;
    Vec3A position_offset = (local_a.translation + local_b.translation) * 0.5f;
    local_a.translation -= position_offset;
    local_b.translation -= position_offset;
    bool check_simplex = false;
    int degenerate_simplex = 0;
    float squared_distance = k_f32_max;
    for (int iter = 0; iter < k_max_iterations; ++iter) {
      Vec3A sep_in_a = local_a.matrix3.transpose_mul_vec3a(-separating_axis);
      Vec3A sep_in_b = local_b.matrix3.transpose_mul_vec3a(separating_axis);
      Vec3A p_in_a = shape_a.support_without_margin(sep_in_a);
      Vec3A p_world = local_a.transform_point3a(p_in_a);
      Vec3A q_in_b = shape_b.support_without_margin(sep_in_b);
      Vec3A q_world = local_b.transform_point3a(q_in_b);
      Vec3A wv = p_world - q_world;
      float delta = separating_axis.dot(wv);
      if (delta > 0.0f &&
          delta * delta > squared_distance * maximum_distance_squared) {
        degenerate_simplex = 10;
        check_simplex = true;
        break;
      }
      if (simplex_solver.in_simplex(wv)) {
        degenerate_simplex = 1;
        check_simplex = true;
        break;
      }
      float f0 = squared_distance - delta;
      float f1 = squared_distance * k_rel_error2;
      if (f0 <= f1) {
        degenerate_simplex = is_sign_negative_f32(f0) ? 2 : 11;
        check_simplex = true;
        break;
      }
      simplex_solver.add_vertex(wv, p_world, q_world);
      Vec3A new_axis = Vec3A::zero();
      if (!simplex_solver.closest(new_axis)) {
        degenerate_simplex = 3;
        check_simplex = true;
        break;
      }
      if (new_axis.length_squared() < k_rel_error2) {
        separating_axis = new_axis;
        degenerate_simplex = 6;
        check_simplex = true;
        break;
      }
      float previous_squared_distance = squared_distance;
      squared_distance = new_axis.length_squared();
      if (previous_squared_distance - squared_distance <=
          k_f32_epsilon * previous_squared_distance) {
        check_simplex = true;
        degenerate_simplex = 12;
        break;
      }
      separating_axis = new_axis;
      if (simplex_solver.full_simplex()) {
        degenerate_simplex = 13;
        break;
      }
    }
    bool is_valid = false;
    float distance = 0.0f;
    Vec3A point_on_b = Vec3A::zero();
    Vec3A normal_in_b = Vec3A::zero();
    if (check_simplex) {
      point_on_b = simplex_solver.compute_points();
      normal_in_b = separating_axis;
      float len_sqr = separating_axis.length_squared();
      if (len_sqr <= 0.0001f) {
        degenerate_simplex = 5;
      }
      if (len_sqr > k_f32_epsilon * k_f32_epsilon) {
        float rlen = 1.0f / sqrt_f32(len_sqr);
        normal_in_b *= rlen;
        float s = sqrt_f32(squared_distance);
        point_on_b += separating_axis * (margin_b / s);
        distance = (1.0f / rlen) - margin;
        is_valid = true;
      }
    }
    bool catch_degenerate =
        degenerate_simplex != 0 && (distance + margin) <= 0.01f;
    GjkEpa2Result pen;
    if ((!is_valid || catch_degenerate) &&
        calc_pen_depth(shape_a, shape_b, local_a, local_b, pen)) {
      Vec3A tmp_a = pen.witnesses[0];
      Vec3A tmp_b = pen.witnesses[1];
      if (pen.penetrating) {
        Vec3A witness_delta = tmp_b - tmp_a;
        Vec3A normal = witness_delta;
        float len_sqr = normal.length_squared();
        if (len_sqr <= k_f32_epsilon * k_f32_epsilon) {
          normal = pen.normal;
          len_sqr = normal.length_squared();
        }
        if (len_sqr > k_f32_epsilon * k_f32_epsilon) {
          Vec3A n = normal * (1.0f / sqrt_f32(len_sqr));
          float distance_2 = -witness_delta.length();
          if (!is_valid || distance_2 < distance) {
            distance = distance_2;
            point_on_b = tmp_b;
            normal_in_b = n;
            is_valid = true;
          }
        }
      } else {
        float nls = pen.normal.length_squared();
        if (nls > 0.0f) {
          float distance_2 = (tmp_a - tmp_b).length() - margin;
          if (!is_valid || distance_2 < distance) {
            distance = distance_2;
            point_on_b = tmp_b + pen.normal * margin_b;
            normal_in_b = pen.normal.normalize();
            is_valid = true;
          }
        }
      }
    }
    if (is_valid &&
        (distance < 0.0f || distance * distance < maximum_distance_squared)) {
      separating_axis = normal_in_b;
      separating_distance = distance;
      output.add_contact_point(normal_in_b, point_on_b + position_offset,
                               distance);
    }
  }
};

struct ConvexCastResult {
  Vec3A normal;
  float fraction;
};

RS_HDN bool calc_time_of_impact(const ConvexShape &convex_b, Vec3A from_a,
                                Vec3A to_a, ConvexCastResult &out) {
  VoronoiSimplexSolver simplex_solver;
  simplex_solver.init();
  float lambda = 0.0f;
  Vec3A interpolated = from_a;
  Vec3A sup_vertex_b = convex_b.support(to_a - from_a);
  Vec3A v = from_a - sup_vertex_b;
  Vec3A n = Vec3A::zero();
  for (int i = 0; i < 32; ++i) {
    Vec3A wv = interpolated - sup_vertex_b;
    float v_dot_w = v.dot(wv);
    if (lambda > 1.0f) {
      return false;
    }
    if (v_dot_w > 0.0f) {
      float v_dot_r = v.dot(to_a - from_a);
      if (v_dot_r >= -(k_f32_epsilon * k_f32_epsilon)) {
        return false;
      }
      lambda -= v_dot_w / v_dot_r;
      interpolated = from_a + (to_a - from_a) * lambda;
      n = v;
    }
    if (!simplex_solver.in_simplex(wv)) {
      simplex_solver.add_vertex(wv, interpolated, sup_vertex_b);
    }
    float dist2 = simplex_solver.closest(v) ? v.length_squared() : 0.0f;
    if (dist2 <= 0.0001f) {
      break;
    }
  }
  Vec3A normal;
  if (!n.try_normalize(normal)) {
    return false;
  }
  if (is_sign_negative_f32(normal.dot(to_a - from_a))) {
    out = {normal, lambda};
    return true;
  }
  return false;
}

template <typename Closest>
RS_HD bool calc_point_time_of_impact(Vec3A from, Vec3A to, Closest &closest,
                                     ConvexCastResult &out) {
  Vec3A delta = to - from;
  float dist;
  Vec3A normal;
  if (!closest(from, dist, normal)) {
    return false;
  }
  float lambda = 0.0f;
  float last_lambda = 0.0f;
  int iteration = 0;
  while (dist > 0.001f) {
    ++iteration;
    if (iteration > 64) {
      return false;
    }
    lambda -= dist / delta.dot(normal);
    if (lambda > 1.0f || lambda < 0.0f || lambda <= last_lambda) {
      return false;
    }
    last_lambda = lambda;
    Vec3A point = from * (1.0f - lambda) + to * lambda;
    float next_dist;
    Vec3A next_normal;
    if (!closest(point, next_dist, next_normal)) {
      return false;
    }
    if (next_dist < 0.0f) {
      out = {next_normal, last_lambda};
      return true;
    }
    dist = next_dist;
    normal = next_normal;
  }
  if (normal.dot(delta) >= 0.0f) {
    return false;
  }
  out = {normal, lambda};
  return true;
}

}
