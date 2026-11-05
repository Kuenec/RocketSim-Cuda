#pragma once

#include <cstdint>

#include "rsc/bullet/manifold.hpp"
#include "rsc/bullet/shapes.hpp"
#include "rsc/math/libm.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

namespace boxbox {

struct Obb {
  Vec3A center;
  Mat3A axis;
  Vec3A extent;
};

struct Hit {
  float depth;
  Vec3A normal;
  int axis_idx;
};

struct Point2 {
  float v[2];
};

RS_HD float line_closest_approach(Vec3A pa, Vec3A ua, Vec3A pb, Vec3A ub) {
  Vec3A p = pb - pa;
  float uaub = ua.dot(ub);
  float q1 = ua.dot(p);
  float q2 = -ub.dot(p);
  float d = 1.0f - uaub * uaub;
  if (d <= 0.0001f) {
    return 0.0f;
  }
  return (uaub * q1 + q2) / d;
}

RS_HD int intersect_rect_quad2(const float *h, const Point2 *poly,
                               Point2 *out) {
  Point2 q[16];
  int q_len = 4;
  for (int i = 0; i < 4; ++i) {
    q[i] = poly[i];
  }
  const float signs[2] = {-1.0f, 1.0f};
  for (int dir = 0; dir < 2; ++dir) {
    for (int si = 0; si < 2; ++si) {
      float sign = signs[si];
      Point2 r[16];
      int r_len = 0;
      float clip_val = sign * h[dir];
      for (int i = 0; i < q_len; ++i) {
        Point2 cur = q[i];
        Point2 next = q[(i + 1) % q_len];
        float cur_val = cur.v[dir];
        float next_val = next.v[dir];
        bool inside_cur = sign * cur_val < h[dir];
        if (inside_cur) {
          r[r_len++] = cur;
        }
        bool inside_next = sign * next_val < h[dir];
        if (inside_cur != inside_next) {
          float p = cur.v[1 - dir] + (next.v[1 - dir] - cur.v[1 - dir]) /
                                         (next_val - cur_val) *
                                         (clip_val - cur_val);
          float p1 = p;
          float p2 = clip_val;
          if (dir == 0) {
            float tmp = p1;
            p1 = p2;
            p2 = tmp;
          }
          r[r_len++] = {{p1, p2}};
        }
      }
      for (int i = 0; i < r_len; ++i) {
        q[i] = r[i];
      }
      q_len = r_len;
    }
  }
  for (int i = 0; i < q_len; ++i) {
    out[i] = q[i];
  }
  return q_len;
}

RS_HD int cull_points2(const Point2 *p, int n, int i0, int m, int *result) {
  float cx, cy;
  if (n == 1) {
    cx = p[0].v[0];
    cy = p[0].v[1];
  } else if (n == 2) {
    cx = (p[0].v[0] + p[1].v[0]) * 0.5f;
    cy = (p[0].v[1] + p[1].v[1]) * 0.5f;
  } else {
    float a_sum = 0.0f;
    float cx_sum = 0.0f;
    float cy_sum = 0.0f;
    for (int i = 0; i < n - 1; ++i) {
      float x0 = p[i].v[0], y0 = p[i].v[1];
      float x1 = p[i + 1].v[0], y1 = p[i + 1].v[1];
      float q = x0 * y1 - x1 * y0;
      a_sum += q;
      cx_sum += q * (x0 + x1);
      cy_sum += q * (y0 + y1);
    }
    float x_last = p[n - 1].v[0], y_last = p[n - 1].v[1];
    float x0 = p[0].v[0], y0 = p[0].v[1];
    float q_last = x_last * y0 - x0 * y_last;
    float area = a_sum + q_last;
    float inv = abs_f32(area) > k_f32_epsilon ? 1.0f / (3.0f * area) : 1e18f;
    cx = inv * (cx_sum + q_last * (x_last + x0));
    cy = inv * (cy_sum + q_last * (y_last + y0));
  }
  float angles[8];
  for (int i = 0; i < n; ++i) {
    angles[i] = atan2f_glibc(p[i].v[1] - cy, p[i].v[0] - cx);
  }
  bool avail[8] = {false, false, false, false, false, false, false, false};
  for (int i = 0; i < n; ++i) {
    avail[i] = true;
  }
  avail[i0] = false;
  int count = 0;
  result[count++] = i0;
  for (int j = 1; j < m; ++j) {
    float target =
        static_cast<float>(j) * (k_tau / static_cast<float>(m)) + angles[i0];
    if (target > k_pi) {
      target -= k_tau;
    }
    int best_idx = i0;
    float best_diff = k_f32_max;
    for (int i = 0; i < n; ++i) {
      if (avail[i]) {
        float diff = abs_f32(angles[i] - target);
        if (diff > k_pi) {
          diff = k_tau - diff;
        }
        if (diff < best_diff) {
          best_diff = diff;
          best_idx = i;
        }
      }
    }
    avail[best_idx] = false;
    result[count++] = best_idx;
  }
  return count;
}

RS_HD bool box_box_sat(const Obb &obb1, const Mat3A &r1t, const Obb &obb2,
                       Hit &hit) {
  constexpr float fudge_factor = 1.05f;
  Vec3A p = obb2.center - obb1.center;
  Vec3A pp = r1t * p;
  Mat3A rij = r1t * obb2.axis;
  Vec3A rij_cols[3] = {rij.x_axis, rij.y_axis, rij.z_axis};
  Mat3A q_cols_mat = rij.abs();
  Vec3A q_cols[3] = {q_cols_mat.x_axis, q_cols_mat.y_axis, q_cols_mat.z_axis};
  Mat3A q_rows_mat = q_cols_mat.transpose();
  Vec3A q_rows[3] = {q_rows_mat.x_axis, q_rows_mat.y_axis, q_rows_mat.z_axis};
  float s = u32_to_f32(0xff800000u);
  float s2 = 0.0f;
  bool has_normal_r = false;
  Vec3A normal_r = Vec3A::zero();
  Vec3A normal_c = Vec3A::zero();
  bool invert_normal = false;
  int code = 0;
  Vec3A axes1[3] = {obb1.axis.x_axis, obb1.axis.y_axis, obb1.axis.z_axis};
  for (int i = 0; i < 3; ++i) {
    float expr1 = pp[i];
    float expr2 = obb1.extent[i] + obb2.extent.dot(q_rows[i]);
    s2 = abs_f32(expr1) - expr2;
    if (s2 > 0.0f) {
      return false;
    }
    if (s2 > s) {
      s = s2;
      has_normal_r = true;
      normal_r = axes1[i];
      invert_normal = expr1 < 0.0f;
      code = i + 1;
    }
  }
  Vec3A axes2[3] = {obb2.axis.x_axis, obb2.axis.y_axis, obb2.axis.z_axis};
  for (int i = 0; i < 3; ++i) {
    float expr1 = axes2[i].dot(p);
    float expr2 = obb1.extent.dot(q_cols[i]) + obb2.extent[i];
    s2 = abs_f32(expr1) - expr2;
    if (s2 > 0.0f) {
      return false;
    }
    if (s2 > s) {
      s = s2;
      has_normal_r = true;
      normal_r = axes2[i];
      invert_normal = expr1 < 0.0f;
      code = i + 4;
    }
  }
  for (int c = 0; c < 3; ++c) {
    q_cols[c] += Vec3A::splat(1e-5f);
  }
  Mat3A q_rows_mat2 =
      Mat3A::from_cols(q_cols[0], q_cols[1], q_cols[2]).transpose();
  q_rows[0] = q_rows_mat2.x_axis;
  q_rows[1] = q_rows_mat2.y_axis;
  q_rows[2] = q_rows_mat2.z_axis;
  for (int i = 0; i < 3; ++i) {
    Vec3A axis = Vec3A::zero();
    axis[i] = 1.0f;
    int i1 = (i + 1) % 3;
    int i2 = (i + 2) % 3;
    for (int j = 0; j < 3; ++j) {
      int j1 = (j + 1) % 3;
      int j2 = (j + 2) % 3;
      Vec3A n = axis.cross(rij_cols[j]);
      float expr2 =
          obb1.extent[i1] * q_cols[j][i2] + obb1.extent[i2] * q_cols[j][i1] +
          obb2.extent[j1] * q_rows[i][j2] + obb2.extent[j2] * q_rows[i][j1];
      float expr1 = pp.dot(n);
      s2 = abs_f32(expr1) - expr2;
      if (s2 > k_f32_epsilon) {
        return false;
      }
      float l = n.length();
      if (l > k_f32_epsilon) {
        s2 /= l;
        if (s2 * fudge_factor > s) {
          s = s2;
          has_normal_r = false;
          normal_c = n / l;
          invert_normal = expr1 < 0.0f;
          code = 7 + i * 3 + j;
        }
      }
    }
  }
  if (code == 0) {
    return false;
  }
  Vec3A normal = has_normal_r ? normal_r : obb1.axis * normal_c;
  if (invert_normal) {
    normal = -normal;
  }
  hit = {-s, normal, code};
  return true;
}

template <typename Callback>
RS_HD void compute_contact_points(const Obb *obb1_in, const Mat3A *r1t_in,
                                  const Obb *obb2_in, const Mat3A *r2t_in,
                                  const Hit &hit, PersistentManifold &m,
                                  const RigidBody &col1, const RigidBody &col2,
                                  Callback &callback) {
  const Obb *obb1 = obb1_in;
  const Obb *obb2 = obb2_in;
  const Mat3A *r2t = r2t_in;
  Vec3A r1t_axes[3] = {r1t_in->x_axis, r1t_in->y_axis, r1t_in->z_axis};
  Vec3A r2t_axes[3] = {r2t_in->x_axis, r2t_in->y_axis, r2t_in->z_axis};
  if (hit.axis_idx > 6) {
    Vec3A pa = obb1->center;
    for (int i = 0; i < 3; ++i) {
      float sign =
          2.0f * (hit.normal.dot(r1t_axes[i]) > 0.0f ? 1.0f : 0.0f) - 1.0f;
      pa += sign * obb1->extent[i] * r1t_axes[i];
    }
    Vec3A pb = obb2->center;
    for (int i = 0; i < 3; ++i) {
      float sign =
          1.0f - 2.0f * (hit.normal.dot(r2t_axes[i]) > 0.0f ? 1.0f : 0.0f);
      pb += sign * obb2->extent[i] * r2t_axes[i];
    }
    Vec3A ua = r1t_axes[(hit.axis_idx - 7) / 3];
    Vec3A ub = r2t_axes[(hit.axis_idx - 7) % 3];
    float beta = line_closest_approach(pa, ua, pb, ub);
    m.add_contact_point(col1, col2, -hit.normal, pb + ub * beta, -hit.depth, -1,
                        callback);
    return;
  }
  if (hit.axis_idx > 3) {
    const Obb *t = obb1;
    obb1 = obb2;
    obb2 = t;
    r2t = r1t_in;
    for (int i = 0; i < 3; ++i) {
      Vec3A tv = r1t_axes[i];
      r1t_axes[i] = r2t_axes[i];
      r2t_axes[i] = tv;
    }
  }
  Vec3A normal_2 = hit.axis_idx <= 3 ? hit.normal : -hit.normal;
  Vec3A nr = r2t->transpose_mul_vec3a(normal_2);
  Vec3A anr = nr.abs();
  int lanr, a1, a2;
  if (anr.y > anr.x) {
    if (anr.y > anr.z) {
      lanr = 1;
      a1 = 0;
      a2 = 2;
    } else {
      lanr = 2;
      a1 = 0;
      a2 = 1;
    }
  } else if (anr.x > anr.z) {
    lanr = 0;
    a1 = 1;
    a2 = 2;
  } else {
    lanr = 2;
    a1 = 0;
    a2 = 1;
  }
  float face_sign = nr[lanr] < 0.0f ? obb2->extent[lanr] : -obb2->extent[lanr];
  Vec3A center = obb2->center - obb1->center + face_sign * r2t_axes[lanr];
  int code_n = hit.axis_idx <= 3 ? hit.axis_idx - 1 : hit.axis_idx - 4;
  int code1, code2;
  if (code_n == 0) {
    code1 = 1;
    code2 = 2;
  } else if (code_n == 1) {
    code1 = 0;
    code2 = 2;
  } else {
    code1 = 0;
    code2 = 1;
  }
  float c1 = center.dot(r1t_axes[code1]);
  float c2 = center.dot(r1t_axes[code2]);
  Vec4 mm{r1t_axes[code1].dot(r2t_axes[a1]), r1t_axes[code1].dot(r2t_axes[a2]),
          r1t_axes[code2].dot(r2t_axes[a1]), r1t_axes[code2].dot(r2t_axes[a2])};
  Vec4 k = mm * Vec4{obb2->extent[a1], obb2->extent[a2], obb2->extent[a1],
                     obb2->extent[a2]};
  Point2 quad[4] = {{{c1 - k.x - k.y, c2 - k.z - k.w}},
                    {{c1 - k.x + k.y, c2 - k.z + k.w}},
                    {{c1 + k.x + k.y, c2 + k.z + k.w}},
                    {{c1 + k.x - k.y, c2 + k.z - k.w}}};
  float rect[2] = {obb1->extent[code1], obb1->extent[code2]};
  Point2 ret[16];
  int ret_len = intersect_rect_quad2(rect, quad, ret);
  if (ret_len == 0) {
    return;
  }
  float det1 = 1.0f / (mm.x * mm.w - mm.y * mm.z);
  mm = mm * det1;
  int cnum = 0;
  Vec3A point[8];
  float dep[8];
  for (int i = 0; i < 8; ++i) {
    point[i] = Vec3A::zero();
    dep[i] = 0.0f;
  }
  for (int j = 0; j < ret_len; ++j) {
    float k1 = mm.w * (ret[j].v[0] - c1) - mm.y * (ret[j].v[1] - c2);
    float k2 = -mm.z * (ret[j].v[0] - c1) + mm.x * (ret[j].v[1] - c2);
    point[cnum] = center + k1 * r2t_axes[a1] + k2 * r2t_axes[a2];
    dep[cnum] = obb1->extent[code_n] - normal_2.dot(point[cnum]);
    if (dep[cnum] >= 0.0f) {
      ret[cnum] = ret[j];
      ++cnum;
    }
  }
  if (cnum == 0) {
    return;
  }
  int maxc = k_manifold_cache_size < cnum ? k_manifold_cache_size : cnum;
  if (cnum <= maxc) {
    for (int i = 0; i < cnum; ++i) {
      Vec3A pos = hit.axis_idx < 4
                      ? point[i] + obb1->center
                      : point[i] + obb1->center - hit.normal * dep[i];
      m.add_contact_point(col1, col2, -hit.normal, pos, -dep[i], -1, callback);
    }
    return;
  }
  int i1 = 0;
  for (int i = 0; i < cnum; ++i) {
    if (dep[i] > dep[i1]) {
      i1 = i;
    }
  }
  int iret[8];
  int n_iret = cull_points2(ret, cnum, i1, maxc, iret);
  for (int t = 0; t < n_iret && t < maxc; ++t) {
    int idx = iret[t];
    Vec3A pos_in_world = point[idx] + obb1->center;
    if (hit.axis_idx < 4) {
      m.add_contact_point(col1, col2, -hit.normal, pos_in_world, -dep[idx], -1,
                          callback);
    } else {
      m.add_contact_point(col1, col2, -hit.normal,
                          pos_in_world - hit.normal * dep[idx], -dep[idx], -1,
                          callback);
    }
  }
}

}

template <typename Callback>
RS_HDN bool obb_obb(const RigidBody &a_obj, const CompoundShape &a_shape,
                   const RigidBody &b_obj, const CompoundShape &b_shape,
                   Callback &callback, PersistentManifold &out) {
  Aabb aabb0 = a_shape.get_aabb(a_obj.world_trans);
  Aabb aabb1 = b_shape.get_aabb(b_obj.world_trans);
  if (!aabb0.intersects(aabb1)) {
    return false;
  }
  Affine3A ta = a_obj.world_trans * a_shape.child_trans;
  Affine3A tb = b_obj.world_trans * b_shape.child_trans;
  Mat3A axis_a = ta.matrix3;
  Mat3A axis_b = tb.matrix3;
  Mat3A axis_a_inv = axis_a.transpose();
  Vec3A side1 = a_shape.child_shape.half_extents + a_shape.child_shape.margin;
  Vec3A side2 = b_shape.child_shape.half_extents + b_shape.child_shape.margin;
  boxbox::Obb obb1{ta.translation, axis_a, side1};
  boxbox::Obb obb2{tb.translation, axis_b, side2};
  boxbox::Hit hit;
  if (!boxbox::box_box_sat(obb1, axis_a_inv, obb2, hit)) {
    return false;
  }
  PersistentManifold m = PersistentManifold::make(a_obj, b_obj);
  boxbox::compute_contact_points(&obb1, &axis_a, &obb2, &axis_b, hit, m, a_obj,
                                 b_obj, callback);
  if (m.num_points == 0) {
    return false;
  }
  m.refresh_contact_points(a_obj, b_obj);
  out = m;
  return true;
}

}
