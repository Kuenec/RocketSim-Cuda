#pragma once

#include "rsc/bullet/shapes.hpp"
#include "rsc/math/vec.hpp"
#ifdef RSC_BOX_TRIANGLE_GJK
#include "rsc/bullet/gjk.hpp"
#endif

namespace rsc {

struct CarToiBox {
  BoxShape shape;

  RS_HD static CarToiBox make(Vec3A half_extents_with_margin) {
    return {BoxShape::make(half_extents_with_margin)};
  }

  RS_HD float angular_motion_disc() const {
    float ex = shape.half_extents.x + shape.margin;
    float ey = shape.half_extents.y + shape.margin;
    float ez = shape.half_extents.z + shape.margin;
    float dx = ex + ex;
    float dy = ey + ey;
    float dz = ez + ez;
    return sqrt_f32((dx * dx + dy * dy) + dz * dz) * 0.5f;
  }
};

struct CarToiContact {
  float distance;
  Vec3A normal;
};

namespace car_toi {

inline constexpr float k_maximum_distance_squared = 1.0e18f;
inline constexpr float k_min_axis_len_sq = 1.0e-12f;
inline constexpr float k_normal_min_core_distance_sq = 1.0e-12f;
inline constexpr float k_min_segment_len_sq = 1.0e-30f;
inline constexpr float k_f32_min_positive = 1.17549435e-38f;

struct AxisSweep {
  float max_gap;
  Vec3A max_gap_axis;

  RS_HD void consider(Vec3A axis, float projection, float radius) {
    float gap = abs_f32(projection) - radius;
    if (gap > max_gap) {
      max_gap = gap;
      max_gap_axis = axis;
    }
  }
};

RS_HD bool unit_axis(Vec3A v, Vec3A &axis) {
  float len_sq = v.length_squared();
  if (len_sq < k_min_axis_len_sq) {
    return false;
  }
  axis = v * (1.0f / sqrt_f32(len_sq));
  return true;
}

RS_HD float projected_radius_a(Vec3A half, Vec3A axis) {
  return half.x * abs_f32(axis.x) + half.y * abs_f32(axis.y) +
         half.z * abs_f32(axis.z);
}

RS_HD float projected_radius_b(Vec3A half, const Mat3A &b_axes, Vec3A axis) {
  float x = half.x * abs_f32(b_axes.x_axis.dot(axis));
  float y = half.y * abs_f32(b_axes.y_axis.dot(axis));
  float z = half.z * abs_f32(b_axes.z_axis.dot(axis));
  return (x + y) + z;
}

RS_HD Vec3A normal_toward_a(Vec3A axis, Vec3A center_rel) {
  return axis.dot(center_rel) > 0.0f ? -axis : axis;
}

RS_HD AxisSweep sat_sweep(Vec3A half_a, Vec3A half_b, const Mat3A &rotation,
                          Vec3A center) {
  AxisSweep sweep{u32_to_f32(0xff800000u), {1.0f, 0.0f, 0.0f}};
  const Mat3A a_axes = Mat3A::identity();
  for (int i = 0; i < 3; ++i) {
    Vec3A axis = a_axes.col(i);
    sweep.consider(axis, center[i],
                   half_a[i] + projected_radius_b(half_b, rotation, axis));
  }
  for (int i = 0; i < 3; ++i) {
    Vec3A axis;
    if (!unit_axis(rotation.col(i), axis)) {
      continue;
    }
    sweep.consider(axis, center.dot(axis),
                   projected_radius_a(half_a, axis) +
                       projected_radius_b(half_b, rotation, axis));
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Vec3A axis;
      if (!unit_axis(a_axes.col(i).cross(rotation.col(j)), axis)) {
        continue;
      }
      sweep.consider(axis, center.dot(axis),
                     projected_radius_a(half_a, axis) +
                         projected_radius_b(half_b, rotation, axis));
    }
  }
  return sweep;
}

struct Nearest {
  float distance_sq;
  Vec3A point_a;
  Vec3A point_b;

  RS_HD void consider(Vec3A a, Vec3A b) {
    float d = (a - b).length_squared();
    if (d < distance_sq) {
      distance_sq = d;
      point_a = a;
      point_b = b;
    }
  }
};

RS_HD Vec3A box_vertex(Vec3A half, int bits) {
  return {(bits & 1) == 0 ? -half.x : half.x,
          (bits & 2) == 0 ? -half.y : half.y,
          (bits & 4) == 0 ? -half.z : half.z};
}

RS_HD Vec3A clamp_to_box(Vec3A point, Vec3A half) {
  return {clamp_f32(point.x, -half.x, half.x),
          clamp_f32(point.y, -half.y, half.y),
          clamp_f32(point.z, -half.z, half.z)};
}

RS_HD void box_edge(Vec3A half, const Mat3A &rotation, Vec3A center, int index,
                    Vec3A &start_world, Vec3A &end_world) {
  int axis = index / 4;
  int bits = index % 4;
  int next_axis = (axis + 1) % 3;
  int other_axis = (axis + 2) % 3;
  Vec3A start = Vec3A::zero();
  start[next_axis] = (bits & 1) == 0 ? -half[next_axis] : half[next_axis];
  start[other_axis] = (bits & 2) == 0 ? -half[other_axis] : half[other_axis];
  Vec3A end = start;
  start[axis] = -half[axis];
  end[axis] = half[axis];
  start_world = center + rotation * start;
  end_world = center + rotation * end;
}

RS_HD void segment_closest(Vec3A p1, Vec3A q1, Vec3A p2, Vec3A q2,
                           Vec3A &point_a, Vec3A &point_b) {
  Vec3A d1 = q1 - p1;
  Vec3A d2 = q2 - p2;
  Vec3A offset = p1 - p2;
  float len1_sq = d1.length_squared();
  float len2_sq = d2.length_squared();
  float len2_offset = d2.dot(offset);
  if (len1_sq <= k_min_segment_len_sq && len2_sq <= k_min_segment_len_sq) {
    point_a = p1;
    point_b = p2;
    return;
  }
  float s;
  float t;
  if (len1_sq <= k_min_segment_len_sq) {
    s = 0.0f;
    t = clamp_f32(len2_offset / len2_sq, 0.0f, 1.0f);
  } else {
    float len1_offset = d1.dot(offset);
    if (len2_sq <= k_min_segment_len_sq) {
      t = 0.0f;
      s = clamp_f32(-len1_offset / len1_sq, 0.0f, 1.0f);
    } else {
      float cross = d1.dot(d2);
      float denom = len1_sq * len2_sq - cross * cross;
      s = denom > k_min_segment_len_sq
              ? clamp_f32((cross * len2_offset - len1_offset * len2_sq) / denom,
                          0.0f, 1.0f)
              : 0.0f;
      t = (cross * s + len2_offset) / len2_sq;
      if (t < 0.0f) {
        t = 0.0f;
        s = clamp_f32(-len1_offset / len1_sq, 0.0f, 1.0f);
      } else if (t > 1.0f) {
        t = 1.0f;
        s = clamp_f32((cross - len1_offset) / len1_sq, 0.0f, 1.0f);
      }
    }
  }
  point_a = p1 + d1 * s;
  point_b = p2 + d2 * t;
}

RS_HD Nearest nearest_features(Vec3A half_a, Vec3A half_b,
                               const Mat3A &rotation, Vec3A center) {
  Nearest best{k_f32_max, Vec3A::zero(), Vec3A::zero()};
  Mat3A inverse_rotation = rotation.transpose();
  for (int bits = 0; bits < 8; ++bits) {
    Vec3A point_a = box_vertex(half_a, bits);
    Vec3A local_b = inverse_rotation * (point_a - center);
    Vec3A point_b = rotation * clamp_to_box(local_b, half_b) + center;
    best.consider(point_a, point_b);
  }
  for (int bits = 0; bits < 8; ++bits) {
    Vec3A point_b = rotation * box_vertex(half_b, bits) + center;
    Vec3A point_a = clamp_to_box(point_b, half_a);
    best.consider(point_a, point_b);
  }
  const Mat3A identity = Mat3A::identity();
  for (int ea = 0; ea < 12; ++ea) {
    Vec3A a0, a1;
    box_edge(half_a, identity, Vec3A::zero(), ea, a0, a1);
    for (int eb = 0; eb < 12; ++eb) {
      Vec3A b0, b1;
      box_edge(half_b, rotation, center, eb, b0, b1);
      Vec3A point_a, point_b;
      segment_closest(a0, a1, b0, b1, point_a, point_b);
      best.consider(point_a, point_b);
    }
  }
  return best;
}

}

#ifdef RSC_BOX_TRIANGLE_GJK

struct CarToiGjkContact {
  bool found;
  CarToiContact contact;

  RS_HD void add_contact_point(Vec3A normal_on_b, Vec3A, float depth) {
    found = true;
    contact = {depth, normal_on_b};
  }
};

RS_HD Vec3A car_toi_bounds_center(const CarToiBox &box, const Affine3A &trans) {
  return CompoundShape::make(box.shape, Affine3A::identity())
      .get_aabb(trans)
      .center();
}

RS_HD bool box_sweep_closest_points(const CarToiBox &shape_a,
                                    const Affine3A &trans_a,
                                    const CarToiBox &shape_b,
                                    const Affine3A &trans_b,
                                    CarToiContact &contact) {
  GjkPairDetector detector;
  detector.init(shape_a.shape.margin, shape_b.shape.margin);
  CarToiGjkContact result{false, {}};
  Epa2 epa;
  detector.get_closest_points(
      trans_a, trans_b, car_toi::k_maximum_distance_squared,
      shape_a.shape.convex(), shape_b.shape.convex(), epa, result);
  if (!result.found) {
    return false;
  }
  Vec3A position_offset = (trans_a.translation + trans_b.translation) * 0.5f;
  Affine3A local_a = trans_a;
  Affine3A local_b = trans_b;
  local_a.translation -= position_offset;
  local_b.translation -= position_offset;
  Vec3A between = car_toi_bounds_center(shape_a, local_a) -
                  car_toi_bounds_center(shape_b, local_b);
  contact = result.contact;
  if ((between.x * contact.normal.x + between.y * contact.normal.y) +
          between.z * contact.normal.z <
      0.0f) {
    contact.normal = -contact.normal;
  }
  return true;
}

#else

RS_HD bool box_sweep_closest_points(const CarToiBox &shape_a,
                                    const Affine3A &trans_a,
                                    const CarToiBox &shape_b,
                                    const Affine3A &trans_b,
                                    CarToiContact &contact) {
  using namespace car_toi;
  const Mat3A identity = Mat3A::identity();
  Vec3A core_center_a = trans_a.transform_point3a(Vec3A::zero());
  Vec3A core_center_b = trans_b.transform_point3a(Vec3A::zero());
  Mat3A rotation_a = trans_a.matrix3 * identity;
  Mat3A rotation_b = trans_b.matrix3 * identity;
  Vec3A half_a = shape_a.shape.half_extents;
  Vec3A half_b = shape_b.shape.half_extents;
  Vec3A offset = (trans_a.translation + trans_b.translation) * 0.5f;
  Vec3A center_a = core_center_a - offset;
  Vec3A center_b = core_center_b - offset;
  Mat3A rotation = rotation_a.transpose() * rotation_b;
  Vec3A center_rel = rotation_a.transpose() * (center_b - center_a);
  AxisSweep sweep = sat_sweep(half_a, half_b, rotation, center_rel);
  float margin_sum = shape_a.shape.margin + shape_b.shape.margin;
  float distance;
  Vec3A normal_local;
  if (sweep.max_gap > 0.0f) {
    Nearest nearest = nearest_features(half_a, half_b, rotation, center_rel);
    Vec3A witness_delta = nearest.point_a - nearest.point_b;
    float witness_len_sq = witness_delta.length_squared();
    normal_local = witness_len_sq > k_normal_min_core_distance_sq
                       ? witness_delta * (1.0f / sqrt_f32(witness_len_sq))
                       : normal_toward_a(sweep.max_gap_axis, center_rel);
    distance = sqrt_f32(nearest.distance_sq) - margin_sum;
  } else {
    distance = -(-sweep.max_gap + margin_sum);
    normal_local = normal_toward_a(sweep.max_gap_axis, center_rel);
  }
  Vec3A normal_world = rotation_a * normal_local;
  float normal_len_sq = normal_world.length_squared();
  if (normal_len_sq <= k_f32_min_positive) {
    return false;
  }
  if (!(distance < 0.0f || distance * distance < k_maximum_distance_squared)) {
    return false;
  }
  contact.distance = distance;
  contact.normal = normal_world * (1.0f / sqrt_f32(normal_len_sq));
  return true;
}

#endif

}
