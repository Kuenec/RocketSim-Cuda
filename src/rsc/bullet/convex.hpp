#pragma once

#include <cstdint>

#include "rsc/math/shared.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

inline constexpr float k_convex_distance_margin = 0.04f;

struct HullView {
  const Vec3A *points;
  uint32_t count;

  RS_HDN Vec3A max_dot(Vec3A direction) const {
    float best = u32_to_f32(0xff800000u);
    Vec3A support_vertex = Vec3A::zero();
    uint32_t chunks = count / 4;
    for (uint32_t i = 0; i < chunks; ++i) {
      const Vec3A *p = points + i * 4;
      Vec4 px{p[0].x, p[1].x, p[2].x, p[3].x};
      Vec4 py{p[0].y, p[1].y, p[2].y, p[3].y};
      Vec4 pz{p[0].z, p[1].z, p[2].z, p[3].z};
      Vec4 dots = px * Vec4::splat(direction.x) +
                  py * Vec4::splat(direction.y) + pz * Vec4::splat(direction.z);
      float this_max = dots.max_element();
      if (this_max > best) {
        best = this_max;
        support_vertex = p[dots.max_position()];
      }
    }
    for (uint32_t i = chunks * 4; i < count; ++i) {
      float d = direction.dot(points[i]);
      if (d > best) {
        support_vertex = points[i];
        best = d;
      }
    }
    return support_vertex;
  }
};

enum class ConvexKind : uint8_t { Sphere, Box, Hull, Triangle };

struct ConvexShape {
  ConvexKind kind;
  float margin;
  Vec3A half_extents;
  HullView hull;
  Vec3A tri[3];

  RS_HDI static ConvexShape sphere(float radius) {
    ConvexShape s{};
    s.kind = ConvexKind::Sphere;
    s.margin = radius;
    return s;
  }

  RS_HDI static ConvexShape box(Vec3A half, float margin) {
    ConvexShape s{};
    s.kind = ConvexKind::Box;
    s.margin = margin;
    s.half_extents = half;
    return s;
  }

  RS_HDI static ConvexShape hull_shape(HullView hull) {
    ConvexShape s{};
    s.kind = ConvexKind::Hull;
    s.margin = k_convex_distance_margin;
    s.hull = hull;
    return s;
  }

  RS_HDI static ConvexShape triangle(Vec3A a, Vec3A b, Vec3A c) {
    ConvexShape s{};
    s.kind = ConvexKind::Triangle;
    s.margin = 0.0f;
    s.tri[0] = a;
    s.tri[1] = b;
    s.tri[2] = c;
    return s;
  }

  RS_HD Vec3A support_without_margin(Vec3A v) const {
    switch (kind) {
    case ConvexKind::Sphere:
      return Vec3A::zero();
    case ConvexKind::Box: {
      Vec3A sign = Vec3A::select(v.cmplt(Vec3A::zero()), Vec3A::splat(-1.0f),
                                 Vec3A::one());
      return half_extents * sign;
    }
    case ConvexKind::Hull:
      return hull.max_dot(v);
    case ConvexKind::Triangle: {
      float d0 = v.dot(tri[0]);
      float d1 = v.dot(tri[1]);
      float d2 = v.dot(tri[2]);
      Vec3A dots{d0, d1, d2};
      return tri[dots.max_position()];
    }
    }
    return Vec3A::zero();
  }

  RS_HD Vec3A support(Vec3A v) const {
    switch (kind) {
    case ConvexKind::Sphere: {
      Vec3A n = Vec3A::zero();
      v.try_normalize(n);
      return margin * n;
    }
    case ConvexKind::Box: {
      Vec3A sign = Vec3A::select(v.cmplt(Vec3A::zero()), Vec3A::splat(-1.0f),
                                 Vec3A::one());
      return (half_extents + margin) * sign;
    }
    case ConvexKind::Hull: {
      Vec3A sup = hull.max_dot(v);
      Vec3A vec_norm = v.normalize_or(Vec3A::splat(-1.0f));
      sup += margin * vec_norm;
      return sup;
    }
    case ConvexKind::Triangle:
      return support_without_margin(v);
    }
    return Vec3A::zero();
  }
};

}
