#pragma once

#include <cstdint>

#include "rsc/math/libm.hpp"
#include "rsc/sim/car.hpp"
#include "rsc/sim/consts.hpp"

namespace rsc {

namespace bump {

RS_HD Vec3A normalize_unless_one(Vec3A v) {
  float len_sq = v.length_squared();
  if (len_sq != 1.0f) {
    if (len_sq < 1e-8f) {
      return Vec3A::zero();
    }
    return v * (1.0f / sqrt_f32(len_sq));
  }
  return v;
}

RS_HD Vec3A hitbox_center(const CarState &s, const CarBodyConfig &c) {
  return s.phys.pos + s.phys.rot_mat.transpose_mul_vec3a(c.hitbox_pos_offset);
}

RS_HD float cone_angle_deg(Vec3A delta, Vec3A fwd, Vec3A axis) {
  float k = delta.dot(axis);
  k = k >= 0.0f ? k / 1.01f : k * 1.01f;
  Vec3A projected = normalize_unless_one(delta - axis * k);
  return acosf_glibc(clamp_f32(fwd.dot(projected), -1.0f, 1.0f)) * 57.29578f;
}

RS_HD bool is_within_forward_cone(const CarState &attacker,
                                  const CarBodyConfig &attacker_config,
                                  const CarState &victim,
                                  const CarBodyConfig &victim_config,
                                  bool reverse, float max_yaw_deg,
                                  float max_pitch_deg) {
  Vec3A fwd = attacker.phys.rot_mat.x_axis;
  Vec3A delta = normalize_unless_one(hitbox_center(victim, victim_config) -
                                     hitbox_center(attacker, attacker_config));
  if (reverse) {
    fwd = -fwd;
  }
  if (cone_angle_deg(delta, fwd, attacker.phys.rot_mat.y_axis) >
      max_pitch_deg) {
    return false;
  }
  return cone_angle_deg(delta, fwd, attacker.phys.rot_mat.z_axis) <=
         max_yaw_deg;
}

RS_HD Vec3A rotate_about(Vec3A x, Vec3A axis, float c, float s) {
  return x * c + axis.cross(x) * s + axis * (axis.dot(x) * (1.0f - c));
}

RS_HD void integrate_pose(CarState &state, Vec3A vel, float t) {
  state.phys.pos += vel * t;
  Vec3A ang_vel = state.phys.ang_vel;
  float ang_speed = ang_vel.length();
  float theta = ang_speed * t;
  if (ang_speed > 0.0f && theta != 0.0f) {
    Vec3A axis = ang_vel / ang_speed;
    float c = cosf_glibc(theta);
    float s = sinf_glibc(theta);
    Mat3A r = state.phys.rot_mat;
    state.phys.rot_mat = Mat3A::from_cols(rotate_about(r.x_axis, axis, c, s),
                                          rotate_about(r.y_axis, axis, c, s),
                                          rotate_about(r.z_axis, axis, c, s));
  }
}

RS_HD float hitbox_separation(const CarState &a, const CarBodyConfig &ac,
                              const CarState &b, const CarBodyConfig &bc) {
  Mat3A a_rot = a.phys.rot_mat;
  Mat3A b_rot = b.phys.rot_mat;
  Vec3A a_half = ac.hitbox_size * 0.5f;
  Vec3A b_half = bc.hitbox_size * 0.5f;
  Vec3A a_center = a.phys.pos + a_rot * ac.hitbox_pos_offset;
  Vec3A b_center = b.phys.pos + b_rot * bc.hitbox_pos_offset;
  Vec3A delta = b_center - a_center;
  float best = -1e30f;
  auto test = [&](Vec3A axis) {
    Vec3A a_reach = a_rot.transpose_mul_vec3a(axis).abs();
    Vec3A b_reach = b_rot.transpose_mul_vec3a(axis).abs();
    float depth = abs_f32(delta.dot(axis));
    for (int i = 0; i < 3; ++i) {
      depth -= a_reach[i] * a_half[i] + b_reach[i] * b_half[i];
    }
    best = max_f32(best, depth);
  };
  Vec3A a_cols[3] = {a_rot.x_axis, a_rot.y_axis, a_rot.z_axis};
  Vec3A b_cols[3] = {b_rot.x_axis, b_rot.y_axis, b_rot.z_axis};
  for (int i = 0; i < 3; ++i) {
    test(a_cols[i]);
  }
  for (int i = 0; i < 3; ++i) {
    test(b_cols[i]);
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Vec3A axis = a_cols[i].cross(b_cols[j]);
      float len = axis.length();
      if (len > 1e-6f) {
        test(axis / len);
      }
    }
  }
  return best;
}

RS_HD void get_time_of_impact_poses(CarState &a, const CarBodyConfig &ac,
                                    Vec3A a_vel, CarState &b,
                                    const CarBodyConfig &bc, Vec3A b_vel,
                                    float tick_time) {
  constexpr float penetration = 0.039f * 50.0f;
  auto separation = [&](float lambda) {
    float t = (2.0f * lambda - 1.0f) * tick_time;
    CarState ai = a;
    CarState bi = b;
    integrate_pose(ai, a_vel, t);
    integrate_pose(bi, b_vel, t);
    return hitbox_separation(ai, ac, bi, bc) + penetration;
  };
  float lambda;
  if (separation(0.0f) <= 0.0f) {
    lambda = 0.0f;
  } else if (separation(1.0f) > 0.0f) {
    return;
  } else {
    float lo = 0.0f;
    float hi = 1.0f;
    for (int i = 0; i < 24; ++i) {
      float mid = (lo + hi) * 0.5f;
      if (separation(mid) > 0.0f) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    lambda = hi;
  }
  float t = (2.0f * lambda - 1.0f) * tick_time;
  integrate_pose(a, a_vel, t);
  integrate_pose(b, b_vel, t);
}

RS_HD bool should_demolish(const CarState &attacker, const CarBodyConfig &ac,
                           const CarState &victim, const CarBodyConfig &vc,
                           float tick_time) {
  using namespace consts::car;
  if (!attacker.is_supersonic) {
    return false;
  }
  Vec3A start_vel = attacker.phys.vel;
  float forward_speed = start_vel.dot(attacker.phys.rot_mat.x_axis);
  if (abs_f32(forward_speed) < supersonic::maintain_min_speed) {
    return false;
  }
  CarState ap = attacker;
  CarState vp = victim;
  get_time_of_impact_poses(ap, ac, start_vel, vp, vc, victim.phys.vel,
                           tick_time);
  return is_within_forward_cone(ap, ac, vp, vc, forward_speed < 0.0f,
                                demo::yaw_limit_deg, demo::pitch_limit_deg);
}

}

}
