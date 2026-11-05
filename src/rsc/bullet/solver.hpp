#pragma once

#include <cstdint>

#include "rsc/bullet/manifold.hpp"
#include "rsc/bullet/rigid_body.hpp"
#include "rsc/bullet/world_types.hpp"

namespace rsc {

inline constexpr int k_solver_iterations = 10;
inline constexpr float k_solver_sor = 1.0f;
inline constexpr float k_wheel_pushback_erp = 0.1f;
inline constexpr float k_erp2 = 0.8f;
inline constexpr float k_split_impulse_penetration_threshold = 1e30f;
inline constexpr float k_split_impulse_turn_erp = 0.1f;
inline constexpr float k_warmstarting_factor = 0.85f;
inline constexpr int k_max_solver_bodies = k_max_dyn_bodies + 1;
inline constexpr int k_max_constraints = 192;

RS_HDI float bullet_dot(Vec3A a, Vec3A b) {
  Vec3A r = a * b;
  return r.x + (r.y + r.z);
}

struct SolverBody {
  Vec3A delta_lin_vel;
  Vec3A delta_ang_vel;
  Vec3A inv_mass;
  Vec3A push_vel;
  Vec3A turn_vel;
  Vec3A lin_vel;
  Vec3A ang_vel;
  Vec3A external_force_impulse;
  Vec3A external_torque_impulse;
  int32_t original_body;

  RS_HD static SolverBody fixed() {
    SolverBody b;
    b.delta_lin_vel = Vec3A::zero();
    b.delta_ang_vel = Vec3A::zero();
    b.inv_mass = Vec3A::zero();
    b.push_vel = Vec3A::zero();
    b.turn_vel = Vec3A::zero();
    b.lin_vel = Vec3A::zero();
    b.ang_vel = Vec3A::zero();
    b.external_force_impulse = Vec3A::zero();
    b.external_torque_impulse = Vec3A::zero();
    b.original_body = -1;
    return b;
  }

  RS_HD static SolverBody from(const RigidBody &rb) {
    SolverBody b = fixed();
    b.inv_mass = rb.inv_mass_splat;
    b.lin_vel = rb.lin_vel;
    b.ang_vel = rb.ang_vel;
    b.external_force_impulse = rb.accum_lin_vel;
    b.external_torque_impulse = rb.accum_ang_vel;
    b.original_body = static_cast<int32_t>(rb.world_array_idx);
    return b;
  }

  RS_HD void internal_apply_impulse(Vec3A linear, Vec3A angular, float mag) {
    delta_lin_vel += linear * mag;
    delta_ang_vel += angular * mag;
  }

  RS_HD Vec3A get_vel_in_local_point_no_delta(Vec3A rel_pos) const {
    return lin_vel + external_force_impulse +
           (ang_vel + external_torque_impulse).cross(rel_pos);
  }
};

RS_HD Vec3A plane_space_1(Vec3A n) {
  if (abs_f32(n.z) > k_frac_1_sqrt_2) {
    float a = n.y * n.y + n.z * n.z;
    float k = 1.0f / sqrt_f32(a);
    return {0.0f, -n.z * k, n.y * k};
  }
  float a = n.x * n.x + n.y * n.y;
  float k = 1.0f / sqrt_f32(a);
  return {-n.y * k, n.x * k, 0.0f};
}

RS_HD Vec3A calc_lat_friction_dir(const ManifoldPoint &cp, const SolverBody &a,
                                  const SolverBody &b, Vec3A rel_pos1,
                                  Vec3A rel_pos2) {
  Vec3A vel1 = a.get_vel_in_local_point_no_delta(rel_pos1);
  Vec3A vel2 = b.get_vel_in_local_point_no_delta(rel_pos2);
  Vec3A vel = vel1 - vel2;
  float rel_vel = cp.normal_world_on_b.dot(vel);
  Vec3A lat = vel - cp.normal_world_on_b * rel_vel;
  float lat_rel_vel = lat.length_squared();
  if (lat_rel_vel > k_f32_epsilon) {
    return lat * (1.0f / sqrt_f32(lat_rel_vel));
  }
  return plane_space_1(cp.normal_world_on_b);
}

struct SolverConstraint {
  Vec3A rel_pos1_cross_normal;
  Vec3A contact_normal_1;
  Vec3A rel_pos2_cross_normal;
  Vec3A contact_normal_2;
  Vec3A angular_component_a;
  Vec3A angular_component_b;
  float applied_push_impulse;
  float applied_impulse;
  float friction;
  float jac_diag_ab_inv;
  float rhs;
  float lower_limit;
  float upper_limit;
  float rhs_penetration;
  int32_t friction_idx;
  int32_t solver_body_id_a;
  int32_t solver_body_id_b;
  bool is_split_only;

  RS_HD static SolverConstraint blank(int a, int b) {
    SolverConstraint c;
    c.rel_pos1_cross_normal = Vec3A::zero();
    c.contact_normal_1 = Vec3A::zero();
    c.rel_pos2_cross_normal = Vec3A::zero();
    c.contact_normal_2 = Vec3A::zero();
    c.angular_component_a = Vec3A::zero();
    c.angular_component_b = Vec3A::zero();
    c.applied_push_impulse = 0.0f;
    c.applied_impulse = 0.0f;
    c.friction = 0.0f;
    c.jac_diag_ab_inv = 0.0f;
    c.rhs = 0.0f;
    c.lower_limit = 0.0f;
    c.upper_limit = 0.0f;
    c.rhs_penetration = 0.0f;
    c.friction_idx = 0;
    c.solver_body_id_a = a;
    c.solver_body_id_b = b;
    c.is_split_only = false;
    return c;
  }

  RS_HD void setup_split(const RigidBody *rb0, const RigidBody *rb1,
                         Vec3A rel_pos1, Vec3A rel_pos2,
                         const ManifoldPoint &cp, float time_step) {
    float denom0 = 0.0f;
    if (rb0) {
      Vec3A torque_axis = rel_pos1.cross(cp.normal_world_on_b);
      angular_component_a =
          rb0->inv_inertia_tensor_world.transpose_mul_vec3a(torque_axis);
      Vec3A vec = angular_component_a.cross(rel_pos1);
      denom0 = rb0->inv_mass + cp.normal_world_on_b.dot(vec);
      contact_normal_1 = cp.normal_world_on_b;
      rel_pos1_cross_normal = torque_axis;
    }
    float denom1 = 0.0f;
    if (rb1) {
      Vec3A torque_axis = rel_pos2.cross(cp.normal_world_on_b);
      angular_component_b =
          rb1->inv_inertia_tensor_world.transpose_mul_vec3a(-torque_axis);
      Vec3A vec = (-angular_component_b).cross(rel_pos2);
      denom1 = rb1->inv_mass + cp.normal_world_on_b.dot(vec);
      contact_normal_2 = -cp.normal_world_on_b;
      rel_pos2_cross_normal = -torque_axis;
    }
    jac_diag_ab_inv = k_solver_sor / (denom0 + denom1);
    float inv_time_step = 1.0f / time_step;
    float penetration = cp.distance_1;
    float positional_error =
        penetration > 0.0f ? 0.0f : -penetration * k_erp2 * inv_time_step;
    float penetration_impulse = positional_error * jac_diag_ab_inv;
    rhs_penetration = penetration > k_split_impulse_penetration_threshold
                          ? 0.0f
                          : penetration_impulse;
  }

  RS_HD void make_dynamic_side_a(bool dynamic_is_b) {
    if (!dynamic_is_b) {
      return;
    }
    int32_t t = solver_body_id_a;
    solver_body_id_a = solver_body_id_b;
    solver_body_id_b = t;
    Vec3A v = rel_pos1_cross_normal;
    rel_pos1_cross_normal = rel_pos2_cross_normal;
    rel_pos2_cross_normal = v;
    v = contact_normal_1;
    contact_normal_1 = contact_normal_2;
    contact_normal_2 = v;
    v = angular_component_a;
    angular_component_a = angular_component_b;
    angular_component_b = v;
  }

  RS_HD void setup_contact(SolverBody &sa, SolverBody &sb, const RigidBody *rb0,
                           const RigidBody *rb1, Vec3A rel_pos1, Vec3A rel_pos2,
                           const ManifoldPoint &cp, float time_step) {
    float inv_time_step = 1.0f / time_step;
    applied_impulse = cp.applied_impulse * k_warmstarting_factor;
    float denom0 = 0.0f;
    Vec3A vel0 = Vec3A::zero();
    float vel_1_dot_n = 0.0f;
    if (rb0) {
      Vec3A torque_axis = rel_pos1.cross(cp.normal_world_on_b);
      angular_component_a =
          rb0->inv_inertia_tensor_world.transpose_mul_vec3a(torque_axis);
      Vec3A vec = angular_component_a.cross(rel_pos1);
      denom0 = rb0->inv_mass + cp.normal_world_on_b.dot(vec);
      vel0 = rb0->get_vel_in_local_point(rel_pos1);
      contact_normal_1 = cp.normal_world_on_b;
      rel_pos1_cross_normal = torque_axis;
      sa.internal_apply_impulse(cp.normal_world_on_b * sa.inv_mass,
                                angular_component_a, applied_impulse);
      vel_1_dot_n =
          contact_normal_1.dot(sa.lin_vel + sa.external_force_impulse) +
          rel_pos1_cross_normal.dot(sa.ang_vel + sa.external_torque_impulse);
    }
    float denom1 = 0.0f;
    Vec3A vel1 = Vec3A::zero();
    float vel_2_dot_n = 0.0f;
    if (rb1) {
      Vec3A torque_axis = rel_pos2.cross(cp.normal_world_on_b);
      angular_component_b =
          rb1->inv_inertia_tensor_world.transpose_mul_vec3a(-torque_axis);
      Vec3A vec = (-angular_component_b).cross(rel_pos2);
      denom1 = rb1->inv_mass + cp.normal_world_on_b.dot(vec);
      vel1 = rb1->get_vel_in_local_point(rel_pos2);
      contact_normal_2 = -cp.normal_world_on_b;
      rel_pos2_cross_normal = -torque_axis;
      sb.internal_apply_impulse(cp.normal_world_on_b * sb.inv_mass,
                                -angular_component_b, -applied_impulse);
      vel_2_dot_n =
          contact_normal_2.dot(sb.lin_vel + sb.external_force_impulse) +
          rel_pos2_cross_normal.dot(sb.ang_vel + sb.external_torque_impulse);
    }
    jac_diag_ab_inv = k_solver_sor / (denom0 + denom1);
    Vec3A vel = vel0 - vel1;
    float rel_vel_a = cp.normal_world_on_b.dot(vel);
    float restitution = max_f32_c(cp.combined_restitution * -rel_vel_a, 0.0f);
    float penetration = cp.distance_1;
    float positional_error =
        penetration > 0.0f ? 0.0f : -penetration * k_erp2 * inv_time_step;
    float rel_vel = vel_1_dot_n + vel_2_dot_n;
    float vel_error = restitution - rel_vel;
    float penetration_impulse = positional_error * jac_diag_ab_inv;
    float vel_impulse = vel_error * jac_diag_ab_inv;
    if (penetration > k_split_impulse_penetration_threshold) {
      rhs = penetration_impulse + vel_impulse;
      rhs_penetration = 0.0f;
    } else {
      rhs = vel_impulse;
      rhs_penetration = penetration_impulse;
    }
  }

  RS_HD void setup_friction(const SolverBody &sa, const SolverBody &sb,
                            const RigidBody *rb0, const RigidBody *rb1,
                            Vec3A rel_pos1, Vec3A rel_pos2, Vec3A normal_axis) {
    float vel_1_dot_n = 0.0f;
    float denom1 = 0.0f;
    if (rb0) {
      contact_normal_1 = normal_axis;
      rel_pos1_cross_normal = rel_pos1.cross(contact_normal_1);
      angular_component_a = rb0->inv_inertia_tensor_world.transpose_mul_vec3a(
          rel_pos1_cross_normal);
      Vec3A vec = angular_component_a.cross(rel_pos1);
      denom1 = rb0->inv_mass + normal_axis.dot(vec);
      vel_1_dot_n =
          contact_normal_1.dot(sa.lin_vel + sa.external_force_impulse) +
          rel_pos1_cross_normal.dot(sa.ang_vel);
    }
    float vel_2_dot_n = 0.0f;
    float denom2 = 0.0f;
    if (rb1) {
      contact_normal_2 = -normal_axis;
      rel_pos2_cross_normal = rel_pos2.cross(contact_normal_2);
      angular_component_b = rb1->inv_inertia_tensor_world.transpose_mul_vec3a(
          rel_pos2_cross_normal);
      Vec3A vec = (-angular_component_b).cross(rel_pos2);
      denom2 = rb1->inv_mass + normal_axis.dot(vec);
      vel_2_dot_n =
          contact_normal_2.dot(sb.lin_vel + sb.external_force_impulse) +
          rel_pos2_cross_normal.dot(sb.ang_vel);
    }
    jac_diag_ab_inv = k_solver_sor / (denom1 + denom2);
    float rel_vel = vel_1_dot_n + vel_2_dot_n;
    float vel_error = -rel_vel;
    rhs = vel_error * jac_diag_ab_inv;
  }

  RS_HD float resolve_lower_limit_one_dynamic(SolverBody &body) {
    float delta_impulse = rhs;
    float delta_vel_dot_n =
        bullet_dot(contact_normal_1, body.delta_lin_vel) +
        bullet_dot(rel_pos1_cross_normal, body.delta_ang_vel);
    delta_impulse -= delta_vel_dot_n * jac_diag_ab_inv;
    float sum = applied_impulse + delta_impulse;
    if (sum < lower_limit) {
      delta_impulse = lower_limit - applied_impulse;
      applied_impulse = lower_limit;
    } else {
      applied_impulse = sum;
    }
    body.delta_lin_vel += contact_normal_1 * body.inv_mass * delta_impulse;
    body.delta_ang_vel += angular_component_a * delta_impulse;
    return delta_impulse / jac_diag_ab_inv;
  }

  RS_HD float resolve_generic_one_dynamic(SolverBody &body) {
    float delta_impulse = rhs;
    float delta_vel_dot_n =
        bullet_dot(contact_normal_1, body.delta_lin_vel) +
        bullet_dot(rel_pos1_cross_normal, body.delta_ang_vel);
    delta_impulse -= delta_vel_dot_n * jac_diag_ab_inv;
    float sum = applied_impulse + delta_impulse;
    if (sum < lower_limit) {
      delta_impulse = lower_limit - applied_impulse;
      applied_impulse = lower_limit;
    } else if (sum > upper_limit) {
      delta_impulse = upper_limit - applied_impulse;
      applied_impulse = upper_limit;
    } else {
      applied_impulse = sum;
    }
    body.delta_lin_vel += contact_normal_1 * body.inv_mass * delta_impulse;
    body.delta_ang_vel += angular_component_a * delta_impulse;
    return delta_impulse / jac_diag_ab_inv;
  }

  RS_HD float resolve_split_one_dynamic(SolverBody &body) {
    if (rhs_penetration == 0.0f) {
      return 0.0f;
    }
    float delta_impulse = rhs_penetration;
    float delta_vel_dot_n = bullet_dot(contact_normal_1, body.push_vel) +
                            bullet_dot(rel_pos1_cross_normal, body.turn_vel);
    delta_impulse -= delta_vel_dot_n * jac_diag_ab_inv;
    float sum = applied_push_impulse + delta_impulse;
    if (sum < lower_limit) {
      delta_impulse = lower_limit - applied_push_impulse;
      applied_push_impulse = lower_limit;
    } else {
      applied_push_impulse = sum;
    }
    body.push_vel += contact_normal_1 * body.inv_mass * delta_impulse;
    body.turn_vel += angular_component_a * delta_impulse;
    return delta_impulse / jac_diag_ab_inv;
  }

  RS_HD float resolve_two(SolverBody &ba, SolverBody &bb, bool upper) {
    float delta_impulse = rhs;
    float d1 = ba.original_body >= 0
                   ? bullet_dot(contact_normal_1, ba.delta_lin_vel) +
                         bullet_dot(rel_pos1_cross_normal, ba.delta_ang_vel)
                   : 0.0f;
    float d2 = bb.original_body >= 0
                   ? bullet_dot(contact_normal_2, bb.delta_lin_vel) +
                         bullet_dot(rel_pos2_cross_normal, bb.delta_ang_vel)
                   : 0.0f;
    delta_impulse -= d1 * jac_diag_ab_inv;
    delta_impulse -= d2 * jac_diag_ab_inv;
    float sum = applied_impulse + delta_impulse;
    if (sum < lower_limit) {
      delta_impulse = lower_limit - applied_impulse;
      applied_impulse = lower_limit;
    } else if (upper && sum > upper_limit) {
      delta_impulse = upper_limit - applied_impulse;
      applied_impulse = upper_limit;
    } else {
      applied_impulse = sum;
    }
    if (ba.original_body >= 0) {
      ba.delta_lin_vel += contact_normal_1 * ba.inv_mass * delta_impulse;
      ba.delta_ang_vel += angular_component_a * delta_impulse;
    }
    if (bb.original_body >= 0) {
      bb.delta_lin_vel += contact_normal_2 * bb.inv_mass * delta_impulse;
      bb.delta_ang_vel += angular_component_b * delta_impulse;
    }
    return delta_impulse / jac_diag_ab_inv;
  }

  RS_HD float resolve_split_two(SolverBody &ba, SolverBody &bb) {
    if (rhs_penetration == 0.0f) {
      return 0.0f;
    }
    float delta_impulse = rhs_penetration;
    float d1 = ba.original_body >= 0
                   ? bullet_dot(contact_normal_1, ba.push_vel) +
                         bullet_dot(rel_pos1_cross_normal, ba.turn_vel)
                   : 0.0f;
    float d2 = bb.original_body >= 0
                   ? bullet_dot(contact_normal_2, bb.push_vel) +
                         bullet_dot(rel_pos2_cross_normal, bb.turn_vel)
                   : 0.0f;
    delta_impulse -= d1 * jac_diag_ab_inv;
    delta_impulse -= d2 * jac_diag_ab_inv;
    float sum = applied_push_impulse + delta_impulse;
    if (sum < lower_limit) {
      delta_impulse = lower_limit - applied_push_impulse;
      applied_push_impulse = lower_limit;
    } else {
      applied_push_impulse = sum;
    }
    if (ba.original_body >= 0) {
      ba.push_vel += contact_normal_1 * ba.inv_mass * delta_impulse;
      ba.turn_vel += angular_component_a * delta_impulse;
    }
    if (bb.original_body >= 0) {
      bb.push_vel += contact_normal_2 * bb.inv_mass * delta_impulse;
      bb.turn_vel += angular_component_b * delta_impulse;
    }
    return delta_impulse / jac_diag_ab_inv;
  }
};

struct SpecialAccumulator {
  uint32_t obj_idx;
  Vec3A total_normal;
  float total_lever_len;
  float total_friction;
  float total_restitution;
  uint32_t count;
};

struct SolverScratch {
  int num_bodies;
  SolverBody bodies[k_max_solver_bodies];
  int num_contacts;
  SolverConstraint contacts[k_max_constraints];
  int num_frictions;
  SolverConstraint frictions[k_max_constraints];
  int num_specials;
  SpecialAccumulator specials[k_max_dyn_bodies];
  bool split_should_run[k_max_constraints];
  int fixed_body_id;
};

}
