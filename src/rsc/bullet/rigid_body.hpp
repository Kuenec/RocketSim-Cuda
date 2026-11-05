#pragma once

#include <cstdint>

#include "rsc/bullet/shapes.hpp"
#include "rsc/math/libm.hpp"
#include "rsc/math/vec.hpp"

namespace rsc {

inline constexpr float k_contact_breaking_threshold = 0.02f;

enum class ActivationState : uint8_t {
  Active,
  Sleeping,
  DisableSimulation,
  WantsDeactivation
};

enum CollisionFlag : uint8_t {
  k_flag_static_object = 1,
  k_flag_custom_material_callback = 1 << 1,
  k_flag_can_sleep = 1 << 2,
  k_flag_no_contact_response = 1 << 3,
  k_flag_no_world_gravity = 1 << 4,
  k_flag_no_angular_motion = 1 << 5,
  k_flag_identity_transform = 1 << 6,
};

enum class UserInfoType : uint8_t { None, Car, Ball, DropshotTile };

RS_HD void integrate_trans_no_rot(Vec3A &cur, Vec3A lin_vel, float time_step) {
  cur += lin_vel * time_step;
}

RS_HD Quat quat_from_axis_angle_simd(Vec3A axis, float angle) {
  float half = angle * 0.5f;
  float s = sinf_glibc(half);
  float c = cosf_glibc(half);
  Vec3A v = axis * s;
  return {v.x, v.y, v.z, c};
}

RS_HD Vec3A quat_mul_vec3a(Quat q, Vec3A rhs) {
  Vec3A b{q.x, q.y, q.z};
  float w = q.w;
  float b2 = b.dot(b);
  float rb = rhs.dot(b);
  Vec3A c = b.cross(rhs);
  float ww_b2 = w * w - b2;
  float rb2 = rb * 2.0f;
  float w2 = w * 2.0f;
  return {(rhs.x * ww_b2 + b.x * rb2) + c.x * w2,
          (rhs.y * ww_b2 + b.y * rb2) + c.y * w2,
          (rhs.z * ww_b2 + b.z * rb2) + c.z * w2};
}

RS_HD void integrate_trans(Affine3A &cur_trans, Quat &cur_rot, Vec3A lin_vel,
                           Vec3A ang_vel, float time_step) {
  integrate_trans_no_rot(cur_trans.translation, lin_vel, time_step);
  float angle = ang_vel.length();
  if (angle * time_step > k_frac_pi_4) {
    angle = k_frac_pi_4 / time_step;
  }
  float half_angle = angle * time_step * 0.5f;
  float scale;
  if (angle < 0.001f) {
    scale = 0.5f * time_step -
            time_step * time_step * time_step * 0.020833334f * angle * angle;
  } else {
    scale = sinf_glibc(half_angle) / angle;
  }
  Vec3A axis = ang_vel * scale;
  Quat dorn{axis.x, axis.y, axis.z, cosf_glibc(half_angle)};
  cur_rot = (dorn * cur_rot).normalize();
  cur_trans.matrix3 = mat3a_from_quat(cur_rot);
}

struct RigidBodyConstructionInfo {
  float mass;
  Affine3A start_world_trans;
  Shape collision_shape;
  Vec3A local_inertia;
  float linear_damping;
  float friction;
  float restitution;
  float linear_sleeping_threshold;
  float angular_sleeping_threshold;

  RS_HD static RigidBodyConstructionInfo make(float mass, const Shape &shape) {
    RigidBodyConstructionInfo info;
    info.mass = mass;
    info.start_world_trans = Affine3A::identity();
    info.collision_shape = shape;
    info.local_inertia = Vec3A::zero();
    info.linear_damping = 0.0f;
    info.friction = 0.5f;
    info.restitution = 0.0f;
    info.linear_sleeping_threshold = 0.0f;
    info.angular_sleeping_threshold = 1.0f;
    return info;
  }
};

struct RigidBody {
  Affine3A world_trans;
  Quat world_quat;
  Shape shape;
  ActivationState activation;
  uint32_t broadphase_handle;
  Affine3A interp_world_trans;
  float contact_processing_threshold;
  float contact_breaking_threshold;
  uint8_t collision_flags;
  int32_t companion_id;
  uint32_t world_array_idx;
  float deactivation_time;
  float friction;
  float restitution;
  uint32_t user_pointer;
  UserInfoType user_idx;
  Mat3A inv_inertia_tensor_world;
  Vec3A lin_vel;
  Vec3A ang_vel;
  float mass;
  float inv_mass;
  Vec3A inv_inertia_local;
  Vec3A accum_lin_vel;
  Vec3A accum_ang_vel;
  float linear_damping;
  float cached_linear_damping;
  float linear_sleeping_threshold;
  float angular_sleeping_threshold;
  Vec3A inv_mass_splat;

  RS_HD static Mat3A get_inertia_tensor(const Mat3A &world_mat,
                                        Vec3A inv_inertia_local) {
    Mat3A scaled = world_mat.transpose();
    scaled.x_axis *= inv_inertia_local;
    scaled.y_axis *= inv_inertia_local;
    scaled.z_axis *= inv_inertia_local;
    return world_mat * scaled;
  }

  RS_HD static Vec3A inv_inertia_from_local(Vec3A local_inertia) {
    return Vec3A::select(local_inertia.cmpeq(Vec3A::zero()), Vec3A::zero(),
                         1.0f / local_inertia);
  }

  RS_HD static RigidBody make(const RigidBodyConstructionInfo &info,
                              float damping_factor) {
    RigidBody rb;
    float inverse_mass;
    uint8_t flags;
    if (info.mass == 0.0f) {
      inverse_mass = 0.0f;
      flags = k_flag_static_object;
    } else {
      inverse_mass = 1.0f / info.mass;
      flags = 0;
    }
    if (info.start_world_trans.matrix3 == Mat3A::identity()) {
      if (info.mass == 0.0f) {
        flags |= k_flag_no_angular_motion;
      }
      if (info.start_world_trans.translation == Vec3A::zero()) {
        flags |= k_flag_identity_transform;
      }
    }
    rb.world_trans = info.start_world_trans;
    rb.world_quat = quat_from_mat3a(info.start_world_trans.matrix3);
    rb.shape = info.collision_shape;
    rb.activation = ActivationState::Active;
    rb.broadphase_handle = 0;
    rb.interp_world_trans = info.start_world_trans;
    rb.contact_processing_threshold = k_f32_max;
    rb.contact_breaking_threshold =
        info.collision_shape.get_contact_breaking_threshold(
            k_contact_breaking_threshold);
    rb.collision_flags = flags;
    rb.companion_id = -1;
    rb.world_array_idx = 0;
    rb.deactivation_time = 0.0f;
    rb.friction = info.friction;
    rb.restitution = info.restitution;
    rb.user_pointer = 0;
    rb.user_idx = UserInfoType::None;
    rb.inv_inertia_local = inv_inertia_from_local(info.local_inertia);
    rb.inv_inertia_tensor_world = get_inertia_tensor(
        info.start_world_trans.matrix3, rb.inv_inertia_local);
    rb.lin_vel = Vec3A::zero();
    rb.ang_vel = Vec3A::zero();
    rb.mass = info.mass;
    rb.inv_mass = inverse_mass;
    rb.accum_lin_vel = Vec3A::zero();
    rb.accum_ang_vel = Vec3A::zero();
    rb.linear_damping = clamp_f32(info.linear_damping, 0.0f, 1.0f);
    rb.cached_linear_damping = damping_factor;
    rb.linear_sleeping_threshold = info.linear_sleeping_threshold;
    rb.angular_sleeping_threshold = info.angular_sleeping_threshold;
    rb.inv_mass_splat = Vec3A::splat(inverse_mass);
    return rb;
  }

  RS_HD bool is_static_obj() const {
    return (collision_flags & k_flag_static_object) != 0;
  }

  RS_HD bool is_active() const {
    return !(activation == ActivationState::Sleeping ||
             activation == ActivationState::DisableSimulation);
  }

  RS_HD void set_activation_state(ActivationState s) {
    if (activation != ActivationState::DisableSimulation) {
      activation = s;
    }
  }

  RS_HD void force_activate() {
    activation = ActivationState::Active;
    deactivation_time = 0.0f;
  }

  RS_HD void update_transform_flags() {
    collision_flags =
        static_cast<uint8_t>(collision_flags & ~k_flag_identity_transform);
    if (world_trans.translation == Vec3A::zero() &&
        ((collision_flags & k_flag_no_angular_motion) != 0 ||
         world_trans.matrix3 == Mat3A::identity())) {
      collision_flags |= k_flag_identity_transform;
    }
  }

  RS_HD void set_world_trans(Affine3A t) {
    Quat q;
    if (collision_flags & k_flag_no_angular_motion) {
      t.matrix3 = Mat3A::identity();
      q = Quat::identity();
    } else {
      q = quat_from_mat3a(t.matrix3);
    }
    world_quat = q;
    world_trans = t;
    update_transform_flags();
  }

  RS_HD void set_world_trans_and_rot(Affine3A t, Quat q) {
    if (collision_flags & k_flag_no_angular_motion) {
      t.matrix3 = Mat3A::identity();
      q = Quat::identity();
    }
    world_trans = t;
    world_quat = q;
    update_transform_flags();
  }

  RS_HD void set_world_pos(Vec3A pos) {
    world_trans.translation = pos;
    update_transform_flags();
  }

  RS_HD void clear_accum_vels() {
    accum_lin_vel = Vec3A::zero();
    accum_ang_vel = Vec3A::zero();
  }

  RS_HD void update_inertia_tensor() {
    inv_inertia_tensor_world =
        get_inertia_tensor(world_trans.matrix3, inv_inertia_local);
  }

  RS_HD void apply_impulse_parts(Vec3A lin_impulse, Vec3A ang_impulse,
                                 bool accum) {
    if (accum) {
      accum_lin_vel += lin_impulse;
      accum_ang_vel += ang_impulse;
    } else {
      lin_vel += lin_impulse;
      ang_vel += ang_impulse;
    }
  }

  RS_HD void add_linear_impulse(Vec3A v, bool massed, bool accum) {
    float scaler = massed ? inv_mass : 1.0f;
    apply_impulse_parts(v * scaler, Vec3A::zero(), accum);
  }

  RS_HD void add_linear_rel_pos_impulse(Vec3A v, Vec3A rel_pos, bool massed,
                                        bool accum) {
    float scaler = massed ? inv_mass : 1.0f;
    Vec3A lin = v * scaler;
    Vec3A ang = inv_inertia_tensor_world * rel_pos.cross(v);
    if (!massed) {
      ang *= mass;
    }
    apply_impulse_parts(lin, ang, accum);
  }

  RS_HD void add_angular_impulse(Vec3A av, bool massed, bool accum) {
    float scaler = massed ? inv_mass : 1.0f;
    apply_impulse_parts(Vec3A::zero(), av * scaler, accum);
  }

  RS_HD void apply_damping() {
    if (linear_damping != 0.0f) {
      lin_vel *= cached_linear_damping;
    }
  }

  RS_HD Affine3A predict_integration_trans(float time_step) const {
    Affine3A trans = world_trans;
    Quat quat = world_quat;
    if (collision_flags & k_flag_no_angular_motion) {
      integrate_trans_no_rot(trans.translation, lin_vel, time_step);
    } else {
      integrate_trans(trans, quat, lin_vel, ang_vel, time_step);
    }
    return trans;
  }

  RS_HD void set_center_of_mass_trans(const Affine3A &xform) {
    set_world_trans(xform);
    interp_world_trans = world_trans;
    update_inertia_tensor();
  }

  RS_HD Vec3A get_vel_in_local_point(Vec3A rel_pos) const {
    return lin_vel + ang_vel.cross(rel_pos);
  }

  RS_HD void update_activation_state(float time_step) {
    if ((collision_flags & k_flag_can_sleep) == 0) {
      set_activation_state(ActivationState::Active);
      return;
    }
    if (activation == ActivationState::Sleeping) {
      return;
    }
    float lin_thresh_sq = linear_sleeping_threshold * linear_sleeping_threshold;
    float ang_thresh_sq =
        angular_sleeping_threshold * angular_sleeping_threshold;
    bool within = lin_vel.length_squared() < lin_thresh_sq &&
                  ang_vel.length_squared() < ang_thresh_sq;
    if (within) {
      deactivation_time += time_step;
      if (deactivation_time > 2.0f && activation == ActivationState::Active) {
        set_activation_state(ActivationState::WantsDeactivation);
      }
    } else {
      deactivation_time = 0.0f;
      set_activation_state(ActivationState::Active);
    }
  }

  RS_HD void sleep() {
    set_activation_state(ActivationState::Sleeping);
    lin_vel = Vec3A::zero();
    ang_vel = Vec3A::zero();
  }

  RS_HD float get_mass() const {
    return inv_mass == 0.0f ? 0.0f : 1.0f / inv_mass;
  }

  RS_HD float compute_impulse_denominator(Vec3A pos, Vec3A normal) const {
    Vec3A r0 = pos - world_trans.translation;
    Vec3A c0 = r0.cross(normal);
    Vec3A vec = inv_inertia_tensor_world.transpose_mul_vec3a(c0).cross(r0);
    return inv_mass + normal.dot(vec);
  }

  RS_HD Vec3A get_up_vector() const { return world_trans.matrix3.z_axis; }

  RS_HD Vec3A get_forward_vector() const { return world_trans.matrix3.x_axis; }

  RS_HD float get_forward_speed() const {
    return lin_vel.dot(get_forward_vector());
  }

  RS_HD void limit_vels(float max_lin_speed, float max_ang_speed) {
    if (lin_vel.length_squared() > max_lin_speed * max_lin_speed) {
      lin_vel = lin_vel.normalize_or_zero() * max_lin_speed;
    }
    if (ang_vel.length_squared() > max_ang_speed * max_ang_speed) {
      ang_vel = ang_vel.normalize_or_zero() * max_ang_speed;
    }
  }

  RS_HD bool has_contact_response() const {
    return (collision_flags & k_flag_no_contact_response) == 0;
  }
};

struct RigidBodyUpdateInfo {
  Shape shape;
  float mass;
  Vec3A local_inertia;
  float friction;
  float restitution;
  float linear_damping;
  float damping_factor;
  bool invalidate_contacts;
};

RS_HD void rigid_body_update_physical_properties(RigidBody &rb,
                                                 const RigidBodyUpdateInfo &p) {
  rb.shape = p.shape;
  rb.mass = p.mass;
  rb.inv_mass = p.mass == 0.0f ? 0.0f : 1.0f / p.mass;
  rb.inv_mass_splat = Vec3A::splat(rb.inv_mass);
  rb.inv_inertia_local = RigidBody::inv_inertia_from_local(p.local_inertia);
  rb.update_inertia_tensor();
  rb.contact_breaking_threshold =
      rb.shape.get_contact_breaking_threshold(k_contact_breaking_threshold);
  rb.friction = p.friction;
  rb.restitution = p.restitution;
  rb.linear_damping = clamp_f32(p.linear_damping, 0.0f, 1.0f);
  rb.cached_linear_damping = p.damping_factor;
  rb.clear_accum_vels();
  rb.force_activate();
}

}
