#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#include "rsc/bullet/world_types.hpp"
#include "rsc/sim/quantize.hpp"

namespace {

using rsc::Mat3A;
using rsc::Quat;
using rsc::RigidBody;
using rsc::Vec3A;

constexpr uint32_t k_max_batch = rsc::k_max_dyn_bodies;

struct SplitMix {
  uint64_t state;

  uint64_t next() {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  uint32_t below(uint32_t n) { return static_cast<uint32_t>(next() % n); }
  float unit() {
    return static_cast<float>(next() >> 40) / static_cast<float>(1u << 24);
  }
  float range(float lo, float hi) { return lo + (hi - lo) * unit(); }
};

float unusual_value(SplitMix &rng) {
  switch (rng.below(8)) {
  case 0:
    return 0.0f;
  case 1:
    return -0.0f;
  case 2:
    return std::numeric_limits<float>::quiet_NaN();
  case 3:
    return std::numeric_limits<float>::infinity();
  case 4:
    return -std::numeric_limits<float>::infinity();
  case 5:
    return rng.range(-1e20f, 1e20f);
  case 6:
    return rng.range(-1e-30f, 1e-30f);
  default:
    return rsc::u32_to_f32(static_cast<uint32_t>(rng.next()));
  }
}

Quat random_quat(SplitMix &rng) {
  Quat q{rng.range(-1.0f, 1.0f), rng.range(-1.0f, 1.0f), rng.range(-1.0f, 1.0f),
         rng.range(-1.0f, 1.0f)};
  switch (rng.below(16)) {
  case 0:
    return {1.0f, 0.0f, 0.0f, 0.0f};
  case 1:
    return {0.0f, 0.0f, 0.70710677f, 0.70710677f};
  case 2:
    return {0.5f, -0.5f, 0.5f, -0.5f};
  case 3:
    q.x = 0.0f;
    q.y = 0.0f;
    break;
  case 4:
    q.w = q.x;
    q.z = -q.y;
    break;
  default:
    break;
  }
  return rsc::game_quat_normalize_scalar(q, 1e-8f);
}

Mat3A random_basis(SplitMix &rng) {
  Mat3A basis = rsc::game_set_rotation_scalar_columns(random_quat(rng));
  uint32_t kind = rng.below(16);
  if (kind == 0) {
    float scale = 1.0f + rng.range(-4e-4f, 4e-4f);
    basis = basis * scale;
  } else if (kind == 1) {
    basis.col(static_cast<int>(rng.below(3))) *=
        1.0f + rng.range(-1e-3f, 1e-3f);
  } else if (kind == 2) {
    Vec3A &column = basis.col(static_cast<int>(rng.below(3)));
    float value = unusual_value(rng);
    switch (rng.below(3)) {
    case 0:
      column.x = value;
      break;
    case 1:
      column.y = value;
      break;
    default:
      column.z = value;
      break;
    }
  } else if (kind == 3) {
    basis = Mat3A::from_cols(Vec3A::zero(), Vec3A::zero(), Vec3A::zero());
  } else if (kind == 4) {
    basis.x_axis = {unusual_value(rng), unusual_value(rng), unusual_value(rng)};
    basis.y_axis = {unusual_value(rng), unusual_value(rng), unusual_value(rng)};
    basis.z_axis = {unusual_value(rng), unusual_value(rng), unusual_value(rng)};
  }
  return basis;
}

Quat random_source(SplitMix &rng) {
  Quat q = random_quat(rng);
  uint32_t kind = rng.below(16);
  if (kind == 0) {
    float scale = rng.range(0.0f, 2.0f);
    return {q.x * scale, q.y * scale, q.z * scale, q.w * scale};
  }
  if (kind == 1) {
    q.y = unusual_value(rng);
  } else if (kind == 2) {
    return {0.0f, 0.0f, 0.0f, 0.0f};
  } else if (kind == 3) {
    return {unusual_value(rng), unusual_value(rng), unusual_value(rng),
            unusual_value(rng)};
  }
  return q;
}

Vec3A random_ang_vel(SplitMix &rng) {
  switch (rng.below(8)) {
  case 0:
    return Vec3A::zero();
  case 1:
    return {rng.range(-1e-4f, 1e-4f), rng.range(-1e-4f, 1e-4f),
            rng.range(-1e-4f, 1e-4f)};
  case 2:
    return {rng.range(-200.0f, 200.0f), rng.range(-200.0f, 200.0f),
            rng.range(-200.0f, 200.0f)};
  case 3:
    return {unusual_value(rng), rng.range(-6.0f, 6.0f), rng.range(-6.0f, 6.0f)};
  default:
    return {rng.range(-6.0f, 6.0f), rng.range(-6.0f, 6.0f),
            rng.range(-6.0f, 6.0f)};
  }
}

bool is_nan_bits(uint32_t bits) { return (bits & 0x7fffffffu) > 0x7f800000u; }

template <typename T> bool same_floats(const T &a, const T &b) {
  constexpr size_t count = sizeof(T) / sizeof(uint32_t);
  uint32_t a_bits[count];
  uint32_t b_bits[count];
  std::memcpy(a_bits, &a, sizeof(a_bits));
  std::memcpy(b_bits, &b, sizeof(b_bits));
  for (size_t i = 0; i < count; ++i) {
    if (a_bits[i] != b_bits[i] &&
        !(is_nan_bits(a_bits[i]) && is_nan_bits(b_bits[i]))) {
      return false;
    }
  }
  return true;
}

struct Tally {
  uint64_t tested = 0;
  uint64_t mismatched = 0;
};

bool report(const char *name, const Tally &tally) {
  bool pass = tally.mismatched == 0;
  std::printf("%-22s %-4s tested=%llu mismatched=%llu\n", name,
              pass ? "PASS" : "FAIL",
              static_cast<unsigned long long>(tally.tested),
              static_cast<unsigned long long>(tally.mismatched));
  return pass;
}

bool check_basis_rotations(uint64_t batches, uint64_t seed) {
  SplitMix rng{seed};
  Tally tally;
  for (uint64_t batch = 0; batch < batches; ++batch) {
    uint32_t n = 1 + rng.below(k_max_batch);
    Mat3A bases[k_max_batch];
    const Mat3A *pointers[k_max_batch];
    Quat batched[k_max_batch];
    for (uint32_t i = 0; i < n; ++i) {
      bases[i] = random_basis(rng);
      pointers[i] = &bases[i];
    }
    rsc::game_basis_rotations(pointers, n, batched);
    for (uint32_t i = 0; i < n; ++i) {
      tally.mismatched +=
          same_floats(batched[i], rsc::game_basis_rotation(bases[i])) ? 0 : 1;
    }
    tally.tested += n;
  }
  return report("basis_rotations", tally);
}

bool check_rotation_steps(uint64_t batches, uint64_t seed) {
  SplitMix rng{seed};
  Tally tally;
  for (uint64_t batch = 0; batch < batches; ++batch) {
    uint32_t n = 1 + rng.below(k_max_batch);
    Quat steps[k_max_batch];
    Quat sources[k_max_batch];
    Quat rotations[k_max_batch];
    Mat3A columns[k_max_batch];
    for (uint32_t i = 0; i < n; ++i) {
      steps[i] = rsc::game_rotation_step(random_ang_vel(rng), 1.0f / 120.0f);
      sources[i] = random_source(rng);
    }
    rsc::game_apply_rotation_steps(steps, sources, n, rotations, columns);
    for (uint32_t i = 0; i < n; ++i) {
      Quat rotation = rsc::game_apply_rotation_step(steps[i], sources[i]);
      bool same = same_floats(rotations[i], rotation) &&
                  same_floats(columns[i],
                              rsc::game_set_rotation_scalar_columns(rotation));
      tally.mismatched += same ? 0 : 1;
    }
    tally.tested += n;
  }
  return report("rotation_steps", tally);
}

RigidBody body_with_basis(const Mat3A &basis) {
  RigidBody body{};
  body.world_trans.matrix3 = basis;
  body.world_trans.translation = {1.0f, 2.0f, 3.0f};
  body.world_quat = Quat::identity();
  return body;
}

bool check_body_rotations(uint64_t batches, uint64_t seed) {
  SplitMix rng{seed};
  Tally tally;
  for (uint64_t batch = 0; batch < batches; ++batch) {
    uint32_t n = 1 + rng.below(k_max_batch);
    RigidBody batched[k_max_batch];
    RigidBody *pointers[k_max_batch];
    RigidBody single[k_max_batch];
    for (uint32_t i = 0; i < n; ++i) {
      batched[i] = body_with_basis(random_basis(rng));
      single[i] = batched[i];
      pointers[i] = &batched[i];
    }
    rsc::quantize_body_rotations(pointers, n);
    for (uint32_t i = 0; i < n; ++i) {
      rsc::quantize_body_rotation(
          single[i],
          rsc::quant::quantize_quat(rsc::body_rotation_from_basis(single[i])));
      bool same = same_floats(batched[i].world_trans, single[i].world_trans) &&
                  same_floats(batched[i].world_quat, single[i].world_quat) &&
                  batched[i].collision_flags == single[i].collision_flags;
      tally.mismatched += same ? 0 : 1;
    }
    tally.tested += n;
  }
  return report("body_rotations", tally);
}

}

int main(int argc, char **argv) {
  uint64_t batches = 1ULL << 22;
  if (argc > 1 && std::string(argv[1]) == "--quick") {
    batches = 1ULL << 18;
  }
  bool pass = true;
  pass &= check_basis_rotations(batches, 1);
  pass &= check_rotation_steps(batches, 2);
  pass &= check_body_rotations(batches, 3);
  std::printf("LANES %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
