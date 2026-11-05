#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rsc/math/bvh.hpp"
#include "rsc/sim/arena_ops.hpp"

namespace rsc {

enum class MemWeightMode : uint8_t { Heavy, Balanced, Light };

struct CollisionMeshFile {
  std::vector<uint32_t> indices;
  std::vector<Vec3A> vertices;
  uint32_t hash;
};

struct ProcessedMesh {
  std::vector<TriangleShape> tris;
  std::vector<TriangleInfo> infos;
  BvhStorage bvh;
  Aabb aabb_ident_cache;
  Vec3A translation;
  uint32_t num_faces;
};

class MeshLibrary {
public:
  static MeshLibrary load_directory(const std::string &root);
  static MeshLibrary from_memory(
      const std::vector<std::pair<GameMode, std::vector<std::vector<uint8_t>>>>
          &files);

  bool has_mode(GameMode mesh_mode) const;
  const std::vector<ProcessedMesh> &meshes(GameMode mesh_mode) const;

private:
  std::vector<ProcessedMesh> soccar_;
  std::vector<ProcessedMesh> hoops_;
  std::vector<ProcessedMesh> dropshot_;
  bool has_soccar_ = false;
  bool has_hoops_ = false;
  bool has_dropshot_ = false;
};

struct CarSpec {
  Team team;
  CarBodyConfig config;
};

struct ArenaSetup {
  GameMode mode = GameMode::Soccar;
  MemWeightMode mem_weight_mode = MemWeightMode::Balanced;
  Vec3A min_pos{-5600.0f, -6000.0f, 0.0f};
  Vec3A max_pos{5600.0f, 6000.0f, 2200.0f};
  float max_aabb_len = 370.0f;
  bool no_ball_rot = false;
  bool has_custom_pads = false;
  std::vector<BoostPadConfig> custom_pads;
};

class HostStatic {
public:
  HostStatic(const MeshLibrary &library, const ArenaSetup &setup);
  HostStatic(const HostStatic &) = delete;
  HostStatic &operator=(const HostStatic &) = delete;

  const ArenaStatic &view() const { return view_; }
  const ArenaSetup &setup() const { return setup_; }
  const ArenaPhysics &template_physics() const { return *template_phys_; }

  std::vector<RigidBody> static_bodies;
  std::vector<BroadphaseProxy> static_proxies;
  std::vector<MeshView> mesh_views;
  std::vector<const BvhStorage *> mesh_bvhs;
  std::vector<Vec3A> hull_points;
  std::vector<uint32_t> cell_offsets;
  std::vector<uint16_t> cell_handles;
  std::vector<BoostPadConfig> pad_configs;
  std::vector<Aabb> pad_aabbs;
  BvhStorage pad_bvh;
  std::vector<float> sin_table;

private:
  void build_statics(const MeshLibrary &library);
  void build_grid();
  void build_pads();

  ArenaSetup setup_;
  ArenaStatic view_{};
  std::unique_ptr<ArenaPhysics> template_phys_;
};

MutatorConfig default_mutators(GameMode mode);
CarBodyConfig car_body_preset(int index);
float damping_factor_for(float linear_damping);

void create_arena(const HostStatic &st, const MutatorConfig &mutators,
                  uint64_t rng_seed, const std::vector<CarSpec> &cars,
                  Arena &out);

uint32_t arena_add_car(const HostStatic &st, Arena &a, const CarSpec &spec);

void arena_set_mutator_config(const HostStatic &st, Arena &a,
                              const MutatorConfig &mutators);

void arena_set_boost_pad_state(const HostStatic &st, Arena &a, uint32_t idx,
                               float cooldown,
                               const std::vector<uint32_t> &touching);

}
