#include "rsc/host/builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <map>
#include <stdexcept>
#include <sys/stat.h>

namespace rsc {

namespace {

uint32_t read_u32(const std::vector<uint8_t> &bytes, size_t &pos) {
  if (pos + 4 > bytes.size()) {
    throw std::runtime_error("collision mesh file is truncated");
  }
  uint32_t v = static_cast<uint32_t>(bytes[pos]) |
               (static_cast<uint32_t>(bytes[pos + 1]) << 8) |
               (static_cast<uint32_t>(bytes[pos + 2]) << 16) |
               (static_cast<uint32_t>(bytes[pos + 3]) << 24);
  pos += 4;
  return v;
}

CollisionMeshFile parse_cmf(const std::vector<uint8_t> &bytes) {
  size_t pos = 0;
  uint32_t num_tris = read_u32(bytes, pos);
  uint32_t num_vertices = read_u32(bytes, pos);
  if (std::min(num_tris, num_vertices) == 0 ||
      std::max(num_tris, num_vertices) > 1000000) {
    throw std::runtime_error("invalid collision mesh file: bad counts " +
                             std::to_string(num_tris) + "/" +
                             std::to_string(num_vertices));
  }
  CollisionMeshFile f;
  f.indices.resize(static_cast<size_t>(num_tris) * 3);
  for (auto &i : f.indices) {
    i = read_u32(bytes, pos);
    if (i >= num_vertices) {
      throw std::runtime_error("invalid collision mesh file: bad index");
    }
  }
  f.vertices.resize(num_vertices);
  for (auto &v : f.vertices) {
    v.x = u32_to_f32(read_u32(bytes, pos));
    v.y = u32_to_f32(read_u32(bytes, pos));
    v.z = u32_to_f32(read_u32(bytes, pos));
  }
  uint64_t num_indices = f.indices.size();
  uint64_t nv = f.vertices.size();
  uint32_t hash = static_cast<uint32_t>(nv + (num_indices / 3 * nv));
  for (uint32_t idx : f.indices) {
    const Vec3A &v = f.vertices[idx];
    const float comps[3] = {v.x, v.y, v.z};
    for (float c : comps) {
      uint32_t cur = static_cast<uint32_t>(sat_f32_to_i32(c));
      cur = ((cur >> 16) ^ cur) * 0x45d9f3bu;
      cur = ((cur >> 16) ^ cur) * 0x45d9f3bu;
      cur = (cur >> 16) ^ cur;
      hash ^= cur + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    }
  }
  f.hash = hash;
  return f;
}

const std::vector<uint32_t> &mode_hashes(GameMode mode) {
  static const std::vector<uint32_t> soccar = {
      0xA160BAF9, 0x2811EEE8, 0xB81AC8B9, 0x760358D3, 0x73AE4940, 0x918F4A4E,
      0x1F8EE550, 0x255BA8C1, 0x14B84668, 0xEC759EBF, 0x94FB0D5C, 0xDEA07102,
      0xBD4FBEA8, 0x39A47F63, 0x3D79D25D, 0xD84C7A68};
  static const std::vector<uint32_t> hoops = {
      0x72F2359E, 0x5ED14A26, 0xFD5A0D07, 0x92AFA5B5, 0x0E4133C7, 0x399E8B5F,
      0xBB9D4FB5, 0x8C87FB93, 0x1CFD0E16, 0xE19E1DF6, 0x9CA179DC, 0x16F3CC19};
  static const std::vector<uint32_t> dropshot = {0x7EB0B2D3, 0x911041D2};
  static const std::vector<uint32_t> none;
  switch (mode) {
  case GameMode::Soccar:
    return soccar;
  case GameMode::Hoops:
    return hoops;
  case GameMode::Dropshot:
    return dropshot;
  default:
    return none;
  }
}

int soccar_component_order(uint32_t hash) {
  static const uint32_t order[16] = {
      0xA160BAF9, 0x2811EEE8, 0x14B84668, 0xEC759EBF, 0x94FB0D5C, 0xDEA07102,
      0xBD4FBEA8, 0x39A47F63, 0x3D79D25D, 0xD84C7A68, 0xB81AC8B9, 0x760358D3,
      0x73AE4940, 0x918F4A4E, 0x1F8EE550, 0x255BA8C1};
  for (int i = 0; i < 16; ++i) {
    if (order[i] == hash) {
      return i;
    }
  }
  throw std::runtime_error("validated Soccar mesh hash has no native order");
}

Vec3A component_translation(const CollisionMeshFile &f) {
  constexpr float goal_translation = 102.4f;
  Vec3A min = Vec3A::splat(k_f32_max);
  Vec3A max = Vec3A::splat(k_f32_min);
  for (const Vec3A &v : f.vertices) {
    min = min.min(v);
    max = max.max(v);
  }
  float center_y = (min.y + max.y) * 0.5f;
  if (center_y > goal_translation) {
    return {0.0f, goal_translation, 0.0f};
  }
  if (center_y < -goal_translation) {
    return {0.0f, -goal_translation, 0.0f};
  }
  return Vec3A::zero();
}

struct BvhQuantization {
  Vec3A min, max, scale;

  void update_scale() { scale = Vec3A::splat(65533.0f) / (max - min); }

  void quantize_point(Vec3A point, bool is_max, uint16_t *out) const {
    Vec3A value = (point - min) * scale;
    for (int axis = 0; axis < 3; ++axis) {
      if (is_max) {
        out[axis] = static_cast<uint16_t>(
            static_cast<uint16_t>(sat_f32_to_i32(value[axis] + 1.0f)) | 1u);
      } else {
        out[axis] = static_cast<uint16_t>(
            static_cast<uint16_t>(sat_f32_to_i32(value[axis])) & 0xfffeu);
      }
    }
  }

  Vec3A unquantize_point(const uint16_t *p) const {
    Vec3A v{static_cast<float>(p[0]), static_cast<float>(p[1]),
            static_cast<float>(p[2])};
    return v / scale + min;
  }

  static BvhQuantization make(Aabb aabb) {
    Vec3A margin = Vec3A::splat(1.0f);
    BvhQuantization q;
    q.min = aabb.min - margin;
    q.max = aabb.max + margin;
    q.update_scale();
    uint16_t qmin[3];
    q.quantize_point(q.min, false, qmin);
    q.min = q.min.min(q.unquantize_point(qmin) - margin);
    uint16_t qmax[3];
    q.quantize_point(q.max, true, qmax);
    q.max = q.max.max(q.unquantize_point(qmax) + margin);
    q.update_scale();
    return q;
  }

  Aabb quantize_aabb(Aabb aabb) const {
    uint16_t a[3];
    uint16_t b[3];
    quantize_point(aabb.min, false, a);
    quantize_point(aabb.max, true, b);
    return {unquantize_point(a), unquantize_point(b)};
  }
};

Aabb update_triangle_aabb(const TriangleShape &tri) {
  constexpr float min_dim = 0.002f;
  constexpr float min_half = min_dim / 2.0f;
  Aabb aabb = tri.aabb();
  BVec3 diff = (aabb.max - aabb.min).cmplt(Vec3A::splat(min_dim));
  const bool d[3] = {diff.x, diff.y, diff.z};
  for (int axis = 0; axis < 3; ++axis) {
    if (d[axis]) {
      aabb.max[axis] += min_half;
      aabb.min[axis] -= min_half;
    }
  }
  return aabb;
}

Vec3A mesh_support_vertex(const std::vector<TriangleShape> &tris, Vec3A vec) {
  Vec3A support = Vec3A::zero();
  float best = k_f32_min;
  for (const TriangleShape &t : tris) {
    Vec3A dots{vec.dot(t.points[0]), vec.dot(t.points[1]),
               vec.dot(t.points[2])};
    float m = dots.max_element();
    if (m > best) {
      best = m;
      support = t.points[dots.max_position()];
    }
  }
  return support;
}

struct EdgeInfoProcessor {
  uint32_t idx;
  const TriangleShape *shape;
  const std::vector<TriangleShape> *tris;
  std::vector<TriangleInfo> *infos;

  void process_node(uint32_t tri_idx) {
    using namespace edge;
    const TriangleShape &tri = (*tris)[tri_idx];
    if (idx == tri_idx || tri.normal_length() < k_equal_vertex_threshold ||
        shape->normal_length() < k_equal_vertex_threshold) {
      return;
    }
    int num_shared = 0;
    int shared_a[2] = {0, 0};
    int shared_b[2] = {0, 0};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        if (shape->points[i].distance_squared(tri.points[j]) <
            k_equal_vertex_threshold) {
          if (num_shared >= 2) {
            throw std::runtime_error("degenerate triangle in collision mesh");
          }
          shared_a[num_shared] = i;
          shared_b[num_shared] = j;
          ++num_shared;
        }
      }
    }
    if (num_shared != 2) {
      return;
    }
    if (shared_a[0] == 0 && shared_a[1] == 2) {
      shared_a[0] = 2;
      shared_a[1] = 0;
      std::swap(shared_b[0], shared_b[1]);
    }
    TriangleInfo &info = (*infos)[idx];
    int sum_a = shared_a[0] + shared_a[1];
    int other_a = 3 - sum_a;
    Vec3A edge_vec =
        (shape->points[shared_a[1]] - shape->points[shared_a[0]]).normalize();
    int other_b = 3 - (shared_b[0] + shared_b[1]);
    Vec3A normal_b =
        TriangleShape::make(tri.points[shared_b[1]], tri.points[shared_b[0]],
                            tri.points[other_b])
            .normal;
    Vec3A edge_cross_a = edge_vec.cross(shape->normal).normalize();
    if (edge_cross_a.dot(shape->points[other_a] - shape->points[shared_a[0]]) <
        0.0f) {
      edge_cross_a *= -1.0f;
    }
    Vec3A edge_cross_b = edge_vec.cross(normal_b).normalize();
    if (edge_cross_b.dot(tri.points[other_b] - tri.points[shared_b[0]]) <
        0.0f) {
      edge_cross_b *= -1.0f;
    }
    Vec3A calculated_edge = edge_cross_a.cross(edge_cross_b);
    float len2 = calculated_edge.length_squared();
    float corrected_angle = 0.0f;
    bool is_convex = false;
    if (len2 >= k_planar_epsilon) {
      calculated_edge = calculated_edge.normalize();
      Vec3A calculated_normal_a =
          calculated_edge.cross(edge_cross_a).normalize();
      float angle2 = get_angle(calculated_normal_a, edge_cross_a, edge_cross_b);
      float ang4 = k_pi - angle2;
      float dot_a = shape->normal.dot(edge_cross_b);
      is_convex = dot_a < 0.0f;
      corrected_angle = is_convex ? ang4 : -ang4;
    }
    int edge_index;
    uint8_t swap_flag;
    uint8_t convex_flag;
    float *angle_slot;
    if (sum_a == 1) {
      edge_index = 0;
      swap_flag = k_v0v1_swap_normal_b;
      convex_flag = k_v0v1_convex;
      angle_slot = &info.edge_v0_v1_angle;
    } else if (sum_a == 2) {
      edge_index = 2;
      swap_flag = k_v2v0_swap_normal_b;
      convex_flag = k_v2v0_convex;
      angle_slot = &info.edge_v2_v0_angle;
    } else {
      edge_index = 1;
      swap_flag = k_v1v2_swap_normal_b;
      convex_flag = k_v1v2_convex;
      angle_slot = &info.edge_v1_v2_angle;
    }
    Vec3A e = (-shape->edge(edge_index)).normalize();
    Quat orn = quat_from_axis_angle_simd(e, -corrected_angle);
    Vec3A computed_normal_b = quat_mul_vec3a(orn, shape->normal);
    if (computed_normal_b.dot(normal_b) < 0.0f) {
      info.flags |= swap_flag;
    }
    *angle_slot = -corrected_angle;
    if (is_convex) {
      info.flags |= convex_flag;
    }
  }
};

ProcessedMesh process_mesh(const CollisionMeshFile &file) {
  ProcessedMesh pm;
  pm.translation = component_translation(file);
  std::vector<Vec3A> local = file.vertices;
  if (pm.translation != Vec3A::zero()) {
    for (Vec3A &v : local) {
      v = v - pm.translation;
    }
  }
  pm.num_faces = static_cast<uint32_t>(file.indices.size() / 3);
  pm.tris.reserve(pm.num_faces);
  for (size_t i = 0; i < file.indices.size(); i += 3) {
    pm.tris.push_back(TriangleShape::make(local[file.indices[i]],
                                          local[file.indices[i + 1]],
                                          local[file.indices[i + 2]]));
  }
  Vec3A min = Vec3A::zero();
  Vec3A max = Vec3A::zero();
  for (int i = 0; i < 3; ++i) {
    Vec3A vec = Vec3A::zero();
    vec[i] = 1.0f;
    max[i] = mesh_support_vertex(pm.tris, vec)[i];
    vec[i] = -1.0f;
    min[i] = mesh_support_vertex(pm.tris, vec)[i];
  }
  Aabb local_aabb{min, max};
  Vec3A half = 0.5f * (local_aabb.max - local_aabb.min);
  Vec3A center = 0.5f * (local_aabb.max + local_aabb.min);
  pm.aabb_ident_cache = {center - half, center + half};
  BvhQuantization q = BvhQuantization::make(local_aabb);
  std::vector<BvhBuildNode> leaves;
  leaves.reserve(pm.tris.size());
  for (size_t i = 0; i < pm.tris.size(); ++i) {
    leaves.push_back({q.quantize_aabb(update_triangle_aabb(pm.tris[i])), true,
                      static_cast<uint32_t>(i)});
  }
  Aabb tree_aabb = leaves[0].aabb;
  for (size_t i = 1; i < leaves.size(); ++i) {
    tree_aabb = tree_aabb.combine(leaves[i].aabb);
  }
  pm.bvh = build_bvh_bullet(tree_aabb, leaves);
  pm.infos.assign(pm.tris.size(), TriangleInfo{0, k_tau, k_tau, k_tau});
  BvhView view = pm.bvh.view();
  for (uint32_t i = 0; i < pm.tris.size(); ++i) {
    EdgeInfoProcessor proc{i, &pm.tris[i], &pm.tris, &pm.infos};
    view.report_aabb_overlapping_node(proc, pm.tris[i].aabb());
  }
  for (TriangleInfo &info : pm.infos) {
    if (info.edge_v0_v1_angle == k_tau) {
      info.edge_v0_v1_angle = 0.0f;
    }
    if (info.edge_v1_v2_angle == k_tau) {
      info.edge_v1_v2_angle = 0.0f;
    }
    if (info.edge_v2_v0_angle == k_tau) {
      info.edge_v2_v0_angle = 0.0f;
    }
  }
  return pm;
}

std::vector<ProcessedMesh>
build_mode_meshes(GameMode mode,
                  const std::vector<std::vector<uint8_t>> &blobs) {
  const std::vector<uint32_t> &targets = mode_hashes(mode);
  std::map<uint32_t, int> counts;
  for (uint32_t h : targets) {
    counts[h] = 0;
  }
  std::vector<CollisionMeshFile> files;
  for (const auto &blob : blobs) {
    CollisionMeshFile f = parse_cmf(blob);
    auto it = counts.find(f.hash);
    if (it == counts.end()) {
      continue;
    }
    if (it->second > 0) {
      throw std::runtime_error("duplicate collision mesh component");
    }
    it->second += 1;
    files.push_back(std::move(f));
  }
  for (const auto &kv : counts) {
    if (kv.second == 0) {
      throw std::runtime_error(
          "incomplete collision mesh set: missing component");
    }
  }
  if (mode == GameMode::Soccar) {
    std::stable_sort(
        files.begin(), files.end(),
        [](const CollisionMeshFile &a, const CollisionMeshFile &b) {
          return soccar_component_order(a.hash) <
                 soccar_component_order(b.hash);
        });
  }
  std::vector<ProcessedMesh> out;
  out.reserve(files.size());
  for (const auto &f : files) {
    out.push_back(process_mesh(f));
  }
  return out;
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot read collision mesh file: " + path);
  }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
}

const char *mode_folder(GameMode mode) {
  switch (mode) {
  case GameMode::Hoops:
    return "hoops";
  case GameMode::Dropshot:
    return "dropshot";
  default:
    return "soccar";
  }
}

GameMode mesh_mode_of(GameMode mode) {
  switch (mode) {
  case GameMode::Hoops:
    return GameMode::Hoops;
  case GameMode::Dropshot:
    return GameMode::Dropshot;
  default:
    return GameMode::Soccar;
  }
}

}

MeshLibrary MeshLibrary::load_directory(const std::string &root) {
  struct stat sb;
  if (stat(root.c_str(), &sb) != 0) {
    throw std::runtime_error(root + " does not exist");
  }
  if (!S_ISDIR(sb.st_mode)) {
    throw std::runtime_error(root + " is not a directory");
  }
  std::vector<std::pair<GameMode, std::vector<std::vector<uint8_t>>>> files;
  const GameMode modes[3] = {GameMode::Soccar, GameMode::Hoops,
                             GameMode::Dropshot};
  for (GameMode mode : modes) {
    std::string folder = root + "/" + mode_folder(mode);
    DIR *dir = opendir(folder.c_str());
    if (!dir) {
      continue;
    }
    std::vector<std::vector<uint8_t>> blobs;
    while (dirent *entry = readdir(dir)) {
      std::string name = entry->d_name;
      if (name == "." || name == "..") {
        continue;
      }
      std::string path = folder + "/" + name;
      struct stat fs;
      if (stat(path.c_str(), &fs) != 0 || !S_ISREG(fs.st_mode)) {
        continue;
      }
      if (name.size() < 4 || name.substr(name.size() - 4) != ".cmf") {
        continue;
      }
      blobs.push_back(read_file(path));
    }
    closedir(dir);
    files.emplace_back(mode, std::move(blobs));
  }
  return from_memory(files);
}

MeshLibrary MeshLibrary::from_memory(
    const std::vector<std::pair<GameMode, std::vector<std::vector<uint8_t>>>>
        &files) {
  MeshLibrary lib;
  for (const auto &entry : files) {
    GameMode mode = mesh_mode_of(entry.first);
    if (entry.first == GameMode::TheVoid) {
      continue;
    }
    std::vector<ProcessedMesh> meshes = build_mode_meshes(mode, entry.second);
    if (mode == GameMode::Soccar) {
      lib.soccar_ = std::move(meshes);
      lib.has_soccar_ = true;
    } else if (mode == GameMode::Hoops) {
      lib.hoops_ = std::move(meshes);
      lib.has_hoops_ = true;
    } else {
      lib.dropshot_ = std::move(meshes);
      lib.has_dropshot_ = true;
    }
  }
  return lib;
}

bool MeshLibrary::has_mode(GameMode mesh_mode) const {
  switch (mesh_mode_of(mesh_mode)) {
  case GameMode::Hoops:
    return has_hoops_;
  case GameMode::Dropshot:
    return has_dropshot_;
  default:
    return has_soccar_;
  }
}

const std::vector<ProcessedMesh> &
MeshLibrary::meshes(GameMode mesh_mode) const {
  switch (mesh_mode_of(mesh_mode)) {
  case GameMode::Hoops:
    return hoops_;
  case GameMode::Dropshot:
    return dropshot_;
  default:
    return soccar_;
  }
}

float damping_factor_for(float linear_damping) {
  float clamped = clamp_f32(linear_damping, 0.0f, 1.0f);
  return ::powf(1.0f - clamped, consts::tick_time);
}

MutatorConfig default_mutators(GameMode mode) {
  using namespace consts;
  MutatorConfig m;
  m.gravity = {0.0f, 0.0f, gravity_z};
  m.car_mass = car::mass_bt;
  m.ball_mass =
      mode == GameMode::Snowday ? snowday::puck_mass_bt : ball::mass_bt;
  m.ball_max_speed = ball::max_speed;
  m.ball_drag = ball::drag;
  m.jump_accel = car::jump::accel;
  m.jump_immediate_force = car::jump::immediate_force;
  m.boost_accel_ground = car::boost::accel_ground;
  m.boost_accel_air = car::boost::accel_air;
  m.boost_used_per_second =
      mode == GameMode::Heatseeker ? 0.0f : car::boost::used_per_second;
  m.respawn_delay = car::spawn::respawn_time;
  m.bump_cooldown_time = car::bump::cooldown_time;
  m.car_max_boost_amount = car::boost::max;
  m.car_spawn_boost_amount =
      (mode == GameMode::Heatseeker || mode == GameMode::Dropshot)
          ? 100.0f
          : car::boost::spawn_amount;
  m.boost_pad_amount_small = boost_pads::boost_amount_small;
  m.boost_pad_amount_big = boost_pads::boost_amount_big;
  m.boost_pad_cooldown_big = boost_pads::cooldown_big;
  m.boost_pad_cooldown_small = boost_pads::cooldown_small;
  m.ball_hit_extra_force_scale = 1.0f;
  m.bump_force_scale = 1.0f;
  m.ball_radius = ball::get_radius(mode);
  m.unlimited_flips = false;
  m.unlimited_double_jumps = false;
  m.recharge_boost_enabled = mode == GameMode::Dropshot;
  m.recharge_boost_per_second = car::boost::recharge_per_second;
  m.recharge_boost_delay = car::boost::recharge_delay;
  m.demo_mode = DemoMode::Normal;
  m.enable_team_demos = false;
  m.goal_base_threshold_y = goal::soccar_goal_score_base_threshold_y;
  m.boost_pads_enabled = true;
  return m;
}

CarBodyConfig car_body_preset(int idx) {
  static const Vec3A hitbox_sizes[7] = {
      {120.50738f, 86.69941f, 38.659073f},  {130.42679f, 85.77995f, 33.8f},
      {131.31978f, 87.170364f, 31.894402f}, {133.99236f, 83.021f, 32.8f},
      {129.5192f, 84.68787f, 36.659073f},   {123.22023f, 79.21031f, 44.159073f},
      {120.64101f, 86.833405f, 38.7931f}};
  static const Vec3A hitbox_offsets[7] = {
      {13.87566f, 0.0f, 20.754988f}, {9.0f, 0.0f, 15.75f},
      {9.008572f, 0.0f, 12.0942f},   {12.5f, 0.0f, 11.75f},
      {13.87566f, 0.0f, 20.754988f}, {11.37566f, 0.0f, 21.504988f},
      {13.8757f, 0.0f, 15.0f}};
  static const float front_rads[7] = {12.5f, 12.0f, 12.5f, 13.5f,
                                      12.5f, 15.0f, 12.5f};
  static const float back_rads[7] = {15.0f, 13.5f, 17.0f, 15.0f,
                                     15.0f, 15.0f, 15.0f};
  static const float front_rest[7] = {38.755f, 33.95f,  31.9242f, 29.7f,
                                      38.755f, 39.505f, 33.0f};
  static const float back_rest[7] = {37.055f, 33.85f,  27.9242f, 29.666f,
                                     37.055f, 39.105f, 31.3f};
  static const Vec3A front_offsets[7] = {
      {51.25f, 25.90f, 20.755f},  {50.30f, 31.10f, 15.75f},
      {49.97f, 27.80f, 12.0942f}, {51.50f, 26.67f, 11.75f},
      {51.25f, 25.90f, 20.755f},  {51.25f, 25.90f, 21.505f},
      {51.25f, 5.000f, 15.000f}};
  static const Vec3A back_offsets[7] = {
      {-33.75f, 29.50f, 20.755f},  {-34.75f, 33.00f, 15.75f},
      {-35.43f, 20.28f, 12.0942f}, {-35.75f, 35.00f, 11.75f},
      {-34.00f, 29.50f, 20.755f},  {-33.75f, 29.50f, 21.505f},
      {-33.75f, 29.50f, 15.000f}};
  if (idx < 0 || idx >= 7) {
    throw std::runtime_error("car body preset index out of range");
  }
  CarBodyConfig c;
  c.hitbox_size = hitbox_sizes[idx];
  c.hitbox_pos_offset = hitbox_offsets[idx];
  c.front_wheels = {front_rads[idx], front_rest[idx], front_offsets[idx]};
  c.back_wheels = {back_rads[idx], back_rest[idx], back_offsets[idx]};
  c.three_wheels = idx == 6;
  c.dodge_deadzone = 0.5f;
  if (idx == 2) {
    c.steer_angle_curve = consts::curves::plank_steer_angle_from_speed();
    c.powerslide_steer_angle_curve =
        consts::curves::plank_powerslide_steer_angle_from_speed();
  } else if (idx == 6) {
    c.steer_angle_curve = consts::curves::steer_angle_from_speed_threewheel();
    c.powerslide_steer_angle_curve =
        consts::curves::powerslide_steer_angle_from_speed();
  } else {
    c.steer_angle_curve = consts::curves::steer_angle_from_speed();
    c.powerslide_steer_angle_curve =
        consts::curves::powerslide_steer_angle_from_speed();
  }
  return c;
}

namespace {

const Vec3A k_pads_small_soccar[28] = {
    {0.0f, -4240.0f, 70.0f},     {-1792.0f, -4184.0f, 70.0f},
    {1792.0f, -4184.0f, 70.0f},  {-940.0f, -3308.0f, 70.0f},
    {940.0f, -3308.0f, 70.0f},   {0.0f, -2816.0f, 70.0f},
    {-3584.0f, -2484.0f, 70.0f}, {3584.0f, -2484.0f, 70.0f},
    {-1788.0f, -2300.0f, 70.0f}, {1788.0f, -2300.0f, 70.0f},
    {-2048.0f, -1036.0f, 70.0f}, {0.0f, -1024.0f, 70.0f},
    {2048.0f, -1036.0f, 70.0f},  {-1024.0f, 0.0f, 70.0f},
    {1024.0f, 0.0f, 70.0f},      {-2048.0f, 1036.0f, 70.0f},
    {0.0f, 1024.0f, 70.0f},      {2048.0f, 1036.0f, 70.0f},
    {-1788.0f, 2300.0f, 70.0f},  {1788.0f, 2300.0f, 70.0f},
    {-3584.0f, 2484.0f, 70.0f},  {3584.0f, 2484.0f, 70.0f},
    {0.0f, 2816.0f, 70.0f},      {-940.0f, 3308.0f, 70.0f},
    {940.0f, 3308.0f, 70.0f},    {-1792.0f, 4184.0f, 70.0f},
    {1792.0f, 4184.0f, 70.0f},   {0.0f, 4240.0f, 70.0f}};

const Vec3A k_pads_big_soccar[6] = {
    {-3584.0f, 0.0f, 73.0f},     {3584.0f, 0.0f, 73.0f},
    {-3072.0f, 4096.0f, 73.0f},  {3072.0f, 4096.0f, 73.0f},
    {-3072.0f, -4096.0f, 73.0f}, {3072.0f, -4096.0f, 73.0f}};

const Vec3A k_pads_small_hoops[14] = {
    {1536.0f, -1024.0f, 64.0f}, {-1280.0f, -2304.0f, 64.0f},
    {0.0f, -2816.0f, 64.0f},    {-1536.0f, -1024.0f, 64.0f},
    {1280.0f, -2304.0f, 64.0f}, {-512.0f, 512.0f, 64.0f},
    {-1536.0f, 1024.0f, 64.0f}, {1536.0f, 1024.0f, 64.0f},
    {1280.0f, 2304.0f, 64.0f},  {0.0f, 2816.0f, 64.0f},
    {512.0f, 512.0f, 64.0f},    {512.0f, -512.0f, 64.0f},
    {-512.0f, -512.0f, 64.0f},  {-1280.0f, 2304.0f, 64.0f}};

const Vec3A k_pads_big_hoops[6] = {
    {-2176.0f, 2944.0f, 72.0f},  {2176.0f, -2944.0f, 72.0f},
    {-2176.0f, -2944.0f, 72.0f}, {-2432.0f, 0.0f, 72.0f},
    {2432.0f, 0.0f, 72.0f},      {2175.99f, 2944.0f, 72.0f}};

const Vec3A k_tile_hexagon_verts_bt[6] = {
    {0.0f, -8.85f, 0.0f}, {7.6643f, -4.425f, 0.0f}, {7.6643f, 4.425f, 0.0f},
    {0.0f, 8.85f, 0.0f},  {-7.6643f, 4.425f, 0.0f}, {-7.6643f, -4.425f, 0.0f}};

constexpr uint32_t k_hoops_net_face_count = 798;

int total_cmp_f32(float a, float b) {
  int32_t ia = static_cast<int32_t>(f32_to_u32(a));
  int32_t ib = static_cast<int32_t>(f32_to_u32(b));
  ia ^= static_cast<int32_t>(static_cast<uint32_t>(ia >> 31) >> 1);
  ib ^= static_cast<int32_t>(static_cast<uint32_t>(ib >> 31) >> 1);
  return ia < ib ? -1 : (ia > ib ? 1 : 0);
}

uint32_t world_add_dynamic_body(World &w, RigidBody body, uint8_t group,
                                uint8_t mask) {
  ArenaPhysics &a = *w.a;
  uint32_t dyn_idx = a.num_dyn_bodies;
  if (dyn_idx >= static_cast<uint32_t>(k_max_dyn_bodies)) {
    throw std::runtime_error("too many dynamic bodies in arena");
  }
  uint32_t body_idx = w.num_static() + dyn_idx;
  body.world_array_idx = body_idx;
  Aabb aabb = body.shape.get_aabb(body.world_trans);
  uint16_t handle;
  if (a.free_count > 0) {
    handle = a.free_list[--a.free_count];
  } else {
    handle = a.next_new_handle++;
  }
  a.num_handles += 1;
  if (handle <= w.num_static() ||
      handle - w.num_static() - 1 >= static_cast<uint32_t>(k_max_dyn_bodies)) {
    throw std::runtime_error("dynamic broadphase handle out of range");
  }
  uint32_t idx[3];
  grid_cell_indices(*w.s, aabb.min, idx);
  BroadphaseProxy &proxy = w.dyn_proxy(handle);
  proxy.client_obj_idx = body_idx;
  proxy.collision_filter_group = group;
  proxy.collision_filter_mask = mask;
  proxy.aabb = aabb;
  proxy.cell_idx = grid_cell_index(*w.s, idx);
  proxy.indices[0] = idx[0];
  proxy.indices[1] = idx[1];
  proxy.indices[2] = idx[2];
  proxy.stamp = a.next_stamp++;
  sweep_add_handle_edges(w, handle, aabb);
  body.broadphase_handle = handle;
  a.dyn_bodies[dyn_idx] = body;
  a.num_dyn_bodies += 1;
  return body_idx;
}

float grid_cell_size_uu(const ArenaSetup &setup) {
  switch (setup.mem_weight_mode) {
  case MemWeightMode::Light:
    return (setup.max_pos - setup.min_pos).max_element();
  case MemWeightMode::Balanced:
    return setup.max_aabb_len * 3.0f;
  case MemWeightMode::Heavy:
    return setup.max_aabb_len;
  }
  throw std::runtime_error("invalid memory weight mode");
}

}

HostStatic::HostStatic(const MeshLibrary &library, const ArenaSetup &setup)
    : setup_(setup), template_phys_(std::make_unique<ArenaPhysics>()) {
  build_statics(library);
  build_grid();
  build_pads();
  sin_table.resize(k_sin_table_size);
  for (int j = 0; j < k_sin_table_size; ++j) {
    float jf = static_cast<float>(j);
    float angle = ((jf + jf) * k_pi) * (1.0f / 16384.0f);
    sin_table[j] = static_cast<float>(::sin(static_cast<double>(angle)));
  }
  view_.world.sin_table = sin_table.data();
  view_.mode = setup_.mode;

  ArenaPhysics &a = *template_phys_;
  std::memset(&a, 0, sizeof(ArenaPhysics));
  a.handle_edges[0] = HandleEdges{{0, 0, 0}, {1, 1, 1}};
  for (int axis = 0; axis < 3; ++axis) {
    a.edges[axis][0] = {0, 0};
    a.edges[axis][1] = {k_sweep_sentinel, 0};
  }
  a.num_edges = 2;
  a.next_new_handle = 1;
  World w{&view_.world, &a};
  for (uint32_t i = 0; i < view_.world.num_static_bodies; ++i) {
    uint16_t handle = a.next_new_handle++;
    a.num_handles += 1;
    sweep_add_handle_edges(w, handle, static_proxies[i].aabb);
  }
  if (a.error_flags != 0) {
    throw std::runtime_error(
        "static broadphase template exceeds the per-arena pair capacity");
  }
}

void HostStatic::build_statics(const MeshLibrary &library) {
  using namespace consts;
  GameMode mode = setup_.mode;
  struct StaticEntry {
    Shape shape;
    Vec3A pos_bt;
    bool has_group;
    uint8_t group;
    UserInfoType user_idx;
    uint32_t user_pointer;
  };
  std::vector<StaticEntry> entries;
  if (mode != GameMode::TheVoid) {
    if (!library.has_mode(mode)) {
      throw std::runtime_error(
          "collision meshes for this game mode are not loaded");
    }
    const std::vector<ProcessedMesh> &meshes = library.meshes(mode);
    if (meshes.empty()) {
      throw std::runtime_error("no arena meshes found for the game mode");
    }
    for (const ProcessedMesh &pm : meshes) {
      MeshView mv;
      mv.bvh = pm.bvh.view();
      mv.tris = pm.tris.data();
      mv.infos = pm.infos.data();
      mv.num_tris = static_cast<uint32_t>(pm.tris.size());
      mv.aabb_ident_cache = pm.aabb_ident_cache;
      uint32_t mesh_idx = static_cast<uint32_t>(mesh_views.size());
      mesh_views.push_back(mv);
      mesh_bvhs.push_back(&pm.bvh);
      bool is_net =
          mode == GameMode::Hoops && pm.num_faces == k_hoops_net_face_count;
      entries.push_back({Shape::make_mesh(mesh_idx, pm.aabb_ident_cache),
                         pm.translation, is_net,
                         static_cast<uint8_t>(k_group_hoops_net),
                         UserInfoType::None, 0});
    }
    Aabb arena_aabb = arena::get_aabb(mode);
    auto add_plane = [&](Vec3A pos_uu, Vec3A normal, bool has_group,
                         uint8_t group) {
      Vec3A pos_bt = pos_uu * uu_to_bt;
      Affine3A trans{Mat3A::identity(), pos_bt};
      entries.push_back(
          {Shape::make_plane(StaticPlaneShape::make(trans, normal)), pos_bt,
           has_group, group, UserInfoType::None, 0});
    };
    bool is_dropshot = mode == GameMode::Dropshot;
    add_plane({0.0f, 0.0f, arena_aabb.min.z}, {0.0f, 0.0f, 1.0f}, is_dropshot,
              static_cast<uint8_t>(k_group_dropshot_floor));
    add_plane({0.0f, 0.0f, arena_aabb.max.z}, {0.0f, 0.0f, -1.0f}, false, 0);
    if (!is_dropshot) {
      add_plane({arena_aabb.min.x, 0.0f, arena_aabb.max.z / 2.0f},
                {1.0f, 0.0f, 0.0f}, false, 0);
      add_plane({arena_aabb.max.x, 0.0f, arena_aabb.max.z / 2.0f},
                {-1.0f, 0.0f, 0.0f}, false, 0);
    }
    if (mode == GameMode::Hoops) {
      add_plane({0.0f, arena_aabb.min.y, arena_aabb.max.z / 2.0f},
                {0.0f, 1.0f, 0.0f}, false, 0);
      add_plane({0.0f, arena_aabb.max.y, arena_aabb.max.z / 2.0f},
                {0.0f, -1.0f, 0.0f}, false, 0);
    }
    if (is_dropshot) {
      constexpr float clamp_y = dropshot::tile_offset_y * uu_to_bt;
      hull_points.reserve(2 * dropshot::num_tiles_per_team * 6);
      uint32_t tile_idx = 0;
      for (int t = 0; t < 2; ++t) {
        Team team = t == 0 ? Team::Blue : Team::Orange;
        for (uint32_t i = 0; i < dropshot::num_tiles_per_team; ++i) {
          Vec3A pos = dropshot_tile_pos(team, i);
          uint32_t offset = static_cast<uint32_t>(hull_points.size());
          for (int j = 0; j < 6; ++j) {
            Vec3A vert = pos * uu_to_bt + k_tile_hexagon_verts_bt[j];
            vert.y = team == Team::Blue ? min_f32_c(vert.y, -clamp_y)
                                        : max_f32_c(vert.y, clamp_y);
            hull_points.push_back(vert);
          }
          HullView view{hull_points.data() + offset, 6};
          ConvexHullShape hull =
              ConvexHullShape::make(HullSource::Static, offset, view);
          entries.push_back({Shape::make_hull(hull),
                             {0.0f, 0.0f, -hull.margin},
                             true,
                             static_cast<uint8_t>(k_group_dropshot_tile),
                             UserInfoType::DropshotTile,
                             tile_idx});
          ++tile_idx;
        }
      }
    }
  }
  if (entries.size() > static_cast<size_t>(k_max_static_bodies)) {
    throw std::runtime_error("too many static bodies for the arena");
  }
  static_bodies.reserve(entries.size());
  static_proxies.reserve(entries.size());
  for (const StaticEntry &e : entries) {
    RigidBodyConstructionInfo info =
        RigidBodyConstructionInfo::make(0.0f, e.shape);
    info.restitution = arena::base_coefs.restitution;
    info.friction = arena::base_coefs.friction;
    info.start_world_trans.translation = e.pos_bt;
    RigidBody rb = RigidBody::make(info, 1.0f);
    uint32_t idx = static_cast<uint32_t>(static_bodies.size());
    rb.world_array_idx = idx;
    rb.broadphase_handle = idx + 1;
    rb.user_idx = e.user_idx;
    rb.user_pointer = e.user_pointer;
    rb.set_activation_state(ActivationState::Sleeping);
    BroadphaseProxy proxy{};
    proxy.client_obj_idx = idx;
    if (e.has_group) {
      proxy.collision_filter_group =
          static_cast<uint8_t>(e.group | k_group_static);
      proxy.collision_filter_mask =
          static_cast<uint8_t>(e.group ^ k_group_static);
    } else {
      proxy.collision_filter_group = k_group_static;
      proxy.collision_filter_mask =
          static_cast<uint8_t>(k_group_all ^ k_group_static);
    }
    proxy.aabb = rb.shape.get_aabb(rb.world_trans);
    static_bodies.push_back(rb);
    static_proxies.push_back(proxy);
  }
  StaticWorldData &sw = view_.world;
  sw.num_static_bodies = static_cast<uint32_t>(static_bodies.size());
  sw.static_bodies = static_bodies.data();
  sw.static_proxies = static_proxies.data();
  sw.meshes = mesh_views.data();
  sw.hull_points = hull_points.data();
}

void HostStatic::build_grid() {
  using namespace consts;
  float cell_size_uu = grid_cell_size_uu(setup_);
  StaticWorldData &sw = view_.world;
  sw.grid_min = setup_.min_pos * uu_to_bt;
  sw.grid_max = setup_.max_pos * uu_to_bt;
  sw.cell_size = cell_size_uu * uu_to_bt;
  sw.cell_size_sq = sw.cell_size * sw.cell_size;
  Vec3A counts = (sw.grid_max - sw.grid_min) / sw.cell_size;
  uint64_t total = 1;
  for (int i = 0; i < 3; ++i) {
    float c = std::ceil(counts[i]);
    uint64_t n = c >= 18446744073709551616.0f ? UINT64_MAX
                 : c > 0.0f                   ? static_cast<uint64_t>(c)
                                              : 0;
    if (n < 1) {
      n = 1;
    }
    if (n > 4096) {
      throw std::runtime_error("ray grid has too many cells");
    }
    sw.num_cells[i] = static_cast<uint32_t>(n);
    total *= n;
  }
  if (total > (1u << 22)) {
    throw std::runtime_error("ray grid has too many cells");
  }
  std::vector<std::vector<uint16_t>> cells(total);
  for (uint32_t s = 0; s < sw.num_static_bodies; ++s) {
    const BroadphaseProxy &proxy = static_proxies[s];
    const RigidBody &body = static_bodies[s];
    uint16_t handle = static_cast<uint16_t>(s + 1);
    uint32_t lo[3];
    uint32_t hi[3];
    grid_cell_indices(sw, proxy.aabb.min.max(sw.grid_min), lo);
    grid_cell_indices(sw, proxy.aabb.max.min(sw.grid_max), hi);
    const MeshView *mesh = body.shape.kind == ShapeKind::TriangleMesh
                               ? &mesh_views[body.shape.mesh.mesh_idx]
                               : nullptr;
    Vec3A pos = body.world_trans.translation;
    for (uint32_t i = lo[0]; i <= hi[0]; ++i) {
      for (uint32_t j = lo[1]; j <= hi[1]; ++j) {
        for (uint32_t k = lo[2]; k <= hi[2]; ++k) {
          if (mesh) {
            Vec3A cell_min = sw.grid_min + Vec3A{static_cast<float>(i),
                                                 static_cast<float>(j),
                                                 static_cast<float>(k)} *
                                               sw.cell_size;
            Aabb cell_aabb{cell_min, cell_min + Vec3A::splat(sw.cell_size)};
            if (!mesh->bvh.check_overlap_with(cell_aabb - pos)) {
              continue;
            }
          }
          for (uint32_t i1 = 0; i1 <= 2; ++i1) {
            for (uint32_t j1 = 0; j1 <= 2; ++j1) {
              for (uint32_t k1 = 0; k1 <= 2; ++k1) {
                uint32_t cell[3] = {i + i1, j + j1, k + k1};
                if (cell[0] == 0 || cell[1] == 0 || cell[2] == 0) {
                  continue;
                }
                cell[0] -= 1;
                cell[1] -= 1;
                cell[2] -= 1;
                if (cell[0] >= sw.num_cells[0] || cell[1] >= sw.num_cells[1] ||
                    cell[2] >= sw.num_cells[2]) {
                  continue;
                }
                std::vector<uint16_t> &list = cells[grid_cell_index(sw, cell)];
                if (std::find(list.begin(), list.end(), handle) != list.end()) {
                  continue;
                }
                list.push_back(handle);
              }
            }
          }
        }
      }
    }
  }
  cell_offsets.assign(total + 1, 0);
  cell_handles.clear();
  for (uint64_t c = 0; c < total; ++c) {
    cell_offsets[c] = static_cast<uint32_t>(cell_handles.size());
    cell_handles.insert(cell_handles.end(), cells[c].begin(), cells[c].end());
  }
  cell_offsets[total] = static_cast<uint32_t>(cell_handles.size());
  sw.cell_static_offsets = cell_offsets.data();
  sw.cell_static_handles = cell_handles.data();
}

void HostStatic::build_pads() {
  using namespace consts;
  GameMode mode = setup_.mode;
  view_.has_pads = mode != GameMode::TheVoid && mode != GameMode::Dropshot;
  view_.pads = PadLayoutView{};
  if (!view_.has_pads) {
    return;
  }
  if (setup_.has_custom_pads) {
    pad_configs = setup_.custom_pads;
  } else {
    bool hoops = mode == GameMode::Hoops;
    const Vec3A *small = hoops ? k_pads_small_hoops : k_pads_small_soccar;
    size_t num_small = hoops ? 14 : 28;
    const Vec3A *big = hoops ? k_pads_big_hoops : k_pads_big_soccar;
    for (size_t i = 0; i < num_small; ++i) {
      pad_configs.push_back({small[i], false});
    }
    for (size_t i = 0; i < 6; ++i) {
      pad_configs.push_back({big[i], true});
    }
  }
  if (pad_configs.empty()) {
    throw std::runtime_error("boost pad layout is empty");
  }
  if (pad_configs.size() > static_cast<size_t>(k_max_pads)) {
    throw std::runtime_error("too many boost pads");
  }
  std::stable_sort(pad_configs.begin(), pad_configs.end(),
                   [](const BoostPadConfig &a, const BoostPadConfig &b) {
                     int c = total_cmp_f32(a.pos.y, b.pos.y);
                     if (c != 0) {
                       return c < 0;
                     }
                     return total_cmp_f32(a.pos.x, b.pos.x) < 0;
                   });
  std::vector<BvhBuildNode> nodes;
  for (size_t i = 0; i < pad_configs.size(); ++i) {
    float r = pad_cyl_radius(pad_configs[i].is_big);
    Vec3A extent{r, r, boost_pads::cyl_height};
    Aabb aabb{pad_configs[i].pos - extent, pad_configs[i].pos + extent};
    pad_aabbs.push_back(aabb);
    nodes.push_back({aabb, true, static_cast<uint32_t>(i)});
  }
  Aabb all = pad_aabbs[0];
  for (size_t i = 1; i < pad_aabbs.size(); ++i) {
    all = all.combine(pad_aabbs[i]);
  }
  pad_bvh = build_bvh_sah(all, nodes);
  view_.pads.num_pads = static_cast<uint32_t>(pad_configs.size());
  view_.pads.configs = pad_configs.data();
  view_.pads.aabbs = pad_aabbs.data();
  view_.pads.bvh = pad_bvh.view();
  view_.pads.max_pad_z = all.max.z;
}

uint32_t arena_add_car(const HostStatic &st, Arena &a, const CarSpec &spec) {
  using namespace consts;
  if (a.num_cars >= static_cast<uint32_t>(k_max_cars)) {
    throw std::runtime_error("arena car capacity exceeded");
  }
  const CarBodyConfig &config = spec.config;
  const MutatorConfig &mut = a.mutators;
  uint32_t idx = a.num_cars;
  Vec3A inertia;
  Shape shape = make_car_shape(config, mut.car_mass, inertia);
  RigidBodyConstructionInfo info =
      RigidBodyConstructionInfo::make(mut.car_mass, shape);
  info.friction = car::base_coefs.friction;
  info.restitution = car::base_coefs.restitution;
  info.start_world_trans = Affine3A::identity();
  info.local_inertia = inertia;
  RigidBody body = RigidBody::make(info, damping_factor_for(0.0f));
  body.user_idx = UserInfoType::Car;
  body.collision_flags |= k_flag_custom_material_callback;
  World w = arena_world(st.view(), a);
  uint32_t body_idx = world_add_dynamic_body(
      w, body, static_cast<uint8_t>(k_group_default | k_group_dropshot_floor),
      k_group_all);
  Car &car = a.cars[idx];
  car.info.idx = idx;
  car.info.team = spec.team;
  car.info.config = config;
  car.rigid_body_idx = body_idx;
  car.vehicle.min_hit_dist_sq = config.wheel_ray_min_hit_dist_sq();
  car.vehicle.chassis_body_idx = body_idx;
  float strengths[4];
  car_suspension_strengths(
      w.dyn_body(body_idx).inv_mass,
      config.front_wheels.connection_point_offset.x * uu_to_bt,
      config.back_wheels.connection_point_offset.x * uu_to_bt, strengths);
  for (int i = 0; i < k_num_wheels; ++i) {
    bool front = i < 2;
    bool left = i % 2 == 0;
    const WheelPairConfig &wheel =
        front ? config.front_wheels : config.back_wheels;
    Vec3A offset = wheel.connection_point_offset;
    if (left) {
      offset.y *= -1.0f;
    }
    float rest =
        wheel.suspension_rest_length - vehicle_consts::max_suspension_travel;
    car.vehicle.wheels[i] = WheelInfo::make_default();
    car.vehicle.wheels[i].set_params(offset * uu_to_bt, rest * uu_to_bt,
                                     wheel.wheel_radius * uu_to_bt,
                                     strengths[i]);
  }
  car.vel_impulse_cache = Vec3A::zero();
  car.sticky_gate_prev = false;
  car.state = car_state_from_core(
      car_core_grounded(car_state_default().phys, mut.car_spawn_boost_amount));
  car.state.cache.is_on_ground = true;
  car_respawn(car, w.dyn_body(body_idx), a.rng, st.view().mode,
              mut.car_spawn_boost_amount);
  w.dyn_body(body_idx).user_pointer = idx;
  a.num_cars += 1;
  a.ball_only = false;
  return idx;
}

void create_arena(const HostStatic &st, const MutatorConfig &mutators,
                  uint64_t rng_seed, const std::vector<CarSpec> &cars,
                  Arena &a) {
  using namespace consts;
  std::memset(&a, 0, sizeof(Arena));
  a.phys = st.template_physics();
  a.phys.gravity = mutators.gravity * uu_to_bt;
  a.mutators = mutators;
  GameMode mode = st.view().mode;
  Shape shape;
  Vec3A inertia;
  make_ball_collision_shape(mode, mutators, a.phys, shape, inertia);
  RigidBodyConstructionInfo info =
      RigidBodyConstructionInfo::make(mutators.ball_mass, shape);
  info.start_world_trans.translation.z = ball::rest_z * uu_to_bt;
  info.local_inertia = inertia;
  info.linear_damping = mutators.ball_drag;
  info.linear_sleeping_threshold = 0.8f;
  PhysicsCoefs coefs =
      mode == GameMode::Snowday ? snowday::puck_coefs : ball::coefs;
  info.friction = coefs.friction;
  info.restitution = coefs.restitution;
  RigidBody body =
      RigidBody::make(info, damping_factor_for(mutators.ball_drag));
  body.user_idx = UserInfoType::Ball;
  body.collision_flags |= k_flag_can_sleep;
  body.collision_flags |= k_flag_custom_material_callback;
  if (st.setup().no_ball_rot && body.shape.kind == ShapeKind::Sphere) {
    body.collision_flags |= k_flag_no_angular_motion;
  }
  World w = arena_world(st.view(), a);
  a.ball.rigid_body_idx = world_add_dynamic_body(
      w, body,
      static_cast<uint8_t>(k_group_default | k_group_hoops_net |
                           k_group_dropshot_tile),
      k_group_all);
  a.ball.state = ball_state_default();
  a.ball.ground_stick_applied = false;
  a.num_pads = st.view().pads.num_pads;
  if (mode == GameMode::Dropshot) {
    a.has_tiles = true;
    a.tiles = TileStates{};
    a.ball_only = false;
    arena_update_tile_flags(a);
  } else {
    a.has_tiles = false;
    a.ball_only = true;
  }
  a.rng = Rng{rng_seed};
  a.tick_count = 0;
  for (const CarSpec &spec : cars) {
    arena_add_car(st, a, spec);
  }
}

MutatorOp make_mutator_op(const MutatorConfig &current,
                          const MutatorConfig &mutators) {
  if (!(is_finite_f32(current.car_mass) && current.car_mass > 0.0f) ||
      !(is_finite_f32(current.ball_mass) && current.ball_mass > 0.0f)) {
    throw std::runtime_error(
        "runtime mutator updates require finite, positive current masses");
  }
  if (!(is_finite_f32(mutators.car_mass) && mutators.car_mass > 0.0f) ||
      !(is_finite_f32(mutators.ball_mass) && mutators.ball_mass > 0.0f)) {
    throw std::runtime_error(
        "runtime car and ball masses must be finite and strictly positive");
  }
  return {mutators, damping_factor_for(mutators.ball_drag),
          damping_factor_for(0.0f)};
}

PadOp make_pad_op(const HostStatic &st, uint32_t num_cars, uint32_t idx,
                  float cooldown, const std::vector<uint32_t> &touching) {
  if (!st.view().has_pads) {
    throw std::runtime_error("this arena has no boost pads");
  }
  if (idx >= st.view().pads.num_pads) {
    throw std::runtime_error("boost pad index out of range");
  }
  PadOp op{};
  op.cooldown = cooldown;
  for (uint32_t car_idx : touching) {
    if (car_idx >= num_cars) {
      continue;
    }
    bool present = false;
    for (uint32_t i = 0; i < op.num_touching; ++i) {
      if (op.touching[i] == car_idx) {
        present = true;
      }
    }
    if (!present) {
      op.touching[op.num_touching++] = static_cast<uint8_t>(car_idx);
    }
  }
  return op;
}

}
