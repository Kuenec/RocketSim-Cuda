#pragma once

#include <cstdint>

#include "rsc/math/vec.hpp"
#include "rsc/sim/consts.hpp"

namespace rsc {

inline constexpr uint8_t k_tile_full = 0;
inline constexpr uint8_t k_tile_damaged = 1;
inline constexpr uint8_t k_tile_broken = 2;

struct TileStates {
  uint8_t states[2][consts::dropshot::num_tiles_per_team];
};

RS_HD Vec3A dropshot_tile_base_pos(uint32_t index) {
  using namespace consts::dropshot;
  uint32_t cur = 0;
  float y = tile_offset_y;
  for (int i = 0; i < num_tile_rows; ++i) {
    int num_tiles = tiles_in_first_row - i;
    float row_size_x = tile_width_x * static_cast<float>(num_tiles);
    float row_start_x = -(row_size_x / 2.0f) + (tile_width_x / 2.0f);
    for (int j = 0; j < num_tiles; ++j) {
      if (cur == index) {
        float x = row_start_x + tile_width_x * static_cast<float>(j);
        return {x, y, 0.0f};
      }
      ++cur;
    }
    y += row_offset_y;
  }
  return Vec3A::zero();
}

RS_HD Vec3A dropshot_tile_pos(Team team, uint32_t index) {
  return dropshot_tile_base_pos(index) *
         static_cast<float>(static_cast<int8_t>(team) * 2 - 1);
}

RS_HD int dropshot_neighbors(uint32_t start, int radius, uint8_t *out) {
  using namespace consts::dropshot;
  float max_r = tile_width_x * 1.2f * static_cast<float>(radius);
  float max_r_sq = max_r * max_r;
  Vec3A pos = dropshot_tile_pos(Team::Blue, start);
  int count = 0;
  for (int j = 0; j < num_tiles_per_team; ++j) {
    Vec3A other = dropshot_tile_pos(Team::Blue, static_cast<uint32_t>(j));
    float dx = pos.x - other.x;
    float dy = pos.y - other.y;
    float dz = pos.z - other.z;
    float dist_sq = dx * dx + dy * dy + dz * dz;
    if (dist_sq < max_r_sq) {
      out[count++] = static_cast<uint8_t>(j);
    }
  }
  return count;
}

RS_HD int dropshot_break_indices(uint32_t tile_idx, uint8_t charge_level,
                                 uint8_t *out) {
  if (charge_level == 3) {
    return dropshot_neighbors(tile_idx, 2, out);
  }
  if (charge_level == 2) {
    return dropshot_neighbors(tile_idx, 1, out);
  }
  out[0] = static_cast<uint8_t>(tile_idx);
  return 1;
}

}
