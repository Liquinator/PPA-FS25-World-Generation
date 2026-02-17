#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "heightmap_gen.hpp"
#include "perlin_noise_cpu.hpp"

struct TreePlacementConfig {
  float treeLine = 0.8;
  float maxSlope = 0.4;
  float treeDensity = 0.4;
  unsigned int moistureSeedOffset = 1;
  double scale = 60.0;
};

inline glm::vec3 surface_normal(
    int i, int j, const std::vector<std::vector<double>>& heightmap,
    const TreePlacementConfig& config) {
  int x = heightmap.size();
  int y = heightmap[0].size();
  float h = heightmap[i][j] * config.scale;
  float h_left = (i > 0) ? heightmap[i - 1][j] * config.scale : h;
  float h_right = (i < x - 1) ? heightmap[i + 1][j] * config.scale : h;
  float h_up = (j < y - 1) ? heightmap[i][j + 1] * config.scale : h;
  float h_down = (j > 0) ? heightmap[i][j - 1] * config.scale : h;

  float dx = (h_right - h_left) / 2.0;
  float dy = (h_up - h_down) / 2.0;

  glm::vec3 normal(-dx, 1.0, -dy);
  return glm::normalize(normal);
}

inline std::vector<glm::vec2> place_trees_seq(
    const std::vector<std::vector<double>>& heightmap,
    const std::vector<std::vector<double>>& moisture_map,
    const TreePlacementConfig& config = TreePlacementConfig{}) {
  std::vector<glm::vec2> treeLocation;
  glm::vec2 dim = glm::vec2(heightmap.size(), heightmap.size());

  for (int x = 0; x < dim.x; x++) {
    for (int y = 0; y < dim.y; y++) {
      float height = heightmap[x][y];
      if (height > config.treeLine) continue;
      glm::vec3 surface_norm = surface_normal(x, y, heightmap, config);
      if (glm::dot(surface_norm, glm::vec3(0.0, 0.0, 1.0)) < config.maxSlope)
        continue;
      if ((float)moisture_map[x][y] < config.treeDensity) continue;
      treeLocation.push_back(glm::vec2(x, y));
    }
  }
  return treeLocation;
}
/*
inline std::vector<glm::vec2> place_trees_par(
    const std::vector<std::vector<double>>& heightmap,
    const TreePlacementConfig& treeConfig = TreePlacementConfig{},
    const HeightmapConfig& moistureMapConfig = HeightmapConfig{}) {
  PerlinNoise* perlinNoise =
      get_perlin_noise_generator(true, moistureMapConfig.seed);

  std::vector<glm::vec2> treeLocation;
  int dimension = heightmap.size();
  parlay::sequence<double> moisture_map(dimension * dimension);

  double adjusted_frequency = moistureMapConfig.frequency * (dimension / 256);
  moisture_map = perlinNoise->generate_heightmap(
      moistureMapConfig.octaves, adjusted_frequency,
      glm::vec2(dimension, dimension));

  auto minmax = parlay::minmax_element(moisture_map);
  double min_val = *minmax.first;
  double max_val = *minmax.second;
  double range = max_val - min_val;

  auto treeRows = parlay::map(parlay::iota(dimension), [&](int x) {
    std::vector<glm::vec2> localTrees;

    for (int y = 0; y < dimension; y++) {
      double raw_value = moisture_map[x * dimension + y];
      double normalized_value = (raw_value - min_val) / range;

      if (normalized_value < treeConfig.treeDensity) continue;
      if (heightmap[x][y] > treeConfig.treeLine) continue;
      glm::vec3 surface_norm = surface_normal(x, y, heightmap, treeConfig);
      if (glm::dot(surface_norm, glm::vec3(0.0, 0.0, 1.0)) <
          treeConfig.maxSlope)
        continue;
      localTrees.push_back(glm::vec2(x, y));
    }
    return localTrees;
  });

  auto flattened = parlay::flatten(treeRows);
  treeLocation.assign(flattened.begin(), flattened.end());
  return treeLocation;
}

inline std::vector<glm::vec2> place_trees(
    const std::vector<std::vector<double>>& heightmap,
    bool use_parallel = false,
    const TreePlacementConfig& treeConfig = TreePlacementConfig{},
    const HeightmapConfig& moistureMapConfig = HeightmapConfig{}) {
  int dimension = heightmap.size();

  if (use_parallel) {
    return place_trees_par(heightmap, treeConfig, moistureMapConfig);
  } else {
    auto moisture_map =
        generate_heightmap(dimension, use_parallel, moistureMapConfig);
    return place_trees_seq(heightmap, moisture_map, treeConfig);
  }
}
  */