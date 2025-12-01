#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "perlin_noise.hpp"

struct HeightmapConfig {
  int octaves = 20;
  double frequency = 2.0;
  unsigned int seed = 42;
  double scale = 60;
};

PerlinNoise* get_perlin_noise_generator(const bool use_parallel,
                                        const unsigned int seed) {
  if (use_parallel) {
    return new PerlinNoisePar(seed);
  } else {
    return new PerlinNoiseSeq(seed);
  }
}

inline std::vector<std::vector<double>> generate_heightmap_seq(
    int dimensions, const HeightmapConfig& config = HeightmapConfig{}) {
  std::vector<std::vector<double>> heightmap(
      dimensions, std::vector<double>(dimensions, 0.0));

  PerlinNoiseSeq PerlinNoise(config.seed);
  double adjusted_frequency = config.frequency * (dimensions / 256);
  return PerlinNoise.generate_normalized_heightmap(
      config.octaves, adjusted_frequency, glm::vec2(dimensions, dimensions),
      heightmap);
}

inline std::vector<std::vector<double>> generate_heightmap_par(
    int dimensions, const HeightmapConfig& config = HeightmapConfig{}) {
  std::vector<std::vector<double>> heightmap(
      dimensions, std::vector<double>(dimensions, 0.0));

  PerlinNoisePar PerlinNoise(config.seed);
  double adjusted_frequency = config.frequency * (dimensions / 256);
  return PerlinNoise.generate_normalized_heightmap(
      config.octaves, adjusted_frequency, glm::vec2(dimensions, dimensions),
      heightmap);
}

inline std::vector<std::vector<double>> generate_heightmap(
    int dimensions, bool use_parallel = false,
    const HeightmapConfig& config = HeightmapConfig{}) {
  if (use_parallel) {
    return generate_heightmap_par(dimensions, config);
  } else {
    return generate_heightmap_seq(dimensions, config);
  }
}