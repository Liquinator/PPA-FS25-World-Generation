#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "perlin_noise.hpp"
#include "perlin_noise_cuda.hpp"

struct HeightmapConfig {
  int width;
  int height;
  int octaves = 20;
  double frequency = 2.0;
  unsigned int seed = 42;
  double scale = 60;
};

class MapGenerator {
 public:
  HeightMap generate_heightmap_seq(const PerlinNoise& noise,
                                   const HeightmapConfig& config) {
    HeightMap heightmap(config.width, config.height);
    for (int x = 0; x < config.width; x++) {
      for (int y = 0; y < config.height; y++) {
        float nx = x * (config.frequency / config.width);
        float ny = y * (config.frequency / config.height);
        heightmap.at(x, y) = noise.octaveNoise(nx, ny, config.octaves);
      }
    }
    normalize(heightmap);
    return heightmap;
  }

  HeightMap generate_heightmap_par(const PerlinNoise& noise,
                                   const HeightmapConfig& config) {
    HeightMap heightmap(config.width, config.height);
    parlay::parallel_for(0, config.width * config.height, [&](size_t i) {
      int x = i % config.width;
      int y = i / config.width;

      float nx = x * (config.frequency / config.width);
      float ny = y * (config.frequency / config.height);

      heightmap.data[i] = noise.octaveNoise(nx, ny, config.octaves);
    });
    normalize(heightmap);
    return heightmap;
  }

  HeightMap generate_heightmap_cuda(const PerlinNoiseCuda& noise,
                                    const HeightmapConfig& config) {
    HeightMap heightmap(config.width, config.height);
    heightmap.data = noise.generate_normalized_heightmap(
        config.octaves, config.frequency,
        glm::vec2(config.width, config.height));
    return heightmap;
  }

 private:
  // TODO Include sequential normalization
  void normalize(HeightMap& heightmap) {
    auto minmax =
        std::minmax_element(heightmap.data.begin(), heightmap.data.end());
    float min_val = *minmax.first;
    float max_val = *minmax.second;
    float range = max_val - min_val;

    if (range <= 0.00001) return;

    parlay::parallel_for(0, heightmap.data.size(), [&](size_t i) {
      heightmap.data[i] = (heightmap.data[i] - min_val) / range;
    });
  }
};