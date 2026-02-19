#pragma once

#include <glm/glm.hpp>

#include "parlay/internal/get_time.h"
#include "perlin_noise_cpu.hpp"
#include "perlin_noise_cuda.hpp"
#include "perlin_noise_hybrid.hpp"

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
  HeightMap generate_heightmap_seq(const PerlinNoise &noise,
                                   const HeightmapConfig &config) {
    parlay::internal::timer t(std::string("sequential"));
    HeightMap heightmap(config.width, config.height);
    for (int x = 0; x < config.width; x++) {
      for (int y = 0; y < config.height; y++) {
        float nx = x * (config.frequency / config.width);
        float ny = y * (config.frequency / config.height);
        heightmap.at(x, y) = noise.octaveNoise(nx, ny, config.octaves);
      }
    }
    t.next("generation");
    normalize_seq(heightmap);
    t.next("normalization");
    t.total();
    return heightmap;
  }

  HeightMap generate_heightmap_par(const PerlinNoise &noise,
                                   const HeightmapConfig &config) {
    parlay::internal::timer t(std::string("parallel"));
    HeightMap heightmap(config.width, config.height);
    parlay::parallel_for(0, config.width * config.height, [&](size_t i) {
      int x = i % config.width;
      int y = i / config.width;

      float nx = x * (config.frequency / config.width);
      float ny = y * (config.frequency / config.height);

      heightmap.data[i] = noise.octaveNoise(nx, ny, config.octaves);
    });
    t.next("generation");
    normalize(heightmap);
    t.next("normalization");
    t.total();
    return heightmap;
  }

  HeightMap generate_heightmap_par_tiled(const PerlinNoise &noise,
                                         const HeightmapConfig &config) {
    parlay::internal::timer t(std::string("tiled"));
    HeightMap heightmap(config.width, config.height);
    parlay::parallel_for(0, heightmap.data.size(),
                         [&](size_t i) { heightmap.data[i] = 0.0f; });

    float freq_x = static_cast<float>(config.frequency / config.width);
    float freq_y = static_cast<float>(config.frequency / config.height);

    noise.octaveNoiseTiled(heightmap.data.data(), config.width, config.height,
                           config.octaves, freq_x, freq_y);
    t.next("generation");
    normalize(heightmap);
    t.next("normalization");
    t.total();
    return heightmap;
  }

  HeightMap generate_heightmap_cuda(const PerlinNoiseCuda &noise,
                                    const HeightmapConfig &config) {
    HeightMap heightmap(config.width, config.height);
    heightmap.data = noise.generate_normalized_heightmap(
        config.octaves, config.frequency,
        glm::vec2(config.width, config.height));
    return heightmap;
  }

  HeightMap generate_heightmap_hybrid(PerlinNoiseHybrid &noise,
                                      const HeightmapConfig &config) {
    HeightMap heightmap(config.width, config.height);
    heightmap.data = noise.generate_normalized_heightmap(
        config.octaves, config.frequency,
        glm::vec2(config.width, config.height));
    return heightmap;
  }

private:
  void normalize_seq(HeightMap &heightmap) {
    auto minmax =
        std::minmax_element(heightmap.data.begin(), heightmap.data.end());
    float min_val = *minmax.first;
    float max_val = *minmax.second;
    float range = max_val - min_val;

    if (range <= 0.00001)
      return;

    for (size_t i = 0; i < heightmap.data.size(); i++) {
      heightmap.data[i] = (heightmap.data[i] - min_val) / range;
    }
  }

  void normalize(HeightMap &heightmap) {
    auto minmax = parlay::minmax_element(heightmap.data);
    float min_val = *minmax.first;
    float max_val = *minmax.second;
    float range = max_val - min_val;

    if (range <= 0.00001)
      return;

    parlay::parallel_for(0, heightmap.data.size(), [&](size_t i) {
      heightmap.data[i] = (heightmap.data[i] - min_val) / range;
    });
  }
};