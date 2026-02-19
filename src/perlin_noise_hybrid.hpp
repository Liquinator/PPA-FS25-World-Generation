#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <parlay/sequence.h>
#include <vector>

class PerlinNoiseHybrid {
 public:
  PerlinNoiseHybrid(unsigned int seed, float gen_split_point_percent,
                    float norm_split_point_percent);
  ~PerlinNoiseHybrid();

  parlay::sequence<float> generate_normalized_heightmap(int32_t octaves,
                                                        float frequency,
                                                        glm::vec2 dim);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;

  parlay::sequence<float> heightmap;
  size_t world_size;

  void generate_heightmap(int32_t octaves, float frequency, glm::vec2 dim);
};