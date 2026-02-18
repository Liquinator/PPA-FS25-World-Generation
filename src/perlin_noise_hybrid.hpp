#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

class PerlinNoiseHybrid {
 public:
  PerlinNoiseHybrid(unsigned int seed, size_t gen_split_point,
                    size_t norm_split_point);
  ~PerlinNoiseHybrid();

  float* generate_normalized_heightmap(int32_t octaves, float frequency,
                                       glm::vec2 dim);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;

  float* heightmap;
  size_t world_size;

  void generate_heightmap(int32_t octaves, float frequency, glm::vec2 dim);
};