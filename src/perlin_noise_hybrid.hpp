#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

class PerlinNoiseHybrid {
 public:
  PerlinNoiseHybrid(unsigned int seed) {};

  std::vector<float> generate_normalized_heightmap(int32_t octaves,
                                                   float frequency,
                                                   glm::vec2 dim) const {
    split_point = dim.length();
  };

 private:
  const size_t split_point;
};