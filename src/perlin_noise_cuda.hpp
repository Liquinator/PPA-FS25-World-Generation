#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

class PerlinNoiseCuda {
 public:
  PerlinNoiseCuda(unsigned int seed);
  ~PerlinNoiseCuda();

  std::vector<float> generate_normalized_heightmap(int32_t octaves,
                                                   float frequency,
                                                   glm::vec2 dim) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};