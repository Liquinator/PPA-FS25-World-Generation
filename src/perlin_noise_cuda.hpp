#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <memory>

class PerlinNoiseCuda {
 public:
  PerlinNoiseCuda(unsigned int seed);
  ~PerlinNoiseCuda();

  std::vector<float> generate_normalized_heightmap(int32_t octaves,
                                                   float frequency, glm::vec2);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};