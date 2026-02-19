#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <parlay/sequence.h>

class PerlinNoiseCuda {
 public:
  PerlinNoiseCuda(unsigned int seed);
  ~PerlinNoiseCuda();

  parlay::sequence<float> generate_normalized_heightmap(int32_t octaves,
                                                        float frequency,
                                                        glm::vec2 dim) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};