#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <glm/glm.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "perlin_common.cuh"
#include "perlin_noise_cuda.hpp"

thrust::device_vector<float> generate_heightmap(
    const thrust::device_vector<int>& device_permutation, int32_t octaves,
    float frequency, glm::vec2 dim) {
  size_t world_size = (size_t)(dim.x * dim.y);
  thrust::device_vector<float> device_results(world_size);

  float freq_x = (float)(frequency / dim.x);
  float freq_y = (float)(frequency / dim.y);

  PerlinFunctor perlinFunctor(
      thrust::raw_pointer_cast(device_permutation.data()), (int)dim.x,
      (int)dim.y, octaves, freq_x, freq_y);

  thrust::transform(thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(world_size),
                    device_results.begin(), perlinFunctor);

  return device_results;
}

struct PerlinNoiseCuda::Impl {
  thrust::device_vector<int> device_permutation;

  Impl(unsigned int seed) {
    std::vector<int> p(256);
    std::iota(p.begin(), p.end(), 0);

    std::default_random_engine engine(seed);
    std::shuffle(p.begin(), p.end(), engine);

    p.insert(p.end(), p.begin(), p.end());
    device_permutation = p;
  }
};

PerlinNoiseCuda::PerlinNoiseCuda(unsigned int seed)
    : impl(std::make_unique<Impl>(seed)) {}

PerlinNoiseCuda::~PerlinNoiseCuda() = default;

std::vector<float> PerlinNoiseCuda::generate_normalized_heightmap(
    int32_t octaves, float frequency, glm::vec2 dim) const {
  thrust::device_vector<float> device_results =
      generate_heightmap(impl->device_permutation, octaves, frequency, dim);
  auto result =
      thrust::minmax_element(device_results.begin(), device_results.end());

  float min_val = *result.first;
  float max_val = *result.second;

  thrust::transform(device_results.begin(), device_results.end(),
                    device_results.begin(), NormalizeFunctor(min_val, max_val));
  std::vector<float> heightmap(device_results.size());
  thrust::copy(device_results.begin(), device_results.end(), heightmap.begin());

  return heightmap;
}
