#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <glm/glm.hpp>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <random>

#include "parlay/internal/get_time.h"
#include "perlin_common.cuh"
#include "perlin_noise_cuda.hpp"

struct PerlinNoiseCuda::Impl {
  parlay::sequence<int> permutation;
  cudaStream_t stream;

  Impl(unsigned int seed) : permutation(512) {
    auto p = parlay::tabulate<int>(256, [](size_t i) { return (int)i; });
    std::default_random_engine engine(seed);
    std::shuffle(p.begin(), p.end(), engine);

    for (int i = 0; i < 512; i++) {
      permutation[i] = p[i % 256];
    }

    cudaStreamCreate(&stream);
  }

  ~Impl() { cudaStreamDestroy(stream); }
};

PerlinNoiseCuda::PerlinNoiseCuda(unsigned int seed)
    : impl(std::make_unique<Impl>(seed)) {}

PerlinNoiseCuda::~PerlinNoiseCuda() = default;

parlay::sequence<float> PerlinNoiseCuda::generate_normalized_heightmap(
    int32_t octaves, float frequency, glm::vec2 dim) const {
  parlay::internal::timer t(std::string("cuda"));
  size_t world_size = (size_t)(dim.x * dim.y);
  parlay::sequence<float> heightmap(world_size);

  float freq_x = (float)(frequency / dim.x);
  float freq_y = (float)(frequency / dim.y);

  PerlinFunctor perlinFunctor(impl->permutation.data(), (int)dim.x, (int)dim.y,
                              octaves, freq_x, freq_y);

  thrust::transform(thrust::cuda::par.on(impl->stream),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(world_size),
                    heightmap.begin(), perlinFunctor);
  cudaStreamSynchronize(impl->stream);
  t.next("generation");

  auto result = thrust::minmax_element(thrust::cuda::par.on(impl->stream),
                                       heightmap.begin(), heightmap.end());
  float min_val = *result.first;
  float max_val = *result.second;
  t.next("minmax");

  thrust::transform(thrust::cuda::par.on(impl->stream), heightmap.begin(),
                    heightmap.end(), heightmap.begin(),
                    NormalizeFunctor(min_val, max_val));
  cudaStreamSynchronize(impl->stream);
  t.next("normalization");
  t.total();

  return heightmap;
}
