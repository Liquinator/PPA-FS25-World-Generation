#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <glm/glm.hpp>
#include <random>
#include <vector>

#include "perlin_common.cuh"
#include "perlin_noise_cpu.hpp"
#include "perlin_noise_hybrid.hpp"

struct PerlinNoiseHybrid::Impl {
  const int PERM_SIZE = 512;
  int* unified_perm = nullptr;
  const size_t gen_split_point;
  const size_t norm_split_point;

  cudaStream_t gpu_stream;

  Impl(unsigned int seed, const size_t _gen_split_point,
       const size_t _norm_split_point)
      : gen_split_point(_gen_split_point), norm_split_point(_norm_split_point) {
    cudaMallocManaged(&unified_perm, PERM_SIZE * sizeof(int));
    std::vector<int> p(256);
    std::iota(p.begin(), p.end(), 0);
    std::default_random_engine engine(seed);
    std::shuffle(p.begin(), p.end(), engine);

    for (int i = 0; i < 512; i++) {
      unified_perm[i] = p[i % 256];
    }

    int device_id;
    cudaGetDevice(&device_id);
    cudaMemLocation location{cudaMemLocationTypeDevice, device_id};
    cudaMemPrefetchAsync(unified_perm, PERM_SIZE * sizeof(int), location, 0);
  }

  ~Impl() {
    if (unified_perm) cudaFree(unified_perm);
  }
};

PerlinNoiseHybrid::PerlinNoiseHybrid(unsigned int seed, size_t gen_split_point,
                                     size_t norm_split_point)
    : impl(std::make_unique<Impl>(seed, gen_split_point, norm_split_point)) {
  cudaStreamCreate(&impl->gpu_stream);
}

PerlinNoiseHybrid::~PerlinNoiseHybrid() {
  if (heightmap) cudaFree(heightmap);
  cudaStreamDestroy(impl->gpu_stream);
};

void PerlinNoiseHybrid::generate_heightmap(int32_t octaves, float frequency,
                                           glm::vec2 dim) {
  world_size = (size_t)(dim.x * dim.y);
  cudaMallocManaged(&heightmap, world_size * sizeof(float));
  thrust::device_ptr<float> dev_ptr(heightmap);

  float freq_x = (float)(frequency / dim.x);
  float freq_y = (float)(frequency / dim.y);

  PerlinFunctor perlinFunctor(impl->unified_perm, (int)dim.x, (int)dim.y,
                              octaves, freq_x, freq_y);

  thrust::transform(
      thrust::cuda::par.on(impl->gpu_stream),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(impl->gen_split_point), dev_ptr,
      perlinFunctor);

  parlay::parallel_for(impl->gen_split_point, world_size,
                       [&](size_t i) { heightmap[i] = perlinFunctor(i); });

  cudaStreamSynchronize(impl->gpu_stream);
}

float* PerlinNoiseHybrid::generate_normalized_heightmap(int32_t octaves,
                                                        float frequency,
                                                        glm::vec2 dim) {
  generate_heightmap(octaves, frequency, dim);

  auto cuda_minmax =
      thrust::minmax_element(thrust::cuda::par.on(impl->gpu_stream), heightmap,
                             heightmap + world_size);

  float gpu_min_val = *cuda_minmax.first;
  float gpu_max_val = *cuda_minmax.second;
  thrust::device_ptr<float> dev_ptr(heightmap);

  thrust::transform(
      thrust::cuda::par.on(impl->gpu_stream),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(impl->norm_split_point), dev_ptr,
      NormalizeFunctor(gpu_min_val, gpu_max_val));

  cudaStreamSynchronize(impl->gpu_stream);

  return heightmap;
}