#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <glm/glm.hpp>
#include <parlay/sequence.h>
#include <random>

#include "perlin_common.cuh"
#include "perlin_noise_cpu.hpp"
#include "perlin_noise_hybrid.hpp"

struct PerlinNoiseHybrid::Impl {
  static constexpr int PERM_SIZE = 512;
  parlay::sequence<int> unified_perm;
  const float gen_split_point_percent;
  const float norm_split_point_percent;

  cudaStream_t gpu_stream;

  Impl(unsigned int seed, const float _gen_split_point_percent,
       const float _norm_split_point_percent)
      : unified_perm(PERM_SIZE),
        gen_split_point_percent(_gen_split_point_percent),
        norm_split_point_percent(_norm_split_point_percent) {
    auto p = parlay::tabulate<int>(256, [](size_t i) { return (int)i; });
    std::default_random_engine engine(seed);
    std::shuffle(p.begin(), p.end(), engine);

    for (int i = 0; i < 512; i++) {
      unified_perm[i] = p[i % 256];
    }
  }
};

PerlinNoiseHybrid::PerlinNoiseHybrid(unsigned int seed,
                                     const float gen_split_point_percent,
                                     const float norm_split_point_percent)
    : impl(std::make_unique<Impl>(seed, gen_split_point_percent,
                                  norm_split_point_percent)) {
  cudaStreamCreate(&impl->gpu_stream);
}

PerlinNoiseHybrid::~PerlinNoiseHybrid() {
  cudaStreamDestroy(impl->gpu_stream);
};

void PerlinNoiseHybrid::generate_heightmap(int32_t octaves, float frequency,
                                           glm::vec2 dim) {
  world_size = (size_t)(dim.x * dim.y);
  size_t gen_split_point = impl->gen_split_point_percent * world_size;

  heightmap = parlay::sequence<float>(world_size);
  auto gpu_part = heightmap.cut(0, gen_split_point);
  auto cpu_part = heightmap.cut(gen_split_point, world_size);

  float freq_x = (float)(frequency / dim.x);
  float freq_y = (float)(frequency / dim.y);

  PerlinFunctor perlinFunctor(impl->unified_perm.data(), (int)dim.x, (int)dim.y,
                              octaves, freq_x, freq_y);
  parlay::par_do(
      [&]() {
        thrust::transform(
            thrust::cuda::par.on(impl->gpu_stream),
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(gen_split_point),
            gpu_part.begin(), perlinFunctor);
      },
      [&]() {
        parlay::parallel_for(0, cpu_part.size(), [&](size_t i) {
          cpu_part[i] = perlinFunctor(gen_split_point + i);
        });
      });
  cudaStreamSynchronize(impl->gpu_stream);
}

parlay::sequence<float> PerlinNoiseHybrid::generate_normalized_heightmap(
    int32_t octaves, float frequency, glm::vec2 dim) {
  generate_heightmap(octaves, frequency, dim);
  size_t norm_split_point = impl->norm_split_point_percent * world_size;
  auto gpu_part = heightmap.cut(0, norm_split_point);
  auto cpu_part = heightmap.cut(norm_split_point, world_size);

  auto cuda_minmax =
      thrust::minmax_element(thrust::cuda::par.on(impl->gpu_stream),
                             heightmap.begin(), heightmap.end());

  float gpu_min_val = *cuda_minmax.first;
  float gpu_max_val = *cuda_minmax.second;

  float range = gpu_max_val - gpu_min_val;

  parlay::par_do(
      [&]() {
        thrust::transform(thrust::cuda::par.on(impl->gpu_stream),
                          gpu_part.begin(), gpu_part.end(), gpu_part.begin(),
                          NormalizeFunctor(gpu_min_val, gpu_max_val));
      },
      [&]() {
        parlay::parallel_for(0, cpu_part.size(), [&](size_t i) {
          cpu_part[i] =
              (range == 0.0f) ? 0.0f : (cpu_part[i] - gpu_min_val) / range;
        });
      });
  cudaStreamSynchronize(impl->gpu_stream);

  return std::move(heightmap);
}