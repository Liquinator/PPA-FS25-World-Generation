#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

#include "perlin_noise_cuda.hpp"

#include <algorithm>
#include <glm/glm.hpp>
#include <numeric>
#include <random>
#include <vector>

struct PerlinFunctor {
  const int* permutation;
  int width;
  int height;
  int octaves;
  float freq_x;
  float freq_y;

  PerlinFunctor(const int* _perm, int _width, int _height, int _octaves,
                float _freq_x, float _freq_y)
      : permutation(_perm),
        width(_width),
        height(_height),
        octaves(_octaves),
        freq_x(_freq_x),
        freq_y(_freq_y) {}

  __device__ float fade(const float t) const {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }

  __device__ float lerp(const float t, const float a, const float b) const {
    return (a + (b - a) * t);
  }

  __device__ float grad(const int hash, const float x,
                        const float y) const noexcept {
    const int h = hash & 15;
    const float u = (h < 8) ? x : y;
    const float v = (h < 4) ? y : (h == 12 || h == 14 ? x : 0.0);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
  }

  __device__ float noise(float x, float y) const {
    int X = ((int)floorf(x)) & 255;
    int Y = ((int)floorf(y)) & 255;

    x -= floorf(x);
    y -= floorf(y);

    float u = fade(x);
    float v = fade(y);

    int aa = permutation[permutation[X] + Y];
    int ab = permutation[permutation[X] + Y + 1];
    int ba = permutation[permutation[X + 1] + Y];
    int bb = permutation[permutation[X + 1] + Y + 1];

    float corner_aa = grad(aa, x, y);
    float corner_ab = grad(ab, x, y - 1);
    float corner_ba = grad(ba, x - 1, y);
    float corner_bb = grad(bb, x - 1, y - 1);

    float top_edge = lerp(u, corner_aa, corner_ba);
    float bottom_edge = lerp(u, corner_ab, corner_bb);

    return lerp(v, top_edge, bottom_edge);
  }

  __device__ float octaveNoise(float x, float y,
                               const float persistence = 0.5) const {
    float total = 0.0;
    float amplitude = 1.0;

    for (int i = 0; i < octaves; ++i) {
      total += noise(x, y) * amplitude;
      x *= 2.0;
      y *= 2.0;
      amplitude *= persistence;
    }

    return total;
  }

  __device__ float operator()(const size_t idx) const {
    int x_idx = idx % height;
    int y_idx = idx / width;

    float x = x_idx * freq_x;
    float y = y_idx * freq_y;

    return octaveNoise(x, y);
  }
};

struct NormalizeFunctor {
  float min_val;
  float range;

  NormalizeFunctor(float _min_val, float _max_val)
      : min_val(_min_val), range(_max_val - _min_val) {}

  __device__ float operator()(const float& val) const {
    if (range == 0.0) return 0.0;
    return (val - min_val) / range;
  }
};

namespace {
  thrust::device_vector<float> generate_heightmap(thrust::device_vector<int>& device_permutation,
    int32_t octaves,
    float frequency,
    glm::vec2 dim) {
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


  std::vector<float> PerlinNoiseCuda::generate_normalized_heightmap(int32_t octaves,
                                                   float frequency,
                                                   glm::vec2 dim) const {
    thrust::device_vector<float> device_results =
        generate_heightmap(impl->device_permutation, octaves, frequency, dim);
    auto result =
        thrust::minmax_element(device_results.begin(), device_results.end());

    float min_val = *result.first;
    float max_val = *result.second;

    thrust::transform(device_results.begin(), device_results.end(),
                      device_results.begin(),
                      NormalizeFunctor(min_val, max_val));
    std::vector<float> heightmap(device_results.size());
    thrust::copy(device_results.begin(), device_results.end(),
                 heightmap.begin());

    // TODO Either reshape to 2D vector or refactor original parlay impl 1D
    // vector
    return heightmap;
  }
