#pragma once

#include <glm/glm.hpp>
#include <random>
#include <vector>

#include "parlay/primitives.h"

class PerlinNoise {
 public:
  PerlinNoise(const unsigned int seed) {
    std::vector<int> p(256);
    std::iota(p.begin(), p.end(), 0);

    std::default_random_engine engine(seed);
    std::shuffle(p.begin(), p.end(), engine);

    p.insert(p.end(), p.begin(), p.end());

    permutation = p;
  };
  virtual std::vector<std::vector<double>> generate_normalized_heightmap(
      int32_t octaves, double frequency, glm::vec2 dim,
      std::vector<std::vector<double>>& heightmap) = 0;

  virtual parlay::sequence<double> generate_heightmap(int32_t octaves,
                                                      double frequency,
                                                      glm::vec2 dim) = 0;

 protected:
  std::vector<int> permutation;
  double fade(const double t) const noexcept {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }
  double lerp(const double t, const double a, const double b) const noexcept {
    return (a + (b - a) * t);
  }

  double grad(const int hash, const double x, const double y) const noexcept {
    const int h = hash & 15;
    const double u = (h < 8) ? x : y;
    const double v = (h < 4) ? y : (h == 12 || h == 14 ? x : 0.0);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
  }
};

class PerlinNoiseSeq : public PerlinNoise {
 public:
  PerlinNoiseSeq(const unsigned int seed) : PerlinNoise(seed) {}

  parlay::sequence<double> generate_heightmap(
      [[maybe_unused]] int32_t octaves, [[maybe_unused]] double frequency,
      [[maybe_unused]] glm::vec2 dim) override {};

  inline virtual std::vector<std::vector<double>> generate_normalized_heightmap(
      int32_t octaves, double frequency, glm::vec2 dim,
      std::vector<std::vector<double>>& heightmap) override {
    float min, max = 0.0;
    for (int x = 0; x < dim.x; x++) {
      for (int y = 0; y < dim.y; y++) {
        double noise_res = octaveNoise(x * (frequency / dim.x),
                                       y * (frequency / dim.y), octaves);

        heightmap[x][y] = noise_res;
        if (noise_res > max) {
          max = noise_res;
        }
        if (noise_res < min) {
          min = noise_res;
        }
      }
    }

    for (int x = 0; x < dim.x; x++) {
      for (int y = 0; y < dim.y; y++) {
        heightmap[x][y] = (heightmap[x][y] - min) / (max - min);
      }
    }

    return heightmap;
  };

 private:
  double noise(double x, double y) const {
    const double _x = std::floor(x);
    const double _y = std::floor(y);
    int X = static_cast<int>(_x) & 255;
    int Y = static_cast<int>(_y) & 255;

    x -= std::floor(x);
    y -= std::floor(y);

    double u = fade(x);
    double v = fade(y);

    int aa = permutation[permutation[X] + Y];
    int ab = permutation[permutation[X] + Y + 1];
    int ba = permutation[permutation[X + 1] + Y];
    int bb = permutation[permutation[X + 1] + Y + 1];

    double corner_aa = grad(aa, x, y);
    double corner_ab = grad(ab, x, y - 1);
    double corner_ba = grad(ba, x - 1, y);
    double corner_bb = grad(bb, x - 1, y - 1);

    double top_edge = lerp(u, corner_aa, corner_ba);
    double bottom_edge = lerp(u, corner_ab, corner_bb);

    return lerp(v, top_edge, bottom_edge);
  };

  double octaveNoise(double x, double y, const int octaves,
                     const double persistence = 0.5) {
    double total = 0.0;
    double amplitude = 1.0;

    for (int i = 0; i < octaves; ++i) {
      total += noise(x, y) * amplitude;
      x *= 2;
      y *= 2;
      amplitude *= persistence;
    }

    return total;
  }
};

class PerlinNoisePar : public PerlinNoise {
 public:
  PerlinNoisePar(const unsigned int seed) : PerlinNoise(seed) {}

  inline parlay::sequence<double> generate_heightmap(int32_t octaves,
                                                     double frequency,
                                                     glm::vec2 dim) override {
    auto results = parlay::tabulate((int)dim.x * (int)dim.y, [&](size_t idx) {
      int x = idx / (int)dim.y;
      int y = idx % (int)dim.y;
      return octaveNoise(x * (frequency / dim.x), y * (frequency / dim.y),
                         octaves);
    });

    return results;
  }

  inline virtual std::vector<std::vector<double>> generate_normalized_heightmap(
      int32_t octaves, double frequency, glm::vec2 dim,
      std::vector<std::vector<double>>& heightmap) override {
    auto results = generate_heightmap(octaves, frequency, dim);
    normalize_heightmap(heightmap, dim, results);

    return heightmap;
  }

 private:
  void normalize_heightmap(std::vector<std::vector<double>>& heightmap,
                           glm::vec2 dim, parlay::sequence<double>& results) {
    auto minmax = parlay::minmax_element(results);
    double min_val = *minmax.first;
    double max_val = *minmax.second;

    parlay::parallel_for(0, (int)dim.x, [&](int x) {
      for (int y = 0; y < dim.y; y++) {
        heightmap[x][y] = results[x * (int)dim.y + y];
        heightmap[x][y] = (heightmap[x][y] - min_val) / (max_val - min_val);
      }
    });
  };

  double noise(double x, double y) const {
    const double _x = std::floor(x);
    const double _y = std::floor(y);
    int X = static_cast<int>(_x) & 255;
    int Y = static_cast<int>(_y) & 255;

    x -= std::floor(x);
    y -= std::floor(y);

    double u = fade(x);
    double v = fade(y);

    int aa = permutation[permutation[X] + Y];
    int ab = permutation[permutation[X] + Y + 1];
    int ba = permutation[permutation[X + 1] + Y];
    int bb = permutation[permutation[X + 1] + Y + 1];

    double corner_aa = grad(aa, x, y);
    double corner_ab = grad(ab, x, y - 1);
    double corner_ba = grad(ba, x - 1, y);
    double corner_bb = grad(bb, x - 1, y - 1);

    double top_edge = lerp(u, corner_aa, corner_ba);
    double bottom_edge = lerp(u, corner_ab, corner_bb);

    return lerp(v, top_edge, bottom_edge);
  };

  double octaveNoise(double x, double y, const int octaves,
                     const double persistence = 0.5) {
    double total = 0.0;
    double amplitude = 1.0;

    for (int i = 0; i < octaves; ++i) {
      total += noise(x, y) * amplitude;
      x *= 2;
      y *= 2;
      amplitude *= persistence;
    }

    return total;
  }
};
