#pragma once

#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include <algorithm>
#include <glm/glm.hpp>
#include <random>

struct HeightMap {
  int width;
  int height;
  parlay::sequence<float> data;

  HeightMap(int _width, int _height)
      : width(_width), height(_height),
        data(parlay::sequence<float>::uninitialized(_width * _height)) {}
  float &at(int x, int y) { return data[y * width + x]; }
};

class PerlinNoise {
public:
  PerlinNoise(const unsigned int seed) : permutation(512) {
    auto p = parlay::tabulate<int>(256, [](size_t i) { return (int)i; });
    std::default_random_engine engine(seed);
    std::shuffle(p.begin(), p.end(), engine);
    for (int i = 0; i < 512; i++) {
      permutation[i] = p[i % 256];
    }
  };

  float octaveNoise(float x, float y, const int octaves,
                    const float persistence = 0.5) const {
    float total = 0.0;
    float amplitude = 1.0;

    for (int i = 0; i < octaves; ++i) {
      total += noise(x, y) * amplitude;
      x *= 2;
      y *= 2;
      amplitude *= persistence;
    }

    return total;
  }

private:
  parlay::sequence<int> permutation;
  float fade(const float t) const noexcept {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }
  float lerp(const float t, const float a, const float b) const noexcept {
    return (a + (b - a) * t);
  }

  float grad(const int hash, const float x, const float y) const noexcept {
    const int h = hash & 15;
    const float u = (h < 8) ? x : y;
    const float v = (h < 4) ? y : (h == 12 || h == 14 ? x : 0.0);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
  }

  float noise(float x, float y) const {
    const float fx = std::floor(x);
    const float fy = std::floor(y);
    int X = static_cast<int>(fx) & 255;
    int Y = static_cast<int>(fy) & 255;

    x -= fx;
    y -= fy;

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
};
