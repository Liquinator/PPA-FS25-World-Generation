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

  // Tiled: octave-first, row-parallel, cell-segmented inner loop.
  // Precomputes corner hashes once per cell so the inner loop is pure
  // arithmetic (no gathers) and auto-vectorizes with NEON/SVE.
  void octaveNoiseTiled(float *data, int width, int height, int octaves,
                        float freq_x, float freq_y,
                        float persistence = 0.5f) const {
    static constexpr float GX[16] = {1,  -1, 1, -1, 1, -1, 1,  -1,
                                     0,  0,  0, 0,  1, 0,  -1, 0};
    static constexpr float GY[16] = {1, 1, -1, -1, 0, 0, 0, 0,
                                     1, -1, 1, -1, 1, -1, 1, -1};

    const int *perm = permutation.data();
    float amplitude = 1.0f;

    for (int oct = 0; oct < octaves; ++oct) {
      float scale = static_cast<float>(1 << oct);
      float sx = freq_x * scale;
      float sy = freq_y * scale;

      parlay::parallel_for(0, (size_t)height, [&](size_t py) {
        float ny = py * sy;
        float fy = std::floor(ny);
        int cell_y = static_cast<int>(fy);
        float frac_y = ny - fy;
        int Y = cell_y & 255;
        float v = fade(frac_y);

        float *row = data + py * width;
        int px = 0;

        while (px < width) {
          float nx0 = px * sx;
          float fx0 = std::floor(nx0);
          int cell_x = static_cast<int>(fx0);
          int X = cell_x & 255;

          // 4 corner hashes — computed ONCE per cell
          int h_aa = perm[perm[X] + Y] & 15;
          int h_ab = perm[perm[X] + Y + 1] & 15;
          int h_ba = perm[perm[X + 1] + Y] & 15;
          int h_bb = perm[perm[X + 1] + Y + 1] & 15;

          // Precomputed gradient coefficients (constant for entire inner loop)
          float cx_aa = GX[h_aa], cy_aa = GY[h_aa];
          float cx_ab = GX[h_ab], cy_ab = GY[h_ab];
          float cx_ba = GX[h_ba], cy_ba = GY[h_ba];
          float cx_bb = GX[h_bb], cy_bb = GY[h_bb];

          // Y-gradient components hoisted out of inner loop
          float gy_aa = cy_aa * frac_y;
          float gy_ab = cy_ab * (frac_y - 1.0f);
          float gy_ba = cy_ba * frac_y;
          float gy_bb = cy_bb * (frac_y - 1.0f);

          // End of this cell in pixel space
          int px_end = static_cast<int>((cell_x + 1) / sx) + 1;
          if (px_end > width) px_end = width;

          // Inner loop: pure FMA, no gathers, SIMD-friendly
          for (int p = px; p < px_end; ++p) {
            float frac_x = p * sx - cell_x;
            float u = fade(frac_x);

            float corner_aa = cx_aa * frac_x + gy_aa;
            float corner_ab = cx_ab * frac_x + gy_ab;
            float corner_ba = cx_ba * (frac_x - 1.0f) + gy_ba;
            float corner_bb = cx_bb * (frac_x - 1.0f) + gy_bb;

            float top = corner_aa + u * (corner_ba - corner_aa);
            float bot = corner_ab + u * (corner_bb - corner_ab);
            float val = top + v * (bot - top);

            row[p] += val * amplitude;
          }

          px = px_end;
        }
      });

      amplitude *= persistence;
    }
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
