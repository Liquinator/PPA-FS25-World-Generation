#pragma once

#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include <algorithm>
#include <glm/glm.hpp>
#include <random>

#if defined(__aarch64__) && !defined(__CUDACC__)
#include <arm_neon.h>
#endif

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
  // arithmetic (no gathers)
  void octaveNoiseTiled(float *__restrict__ data, int width, int height,
                        int octaves, float freq_x, float freq_y,
                        float persistence = 0.5f) const {
    static constexpr float GX[16] = {1, -1, 1, -1, 1, -1, 1,  -1,
                                     0, 0,  0, 0,  1, 0,  -1, 0};
    static constexpr float GY[16] = {1, 1,  -1, -1, 0, 0,  0, 0,
                                     1, -1, 1,  -1, 1, -1, 1, -1};

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
          if (px_end > width)
            px_end = width;

          float base_frac = px * sx - cell_x;
          int count = px_end - px;

#if defined(__aarch64__) && !defined(__CUDACC__)
          {
            const float32x4_t v_cx_aa = vdupq_n_f32(cx_aa);
            const float32x4_t v_cx_ab = vdupq_n_f32(cx_ab);
            const float32x4_t v_cx_ba = vdupq_n_f32(cx_ba);
            const float32x4_t v_cx_bb = vdupq_n_f32(cx_bb);
            const float32x4_t v_gy_aa = vdupq_n_f32(gy_aa);
            const float32x4_t v_gy_ab = vdupq_n_f32(gy_ab);
            const float32x4_t v_gy_ba = vdupq_n_f32(gy_ba);
            const float32x4_t v_gy_bb = vdupq_n_f32(gy_bb);
            const float32x4_t v_v = vdupq_n_f32(v);
            const float32x4_t v_amp = vdupq_n_f32(amplitude);
            const float32x4_t v_one = vdupq_n_f32(1.0f);
            const float32x4_t v_six = vdupq_n_f32(6.0f);
            const float32x4_t v_neg15 = vdupq_n_f32(-15.0f);
            const float32x4_t v_ten = vdupq_n_f32(10.0f);
            const float32x4_t v_step4 = vdupq_n_f32(4.0f * sx);

            float init[4] = {base_frac, base_frac + sx, base_frac + 2.0f * sx,
                             base_frac + 3.0f * sx};
            float32x4_t v_frac = vld1q_f32(init);

            int i = 0;
            for (; i + 4 <= count; i += 4) {
              float32x4_t fm1 = vsubq_f32(v_frac, v_one);

              // fade(t) = t^3 * (t * (t * 6 - 15) + 10)
              float32x4_t t2 = vmulq_f32(v_frac, v_frac);
              float32x4_t t3 = vmulq_f32(t2, v_frac);
              float32x4_t poly = vfmaq_f32(v_neg15, v_frac, v_six);
              poly = vfmaq_f32(v_ten, v_frac, poly);
              float32x4_t u = vmulq_f32(t3, poly);

              // Dot products at 4 corners
              float32x4_t d_aa = vfmaq_f32(v_gy_aa, v_cx_aa, v_frac);
              float32x4_t d_ab = vfmaq_f32(v_gy_ab, v_cx_ab, v_frac);
              float32x4_t d_ba = vfmaq_f32(v_gy_ba, v_cx_ba, fm1);
              float32x4_t d_bb = vfmaq_f32(v_gy_bb, v_cx_bb, fm1);

              // Bilinear interpolation
              float32x4_t top = vfmaq_f32(d_aa, u, vsubq_f32(d_ba, d_aa));
              float32x4_t bot = vfmaq_f32(d_ab, u, vsubq_f32(d_bb, d_ab));
              float32x4_t interp = vfmaq_f32(top, v_v, vsubq_f32(bot, top));

              // Accumulate: row[px+i] += interp * amplitude
              float32x4_t prev = vld1q_f32(row + px + i);
              vst1q_f32(row + px + i, vfmaq_f32(prev, interp, v_amp));

              v_frac = vaddq_f32(v_frac, v_step4);
            }

            // Scalar tail for remaining 0-3 pixels
            for (; i < count; ++i) {
              float frac_x = base_frac + i * sx;
              float fm1 = frac_x - 1.0f;
              float u = frac_x * frac_x * frac_x *
                        (frac_x * (frac_x * 6.0f - 15.0f) + 10.0f);
              float top =
                  (cx_aa * frac_x + gy_aa) +
                  u * ((cx_ba * fm1 + gy_ba) - (cx_aa * frac_x + gy_aa));
              float bot =
                  (cx_ab * frac_x + gy_ab) +
                  u * ((cx_bb * fm1 + gy_bb) - (cx_ab * frac_x + gy_ab));
              row[px + i] += (top + v * (bot - top)) * amplitude;
            }
          }
#else
          for (int i = 0; i < count; ++i) {
            float frac_x = base_frac + i * sx;
            float fm1 = frac_x - 1.0f;
            float u = frac_x * frac_x * frac_x *
                      (frac_x * (frac_x * 6.0f - 15.0f) + 10.0f);

            float top = (cx_aa * frac_x + gy_aa) +
                        u * ((cx_ba * fm1 + gy_ba) - (cx_aa * frac_x + gy_aa));
            float bot = (cx_ab * frac_x + gy_ab) +
                        u * ((cx_bb * fm1 + gy_bb) - (cx_ab * frac_x + gy_ab));

            row[px + i] += (top + v * (bot - top)) * amplitude;
          }
#endif

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
