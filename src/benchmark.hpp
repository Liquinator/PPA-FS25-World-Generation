#include <chrono>
#include <glm/glm.hpp>
#include <iostream>
#include <numeric>
#include <vector>

#include "cmd_parser.hpp"
#include "heightmap_gen.hpp"
#include "tree_gen.hpp"

double WARMUP_TIME_THRESHOLD = 3.0;
int BENCHMARK_QUANTITY = 10;

bool test_correctness(const std::vector<std::vector<double>>& a,
                      const std::vector<std::vector<double>>& b,
                      double delta = 1e-7) {
  if (a.size() != b.size()) return false;

  for (size_t i = 0; i < a.size(); i++) {
    if (a[i].size() != b[i].size()) return false;

    for (size_t j = 0; j < a[i].size(); j++) {
      if (std::abs(a[i][j] - b[i][j]) > delta) return false;
    }
  }
  return true;
}

bool compare_tree_locations(const std::vector<glm::vec2>& a,
                            const std::vector<glm::vec2>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

void warmup() {
  std::cout << "Perlin generation only:" << std::endl;
  std::cout << "Warming up for 3 seconds..." << std::endl;

  auto threshold = std::chrono::duration<double>(WARMUP_TIME_THRESHOLD);
  auto warmup_start = std::chrono::high_resolution_clock::now();

  while (std::chrono::high_resolution_clock::now() - warmup_start < threshold) {
  }
}

inline std::vector<glm::vec2> benchmark_tree_generation(
    const bool use_parallel, const CMDSettings& settings,
    const TreePlacementConfig& treeConfig = TreePlacementConfig{},
    const HeightmapConfig& heightmapConfig = HeightmapConfig{}) {
  PerlinNoisePar* perlinNoise;
  std::vector<std::vector<double>> heightmap;
  heightmap.assign(settings.dimension,
                   std::vector<double>(settings.dimension, 0.0));

  perlinNoise->generate_normalized_heightmap(
      heightmapConfig.octaves, heightmapConfig.frequency,
      glm::vec2(settings.dimension, settings.dimension), heightmap);

  std::vector<glm::vec2> tree_placement =
      place_trees(heightmap, use_parallel, treeConfig, heightmapConfig);

  return tree_placement;
};

inline std::vector<std::vector<double>> benchmark_heightmap(
    const bool use_parallel, CMDSettings& settings,
    const HeightmapConfig& heightmap_config = HeightmapConfig{}) {
  std::vector<std::chrono::duration<double>> times;
  for (int r = 0; r < BENCHMARK_QUANTITY; ++r) {
    auto start = std::chrono::high_resolution_clock::now();
    generate_heightmap(settings.dimension, use_parallel, heightmap_config);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = stop - start;
    std::cout << "Time " << diff.count() << std::endl;
    times.push_back(diff);
  }

  std::chrono::duration<double> total_time = std::accumulate(
      times.begin(), times.end(), std::chrono::duration<double>(0.0));
  std::cout << "average " << (total_time / times.size()).count() << std::endl;
  return generate_heightmap(settings.dimension, use_parallel, heightmap_config);
}

void benchmark(CMDSettings& settings) {
  switch (settings.mode) {
    case GenerationMode::BOTH: {
      auto seq_res_height =
          benchmark_heightmap(false, settings, HeightmapConfig{});
      auto par_res_height =
          benchmark_heightmap(true, settings, HeightmapConfig{});

      auto seq_res_tree = benchmark_tree_generation(
          false, settings, TreePlacementConfig{}, HeightmapConfig{});
      auto par_res_tree = benchmark_tree_generation(
          true, settings, TreePlacementConfig{}, HeightmapConfig{});

      if (test_correctness(seq_res_height, par_res_height)) {
        std::cout << "Sequential and parallel heightmap versions are equal!"
                  << std::endl;
      } else {
        std::cout << "Sequential and parallel heightmap versions are NOT equal!"
                  << std::endl;
      }

      if (compare_tree_locations(seq_res_tree, par_res_tree)) {
        std::cout << "Sequential and parallel tree loactions are equal!"
                  << std::endl;
      } else {
        std::cout << "Sequential and parallel tree loactions are NOT equal!"
                  << std::endl;
      }
      break;
    }

    case GenerationMode::PARALLEL: {
      benchmark_heightmap(true, settings, HeightmapConfig{});
      benchmark_tree_generation(true, settings, TreePlacementConfig{},
                                HeightmapConfig{});
      break;
    }
    case GenerationMode::SEQUENTIAL: {
      benchmark_heightmap(false, settings, HeightmapConfig{});
      benchmark_tree_generation(false, settings, TreePlacementConfig{},
                                HeightmapConfig{});
      break;
    }

    default:
      break;
  }
}
