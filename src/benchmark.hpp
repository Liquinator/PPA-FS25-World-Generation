#include <chrono>
#include <glm/glm.hpp>
#include <iostream>
#include <numeric>
#include <vector>

#include "cmd_parser.hpp"
#include "heightmap_gen.hpp"
#include "perlin_noise_cuda.hpp"
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

template <typename F>
inline HeightMap benchmark_heightmap(const std::string& label, F&& generate) {
  std::cout << "Starting" << label << "heightmap generation benchmark"
            << std::endl;
  std::vector<std::chrono::duration<double>> times;

  for (int r = 0; r < BENCHMARK_QUANTITY; ++r) {
    auto start = std::chrono::high_resolution_clock::now();
    generate();

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = stop - start;
    std::cout << "Time " << diff.count() << std::endl;
    times.push_back(diff);
  }

  std::chrono::duration<double> total_time = std::accumulate(
      times.begin(), times.end(), std::chrono::duration<double>(0.0));
  std::cout << "average " << (total_time / times.size()).count() << std::endl;
  return generate();
}

inline HeightMap benchmark_heightmap_seq(MapGenerator& mapGenerator,
                                         const PerlinNoise& perlinNoise,
                                         const HeightmapConfig& config) {
  return benchmark_heightmap("sequential", [&] {
    return mapGenerator.generate_heightmap_seq(perlinNoise, config);
  });
}

inline HeightMap benchmark_heightmap_par(MapGenerator& mapGenerator,
                                         const PerlinNoise& perlinNoise,
                                         const HeightmapConfig& config) {
  return benchmark_heightmap("parallel", [&] {
    return mapGenerator.generate_heightmap_par(perlinNoise, config);
  });
}
/*
inline std::vector<glm::vec2> benchmark_tree_generation(
    const bool use_parallel, CMDSettings& settings,
    const TreePlacementConfig& treeConfig = TreePlacementConfig{},
    const HeightmapConfig& heightmapConfig = HeightmapConfig{}) {
  PerlinNoise* perlinNoise =
      get_perlin_noise_generator(false, heightmapConfig.seed);
  std::vector<std::vector<double>> heightmap;
  heightmap.assign(settings.dimension,
                   std::vector<double>(settings.dimension, 0.0));

  perlinNoise->generate_normalized_heightmap(
      heightmapConfig.octaves, heightmapConfig.frequency,
      glm::vec2(settings.dimension, settings.dimension), heightmap);

  if (use_parallel)
    std::cout << "Starting parallel tree placement benchmark" << std::endl;
  if (!use_parallel)
    std::cout << "Starting sequential tree placement benchmark" << std::endl;

  std::vector<std::chrono::duration<double>> times;
  for (int r = 0; r < BENCHMARK_QUANTITY; ++r) {
    auto start = std::chrono::high_resolution_clock::now();
    place_trees(heightmap, use_parallel, treeConfig, heightmapConfig);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = stop - start;
    std::cout << "Time " << diff.count() << std::endl;
    times.push_back(diff);
  }

  std::chrono::duration<double> total_time = std::accumulate(
      times.begin(), times.end(), std::chrono::duration<double>(0.0));
  std::cout << "average " << (total_time / times.size()).count() << std::endl;
  std::vector<glm::vec2> tree_placement =
      place_trees(heightmap, use_parallel, treeConfig, heightmapConfig);

  return tree_placement;
};
*/
void benchmark(CMDSettings& settings) {
  PerlinNoise perlinNoise(settings.seed);
  HeightmapConfig config{settings.dimension, settings.dimension};
  MapGenerator mapGenerator;

  switch (settings.mode) {
    case GenerationMode::SEQUENTIAL: {
      auto seq_res_height =
          benchmark_heightmap_seq(mapGenerator, perlinNoise, config);
    }
    case GenerationMode::PARALLEL: {
      auto par_res_height =
          benchmark_heightmap_par(mapGenerator, perlinNoise, config);
    }
    default:
      break;
  }
}
