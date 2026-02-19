#pragma once

#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

enum class GenerationMode { SEQUENTIAL, PARALLEL, TILED, CUDA, HYBRID };

struct CMDSettings {
  int seed{42};
  int dimension{256};
  bool gen_tree{false};
  float hybrid_gen_split{0.95f};
  float hybrid_norm_split{0.95f};
  GenerationMode mode{GenerationMode::PARALLEL};
};

typedef std::function<void(CMDSettings&)> NoArgHandle;
typedef std::function<void(CMDSettings&, const std::string&)> OneArgHandle;

const std::unordered_map<std::string, NoArgHandle> NoArgs{
    {"-p", [](CMDSettings& s) { s.mode = GenerationMode::PARALLEL; }},
    {"-s", [](CMDSettings& s) { s.mode = GenerationMode::SEQUENTIAL; }},
    {"-c", [](CMDSettings& s) { s.mode = GenerationMode::CUDA; }},
    {"-t", [](CMDSettings& s) { s.mode = GenerationMode::TILED; }},
    {"-h", [](CMDSettings& s) { s.mode = GenerationMode::HYBRID; }},
    {"-gen_tree", [](CMDSettings& s) { s.gen_tree = true; }},
};

const std::unordered_map<std::string, OneArgHandle> OneArg{
    {"-seed",
     [](CMDSettings& s, const std::string& arg) { s.seed = std::stoi(arg); }},
    {"-dim", [](CMDSettings& s,
                const std::string& arg) { s.dimension = std::stoi(arg); }},
    {"-hybrid_gen_split",  // percentage of GPU for heightmap generation
     [](CMDSettings& s, const std::string& arg) {
       s.hybrid_gen_split = std::stof(arg);
     }},
    {"-hybrid_norm_split",  // percentage of GPU for heightmap normalization
     [](CMDSettings& s, const std::string& arg) {
       s.hybrid_norm_split = std::stof(arg);
     }},
};

inline CMDSettings parse_settings(int argc, char* args[]) {
  CMDSettings settings;
  std::vector<std::string> positional_argument;
  for (int i = 1; i < argc; ++i) {
    std::string arg = args[i];

    if (arg.find("-", 0) == 0) {
      auto no_args_iterator = NoArgs.find(arg);
      auto args_iterator = OneArg.find(arg);

      if (no_args_iterator != NoArgs.end()) {
        no_args_iterator->second(settings);
      } else if (args_iterator != OneArg.end()) {
        ++i;
        args_iterator->second(settings, {args[i]});
      }
    }
  }

  return settings;
};
