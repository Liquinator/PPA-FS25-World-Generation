#pragma once

#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

enum class GenerationMode { SEQUENTIAL, PARALLEL, BOTH };

struct CMDSettings {
  int seed{42};
  int dimension{256};
  GenerationMode mode{GenerationMode::BOTH};
};

typedef std::function<void(CMDSettings&)> NoArgHandle;
typedef std::function<void(CMDSettings&, const std::string&)> OneArgHandle;

const std::unordered_map<std::string, NoArgHandle> NoArgs{
    {"-p", [](CMDSettings& s) { s.mode = GenerationMode::PARALLEL; }},
    {"-s", [](CMDSettings& s) { s.mode = GenerationMode::SEQUENTIAL; }},
    {"-full", [](CMDSettings& s) { s.mode = GenerationMode::BOTH; }},
};

const std::unordered_map<std::string, OneArgHandle> OneArg{
    {"-seed",
     [](CMDSettings& s, const std::string& arg) { s.seed = std::stoi(arg); }},
    {"-dim", [](CMDSettings& s,
                const std::string& arg) { s.dimension = std::stoi(arg); }},

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
