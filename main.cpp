#include "main.hpp"

int main(int argc, char* args[]) {
  CMDSettings settings = parse_settings(argc, args);
  benchmark(settings);

  return 0;
}
