CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -I/usr/local/include -Isrc -O2 -Wall -Wextra

BUILDDIR := build
TARGET := $(BUILDDIR)/benchmark_world_gen

all: $(TARGET)

$(TARGET): main.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	@rm -rf $(BUILDDIR)

.PHONY: all clean
