cmake_minimum_required(VERSION 3.15)
project(benchmark_world_gen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(glm CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(benchmark_world_gen
    main.cpp
    src/perlin_noise_cuda.cu
    src/perlin_noise_hybrid.cu
)

target_include_directories(benchmark_world_gen
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(benchmark_world_gen
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -march=native
            -O3
            -pthread
            -Wno-subobject-linkage
            -Wno-overflow
            -Wfatal-errors
        >
)

target_compile_options(benchmark_world_gen
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-extended-lambda
            --expt-relaxed-constexpr
            --use_fast_math
            --ptxas-options=-v
            --generate-line-info
            -O3
        >
)

set_target_properties(benchmark_world_gen PROPERTIES
    CUDA_ARCHITECTURES 120
)

target_link_libraries(benchmark_world_gen
    PRIVATE
        glm::glm
)

