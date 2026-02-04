cmake_minimum_required(VERSION 3.15)
project(benchmark_world_gen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(benchmark_world_gen
    main.cpp
)

target_include_directories(benchmark_world_gen
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

find_package(glm CONFIG REQUIRED)

target_link_libraries(benchmark_world_gen
    PRIVATE
        glm::glm
)
