cmake_minimum_required(VERSION 3.18)
project(BVHBenchmark LANGUAGES CXX CUDA)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find CUDA (should be found automatically with LANGUAGES CUDA)
find_package(CUDAToolkit REQUIRED)

# Source files
set(SOURCES
    src/main.cpp
    src/loader.cpp
    src/evaluator.cpp
    src/cuda/lbvh_builder.cu
)

# Executable
add_executable(bvh_benchmark ${SOURCES})

# Link CUDA runtime
target_link_libraries(bvh_benchmark
    CUDA::cudart
)

# CUDA compiler flags
set_target_properties(bvh_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(bvh_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
else()
    target_compile_options(bvh_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()

# Platform-specific settings
if(WIN32)
    target_compile_definitions(bvh_benchmark PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Print configuration
message(STATUS "BVH Benchmark Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CXX Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
