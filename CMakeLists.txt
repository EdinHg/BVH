cmake_minimum_required(VERSION 3.18)
project(BVHBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(CUDAToolkit REQUIRED)

set(SOURCES
    src/main.cpp
    src/loader.cpp
    src/evaluator.cpp
    src/cuda/lbvh_builder.cu
    src/cuda/lbvh_builder_nothrust.cu
    src/cuda/ploc_builder.cu
)

set_source_files_properties(
    src/main.cpp
    src/evaluator.cpp
    PROPERTIES LANGUAGE CUDA
)

add_executable(bvh_benchmark ${SOURCES})

target_link_libraries(bvh_benchmark
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(bvh_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(bvh_benchmark PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()