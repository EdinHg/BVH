cmake_minimum_required(VERSION 3.16)
project(bvh_test LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA Support (optional)
# ============================================================================
option(USE_CUDA "Enable CUDA LBVH builder" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ============================================================================
# Fetch dependencies
# ============================================================================
include(FetchContent)

# GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ============================================================================
# GLAD (OpenGL loader)
# ============================================================================
add_library(glad STATIC extern/glad/src/glad.c)
target_include_directories(glad PUBLIC extern/glad/include)

# ============================================================================
# Core BVH library (shared between executables)
# ============================================================================
add_library(bvh_core INTERFACE)
target_include_directories(bvh_core INTERFACE src)

# ============================================================================
# CUDA LBVH library (optional)
# ============================================================================
if(USE_CUDA)
    add_library(lbvh_cuda STATIC cuda/lbvh_cuda_builder.cu)
    target_include_directories(lbvh_cuda PUBLIC src)
    target_link_libraries(lbvh_cuda PUBLIC CUDA::cudart)
    set_target_properties(lbvh_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "61;70;75;80;86"
    )
endif()

# ============================================================================
# BVH Test executable (CLI benchmark)
# ============================================================================
add_executable(bvh_test src/main.cpp)
target_link_libraries(bvh_test PRIVATE bvh_core)

if(USE_CUDA)
    target_link_libraries(bvh_test PRIVATE lbvh_cuda)
    target_compile_definitions(bvh_test PRIVATE USE_CUDA)
endif()

# ============================================================================
# CUDA Standalone executable (optional, for independent CUDA testing)
# ============================================================================
if(USE_CUDA)
    add_executable(lbvh_cuda_standalone cuda/lbvh_cuda_builder.cu)
    target_include_directories(lbvh_cuda_standalone PRIVATE src)
    target_link_libraries(lbvh_cuda_standalone PRIVATE CUDA::cudart)
    target_compile_definitions(lbvh_cuda_standalone PRIVATE LBVH_CUDA_STANDALONE)
    set_target_properties(lbvh_cuda_standalone PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "61;70;75;80;86"
    )
endif()

# ============================================================================
# BVH Visualizer executable (OpenGL heatmap)
# ============================================================================
add_executable(bvh_visualizer src/visualizer_main.cpp)
target_link_libraries(bvh_visualizer PRIVATE 
    bvh_core 
    glfw 
    glad
)

# Platform-specific OpenGL linking
if(WIN32)
    target_link_libraries(bvh_visualizer PRIVATE opengl32)
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(bvh_visualizer PRIVATE OpenGL::GL)
endif()
