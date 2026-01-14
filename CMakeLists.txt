cmake_minimum_required(VERSION 3.16)
project(bvh_test LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
# BVH Test executable (CLI benchmark)
# ============================================================================
add_executable(bvh_test src/main.cpp)
target_link_libraries(bvh_test PRIVATE bvh_core)

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
