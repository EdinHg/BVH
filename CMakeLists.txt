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
# OpenCL Support (for OpenCL LBVH Builder)
# ============================================================================
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    message(STATUS "OpenCL found via find_package: ${OpenCL_INCLUDE_DIRS}")
    set(HAVE_OPENCL ON)
else()
    # Fallback: Fetch OpenCL headers from Khronos and link against system OpenCL.dll
    message(STATUS "OpenCL SDK not found - attempting to fetch Khronos headers")
    
    if(WIN32 AND EXISTS "C:/Windows/System32/OpenCL.dll")
        # Fetch OpenCL headers only (don't configure - they have outdated CMake)
        FetchContent_Declare(
            opencl_headers
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
            GIT_TAG        v2023.12.14
        )
        FetchContent_GetProperties(opencl_headers)
        if(NOT opencl_headers_POPULATED)
            FetchContent_Populate(opencl_headers)
        endif()
        
        set(OpenCL_INCLUDE_DIRS ${opencl_headers_SOURCE_DIR})
        
        # Create an imported library for OpenCL.dll
        add_library(OpenCL_Lib SHARED IMPORTED)
        set_target_properties(OpenCL_Lib PROPERTIES
            IMPORTED_LOCATION "C:/Windows/System32/OpenCL.dll"
            IMPORTED_IMPLIB "C:/Windows/System32/OpenCL.dll"
        )
        set(OpenCL_LIBRARIES OpenCL_Lib)
        set(HAVE_OPENCL ON)
        message(STATUS "OpenCL headers fetched from Khronos, using system OpenCL.dll")
    else()
        message(STATUS "OpenCL not found - OpenCL LBVH Builder will be disabled")
        set(HAVE_OPENCL OFF)
    endif()
endif()

if(HAVE_OPENCL)
    add_compile_definitions(HAVE_OPENCL=1)
endif()

# Boost.Compute (header-only, uses Boost)
if(HAVE_OPENCL)
    find_package(Boost QUIET)
    if(Boost_FOUND)
        message(STATUS "Boost found: ${Boost_INCLUDE_DIRS}")
        set(BOOST_AVAILABLE ON)
    else()
        # Fetch minimal Boost headers needed for Boost.Compute
        message(STATUS "Boost not found - fetching from GitHub")
        
        FetchContent_Declare(
            boost_headers
            GIT_REPOSITORY https://github.com/boostorg/boost.git
            GIT_TAG        boost-1.83.0
            GIT_SHALLOW    TRUE
        )
        # Only fetch, don't configure (it's header-only for our needs)
        FetchContent_GetProperties(boost_headers)
        if(NOT boost_headers_POPULATED)
            FetchContent_Populate(boost_headers)
        endif()
        
        set(Boost_INCLUDE_DIRS ${boost_headers_SOURCE_DIR})
        set(BOOST_AVAILABLE ON)
        message(STATUS "Boost headers fetched: ${Boost_INCLUDE_DIRS}")
    endif()
    
    if(BOOST_AVAILABLE)
        # Fetch Boost.Compute (header-only library)
        FetchContent_Declare(
            boost_compute
            GIT_REPOSITORY https://github.com/boostorg/compute.git
            GIT_TAG        boost-1.83.0
        )
        FetchContent_MakeAvailable(boost_compute)
        
        set(HAVE_BOOST_COMPUTE ON)
        add_compile_definitions(HAVE_BOOST_COMPUTE=1)
    else()
        message(STATUS "Boost setup failed - OpenCL LBVH Builder will be disabled")
        set(HAVE_OPENCL OFF)
        set(HAVE_BOOST_COMPUTE OFF)
    endif()
endif()

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

# Add OpenCL support to bvh_core if available
if(HAVE_OPENCL AND HAVE_BOOST_COMPUTE)
    target_include_directories(bvh_core INTERFACE 
        ${OpenCL_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${boost_compute_SOURCE_DIR}/include
    )
    target_link_libraries(bvh_core INTERFACE ${OpenCL_LIBRARIES})
    if(Boost_LIBRARIES)
        target_link_libraries(bvh_core INTERFACE ${Boost_LIBRARIES})
    endif()
endif()

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

# ============================================================================
# OpenCL LBVH Benchmark executable (optional)
# ============================================================================
if(HAVE_OPENCL AND HAVE_BOOST_COMPUTE)
    add_executable(bvh_opencl_test src/opencl_main.cpp)
    target_link_libraries(bvh_opencl_test PRIVATE bvh_core)
    
    message(STATUS "OpenCL LBVH Builder enabled")
else()
    message(STATUS "OpenCL LBVH Builder disabled (missing OpenCL or Boost)")
endif()
