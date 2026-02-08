#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include <cmath>
#include <algorithm>

// Aligned 3D vector for CPU/GPU interop
struct __align__(16) float3_cw {
    float x, y, z;
    
    __host__ __device__ float3_cw() : x(0), y(0), z(0) {}
    __host__ __device__ float3_cw(float a, float b, float c) : x(a), y(b), z(c) {}
    
    __host__ __device__ float3_cw operator+(const float3_cw& b) const { 
        return float3_cw(x + b.x, y + b.y, z + b.z); 
    }
    
    __host__ __device__ float3_cw operator-(const float3_cw& b) const { 
        return float3_cw(x - b.x, y - b.y, z - b.z); 
    }
    
    __host__ __device__ float3_cw operator*(float s) const { 
        return float3_cw(x * s, y * s, z * s); 
    }
};

// Axis-Aligned Bounding Box
struct __align__(16) AABB_cw {
    float3_cw min;
    float3_cw max;
    
    __host__ __device__ AABB_cw() {
        min = float3_cw(FLT_MAX, FLT_MAX, FLT_MAX);
        max = float3_cw(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
    
    __host__ __device__ AABB_cw(const float3_cw& min_, const float3_cw& max_) 
        : min(min_), max(max_) {}
    
    __host__ __device__ float surfaceArea() const {
        float3_cw d = max - min;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
    
    __host__ __device__ AABB_cw merge(const AABB_cw& other) const {
        AABB_cw result;
        #ifdef __CUDA_ARCH__
        result.min.x = fminf(min.x, other.min.x);
        result.min.y = fminf(min.y, other.min.y);
        result.min.z = fminf(min.z, other.min.z);
        result.max.x = fmaxf(max.x, other.max.x);
        result.max.y = fmaxf(max.y, other.max.y);
        result.max.z = fmaxf(max.z, other.max.z);
        #else
        result.min.x = std::min(min.x, other.min.x);
        result.min.y = std::min(min.y, other.min.y);
        result.min.z = std::min(min.z, other.min.z);
        result.max.x = std::max(max.x, other.max.x);
        result.max.y = std::max(max.y, other.max.y);
        result.max.z = std::max(max.z, other.max.z);
        #endif
        return result;
    }
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
