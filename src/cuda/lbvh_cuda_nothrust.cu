#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <iomanip>
#include <string>
#include <fstream>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Thrust for radix sort
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// Project includes for OBJ loading/exporting
#include "../src/mesh/obj_loader.hpp"
#include "../src/bvh/bvh_export.hpp"

// --- Minimal Geometry Definitions for Standalone Compilation ---
struct __align__(16) float3_cw {
    float x, y, z;
    __host__ __device__ float3_cw() : x(0), y(0), z(0) {}
    __host__ __device__ float3_cw(float a, float b, float c) : x(a), y(b), z(c) {}
    __host__ __device__ float3_cw operator+(const float3_cw& b) const { return float3_cw(x+b.x, y+b.y, z+b.z); }
    __host__ __device__ float3_cw operator-(const float3_cw& b) const { return float3_cw(x-b.x, y-b.y, z-b.z); }
    __host__ __device__ float3_cw operator*(float s) const { return float3_cw(x*s, y*s, z*s); }
};

struct __align__(16) AABB_cw {
    float3_cw min;
    float3_cw max;
    __host__ __device__ AABB_cw() {
        min = float3_cw(FLT_MAX, FLT_MAX, FLT_MAX);
        max = float3_cw(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
};

struct LBVHNode {
    AABB_cw bbox;
    uint32_t leftChild;  // Leaf if (leftChild & 0x80000000)
    uint32_t rightChild;
    uint32_t parent;     // Needed for bottom-up traversal
};

// Device-side pointers for Triangles (Passed to kernels)
struct TrianglesSoADevice {
    float *v0x, *v0y, *v0z;
    float *v1x, *v1y, *v1z;
    float *v2x, *v2y, *v2z;
};

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ------------------------------------------------------------------
// Device Functions: Morton Codes
// ------------------------------------------------------------------

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__device__ __forceinline__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
__device__ __forceinline__ uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    uint32_t xx = expandBits((uint32_t)x);
    uint32_t yy = expandBits((uint32_t)y);
    uint32_t zz = expandBits((uint32_t)z);
    return xx * 4 + yy * 2 + zz;
}

// ------------------------------------------------------------------
// Device Functions: Tree Topology Logic (Karras 2012)
// ------------------------------------------------------------------

__device__ __forceinline__ int clz_custom(uint32_t x) {
    return x == 0 ? 32 : __clz(x);
}

// Calculates the length of the longest common prefix between two keys.
__device__ int delta(const uint32_t* sortedMortonCodes, const uint32_t* sortedIndices, int numObjects, int i, int j) {
    if (j < 0 || j >= numObjects) return -1;

    uint32_t codeI = sortedMortonCodes[i];
    uint32_t codeJ = sortedMortonCodes[j];

    if (codeI == codeJ) {
        // Use indices to disambiguate identical Morton codes
        uint32_t idxI = sortedIndices[i];
        uint32_t idxJ = sortedIndices[j];
        return 32 + clz_custom(idxI ^ idxJ);
    }
    return clz_custom(codeI ^ codeJ);
}

// ------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------

// 1. Compute Centroids and Scene Bounds (Atomic Reduction)
__global__ void kComputeBoundsAndCentroids(
    TrianglesSoADevice tris,
    int n,
    AABB_cw* triBBoxes,
    float3_cw* centroids)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Load triangle vertices
    float3_cw v0(tris.v0x[i], tris.v0y[i], tris.v0z[i]);
    float3_cw v1(tris.v1x[i], tris.v1y[i], tris.v1z[i]);
    float3_cw v2(tris.v2x[i], tris.v2y[i], tris.v2z[i]);

    // Compute Triangle AABB
    float3_cw minVal, maxVal;
    minVal.x = fminf(v0.x, fminf(v1.x, v2.x));
    minVal.y = fminf(v0.y, fminf(v1.y, v2.y));
    minVal.z = fminf(v0.z, fminf(v1.z, v2.z));
    maxVal.x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    maxVal.y = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    maxVal.z = fmaxf(v0.z, fmaxf(v1.z, v2.z));

    triBBoxes[i].min = minVal;
    triBBoxes[i].max = maxVal;

    // Compute Centroid
    centroids[i] = (minVal + maxVal) * 0.5f;
}

// 2. Initialize indices array
__global__ void kInitializeIndices(uint32_t* indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) indices[i] = i;
}

// 3. Fill integer array with zero
__global__ void kFillZero(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 0;
}

// 4. Parallel reduction for scene bounds (Step 1: Block-level reduction)
__global__ void kReduceBounds_Step1(const AABB_cw* input, AABB_cw* output, int n) {
    __shared__ AABB_cw sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        // Initialize with invalid bounds
        sdata[tid].min = float3_cw(FLT_MAX, FLT_MAX, FLT_MAX);
        sdata[tid].max = float3_cw(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid].min.x = fminf(sdata[tid].min.x, sdata[tid + s].min.x);
            sdata[tid].min.y = fminf(sdata[tid].min.y, sdata[tid + s].min.y);
            sdata[tid].min.z = fminf(sdata[tid].min.z, sdata[tid + s].min.z);

            sdata[tid].max.x = fmaxf(sdata[tid].max.x, sdata[tid + s].max.x);
            sdata[tid].max.y = fmaxf(sdata[tid].max.y, sdata[tid + s].max.y);
            sdata[tid].max.z = fmaxf(sdata[tid].max.z, sdata[tid + s].max.z);
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 5. Final reduction (Step 2: Reduce block results)
__global__ void kReduceBounds_Step2(AABB_cw* data, int n) {
    __shared__ AABB_cw sdata[256];

    int tid = threadIdx.x;

    if (tid < n) {
        sdata[tid] = data[tid];
    } else {
        sdata[tid].min = float3_cw(FLT_MAX, FLT_MAX, FLT_MAX);
        sdata[tid].max = float3_cw(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            sdata[tid].min.x = fminf(sdata[tid].min.x, sdata[tid + s].min.x);
            sdata[tid].min.y = fminf(sdata[tid].min.y, sdata[tid + s].min.y);
            sdata[tid].min.z = fminf(sdata[tid].min.z, sdata[tid + s].min.z);

            sdata[tid].max.x = fmaxf(sdata[tid].max.x, sdata[tid + s].max.x);
            sdata[tid].max.y = fmaxf(sdata[tid].max.y, sdata[tid + s].max.y);
            sdata[tid].max.z = fmaxf(sdata[tid].max.z, sdata[tid + s].max.z);
        }
        __syncthreads();
    }

    if (tid == 0) {
        data[0] = sdata[0];
    }
}

// 6. Compute Morton Codes
__global__ void kComputeMortonCodes(
    const float3_cw* centroids,
    int n,
    AABB_cw sceneBounds,
    uint32_t* mortonCodes,
    uint32_t* indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3_cw c = centroids[i];
    float3_cw minB = sceneBounds.min;
    float3_cw extents = sceneBounds.max - sceneBounds.min;

    // Normalize centroid to [0,1]
    float x = (c.x - minB.x) / extents.x;
    float y = (c.y - minB.y) / extents.y;
    float z = (c.z - minB.z) / extents.z;

    mortonCodes[i] = morton3D(x, y, z);
    indices[i] = i;
}

// 7. Build Internal Nodes (Topology)
__global__ void kBuildInternalNodes(
    const uint32_t* sortedMortonCodes,
    const uint32_t* sortedIndices,
    LBVHNode* nodes,
    int numObjects)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects - 1) return; // Run for N-1 internal nodes

    // Determine direction of the range (+1 or -1)
    int d = (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + 1) -
             delta(sortedMortonCodes, sortedIndices, numObjects, i, i - 1)) >= 0 ? 1 : -1;

    // Compute lower bound of the length of the range
    int min_delta = delta(sortedMortonCodes, sortedIndices, numObjects, i, i - d);
    int l_max = 2;
    while (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + l_max * d) > min_delta) {
        l_max *= 2;
    }

    // Find the other end of the range using binary search
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + (l + t) * d) > min_delta) {
            l += t;
        }
    }
    int j = i + l * d;

    // Find the split position using binary search
    int delta_node = delta(sortedMortonCodes, sortedIndices, numObjects, i, j);
    int s = 0;

    // Divide work based on direction
    int first, last;
    if (d > 0) { first = i; last = j; }
    else       { first = j; last = i; }

    int len = last - first;
    int t = len;

    // Common prefix of the range
    // Efficient binary search to find split
    do {
        t = (t + 1) >> 1;
        if (delta(sortedMortonCodes, sortedIndices, numObjects, first, first + s + t) > delta_node) {
            s += t;
        }
    } while (t > 1);

    int split = first + s;

    // Output children
    uint32_t leftIdx, rightIdx;

    // Left Child
    if (split == first) leftIdx = (numObjects - 1) + split; // Leaf
    else                leftIdx = split;                   // Internal

    // Right Child
    if (split + 1 == last) rightIdx = (numObjects - 1) + split + 1; // Leaf
    else                   rightIdx = split + 1;                   // Internal

    nodes[i].leftChild = leftIdx;
    nodes[i].rightChild = rightIdx;

    // Set parents
    nodes[leftIdx].parent = i;
    nodes[rightIdx].parent = i;
}

// 8. Initialize Leaf Nodes
__global__ void kInitLeafNodes(
    LBVHNode* nodes,
    const AABB_cw* triBBoxes,
    const uint32_t* sortedIndices,
    int numObjects)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    int leafIdx = (numObjects - 1) + i; // Leaves are stored after internal nodes
    uint32_t originalIndex = sortedIndices[i];

    // Store BBox
    nodes[leafIdx].bbox = triBBoxes[originalIndex];
    // Mark as leaf using high bit or just rely on structure
    nodes[leafIdx].leftChild = originalIndex | 0x80000000;
    nodes[leafIdx].rightChild = 0xFFFFFFFF;
}

// 9. Compute Bounding Boxes (Bottom-Up)
__global__ void kRefitHierarchy(
    LBVHNode* nodes,
    int* atomicCounters,
    int numObjects)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    // Start at leaf
    uint32_t idx = (numObjects - 1) + i;

    // Walk up the tree
    while (idx != 0) { // While not root
        uint32_t parent = nodes[idx].parent;

        // Atomic increment: returns old value
        // 0 -> First thread arriving. Terminate.
        // 1 -> Second thread arriving. Process node.
        int oldVal = atomicAdd(&atomicCounters[parent], 1);

        if (oldVal == 0) {
            // First child arrived, exit.
            // The second child will process the parent.
            return;
        }

        // Processing parent (merging children)
        uint32_t left = nodes[parent].leftChild;
        uint32_t right = nodes[parent].rightChild;

        AABB_cw leftBox = nodes[left].bbox;
        AABB_cw rightBox = nodes[right].bbox;

        AABB_cw unionBox;
        unionBox.min.x = fminf(leftBox.min.x, rightBox.min.x);
        unionBox.min.y = fminf(leftBox.min.y, rightBox.min.y);
        unionBox.min.z = fminf(leftBox.min.z, rightBox.min.z);

        unionBox.max.x = fmaxf(leftBox.max.x, rightBox.max.x);
        unionBox.max.y = fmaxf(leftBox.max.y, rightBox.max.y);
        unionBox.max.z = fmaxf(leftBox.max.z, rightBox.max.z);

        // Store result
        nodes[parent].bbox = unionBox;

        // Move up
        idx = parent;
    }
}

// ------------------------------------------------------------------
// Host Builder Class
// ------------------------------------------------------------------

class LBVHBuilderCUDA {
private:
    // Raw CUDA pointers
    float *d_v0x, *d_v0y, *d_v0z;
    float *d_v1x, *d_v1y, *d_v1z;
    float *d_v2x, *d_v2y, *d_v2z;

    AABB_cw* d_triBBoxes;
    float3_cw* d_centroids;
    uint32_t* d_mortonCodes;
    uint32_t* d_indices;

    LBVHNode* d_nodes;
    int* d_atomicFlags;

    AABB_cw* d_boundsReduction;  // Temporary for reduction

    int numTriangles;

    // Profiling Events
    cudaEvent_t start, e_centroids, e_morton, e_sort, e_topology, stop;

    TrianglesSoADevice getDevicePtrs() {
        return {d_v0x, d_v0y, d_v0z, d_v1x, d_v1y, d_v1z, d_v2x, d_v2y, d_v2z};
    }

public:
    LBVHBuilderCUDA() :
        d_v0x(nullptr), d_v0y(nullptr), d_v0z(nullptr),
        d_v1x(nullptr), d_v1y(nullptr), d_v1z(nullptr),
        d_v2x(nullptr), d_v2y(nullptr), d_v2z(nullptr),
        d_triBBoxes(nullptr), d_centroids(nullptr),
        d_mortonCodes(nullptr), d_indices(nullptr),
        d_nodes(nullptr), d_atomicFlags(nullptr),
        d_boundsReduction(nullptr), numTriangles(0) {
        cudaEventCreate(&start);
        cudaEventCreate(&e_centroids);
        cudaEventCreate(&e_morton);
        cudaEventCreate(&e_sort);
        cudaEventCreate(&e_topology);
        cudaEventCreate(&stop);
    }

    ~LBVHBuilderCUDA() {
        // Free all allocated memory
        if (d_v0x) cudaFree(d_v0x);
        if (d_v0y) cudaFree(d_v0y);
        if (d_v0z) cudaFree(d_v0z);
        if (d_v1x) cudaFree(d_v1x);
        if (d_v1y) cudaFree(d_v1y);
        if (d_v1z) cudaFree(d_v1z);
        if (d_v2x) cudaFree(d_v2x);
        if (d_v2y) cudaFree(d_v2y);
        if (d_v2z) cudaFree(d_v2z);
        if (d_triBBoxes) cudaFree(d_triBBoxes);
        if (d_centroids) cudaFree(d_centroids);
        if (d_mortonCodes) cudaFree(d_mortonCodes);
        if (d_indices) cudaFree(d_indices);
        if (d_nodes) cudaFree(d_nodes);
        if (d_atomicFlags) cudaFree(d_atomicFlags);
        if (d_boundsReduction) cudaFree(d_boundsReduction);

        // Destroy profiling events
        cudaEventDestroy(start);
        cudaEventDestroy(e_centroids);
        cudaEventDestroy(e_morton);
        cudaEventDestroy(e_sort);
        cudaEventDestroy(e_topology);
        cudaEventDestroy(stop);
    }

    // PHASE 1: Data Upload (Not part of compute benchmark)
    void prepareData(const TriangleMesh& tris) {
        numTriangles = tris.size();
        if (numTriangles == 0) return;

        // Allocate and Upload Data
        size_t floatSize = numTriangles * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_v0x, floatSize));
        CUDA_CHECK(cudaMalloc(&d_v0y, floatSize));
        CUDA_CHECK(cudaMalloc(&d_v0z, floatSize));
        CUDA_CHECK(cudaMalloc(&d_v1x, floatSize));
        CUDA_CHECK(cudaMalloc(&d_v1y, floatSize));
        CUDA_CHECK(cudaMalloc(&d_v1z, floatSize));
        CUDA_CHECK(cudaMalloc(&d_v2x, floatSize));
        CUDA_CHECK(cudaMalloc(&d_v2y, floatSize));
        CUDA_CHECK(cudaMalloc(&d_v2z, floatSize));

        CUDA_CHECK(cudaMemcpy(d_v0x, tris.v0x.data(), floatSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v0y, tris.v0y.data(), floatSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v0z, tris.v0z.data(), floatSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v1x, tris.v1x.data(), floatSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v1y, tris.v1y.data(), floatSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v1z, tris.v1z.data(), floatSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v2x, tris.v2x.data(), floatSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v2y, tris.v2y.data(), floatSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v2z, tris.v2z.data(), floatSize, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_triBBoxes, numTriangles * sizeof(AABB_cw)));
        CUDA_CHECK(cudaMalloc(&d_centroids, numTriangles * sizeof(float3_cw)));
        CUDA_CHECK(cudaMalloc(&d_mortonCodes, numTriangles * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_indices, numTriangles * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_nodes, (2 * numTriangles - 1) * sizeof(LBVHNode)));
        CUDA_CHECK(cudaMalloc(&d_atomicFlags, (2 * numTriangles - 1) * sizeof(int)));

        int blockSize = 256;
        int gridSize = (numTriangles + blockSize - 1) / blockSize;

        // Reset atomic flags to 0
        kFillZero<<<gridSize, blockSize>>>(d_atomicFlags, 2 * numTriangles - 1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // PHASE 2: Pure Compute (Benchmarked)
    void runCompute() {
        if (numTriangles == 0) return;

        int blockSize = 256;
        int gridSize = (numTriangles + blockSize - 1) / blockSize;

        cudaEventRecord(start);

        // 1. Compute Bounds and Centroids
        kComputeBoundsAndCentroids<<<gridSize, blockSize>>>(
            getDevicePtrs(), numTriangles, d_triBBoxes, d_centroids);

        // 2. Reduce scene bounds
        int numBlocks = gridSize;
        CUDA_CHECK(cudaMalloc(&d_boundsReduction, numBlocks * sizeof(AABB_cw)));

        kReduceBounds_Step1<<<numBlocks, blockSize>>>(d_triBBoxes, d_boundsReduction, numTriangles);

        // Final reduction on single block
        if (numBlocks > 1) {
            kReduceBounds_Step2<<<1, blockSize>>>(d_boundsReduction, numBlocks);
        }

        // Copy final result back
        AABB_cw sceneBounds;
        CUDA_CHECK(cudaMemcpy(&sceneBounds, d_boundsReduction, sizeof(AABB_cw), cudaMemcpyDeviceToHost));

        cudaEventRecord(e_centroids);

        // 3. Compute Morton Codes
        kComputeMortonCodes<<<gridSize, blockSize>>>(
            d_centroids, numTriangles, sceneBounds, d_mortonCodes, d_indices);

        cudaEventRecord(e_morton);

        // 4. Sort (Radix Sort)
        thrust::device_ptr<uint32_t> mortonPtr(d_mortonCodes);
        thrust::device_ptr<uint32_t> indicesPtr(d_indices);
        thrust::sort_by_key(mortonPtr, mortonPtr + numTriangles, indicesPtr);

        cudaEventRecord(e_sort);

        // 5. Build Hierarchy
        int internalGridSize = (numTriangles - 1 + blockSize - 1) / blockSize;
        kBuildInternalNodes<<<internalGridSize, blockSize>>>(
            d_mortonCodes, d_indices, d_nodes, numTriangles);

        // 6. Init Leaves
        kInitLeafNodes<<<gridSize, blockSize>>>(
            d_nodes, d_triBBoxes, d_indices, numTriangles);

        cudaEventRecord(e_topology);

        // 7. Refit BBoxes (Bottom Up)
        kRefitHierarchy<<<gridSize, blockSize>>>(
            d_nodes, d_atomicFlags, numTriangles);

        cudaEventRecord(stop);

        // Print Breakdown
        cudaEventSynchronize(stop);
        float t_c, t_m, t_s, t_t, t_r;
        cudaEventElapsedTime(&t_c, start, e_centroids);
        cudaEventElapsedTime(&t_m, e_centroids, e_morton);
        cudaEventElapsedTime(&t_s, e_morton, e_sort);
        cudaEventElapsedTime(&t_t, e_sort, e_topology);
        cudaEventElapsedTime(&t_r, e_topology, stop);
        float total = t_c + t_m + t_s + t_t + t_r;

        std::cout << "\n   [GPU Phase Breakdown]\n";
        std::cout << "   Bounds/Centroids: " << std::setw(6) << t_c << " ms\n";
        std::cout << "   Morton Codes:     " << std::setw(6) << t_m << " ms\n";
        std::cout << "   Radix Sort:       " << std::setw(6) << t_s << " ms\n";
        std::cout << "   Topology Build:   " << std::setw(6) << t_t << " ms\n";
        std::cout << "   Bottom-Up Refit:  " << std::setw(6) << t_r << " ms\n";
        std::cout << "   Total Kernel Time:" << std::setw(6) << total << " ms\n\n";
    }

    // Helper to verify root bounds on host
    void verify() {
        if(!d_nodes) return;
        LBVHNode root;
        CUDA_CHECK(cudaMemcpy(&root, d_nodes, sizeof(LBVHNode), cudaMemcpyDeviceToHost));
        std::cout << "Root Bounds: "
                  << "[" << root.bbox.min.x << ", " << root.bbox.min.y << ", " << root.bbox.min.z << "] - "
                  << "[" << root.bbox.max.x << ", " << root.bbox.max.y << ", " << root.bbox.max.z << "]\n";
    }

    // Get raw LBVH nodes for export
    std::vector<LBVHNode> getRawNodes() const {
        if (!d_nodes) return std::vector<LBVHNode>();

        int totalNodes = 2 * numTriangles - 1;
        std::vector<LBVHNode> hostNodes(totalNodes);
        CUDA_CHECK(cudaMemcpy(hostNodes.data(), d_nodes, totalNodes * sizeof(LBVHNode), cudaMemcpyDeviceToHost));

        return hostNodes;
    }

    // Get BVH nodes for export (converts to BVHNode format)
    std::vector<BVHNode> getNodes() const {
        if (!d_nodes) return std::vector<BVHNode>();

        int totalNodes = 2 * numTriangles - 1;
        std::vector<LBVHNode> hostNodes(totalNodes);
        CUDA_CHECK(cudaMemcpy(hostNodes.data(), d_nodes, totalNodes * sizeof(LBVHNode), cudaMemcpyDeviceToHost));

        std::vector<BVHNode> result;
        result.reserve(totalNodes);

        for (size_t i = 0; i < hostNodes.size(); ++i) {
            const auto& node = hostNodes[i];
            BVHNode bvhNode;
            bvhNode.bounds.min = Vec3(node.bbox.min.x, node.bbox.min.y, node.bbox.min.z);
            bvhNode.bounds.max = Vec3(node.bbox.max.x, node.bbox.max.y, node.bbox.max.z);

            // Check if this is a leaf node (high bit set in leftChild)
            if (node.leftChild & 0x80000000) {
                bvhNode.childCount = 0;  // Leaf
                bvhNode.primOffset = node.leftChild & 0x7FFFFFFF;  // Strip high bit
                bvhNode.primCount = 1;
            } else {
                bvhNode.childCount = 2;  // Binary internal node
                bvhNode.childOffset = node.leftChild;  // Left child index
                bvhNode.primCount = 0;
            }
            bvhNode.axis = 0;
            result.push_back(bvhNode);
        }
        return result;
    }
};

// ------------------------------------------------------------------
// Command-line argument parsing
// ------------------------------------------------------------------

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  -i, --input <file.obj>    Input OBJ file to load\n"
              << "  -o, --output <file>       Output file for BVH export\n"
              << "  -n, --triangles <count>   Number of random triangles (default: 10000000)\n"
              << "  -l, --leaves-only         Export only leaf node bounding boxes\n"
              << "  -c, --colab-export        Export binary dump for Colab visualization\n"
              << "  -h, --help                Show this help message\n"
              << "\nExamples:\n"
              << "  " << programName << " -i model.obj -o bvh.obj\n"
              << "  " << programName << " -n 1000000 -o bvh.obj\n"
              << "  " << programName << " -i model.obj -o bvh.obj -l\n"
              << "  " << programName << " -i model.obj -o bvh.bin -c\n";
}

// ------------------------------------------------------------------
// Binary Export for Colab Visualization
// ------------------------------------------------------------------

struct VizNode {
    float min[3];
    float max[3];
    int leftIdx;
    int rightIdx;
};

void exportBVHToBinaryLBVH(const std::string& filename, const std::vector<LBVHNode>& nodes) {
    std::ofstream file(filename, std::ios::binary);

    std::vector<VizNode> exportNodes;
    exportNodes.reserve(nodes.size());

    for(const auto& node : nodes) {
        VizNode vn;
        vn.min[0] = node.bbox.min.x;
        vn.min[1] = node.bbox.min.y;
        vn.min[2] = node.bbox.min.z;

        vn.max[0] = node.bbox.max.x;
        vn.max[1] = node.bbox.max.y;
        vn.max[2] = node.bbox.max.z;

        vn.leftIdx = static_cast<int>(node.leftChild);
        vn.rightIdx = static_cast<int>(node.rightChild);

        exportNodes.push_back(vn);
    }

    file.write(reinterpret_cast<const char*>(exportNodes.data()), exportNodes.size() * sizeof(VizNode));
    std::cout << "Exported " << exportNodes.size() << " nodes to " << filename
              << " (" << (exportNodes.size() * sizeof(VizNode))/1024/1024 << " MB)\n";
}

// ------------------------------------------------------------------
// OBJ Export for Visualization (LBVH version)
// ------------------------------------------------------------------

void exportLBVHToOBJ(const std::string& filename, const std::vector<LBVHNode>& nodes, bool leavesOnly) {
    std::ofstream file(filename);
    int v_offset = 1;

    auto writeBox = [&](const AABB_cw& b) {
        float x0 = b.min.x, y0 = b.min.y, z0 = b.min.z;
        float x1 = b.max.x, y1 = b.max.y, z1 = b.max.z;

        file << "v " << x0 << " " << y0 << " " << z0 << "\n";
        file << "v " << x1 << " " << y0 << " " << z0 << "\n";
        file << "v " << x1 << " " << y1 << " " << z0 << "\n";
        file << "v " << x0 << " " << y1 << " " << z0 << "\n";
        file << "v " << x0 << " " << y0 << " " << z1 << "\n";
        file << "v " << x1 << " " << y0 << " " << z1 << "\n";
        file << "v " << x1 << " " << y1 << " " << z1 << "\n";
        file << "v " << x0 << " " << y1 << " " << z1 << "\n";

        int o = v_offset;
        file << "l " << o << " " << o+1 << " " << o+2 << " " << o+3 << " " << o << "\n";
        file << "l " << o+4 << " " << o+5 << " " << o+6 << " " << o+7 << " " << o+4 << "\n";
        file << "l " << o << " " << o+4 << "\n";
        file << "l " << o+1 << " " << o+5 << "\n";
        file << "l " << o+2 << " " << o+6 << "\n";
        file << "l " << o+3 << " " << o+7 << "\n";
        v_offset += 8;
    };

    int numLeafs = (nodes.size() + 1) / 2;
    for (size_t i = 0; i < nodes.size(); ++i) {
        bool nodeIsLeaf = (i >= (nodes.size() + 1) / 2 - 1);
        if (leavesOnly && !nodeIsLeaf) continue;
        writeBox(nodes[i].bbox);
    }
}

// ------------------------------------------------------------------
// Random triangle generation
// ------------------------------------------------------------------

TriangleMesh generateRandomTriangles(int n) {
    TriangleMesh mesh;
    mesh.reserve(n);

    for (int i = 0; i < n; ++i) {
        float x = (rand() % 1000) / 10.0f;
        float y = (rand() % 1000) / 10.0f;
        float z = (rand() % 1000) / 10.0f;

        mesh.addTriangle(
            Vec3(x, y, z),
            Vec3(x + 1, y, z),
            Vec3(x, y + 1, z)
        );
    }

    return mesh;
}

// ------------------------------------------------------------------
// Main / Test
// ------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string inputFile;
    std::string outputFile;
    int numTriangles = 10000000;  // Default: 10 million
    bool leavesOnly = false;
    bool colabExport = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                inputFile = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a file path\n";
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                outputFile = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a file path\n";
                return 1;
            }
        } else if (arg == "-n" || arg == "--triangles") {
            if (i + 1 < argc) {
                numTriangles = std::atoi(argv[++i]);
                if (numTriangles <= 0) {
                    std::cerr << "Error: Invalid triangle count\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: " << arg << " requires a number\n";
                return 1;
            }
        } else if (arg == "-l" || arg == "--leaves-only") {
            leavesOnly = true;
        } else if (arg == "-c" || arg == "--colab-export") {
            colabExport = true;
        } else {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Load or generate mesh
    TriangleMesh mesh;

    if (!inputFile.empty()) {
        std::cout << "Loading OBJ file: " << inputFile << std::endl;
        try {
            mesh = loadOBJ(inputFile);
            std::cout << "Loaded " << mesh.size() << " triangles from OBJ file\n";
        } catch (const std::exception& e) {
            std::cerr << "Error loading OBJ: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "Generating " << numTriangles << " random triangles..." << std::endl;
        mesh = generateRandomTriangles(numTriangles);
    }

    if (mesh.size() == 0) {
        std::cerr << "Error: No triangles to process\n";
        return 1;
    }

    LBVHBuilderCUDA builder;

    // 1. Prepare Data (Allocation + Copy) - NOT TIMED
    std::cout << "Uploading data to GPU and allocating memory..." << std::endl;
    builder.prepareData(mesh);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Starting build..." << std::endl;
    cudaEventRecord(start);

    // 2. Run Compute - TIMED
    builder.runCompute();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Triangles: " << mesh.size() << std::endl;
    std::cout << "Compute Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Throughput: " << (mesh.size() / 1000000.0f) / (milliseconds / 1000.0f) << " MTris/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    builder.verify();

    // Export BVH if output file specified
    if (!outputFile.empty()) {
        if (colabExport) {
            std::cout << "Exporting BVH to Colab binary format: " << outputFile << std::endl;
            try {
                std::vector<LBVHNode> nodes = builder.getRawNodes();
                exportBVHToBinaryLBVH(outputFile, nodes);
            } catch (const std::exception& e) {
                std::cerr << "Error exporting BVH: " << e.what() << std::endl;
                return 1;
            }
        } else {
            std::cout << "Exporting BVH to OBJ format: " << outputFile << std::endl;
            try {
                std::vector<LBVHNode> nodes = builder.getRawNodes();
                exportLBVHToOBJ(outputFile, nodes, leavesOnly);
                std::cout << "Exported " << nodes.size() << " BVH nodes"
                          << (leavesOnly ? " (leaves only)" : "") << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error exporting BVH: " << e.what() << std::endl;
                return 1;
            }
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
