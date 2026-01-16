#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <float.h>
#include <iomanip>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Thrust (C++ STL-like library for CUDA)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

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

// Host-side Triangle Structure of Arrays
struct TrianglesSoA {
    std::vector<float> v0x, v0y, v0z;
    std::vector<float> v1x, v1y, v1z;
    std::vector<float> v2x, v2y, v2z;

    void resize(size_t n) {
        v0x.resize(n); v0y.resize(n); v0z.resize(n);
        v1x.resize(n); v1y.resize(n); v1z.resize(n);
        v2x.resize(n); v2y.resize(n); v2z.resize(n);
    }
    size_t size() const { return v0x.size(); }
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
    float3_cw* centroids, 
    AABB_cw* globalBoundsBlock) // Shared memory reduction usually better, but keeping simple
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

// 2. Compute Morton Codes
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

// 3. Build Internal Nodes (Topology)
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

// 4. Initialize Leaf Nodes
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

// 5. Compute Bounding Boxes (Bottom-Up)
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
    // Device Memory containers
    thrust::device_vector<float> d_v0x, d_v0y, d_v0z;
    thrust::device_vector<float> d_v1x, d_v1y, d_v1z;
    thrust::device_vector<float> d_v2x, d_v2y, d_v2z;
    
    thrust::device_vector<AABB_cw> d_triBBoxes;
    thrust::device_vector<float3_cw> d_centroids;
    thrust::device_vector<uint32_t> d_mortonCodes;
    thrust::device_vector<uint32_t> d_indices;
    
    thrust::device_vector<LBVHNode> d_nodes;
    thrust::device_vector<int> d_atomicFlags;

    TrianglesSoADevice getDevicePtrs() {
        return {
            thrust::raw_pointer_cast(d_v0x.data()), thrust::raw_pointer_cast(d_v0y.data()), thrust::raw_pointer_cast(d_v0z.data()),
            thrust::raw_pointer_cast(d_v1x.data()), thrust::raw_pointer_cast(d_v1y.data()), thrust::raw_pointer_cast(d_v1z.data()),
            thrust::raw_pointer_cast(d_v2x.data()), thrust::raw_pointer_cast(d_v2y.data()), thrust::raw_pointer_cast(d_v2z.data())
        };
    }

public:
    void build(const TrianglesSoA& tris) {
        int n = tris.size();
        if (n == 0) return;

        // 1. Upload Data
        d_v0x = tris.v0x; d_v0y = tris.v0y; d_v0z = tris.v0z;
        d_v1x = tris.v1x; d_v1y = tris.v1y; d_v1z = tris.v1z;
        d_v2x = tris.v2x; d_v2y = tris.v2y; d_v2z = tris.v2z;

        d_triBBoxes.resize(n);
        d_centroids.resize(n);
        d_mortonCodes.resize(n);
        d_indices.resize(n);
        d_nodes.resize(2 * n - 1);
        d_atomicFlags.resize(2 * n - 1);
        
        // Reset atomic flags to 0
        thrust::fill(d_atomicFlags.begin(), d_atomicFlags.end(), 0);

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        // 2. Compute Bounds and Centroids
        kComputeBoundsAndCentroids<<<gridSize, blockSize>>>(getDevicePtrs(), n, 
            thrust::raw_pointer_cast(d_triBBoxes.data()), 
            thrust::raw_pointer_cast(d_centroids.data()), 
            nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reduction for scene bounds
        // Using Thrust for global reduction (easier than writing custom kernel)
        // Transform centroids to min/max components then reduce
        // For simplicity in this demo, we will download centroids to find bounds (or use a simple reduction)
        // Ideally: Use thrust::transform_reduce.
        
        // Simplified: Transform AABBs to scene bounds
        AABB_cw init; 
        init.min = float3_cw(FLT_MAX, FLT_MAX, FLT_MAX);
        init.max = float3_cw(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        // This is a rough way to do it with thrust, usually you write a custom reduction kernel
        // For brevity/correctness in demo, we'll assume a kernel did it or copy back small data
        // Let's implement a quick GPU reduction via thrust
        struct AABBReduce {
            __host__ __device__ AABB_cw operator()(const AABB_cw& a, const AABB_cw& b) {
                AABB_cw res;
                res.min.x = fminf(a.min.x, b.min.x);
                res.min.y = fminf(a.min.y, b.min.y);
                res.min.z = fminf(a.min.z, b.min.z);
                res.max.x = fmaxf(a.max.x, b.max.x);
                res.max.y = fmaxf(a.max.y, b.max.y);
                res.max.z = fmaxf(a.max.z, b.max.z);
                return res;
            }
        };

        AABB_cw sceneBounds = thrust::reduce(d_triBBoxes.begin(), d_triBBoxes.end(), init, AABBReduce());

        // 3. Compute Morton Codes
        kComputeMortonCodes<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_centroids.data()), 
            n, sceneBounds, 
            thrust::raw_pointer_cast(d_mortonCodes.data()), 
            thrust::raw_pointer_cast(d_indices.data()));
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. Sort (Radix Sort via Thrust)
        thrust::sort_by_key(d_mortonCodes.begin(), d_mortonCodes.end(), d_indices.begin());

        // 5. Build Hierarchy
        kBuildInternalNodes<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_mortonCodes.data()),
            thrust::raw_pointer_cast(d_indices.data()),
            thrust::raw_pointer_cast(d_nodes.data()),
            n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 6. Init Leaves
        kInitLeafNodes<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_triBBoxes.data()),
            thrust::raw_pointer_cast(d_indices.data()),
            n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 7. Refit BBoxes (Bottom Up)
        kRefitHierarchy<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_atomicFlags.data()),
            n);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::cout << "BVH Build Complete on GPU. Nodes: " << (2*n-1) << std::endl;
    }

    // Helper to verify root bounds on host
    void verify() {
        if(d_nodes.empty()) return;
        LBVHNode root = d_nodes[0];
        std::cout << "Root Bounds: " 
                  << "[" << root.bbox.min.x << ", " << root.bbox.min.y << ", " << root.bbox.min.z << "] - "
                  << "[" << root.bbox.max.x << ", " << root.bbox.max.y << ", " << root.bbox.max.z << "]\n";
    }
};

// ------------------------------------------------------------------
// Main / Test
// ------------------------------------------------------------------

int main() {
    int n = 1000000; // 1 Million triangles
    std::cout << "Generating " << n << " random triangles..." << std::endl;

    TrianglesSoA tris;
    tris.resize(n);

    // Fill with random data
    for(int i=0; i<n; ++i) {
        float x = (rand() % 1000) / 10.0f;
        float y = (rand() % 1000) / 10.0f;
        float z = (rand() % 1000) / 10.0f;
        
        tris.v0x[i] = x; tris.v0y[i] = y; tris.v0z[i] = z;
        tris.v1x[i] = x + 1; tris.v1y[i] = y; tris.v1z[i] = z;
        tris.v2x[i] = x; tris.v2y[i] = y + 1; tris.v2z[i] = z;
    }

    LBVHBuilderCUDA builder;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    builder.build(tris);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    builder.verify();

    return 0;
}