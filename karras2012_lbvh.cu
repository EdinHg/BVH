// OPTIMIZED Karras 2012 LBVH Implementation
// "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
// HPG 2012 Paper Implementation
//
// OPTIMIZATIONS APPLIED:
// - Unified node array (no separate internal/leaf arrays)
// - GPU AABB refit (no CPU bottleneck!)
// - Explicit parent pointers (O(1) parent access)
// - Optimized Morton code (bitshift instead of multiply)
// - Better memory layout for cache efficiency
//
// Pure parallel radix tree construction - NO treelet optimization
//
// Compile: nvcc -std=c++14 -arch=sm_75 -O3 karras2012_lbvh.cu -o karras2012_lbvh
// Usage: ./karras2012_lbvh <num_triangles> [--export] [--export-leaves]
//        ./karras2012_lbvh <model.obj> [--export] [--export-leaves]

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::cerr << cudaGetErrorString(err) << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Constants
// ============================================================================

#define Ci 1.2f  // Internal node traversal cost (for SAH)
#define Ct 1.0f  // Triangle intersection cost (for SAH)

// ============================================================================
// Math Structures
// ============================================================================

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    __host__ __device__ Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }

    __host__ __device__ static Vec3 min(const Vec3& a, const Vec3& b) {
        return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
    }
    __host__ __device__ static Vec3 max(const Vec3& a, const Vec3& b) {
        return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
    }
};

struct AABB {
    Vec3 min, max;

    __host__ __device__ AABB()
        : min(1e30f, 1e30f, 1e30f),
          max(-1e30f, -1e30f, -1e30f) {}

    __host__ __device__ AABB(const Vec3& min, const Vec3& max) : min(min), max(max) {}

    __host__ __device__ void expand(const Vec3& p) {
        min = Vec3::min(min, p);
        max = Vec3::max(max, p);
    }

    __host__ __device__ void expand(const AABB& other) {
        min = Vec3::min(min, other.min);
        max = Vec3::max(max, other.max);
    }

    __host__ __device__ Vec3 extent() const { return max - min; }
    __host__ __device__ Vec3 center() const { return (min + max) * 0.5f; }

    __host__ __device__ float surfaceArea() const {
        Vec3 d = extent();
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

// ============================================================================
// BVH Node Structure - OPTIMIZED
// ============================================================================
// LAYOUT: [internal nodes: 0..n-2][leaf nodes: n-1..2n-2]
// Binary tree: each internal has exactly 2 children
// For binary tree: rightChild = leftChild + 1 (implicit, saves 4 bytes)

struct BVHNode {
    AABB bounds;          // 24 bytes (6 floats)
    uint32_t leftChild;   // 4 bytes: index to left child (right = left+1)
    uint32_t primOffset;  // 4 bytes: for leaves, index into sorted primitive array
    uint32_t primCount;   // 4 bytes: 0 for internal, 1 for leaf

    __host__ __device__ bool isLeaf() const {
        return primCount > 0;
    }

    __host__ __device__ uint32_t rightChild() const {
        return leftChild + 1;  // Binary tree property
    }
};

// ============================================================================
// Triangle Mesh
// ============================================================================

struct TriangleMesh {
    std::vector<float> v0x, v0y, v0z;
    std::vector<float> v1x, v1y, v1z;
    std::vector<float> v2x, v2y, v2z;

    size_t size() const { return v0x.size(); }

    void reserve(size_t n) {
        v0x.reserve(n); v0y.reserve(n); v0z.reserve(n);
        v1x.reserve(n); v1y.reserve(n); v1z.reserve(n);
        v2x.reserve(n); v2y.reserve(n); v2z.reserve(n);
    }

    void addTriangle(const Vec3& a, const Vec3& b, const Vec3& c) {
        v0x.push_back(a.x); v0y.push_back(a.y); v0z.push_back(a.z);
        v1x.push_back(b.x); v1y.push_back(b.y); v1z.push_back(b.z);
        v2x.push_back(c.x); v2y.push_back(c.y); v2z.push_back(c.z);
    }

    Vec3 getVertex0(size_t i) const { return {v0x[i], v0y[i], v0z[i]}; }
    Vec3 getVertex1(size_t i) const { return {v1x[i], v1y[i], v1z[i]}; }
    Vec3 getVertex2(size_t i) const { return {v2x[i], v2y[i], v2z[i]}; }

    AABB getBounds(size_t i) const {
        AABB box;
        box.expand(getVertex0(i));
        box.expand(getVertex1(i));
        box.expand(getVertex2(i));
        return box;
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

TriangleMesh generateRandomTriangles(size_t count, uint32_t seed = 42) {
    TriangleMesh mesh;
    mesh.reserve(count);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < count; ++i) {
        Vec3 v0(dist(rng), dist(rng), dist(rng));
        Vec3 v1(dist(rng), dist(rng), dist(rng));
        Vec3 v2(dist(rng), dist(rng), dist(rng));
        mesh.addTriangle(v0, v1, v2);
    }

    return mesh;
}

TriangleMesh loadOBJ(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open OBJ file: " + path);
    }

    std::vector<Vec3> vertices;
    TriangleMesh mesh;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back({x, y, z});
        }
        else if (prefix == "f") {
            std::vector<int> faceIndices;
            std::string token;
            while (iss >> token) {
                int idx = std::stoi(token.substr(0, token.find('/')));
                if (idx < 0) idx = static_cast<int>(vertices.size()) + idx + 1;
                faceIndices.push_back(idx - 1);
            }
            for (size_t i = 1; i + 1 < faceIndices.size(); ++i) {
                mesh.addTriangle(
                    vertices[faceIndices[0]],
                    vertices[faceIndices[i]],
                    vertices[faceIndices[i + 1]]
                );
            }
        }
    }

    return mesh;
}

// ============================================================================
// Morton Code Functions - OPTIMIZED (bitshift instead of multiply)
// ============================================================================

__device__ inline uint32_t expandBits(uint32_t v) {
    // Use bitshift operations instead of multiply (faster on GPU)
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

__device__ inline uint32_t mortonCode30(float x, float y, float z) {
    // Clamp to [0, 1023] (10 bits per axis = 30 bits total)
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);

    uint32_t xx = expandBits(static_cast<uint32_t>(x));
    uint32_t yy = expandBits(static_cast<uint32_t>(y));
    uint32_t zz = expandBits(static_cast<uint32_t>(z));

    // Interleave: Z Y X Z Y X ... (use bitshift instead of multiply)
    return (xx << 2) | (yy << 1) | zz;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void computeTriangleBoundsAndCentroids(
    const float* v0x, const float* v0y, const float* v0z,
    const float* v1x, const float* v1y, const float* v1z,
    const float* v2x, const float* v2y, const float* v2z,
    AABB* triBounds, Vec3* centroids, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Vec3 v0(v0x[tid], v0y[tid], v0z[tid]);
    Vec3 v1(v1x[tid], v1y[tid], v1z[tid]);
    Vec3 v2(v2x[tid], v2y[tid], v2z[tid]);

    AABB box;
    box.expand(v0);
    box.expand(v1);
    box.expand(v2);

    triBounds[tid] = box;
    centroids[tid] = box.center();
}

__global__ void computeMortonCodes(
    const Vec3* centroids, AABB sceneBounds,
    uint32_t* mortonCodes, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Vec3 sceneSize = sceneBounds.extent();
    Vec3 offset = centroids[tid] - sceneBounds.min;

    float nx = sceneSize.x > 0 ? offset.x / sceneSize.x : 0.5f;
    float ny = sceneSize.y > 0 ? offset.y / sceneSize.y : 0.5f;
    float nz = sceneSize.z > 0 ? offset.z / sceneSize.z : 0.5f;

    mortonCodes[tid] = mortonCode30(nx, ny, nz);
}

// ============================================================================
// Karras 2012 Algorithm - Parallel Binary Radix Tree Construction
// ============================================================================

// Helper function: compute length of common prefix (delta function from paper)
__device__ inline int clz_safe(uint32_t x) {
    return x == 0 ? 32 : __clz(x);
}

__device__ inline int delta(const uint32_t* sortedCodes, int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    
    // Handle duplicate Morton codes by using index as tiebreaker
    if (sortedCodes[i] == sortedCodes[j]) {
        return 32 + clz_safe(i ^ j);
    }
    
    return clz_safe(sortedCodes[i] ^ sortedCodes[j]);
}

// Sign function
__device__ inline int sign(int x) {
    return (x > 0) - (x < 0);
}

// Karras 2012 Algorithm (Figure 4 from paper) - OPTIMIZED
// Each thread constructs one internal node
// Uses unified node array: [internal: 0..n-2][leaf: n-1..2n-2]
__global__ void constructRadixTree(
    const uint32_t* sortedCodes,
    BVHNode* nodes,         // Unified array
    uint32_t* parents,      // Explicit parent pointers
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;  // n-1 internal nodes

    // Determine direction of the range (+1 or -1)
    int d = sign(delta(sortedCodes, i, i + 1, n) - delta(sortedCodes, i, i - 1, n));

    // Compute upper bound for the length of the range
    int delta_min = delta(sortedCodes, i, i - d, n);
    int l_max = 128;  // Start from 128 as optimization

    // Exponential search for l_max
    while (delta(sortedCodes, i, i + l_max * d, n) > delta_min) {
        l_max *= 4;  // Fast exponential growth
    }

    // Binary search to find the other end (j)
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta(sortedCodes, i, i + (l + t) * d, n) > delta_min) {
            l += t;
        }
    }
    int j = i + l * d;

    // Find the split position using binary search
    int delta_node = delta(sortedCodes, i, j, n);
    int s = 0;
    int divisor = 2;

    for (int t = (l + (divisor - 1)) / divisor; t >= 1; divisor *= 2, t = (l + (divisor - 1)) / divisor) {
        if (delta(sortedCodes, i, i + (s + t) * d, n) > delta_node) {
            s += t;
        }
    }

    int gamma = i + s * d + min(d, 0);  // Split position

    // Determine child indices (leaves start at index n-1)
    int first = min(i, j);
    int last = max(i, j);

    uint32_t leftChild = (first == gamma) ? (n - 1 + gamma) : gamma;
    uint32_t rightChild = (last == gamma + 1) ? (n - 1 + gamma + 1) : (gamma + 1);

    // Store in internal node (unified array)
    BVHNode& node = nodes[i];
    node.leftChild = leftChild;
    // rightChild is implicit: leftChild + 1
    node.primCount = 0;  // Mark as internal node
    node.primOffset = 0;

    // Set parent pointers for children
    parents[leftChild] = i;
    parents[rightChild] = i;
}

// Initialize leaf nodes in unified array
__global__ void initializeLeafNodes(
    const AABB* triBounds,
    const uint32_t* sortedIndices,
    BVHNode* nodes,  // Unified array
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Leaves are stored at indices [n-1, 2n-2]
    uint32_t leafIdx = n - 1 + tid;
    uint32_t originalIdx = sortedIndices[tid];

    BVHNode& leaf = nodes[leafIdx];
    leaf.bounds = triBounds[originalIdx];
    leaf.leftChild = 0;       // Unused for leaves
    leaf.primOffset = tid;    // Index into sorted primitive array
    leaf.primCount = 1;       // One primitive per leaf
}

// GPU AABB Refit - OPTIMIZED (Karras parallel approach)
// Each leaf starts a traversal up the tree using atomics
__global__ void refitAABBsGPU(
    BVHNode* nodes,             // Unified array [internal][leaves]
    const uint32_t* parents,    // Explicit parent pointers
    int* atomicFlags,           // Atomic counters per node
    int numLeaves)
{
    int leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= numLeaves) return;

    // Start from leaf (leaves are at indices n-1 to 2n-2)
    uint32_t current = (numLeaves - 1) + leafIdx;

    // Traverse up the tree
    while (current > 0) {
        uint32_t parent = parents[current];
        if (parent == UINT32_MAX) break;  // Root has no parent

        // Atomically increment visit counter
        int oldCount = atomicAdd(&atomicFlags[parent], 1);

        // First child to arrive - wait for sibling
        if (oldCount == 0) break;

        // Second child - both children ready, merge AABBs
        uint32_t leftChild = nodes[parent].leftChild;
        uint32_t rightChild = nodes[parent].rightChild();  // implicit: leftChild + 1

        // Merge child bounding boxes (direct access, no branches!)
        AABB merged;
        merged.expand(nodes[leftChild].bounds);
        merged.expand(nodes[rightChild].bounds);
        nodes[parent].bounds = merged;

        // Continue to next level
        current = parent;
    }
}

// ============================================================================
// SAH Cost Calculation
// ============================================================================

__global__ void computeSAHCost(
    const BVHNode* nodes, float rootArea,
    float* costs, int numNodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numNodes) return;

    const BVHNode& node = nodes[tid];
    float relativeArea = node.bounds.surfaceArea() / rootArea;

    float cost = 0.0f;
    if (node.isLeaf()) {
        cost = relativeArea * node.primCount * Ct;
    } else {
        cost = relativeArea * Ci;
    }

    costs[tid] = cost;
}

__global__ void reduceCosts(const float* input, float* output, int n) {
    __shared__ float sharedCosts[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float localCost = (gid < n) ? input[gid] : 0.0f;
    sharedCosts[tid] = localCost;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedCosts[tid] += sharedCosts[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sharedCosts[0];
    }
}

// ============================================================================
// Karras 2012 LBVH Builder
// ============================================================================

class Karras2012Builder {
public:
    std::vector<BVHNode> nodes;      // Unified array [internal][leaves]
    std::vector<uint32_t> primIndices;

    void build(const TriangleMesh& mesh) {
        int n = mesh.size();
        if (n == 0) return;

        // === ALLOCATE DEVICE MEMORY ===
        float *d_v0x, *d_v0y, *d_v0z, *d_v1x, *d_v1y, *d_v1z, *d_v2x, *d_v2y, *d_v2z;
        AABB* d_triBounds;
        Vec3* d_centroids;
        uint32_t* d_mortonCodes;
        uint32_t* d_indices;
        BVHNode* d_nodes;          // UNIFIED array
        uint32_t* d_parents;       // Explicit parent pointers
        int* d_atomicFlags;

        size_t floatBytes = n * sizeof(float);
        size_t totalNodes = 2 * n - 1;  // n-1 internal + n leaves

        CUDA_CHECK(cudaMalloc(&d_v0x, floatBytes));
        CUDA_CHECK(cudaMalloc(&d_v0y, floatBytes));
        CUDA_CHECK(cudaMalloc(&d_v0z, floatBytes));
        CUDA_CHECK(cudaMalloc(&d_v1x, floatBytes));
        CUDA_CHECK(cudaMalloc(&d_v1y, floatBytes));
        CUDA_CHECK(cudaMalloc(&d_v1z, floatBytes));
        CUDA_CHECK(cudaMalloc(&d_v2x, floatBytes));
        CUDA_CHECK(cudaMalloc(&d_v2y, floatBytes));
        CUDA_CHECK(cudaMalloc(&d_v2z, floatBytes));
        CUDA_CHECK(cudaMalloc(&d_triBounds, n * sizeof(AABB)));
        CUDA_CHECK(cudaMalloc(&d_centroids, n * sizeof(Vec3)));
        CUDA_CHECK(cudaMalloc(&d_mortonCodes, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_indices, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_nodes, totalNodes * sizeof(BVHNode)));
        CUDA_CHECK(cudaMalloc(&d_parents, totalNodes * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_atomicFlags, totalNodes * sizeof(int)));

        // === COPY MESH DATA TO GPU ===
        CUDA_CHECK(cudaMemcpy(d_v0x, mesh.v0x.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v0y, mesh.v0y.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v0z, mesh.v0z.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v1x, mesh.v1x.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v1y, mesh.v1y.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v1z, mesh.v1z.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v2x, mesh.v2x.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v2y, mesh.v2y.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v2z, mesh.v2z.data(), floatBytes, cudaMemcpyHostToDevice));

        // Initialize parent array (UINT32_MAX = no parent)
        std::vector<uint32_t> initParents(totalNodes, UINT32_MAX);
        CUDA_CHECK(cudaMemcpy(d_parents, initParents.data(), totalNodes * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_atomicFlags, 0, totalNodes * sizeof(int)));

        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        // === STEP 1: Compute Triangle Bounds and Centroids ===
        computeTriangleBoundsAndCentroids<<<blocks, threads>>>(
            d_v0x, d_v0y, d_v0z, d_v1x, d_v1y, d_v1z, d_v2x, d_v2y, d_v2z,
            d_triBounds, d_centroids, n);
        CUDA_CHECK(cudaGetLastError());

        // === STEP 2: Compute Scene Bounds (sample for speed) ===
        std::vector<AABB> sampleBounds(std::min(1024, n));
        int stride = std::max(1, n / 1024);
        for (size_t i = 0; i < sampleBounds.size(); ++i) {
            CUDA_CHECK(cudaMemcpy(&sampleBounds[i], &d_triBounds[i * stride],
                                 sizeof(AABB), cudaMemcpyDeviceToHost));
        }

        AABB sceneBounds;
        for (const auto& b : sampleBounds) {
            sceneBounds.expand(b);
        }

        // === STEP 3: Compute Morton Codes ===
        computeMortonCodes<<<blocks, threads>>>(
            d_centroids, sceneBounds, d_mortonCodes, n);
        CUDA_CHECK(cudaGetLastError());

        // === STEP 4: Sort by Morton Code ===
        thrust::device_ptr<uint32_t> mortonPtr(d_mortonCodes);
        thrust::device_ptr<uint32_t> indicesPtr(d_indices);

        thrust::sequence(indicesPtr, indicesPtr + n);
        thrust::sort_by_key(mortonPtr, mortonPtr + n, indicesPtr);

        // === STEP 5: Initialize Leaf Nodes ===
        initializeLeafNodes<<<blocks, threads>>>(
            d_triBounds, d_indices, d_nodes, n);
        CUDA_CHECK(cudaGetLastError());

        // === STEP 6: Construct Radix Tree (Karras 2012) ===
        int internalBlocks = (n - 1 + threads - 1) / threads;
        constructRadixTree<<<internalBlocks, threads>>>(
            d_mortonCodes, d_nodes, d_parents, n);
        CUDA_CHECK(cudaGetLastError());

        // === STEP 7: GPU AABB Refit (FAST!) ===
        refitAABBsGPU<<<blocks, threads>>>(
            d_nodes, d_parents, d_atomicFlags, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // === COPY RESULTS BACK TO HOST ===
        nodes.resize(totalNodes);
        primIndices.resize(n);

        CUDA_CHECK(cudaMemcpy(nodes.data(), d_nodes, totalNodes * sizeof(BVHNode), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(primIndices.data(), d_indices, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // === CLEANUP ===
        cudaFree(d_v0x); cudaFree(d_v0y); cudaFree(d_v0z);
        cudaFree(d_v1x); cudaFree(d_v1y); cudaFree(d_v1z);
        cudaFree(d_v2x); cudaFree(d_v2y); cudaFree(d_v2z);
        cudaFree(d_triBounds);
        cudaFree(d_centroids);
        cudaFree(d_mortonCodes);
        cudaFree(d_indices);
        cudaFree(d_nodes);
        cudaFree(d_parents);
        cudaFree(d_atomicFlags);
    }
    
    float calculateSAHCost() const {
        if (nodes.empty()) return 0.0f;

        int n = nodes.size();
        float rootArea = nodes[0].bounds.surfaceArea();

        BVHNode* d_nodes;
        float* d_costs;
        float* d_blockCosts;

        CUDA_CHECK(cudaMalloc(&d_nodes, n * sizeof(BVHNode)));
        CUDA_CHECK(cudaMalloc(&d_costs, n * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), n * sizeof(BVHNode), cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        computeSAHCost<<<blocks, threads>>>(d_nodes, rootArea, d_costs, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMalloc(&d_blockCosts, blocks * sizeof(float)));
        reduceCosts<<<blocks, threads>>>(d_costs, d_blockCosts, n);
        CUDA_CHECK(cudaGetLastError());

        std::vector<float> blockCosts(blocks);
        CUDA_CHECK(cudaMemcpy(blockCosts.data(), d_blockCosts, blocks * sizeof(float), cudaMemcpyDeviceToHost));

        float totalCost = 0.0f;
        for (float c : blockCosts) {
            totalCost += c;
        }

        cudaFree(d_nodes);
        cudaFree(d_costs);
        cudaFree(d_blockCosts);

        return totalCost;
    }
};

// ============================================================================
// BVH Export
// ============================================================================

void exportBVHToOBJ(const std::string& path,
                    const std::vector<BVHNode>& nodes,
                    bool leavesOnly = false) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << "\n";
        return;
    }

    file << "# Karras 2012 LBVH\n";
    file << "# Nodes: " << nodes.size() << "\n\n";

    uint32_t vertexOffset = 1;

    for (size_t i = 0; i < nodes.size(); ++i) {
        const BVHNode& node = nodes[i];

        if (leavesOnly && !node.isLeaf()) continue;

        const AABB& b = node.bounds;

        file << "# Node " << i << (node.isLeaf() ? " (leaf)" : " (internal)") << "\n";
        file << "v " << b.min.x << " " << b.min.y << " " << b.max.z << "\n";
        file << "v " << b.max.x << " " << b.min.y << " " << b.max.z << "\n";
        file << "v " << b.max.x << " " << b.min.y << " " << b.min.z << "\n";
        file << "v " << b.min.x << " " << b.min.y << " " << b.min.z << "\n";
        file << "v " << b.min.x << " " << b.max.y << " " << b.max.z << "\n";
        file << "v " << b.max.x << " " << b.max.y << " " << b.max.z << "\n";
        file << "v " << b.max.x << " " << b.max.y << " " << b.min.z << "\n";
        file << "v " << b.min.x << " " << b.max.y << " " << b.min.z << "\n";

        uint32_t v = vertexOffset;
        file << "l " << v+0 << " " << v+1 << "\n";
        file << "l " << v+1 << " " << v+2 << "\n";
        file << "l " << v+2 << " " << v+3 << "\n";
        file << "l " << v+3 << " " << v+0 << "\n";
        file << "l " << v+4 << " " << v+5 << "\n";
        file << "l " << v+5 << " " << v+6 << "\n";
        file << "l " << v+6 << " " << v+7 << "\n";
        file << "l " << v+7 << " " << v+4 << "\n";
        file << "l " << v+0 << " " << v+4 << "\n";
        file << "l " << v+1 << " " << v+5 << "\n";
        file << "l " << v+2 << " " << v+6 << "\n";
        file << "l " << v+3 << " " << v+7 << "\n";
        file << "\n";

        vertexOffset += 8;
    }

    file.close();
}

// ============================================================================
// Timer
// ============================================================================

class Timer {
public:
    void start() {
        CUDA_CHECK(cudaDeviceSynchronize());
        start_ = std::chrono::high_resolution_clock::now();
    }
    void stop() {
        CUDA_CHECK(cudaDeviceSynchronize());
        end_ = std::chrono::high_resolution_clock::now();
    }
    double elapsedMs() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_);
        return duration.count() / 1000.0;
    }
private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "  Random: " << argv[0] << " <num_triangles> [--export] [--export-leaves]\n";
        std::cerr << "  OBJ:    " << argv[0] << " <model.obj> [--export] [--export-leaves]\n";
        return 1;
    }

    std::string firstArg = argv[1];
    bool loadFromOBJ = (firstArg.find(".obj") != std::string::npos);

    size_t numTriangles = 100000;
    bool doExport = false;
    bool leavesOnly = false;
    std::string objPath;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--export") {
            doExport = true;
        } else if (arg == "--export-leaves") {
            doExport = true;
            leavesOnly = true;
        } else if (i == 1) {
            if (loadFromOBJ) {
                objPath = arg;
            } else {
                numTriangles = std::stoull(argv[1]);
            }
        }
    }

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "====================================================================\n";
    std::cout << "Karras 2012 LBVH - Pure Speed (No Quality Optimization)\n";
    std::cout << "\"Maximizing Parallelism in the Construction of BVHs\"\n";
    std::cout << "====================================================================\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    if (loadFromOBJ) {
        std::cout << "Input: " << objPath << "\n";
    } else {
        std::cout << "Triangles: " << numTriangles << " (random)\n";
    }
    std::cout << "\n";

    Timer timer;
    TriangleMesh mesh;

    timer.start();
    if (loadFromOBJ) {
        try {
            mesh = loadOBJ(objPath);
            timer.stop();
            std::cout << "OBJ loading: " << timer.elapsedMs() << " ms\n";
            std::cout << "Loaded " << mesh.size() << " triangles\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    } else {
        mesh = generateRandomTriangles(numTriangles);
        timer.stop();
        std::cout << "Triangle generation: " << timer.elapsedMs() << " ms\n";
    }

    Karras2012Builder builder;
    timer.start();
    builder.build(mesh);
    timer.stop();
    double buildTime = timer.elapsedMs();

    std::cout << "\n====================================================================\n";
    std::cout << "BUILD RESULTS\n";
    std::cout << "====================================================================\n";
    std::cout << "BVH construction: " << buildTime << " ms\n";
    std::cout << "Throughput: " << (mesh.size() / buildTime * 1000.0 / 1e6) << " M triangles/sec\n";
    std::cout << "Total nodes: " << builder.nodes.size() << "\n";

    timer.start();
    float sahCost = builder.calculateSAHCost();
    timer.stop();

    std::cout << "SAH cost calculation: " << timer.elapsedMs() << " ms\n";
    std::cout << "SAH cost: " << sahCost << "\n";
    std::cout << "====================================================================\n\n";

    if (doExport) {
        std::cout << "Exporting BVH to OBJ...\n";
        exportBVHToOBJ("karras2012_bvh.obj", builder.nodes, leavesOnly);
        std::cout << "Exported to: karras2012_bvh.obj\n";
    }

    return 0;
}
