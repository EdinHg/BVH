#include "lbvh_builder.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <iomanip>
#include <sstream>

// --- DEVICE FUNCTIONS ---

__device__ __forceinline__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    
    uint32_t xx = expandBits((uint32_t)x);
    uint32_t yy = expandBits((uint32_t)y);
    uint32_t zz = expandBits((uint32_t)z);
    
    return xx * 4 + yy * 2 + zz;
}

__device__ __forceinline__ int clz_custom(uint32_t x) {
    return x == 0 ? 32 : __clz(x);
}

__device__ int delta(const uint32_t* sortedMortonCodes, const uint32_t* sortedIndices, int numObjects, int i, int j) {
    if (j < 0 || j >= numObjects) return -1;

    uint32_t codeI = sortedMortonCodes[i];
    uint32_t codeJ = sortedMortonCodes[j];

    if (codeI == codeJ) {
        uint32_t idxI = sortedIndices[i];
        uint32_t idxJ = sortedIndices[j];
        return 32 + clz_custom(idxI ^ idxJ);
    }

    return clz_custom(codeI ^ codeJ);
}

// --- KERNELS ---

__global__ void kComputeBoundsAndCentroids(TrianglesSoADevice tris, int n, AABB_cw* triBBoxes, float3_cw* centroids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3_cw v0(tris.v0x[i], tris.v0y[i], tris.v0z[i]);
    float3_cw v1(tris.v1x[i], tris.v1y[i], tris.v1z[i]);
    float3_cw v2(tris.v2x[i], tris.v2y[i], tris.v2z[i]);

    float3_cw minVal, maxVal;
    minVal.x = fminf(v0.x, fminf(v1.x, v2.x)); 
    minVal.y = fminf(v0.y, fminf(v1.y, v2.y)); 
    minVal.z = fminf(v0.z, fminf(v1.z, v2.z));
    maxVal.x = fmaxf(v0.x, fmaxf(v1.x, v2.x)); 
    maxVal.y = fmaxf(v0.y, fmaxf(v1.y, v2.y)); 
    maxVal.z = fmaxf(v0.z, fmaxf(v1.z, v2.z));

    triBBoxes[i].min = minVal; 
    triBBoxes[i].max = maxVal;
    centroids[i] = (minVal + maxVal) * 0.5f;
}

__global__ void kComputeMortonCodes(const float3_cw* centroids, int n, AABB_cw sceneBounds, uint32_t* mortonCodes, uint32_t* indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3_cw c = centroids[i];
    float3_cw minB = sceneBounds.min;
    float3_cw extents = sceneBounds.max - sceneBounds.min;

    float x = (c.x - minB.x) / extents.x;
    float y = (c.y - minB.y) / extents.y;
    float z = (c.z - minB.z) / extents.z;

    mortonCodes[i] = morton3D(x, y, z);
    indices[i] = i;
}

__global__ void kBuildInternalNodes(const uint32_t* sortedMortonCodes, const uint32_t* sortedIndices, LBVHNode* nodes, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects - 1) return;

    int d = (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + 1) - 
             delta(sortedMortonCodes, sortedIndices, numObjects, i, i - 1)) >= 0 ? 1 : -1;

    int min_delta = delta(sortedMortonCodes, sortedIndices, numObjects, i, i - d);
    int l_max = 2;
    while (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + l_max * d) > min_delta) l_max *= 2;

    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + (l + t) * d) > min_delta) l += t;
    }

    int j = i + l * d;
    int delta_node = delta(sortedMortonCodes, sortedIndices, numObjects, i, j);
    int s = 0;
    int first = (d > 0) ? i : j;
    int last = (d > 0) ? j : i;
    int len = last - first;
    int t = len;

    do {
        t = (t + 1) >> 1;
        if (delta(sortedMortonCodes, sortedIndices, numObjects, first, first + s + t) > delta_node) s += t;
    } while (t > 1);

    int split = first + s;
    uint32_t leftIdx = (split == first) ? (numObjects - 1) + split : split;
    uint32_t rightIdx = (split + 1 == last) ? (numObjects - 1) + split + 1 : split + 1;

    nodes[i].leftChild = leftIdx;
    nodes[i].rightChild = rightIdx;
    nodes[leftIdx].parent = i;
    nodes[rightIdx].parent = i;
}

__global__ void kInitLeafNodes(LBVHNode* nodes, const AABB_cw* triBBoxes, const uint32_t* sortedIndices, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    int leafIdx = (numObjects - 1) + i;
    uint32_t originalIndex = sortedIndices[i];

    nodes[leafIdx].bbox = triBBoxes[originalIndex];
    nodes[leafIdx].leftChild = originalIndex | 0x80000000;
    nodes[leafIdx].rightChild = 0xFFFFFFFF;
}

__global__ void kRefitHierarchy(LBVHNode* nodes, int* atomicCounters, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    uint32_t idx = (numObjects - 1) + i;

    while (idx != 0) {
        uint32_t parent = nodes[idx].parent;
        int oldVal = atomicAdd(&atomicCounters[parent], 1);
        if (oldVal == 0) return;

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
     
        nodes[parent].bbox = unionBox;
        idx = parent;
    }
}

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

// --- BUILDER IMPLEMENTATION ---

LBVHBuilderCUDA::LBVHBuilderCUDA() : lastBuildTimeMs(0.0f),
    time_centroids(0), time_morton(0), time_sort(0), time_topology(0), time_refit(0) {
    cudaEventCreate(&start);
    cudaEventCreate(&e_centroids);
    cudaEventCreate(&e_morton);
    cudaEventCreate(&e_sort);
    cudaEventCreate(&e_topology);
    cudaEventCreate(&stop);
}

LBVHBuilderCUDA::~LBVHBuilderCUDA() {
    cudaEventDestroy(start);
    cudaEventDestroy(e_centroids);
    cudaEventDestroy(e_morton);
    cudaEventDestroy(e_sort);
    cudaEventDestroy(e_topology);
    cudaEventDestroy(stop);
}

TrianglesSoADevice LBVHBuilderCUDA::getDevicePtrs() {
    return {
        thrust::raw_pointer_cast(d_v0x.data()), thrust::raw_pointer_cast(d_v0y.data()), thrust::raw_pointer_cast(d_v0z.data()),
        thrust::raw_pointer_cast(d_v1x.data()), thrust::raw_pointer_cast(d_v1y.data()), thrust::raw_pointer_cast(d_v1z.data()),
        thrust::raw_pointer_cast(d_v2x.data()), thrust::raw_pointer_cast(d_v2y.data()), thrust::raw_pointer_cast(d_v2z.data())
    };
}

void LBVHBuilderCUDA::prepareData(const TriangleMesh& mesh) {
    int n = mesh.size();
    d_v0x = mesh.v0x; d_v0y = mesh.v0y; d_v0z = mesh.v0z;
    d_v1x = mesh.v1x; d_v1y = mesh.v1y; d_v1z = mesh.v1z;
    d_v2x = mesh.v2x; d_v2y = mesh.v2y; d_v2z = mesh.v2z;
    
    d_triBBoxes.resize(n);  
    d_centroids.resize(n);
    d_mortonCodes.resize(n);
    d_indices.resize(n);
    d_nodes.resize(2 * n - 1);
    d_atomicFlags.resize(2 * n - 1);
}

void LBVHBuilderCUDA::runCompute(int n) {
    if (n == 0) return;

    thrust::fill(d_atomicFlags.begin(), d_atomicFlags.end(), 0);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEventRecord(start);

    // 1. Centroids
    kComputeBoundsAndCentroids<<<gridSize, blockSize>>>(getDevicePtrs(), n,
        thrust::raw_pointer_cast(d_triBBoxes.data()),
        thrust::raw_pointer_cast(d_centroids.data()));
    
    AABB_cw init = AABB_cw::empty(); 
    AABB_cw sceneBounds = thrust::reduce(d_triBBoxes.begin(), d_triBBoxes.end(), init, AABBReduce());
    
    cudaEventRecord(e_centroids);

    // 2. Morton Codes
    kComputeMortonCodes<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_centroids.data()),
        n, sceneBounds,
        thrust::raw_pointer_cast(d_mortonCodes.data()),
        thrust::raw_pointer_cast(d_indices.data()));
        
    cudaEventRecord(e_morton);

    // 3. Sort
    thrust::sort_by_key(d_mortonCodes.begin(), d_mortonCodes.end(), d_indices.begin());
    
    cudaEventRecord(e_sort);

    // 4. Topology
    kBuildInternalNodes<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_mortonCodes.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        thrust::raw_pointer_cast(d_nodes.data()),
        n);

    kInitLeafNodes<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_triBBoxes.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        n);
        
    cudaEventRecord(e_topology);

    // 5. Refit
    kRefitHierarchy<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_atomicFlags.data()),
        n);
        
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Record timing
    cudaEventElapsedTime(&time_centroids, start, e_centroids);
    cudaEventElapsedTime(&time_morton, e_centroids, e_morton);
    cudaEventElapsedTime(&time_sort, e_morton, e_sort);
    cudaEventElapsedTime(&time_topology, e_sort, e_topology);
    cudaEventElapsedTime(&time_refit, e_topology, stop);
    lastBuildTimeMs = time_centroids + time_morton + time_sort + time_topology + time_refit;
}

void LBVHBuilderCUDA::downloadResults(int n) {
    // Download internal LBVH nodes
    std::vector<LBVHNode> lbvh_nodes(d_nodes.size());
    thrust::copy(d_nodes.begin(), d_nodes.end(), lbvh_nodes.begin());
    
    // Download indices
    h_indices.resize(d_indices.size());
    thrust::copy(d_indices.begin(), d_indices.end(), h_indices.begin());
    
    // Convert to unified BVHNode format
    h_nodes.clear();
    h_nodes.reserve(lbvh_nodes.size());
    
    for (const auto& lnode : lbvh_nodes) {
        BVHNode node;
        node.bbox = lnode.bbox;
        node.leftChild = static_cast<int32_t>(lnode.leftChild);
        node.rightChild = static_cast<int32_t>(lnode.rightChild);
        h_nodes.push_back(node);
    }
}

void LBVHBuilderCUDA::build(const TriangleMesh& mesh) {
    prepareData(mesh);
    runCompute(mesh.size());
    downloadResults(mesh.size());
}

std::string LBVHBuilderCUDA::getTimingBreakdown() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "  Bounds/Centroids: " << time_centroids << " ms\n";
    oss << "  Morton Codes:     " << time_morton << " ms\n";
    oss << "  Radix Sort:       " << time_sort << " ms\n";
    oss << "  Topology Build:   " << time_topology << " ms\n";
    oss << "  Bottom-Up Refit:  " << time_refit << " ms";
    return oss.str();
}
