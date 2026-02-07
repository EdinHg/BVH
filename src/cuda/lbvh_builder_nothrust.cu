#include "lbvh_builder_nothrust.cuh"
#include "radix_sort_kv.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>
#include <sstream>

// --- DEVICE FUNCTIONS ---

__device__ __forceinline__ uint32_t expandBits_nt(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ uint32_t morton3D_nt(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    uint32_t xx = expandBits_nt((uint32_t)x);
    uint32_t yy = expandBits_nt((uint32_t)y);
    uint32_t zz = expandBits_nt((uint32_t)z);
    return xx * 4 + yy * 2 + zz;
}

__device__ __forceinline__ int clz_custom_nt(uint32_t x) {
    return x == 0 ? 32 : __clz(x);
}

__device__ int delta_nt(const uint32_t* sortedMortonCodes, const uint32_t* sortedIndices, int numObjects, int i, int j) {
    if (j < 0 || j >= numObjects) return -1;
    uint32_t codeI = sortedMortonCodes[i];
    uint32_t codeJ = sortedMortonCodes[j];
    if (codeI == codeJ) {
        uint32_t idxI = sortedIndices[i];
        uint32_t idxJ = sortedIndices[j];
        return 32 + clz_custom_nt(idxI ^ idxJ);
    }
    return clz_custom_nt(codeI ^ codeJ);
}

// --- KERNELS ---

__global__ void kComputeBoundsAndCentroids_nt(TrianglesSoADevice tris, int n, AABB_cw* triBBoxes, float3_cw* centroids) {
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

__global__ void kFillZero_nt(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 0;
}

__global__ void kReduceBounds_Step1_nt(const AABB_cw* input, AABB_cw* output, int n) {
    __shared__ AABB_cw sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid].min = float3_cw(FLT_MAX, FLT_MAX, FLT_MAX);
        sdata[tid].max = float3_cw(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
    __syncthreads();

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

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void kReduceBounds_Step2_nt(AABB_cw* data, int n) {
    __shared__ float sdata_min_x[256];
    __shared__ float sdata_min_y[256];
    __shared__ float sdata_min_z[256];
    __shared__ float sdata_max_x[256];
    __shared__ float sdata_max_y[256];
    __shared__ float sdata_max_z[256];

    int tid = threadIdx.x;

    float local_min_x = FLT_MAX;
    float local_min_y = FLT_MAX;
    float local_min_z = FLT_MAX;
    float local_max_x = -FLT_MAX;
    float local_max_y = -FLT_MAX;
    float local_max_z = -FLT_MAX;

    // Loop over the input array (stride by blockDim.x)
    // process all blocks from Step 1, not just the first 256.
    int i = tid;
    while (i < n) {
        local_min_x = fminf(local_min_x, data[i].min.x);
        local_min_y = fminf(local_min_y, data[i].min.y);
        local_min_z = fminf(local_min_z, data[i].min.z);
        local_max_x = fmaxf(local_max_x, data[i].max.x);
        local_max_y = fmaxf(local_max_y, data[i].max.y);
        local_max_z = fmaxf(local_max_z, data[i].max.z);
        i += blockDim.x;
    }

    // Store accumulated local result into shared memory
    sdata_min_x[tid] = local_min_x;
    sdata_min_y[tid] = local_min_y;
    sdata_min_z[tid] = local_min_z;
    sdata_max_x[tid] = local_max_x;
    sdata_max_y[tid] = local_max_y;
    sdata_max_z[tid] = local_max_z;
    
    __syncthreads();

    // Standard Block Reduction (256 -> 1)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_min_x[tid] = fminf(sdata_min_x[tid], sdata_min_x[tid + s]);
            sdata_min_y[tid] = fminf(sdata_min_y[tid], sdata_min_y[tid + s]);
            sdata_min_z[tid] = fminf(sdata_min_z[tid], sdata_min_z[tid + s]);
            sdata_max_x[tid] = fmaxf(sdata_max_x[tid], sdata_max_x[tid + s]);
            sdata_max_y[tid] = fmaxf(sdata_max_y[tid], sdata_max_y[tid + s]);
            sdata_max_z[tid] = fmaxf(sdata_max_z[tid], sdata_max_z[tid + s]);
        }
        __syncthreads();
    }

    // Write final result to index 0
    if (tid == 0) {
        data[0].min.x = sdata_min_x[0];
        data[0].min.y = sdata_min_y[0];
        data[0].min.z = sdata_min_z[0];
        data[0].max.x = sdata_max_x[0];
        data[0].max.y = sdata_max_y[0];
        data[0].max.z = sdata_max_z[0];
    }
}

__global__ void kComputeMortonCodes_nt(const float3_cw* centroids, int n, AABB_cw sceneBounds, uint32_t* mortonCodes, uint32_t* indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3_cw c = centroids[i];
    float3_cw minB = sceneBounds.min;
    float3_cw extents = sceneBounds.max - sceneBounds.min;

    float x = (c.x - minB.x) / extents.x;
    float y = (c.y - minB.y) / extents.y;
    float z = (c.z - minB.z) / extents.z;

    mortonCodes[i] = morton3D_nt(x, y, z);
    indices[i] = i;
}

__global__ void kBuildInternalNodes_nt(const uint32_t* sortedMortonCodes, const uint32_t* sortedIndices, LBVHNodeNoThrust* nodes, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects - 1) return;

    int d = (delta_nt(sortedMortonCodes, sortedIndices, numObjects, i, i + 1) - 
             delta_nt(sortedMortonCodes, sortedIndices, numObjects, i, i - 1)) >= 0 ? 1 : -1;

    int min_delta = delta_nt(sortedMortonCodes, sortedIndices, numObjects, i, i - d);
    int l_max = 2;
    while (delta_nt(sortedMortonCodes, sortedIndices, numObjects, i, i + l_max * d) > min_delta) l_max *= 2;

    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta_nt(sortedMortonCodes, sortedIndices, numObjects, i, i + (l + t) * d) > min_delta) l += t;
    }

    int j = i + l * d;
    int delta_node = delta_nt(sortedMortonCodes, sortedIndices, numObjects, i, j);
    int s = 0;
    int first, last;
    if (d > 0) { first = i; last = j; }
    else       { first = j; last = i; }

    int len = last - first;
    int t = len;

    do {
        t = (t + 1) >> 1;
        if (delta_nt(sortedMortonCodes, sortedIndices, numObjects, first, first + s + t) > delta_node) {
            s += t;
        }
    } while (t > 1);

    int split = first + s;
    uint32_t leftIdx, rightIdx;

    if (split == first) leftIdx = (numObjects - 1) + split;
    else                leftIdx = split;

    if (split + 1 == last) rightIdx = (numObjects - 1) + split + 1;
    else                   rightIdx = split + 1;

    nodes[i].leftChild = leftIdx;
    nodes[i].rightChild = rightIdx;
    nodes[leftIdx].parent = i;
    nodes[rightIdx].parent = i;
}

__global__ void kInitLeafNodes_nt(LBVHNodeNoThrust* nodes, const AABB_cw* triBBoxes, const uint32_t* sortedIndices, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    int leafIdx = (numObjects - 1) + i;
    uint32_t originalIndex = sortedIndices[i];

    nodes[leafIdx].bbox = triBBoxes[originalIndex];
    nodes[leafIdx].leftChild = originalIndex | 0x80000000;
    nodes[leafIdx].rightChild = 0xFFFFFFFF;
}

__global__ void kRefitHierarchy_nt(LBVHNodeNoThrust* nodes, int* atomicCounters, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    uint32_t idx = (numObjects - 1) + i;

    while (idx != 0) {
        uint32_t parent = nodes[idx].parent;
        
        __threadfence();
        
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

// --- BUILDER IMPLEMENTATION ---

LBVHBuilderNoThrust::LBVHBuilderNoThrust() :
    d_v0x(nullptr), d_v0y(nullptr), d_v0z(nullptr),
    d_v1x(nullptr), d_v1y(nullptr), d_v1z(nullptr),
    d_v2x(nullptr), d_v2y(nullptr), d_v2z(nullptr),
    d_triBBoxes(nullptr), d_centroids(nullptr),
    d_mortonCodes(nullptr), d_indices(nullptr),
    d_nodes(nullptr), d_atomicFlags(nullptr),
    d_boundsReduction(nullptr), numTriangles(0),
    lastBuildTimeMs(0.0f),
    time_centroids(0), time_morton(0), time_sort(0), time_topology(0), time_refit(0) {
    cudaEventCreate(&start);
    cudaEventCreate(&e_centroids);
    cudaEventCreate(&e_morton);
    cudaEventCreate(&e_sort);
    cudaEventCreate(&e_topology);
    cudaEventCreate(&stop);
}

LBVHBuilderNoThrust::~LBVHBuilderNoThrust() {
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

    cudaEventDestroy(start);
    cudaEventDestroy(e_centroids);
    cudaEventDestroy(e_morton);
    cudaEventDestroy(e_sort);
    cudaEventDestroy(e_topology);
    cudaEventDestroy(stop);
}

TrianglesSoADevice LBVHBuilderNoThrust::getDevicePtrs() {
    return {d_v0x, d_v0y, d_v0z, d_v1x, d_v1y, d_v1z, d_v2x, d_v2y, d_v2z};
}

void LBVHBuilderNoThrust::prepareData(const TriangleMesh& tris) {
    numTriangles = tris.size();
    if (numTriangles == 0) return;

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
    CUDA_CHECK(cudaMalloc(&d_nodes, (2 * numTriangles - 1) * sizeof(LBVHNodeNoThrust)));
    CUDA_CHECK(cudaMalloc(&d_atomicFlags, (2 * numTriangles - 1) * sizeof(int)));

    int blockSize = 256;
    int gridSize = (numTriangles + blockSize - 1) / blockSize;
    kFillZero_nt<<<gridSize, blockSize>>>(d_atomicFlags, 2 * numTriangles - 1);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void LBVHBuilderNoThrust::runCompute(int n) {
    if (n == 0) return;

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEventRecord(start);

    // 1. Compute Bounds and Centroids
    kComputeBoundsAndCentroids_nt<<<gridSize, blockSize>>>(getDevicePtrs(), n, d_triBBoxes, d_centroids);

    // 2. Reduce scene bounds
    int numBlocks = gridSize;
    CUDA_CHECK(cudaMalloc(&d_boundsReduction, numBlocks * sizeof(AABB_cw)));

    kReduceBounds_Step1_nt<<<numBlocks, blockSize>>>(d_triBBoxes, d_boundsReduction, n);

    if (numBlocks > 1) {
        kReduceBounds_Step2_nt<<<1, blockSize>>>(d_boundsReduction, numBlocks);
    }

    AABB_cw sceneBounds;
    CUDA_CHECK(cudaMemcpy(&sceneBounds, d_boundsReduction, sizeof(AABB_cw), cudaMemcpyDeviceToHost));

    cudaEventRecord(e_centroids);

    // 3. Compute Morton Codes
    kComputeMortonCodes_nt<<<gridSize, blockSize>>>(d_centroids, n, sceneBounds, d_mortonCodes, d_indices);

    cudaEventRecord(e_morton);

    // 4. Sort (Custom Radix Sort)
    radixSortKeyValue30bit(d_mortonCodes, d_indices, n);

    cudaEventRecord(e_sort);

    // 5. Build Hierarchy
    int internalGridSize = (n - 1 + blockSize - 1) / blockSize;
    kBuildInternalNodes_nt<<<internalGridSize, blockSize>>>(d_mortonCodes, d_indices, d_nodes, n);

    kInitLeafNodes_nt<<<gridSize, blockSize>>>(d_nodes, d_triBBoxes, d_indices, n);

    cudaEventRecord(e_topology);

    // 6. Refit BBoxes
    kRefitHierarchy_nt<<<gridSize, blockSize>>>(d_nodes, d_atomicFlags, n);

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

void LBVHBuilderNoThrust::downloadResults(int n) {
    std::vector<LBVHNodeNoThrust> lbvh_nodes(2 * n - 1);
    CUDA_CHECK(cudaMemcpy(lbvh_nodes.data(), d_nodes, (2 * n - 1) * sizeof(LBVHNodeNoThrust), cudaMemcpyDeviceToHost));

    h_indices.resize(n);
    CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

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

void LBVHBuilderNoThrust::build(const TriangleMesh& mesh) {
    prepareData(mesh);
    runCompute(mesh.size());
    downloadResults(mesh.size());
}

std::string LBVHBuilderNoThrust::getTimingBreakdown() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "  Bounds/Centroids: " << time_centroids << " ms\n";
    oss << "  Morton Codes:     " << time_morton << " ms\n";
    oss << "  Radix Sort:       " << time_sort << " ms\n";
    oss << "  Topology Build:   " << time_topology << " ms\n";
    oss << "  Bottom-Up Refit:  " << time_refit << " ms";
    return oss.str();
}
