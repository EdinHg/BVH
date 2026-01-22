#include "lbvh_plus_builder.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <iomanip>
#include <sstream>

#define MAX_TREELET_SIZE 7
#define INF 1e30f

namespace {

__device__ __forceinline__ uint32_t expandBits_P(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ uint32_t morton3D_P(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    return expandBits_P((uint32_t)x) * 4 + expandBits_P((uint32_t)y) * 2 + expandBits_P((uint32_t)z);
}

__device__ __forceinline__ int clz_custom_P(uint32_t x) {
    return x == 0 ? 32 : __clz(x);
}

__device__ int delta_P(const uint32_t* sortedMortonCodes, const uint32_t* sortedIndices, int numObjects, int i, int j) {
    if (j < 0 || j >= numObjects) return -1;
    uint32_t codeI = sortedMortonCodes[i];
    uint32_t codeJ = sortedMortonCodes[j];
    if (codeI == codeJ) return 32 + clz_custom_P(sortedIndices[i] ^ sortedIndices[j]);
    return clz_custom_P(codeI ^ codeJ);
}

__device__ __forceinline__ AABB_cw unionAABB_P(const AABB_cw& a, const AABB_cw& b) {
    AABB_cw res;
    res.min.x = fminf(a.min.x, b.min.x);
    res.min.y = fminf(a.min.y, b.min.y);
    res.min.z = fminf(a.min.z, b.min.z);
    res.max.x = fmaxf(a.max.x, b.max.x);
    res.max.y = fmaxf(a.max.y, b.max.y);
    res.max.z = fmaxf(a.max.z, b.max.z);
    return res;
}

__device__ __forceinline__ float area_P(const AABB_cw& b) {
    float3_cw d = b.max - b.min;
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

// Safe bitmask DP optimization with bounds checking
__device__ void optimizeTreeletDP(LBVHNode* nodes, int rootIdx) {
    // Local stack for traversing the treelet.
    int leafNodes[MAX_TREELET_SIZE];
    AABB_cw leafBoxes[MAX_TREELET_SIZE];
    int internalNodes[MAX_TREELET_SIZE];
    int leafCount = 0, internalCount = 0;
    
    // Collect all nodes within the treelet using a depth-limited search.
    int stack[32];
    int top = 0;
    stack[top++] = rootIdx;
    
    while(top > 0 && leafCount < MAX_TREELET_SIZE) {
        int curr = stack[--top];
        LBVHNode& n = nodes[curr];
        
        if (n.leftChild & 0x80000000) {
            // Leaf node
            if (leafCount < MAX_TREELET_SIZE) {
                leafNodes[leafCount] = curr;
                leafBoxes[leafCount] = n.bbox;
                leafCount++;
            }
        } else {
            // Internal node
            if (curr != rootIdx && internalCount < MAX_TREELET_SIZE - 1) {
                internalNodes[internalCount++] = curr;
            }
            // Add children to stack if we haven't exceeded our depth limit.
            if (top < 30) {
                stack[top++] = n.rightChild;
                stack[top++] = n.leftChild;
            }
        }
    }
    
    // proceed only if we have a valid cluster of nodes.
    if (leafCount < 3 || leafCount > MAX_TREELET_SIZE) return;
    
    int n = leafCount;
    int fullMask = (1 << n) - 1;
    
    // Precompute SAH costs for all subsets.
    AABB_cw subsetBox[128];
    float subsetArea[128];
    float cost[128];
    int partition[128];
    
    // Initialize single-leaf subsets
    for (int i = 0; i < n; i++) {
        int mask = 1 << i;
        subsetBox[mask] = leafBoxes[i];
        subsetArea[mask] = area_P(leafBoxes[i]);
        cost[mask] = 0.0f;
        partition[mask] = 0;
    }
    
    // Build up larger subsets
    for (int mask = 1; mask <= fullMask; mask++) {
        int popcount = __popc(mask);
        if (popcount == 1) continue;
        
        int firstBit = mask & (-mask);
        int rest = mask ^ firstBit;
        subsetBox[mask] = unionAABB_P(subsetBox[firstBit], subsetBox[rest]);
        subsetArea[mask] = area_P(subsetBox[mask]);
    }
    
    // Solve DP: iterate by subset size to guarantee sub-problems are ready.
    for (int size = 2; size <= n; size++) {
        for (int mask = 1; mask <= fullMask; mask++) {
            if (__popc(mask) != size) continue;
            
            float bestCost = INF;
            int bestPart = 0;
            
            // Enumerate all partitions
            for (int left = (mask - 1) & mask; left > 0; left = (left - 1) & mask) {
                int right = mask ^ left;
                if (left >= right) continue; // Avoid duplicates
                
                float c = subsetArea[mask] + cost[left] + cost[right];
                if (c < bestCost) {
                    bestCost = c;
                    bestPart = left;
                }
            }
            
            cost[mask] = bestCost;
            partition[mask] = bestPart;
        }
    }
    
    // Reconstruct optimal tree
    struct Task { int nodeIdx, mask; };
    Task tasks[MAX_TREELET_SIZE];
    int taskTop = 0, nextInt = 0;
    tasks[taskTop++] = {rootIdx, fullMask};
    
    while (taskTop > 0) {
        Task t = tasks[--taskTop];
        int pop = __popc(t.mask);
        if (pop <= 1) continue;
        
        int leftMask = partition[t.mask];
        int rightMask = t.mask ^ leftMask;
        
        // Safety check
        if (leftMask == 0 || rightMask == 0) continue;
        
        int leftNode, rightNode;
        
        // Left child
        if (__popc(leftMask) == 1) {
            int idx = __ffs(leftMask) - 1;
            if (idx >= 0 && idx < leafCount)
                leftNode = leafNodes[idx];
            else continue; // Safety
        } else {
            if (nextInt < internalCount)
                leftNode = internalNodes[nextInt++];
            else continue; // Safety
        }
        
        // Right child  
        if (__popc(rightMask) == 1) {
            int idx = __ffs(rightMask) - 1;
            if (idx >= 0 && idx < leafCount)
                rightNode = leafNodes[idx];
            else continue; // Safety
        } else {
            if (nextInt < internalCount)
                rightNode = internalNodes[nextInt++];
            else continue; // Safety
        }
        
        // Push tasks for multi-leaf children
        if (__popc(leftMask) > 1 && taskTop < MAX_TREELET_SIZE - 1)
            tasks[taskTop++] = {leftNode, leftMask};
        if (__popc(rightMask) > 1 && taskTop < MAX_TREELET_SIZE - 1)
            tasks[taskTop++] = {rightNode, rightMask};
        
        // Update tree structure
        nodes[t.nodeIdx].leftChild = leftNode;
        nodes[t.nodeIdx].rightChild = rightNode;
        nodes[t.nodeIdx].bbox = subsetBox[t.mask];
        nodes[leftNode].parent = t.nodeIdx;
        nodes[rightNode].parent = t.nodeIdx;
    }
}

struct AABBReduceP {
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

} // End anonymous namespace

__global__ void kRefitAndComputeSizes(LBVHNode* nodes, int* atomicCounters, int* subtreeSizes, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    uint32_t idx = (numObjects - 1) + i;
    subtreeSizes[idx] = 1;

    while (idx != 0) {
        uint32_t parent = nodes[idx].parent;
        int oldVal = atomicAdd(&atomicCounters[parent], 1);
        if (oldVal == 0) return;

        uint32_t left = nodes[parent].leftChild;
        uint32_t right = nodes[parent].rightChild;
        subtreeSizes[parent] = subtreeSizes[left] + subtreeSizes[right];
        nodes[parent].bbox = unionAABB_P(nodes[left].bbox, nodes[right].bbox);
        idx = parent;
    }
}

__global__ void kOptimizeMaximalTreelets(LBVHNode* nodes, int* subtreeSizes, int numInternalNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numInternalNodes) return;

    int mySize = subtreeSizes[i];
    if (mySize < 4 || mySize > MAX_TREELET_SIZE) return;
    
    // Identify if this node is the root of a maximal treelet (either it is the BVH root, or its parent is outside the size threshold).
    bool isMaximal = false;
    if (i == 0) {
        isMaximal = true;
    } else {
        uint32_t parentIdx = nodes[i].parent;
        if (parentIdx < (uint32_t)numInternalNodes) {  // Safety check
            isMaximal = (subtreeSizes[parentIdx] > MAX_TREELET_SIZE);
        }
    }
    
    if (!isMaximal) return;
    
    optimizeTreeletDP(nodes, i);
}

__global__ void kComputeBoundsAndCentroids_P(TrianglesSoADevice tris, int n, AABB_cw* triBBoxes, float3_cw* centroids) {
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

__global__ void kComputeMortonCodes_P(const float3_cw* centroids, int n, AABB_cw sceneBounds, uint32_t* mortonCodes, uint32_t* indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3_cw c = centroids[i];
    float3_cw minB = sceneBounds.min;
    float3_cw extents = sceneBounds.max - sceneBounds.min;
    mortonCodes[i] = morton3D_P((c.x - minB.x) / extents.x, (c.y - minB.y) / extents.y, (c.z - minB.z) / extents.z);
    indices[i] = i;
}

__global__ void kBuildInternalNodes_P(const uint32_t* sortedMortonCodes, const uint32_t* sortedIndices, LBVHNode* nodes, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects - 1) return;

    int d = (delta_P(sortedMortonCodes, sortedIndices, numObjects, i, i + 1) - 
             delta_P(sortedMortonCodes, sortedIndices, numObjects, i, i - 1)) >= 0 ? 1 : -1;

    int min_delta = delta_P(sortedMortonCodes, sortedIndices, numObjects, i, i - d);
    int l_max = 2;
    while (delta_P(sortedMortonCodes, sortedIndices, numObjects, i, i + l_max * d) > min_delta) l_max *= 2;

    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2)
        if (delta_P(sortedMortonCodes, sortedIndices, numObjects, i, i + (l + t) * d) > min_delta) l += t;

    int j = i + l * d;
    int delta_node = delta_P(sortedMortonCodes, sortedIndices, numObjects, i, j);
    int s = 0, first = (d > 0) ? i : j, last = (d > 0) ? j : i, t = last - first;

    do {
        t = (t + 1) >> 1;
        if (delta_P(sortedMortonCodes, sortedIndices, numObjects, first, first + s + t) > delta_node) s += t;
    } while (t > 1);

    int split = first + s;
    uint32_t leftIdx = (split == first) ? (numObjects - 1) + split : split;
    uint32_t rightIdx = (split + 1 == last) ? (numObjects - 1) + split + 1 : split + 1;

    nodes[i].leftChild = leftIdx;
    nodes[i].rightChild = rightIdx;
    nodes[leftIdx].parent = i;
    nodes[rightIdx].parent = i;
}

__global__ void kInitLeafNodes_P(LBVHNode* nodes, const AABB_cw* triBBoxes, const uint32_t* sortedIndices, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    int leafIdx = (numObjects - 1) + i;
    uint32_t originalIndex = sortedIndices[i];
    nodes[leafIdx].bbox = triBBoxes[originalIndex];
    nodes[leafIdx].leftChild = originalIndex | 0x80000000;
    nodes[leafIdx].rightChild = 0xFFFFFFFF;
}

// --- CLASS IMPLEMENTATION ---

LBVHPlusBuilderCUDA::LBVHPlusBuilderCUDA() : lastBuildTimeMs(0.0f),
    time_centroids(0), time_morton(0), time_sort(0), time_topology(0), time_refit(0), time_optimize(0) {
    cudaEventCreate(&start);
    cudaEventCreate(&e_centroids);
    cudaEventCreate(&e_morton);
    cudaEventCreate(&e_sort);
    cudaEventCreate(&e_topology);
    cudaEventCreate(&e_refit);
    cudaEventCreate(&stop);
}

LBVHPlusBuilderCUDA::~LBVHPlusBuilderCUDA() {
    cudaEventDestroy(start);
    cudaEventDestroy(e_centroids);
    cudaEventDestroy(e_morton);
    cudaEventDestroy(e_sort);
    cudaEventDestroy(e_topology);
    cudaEventDestroy(e_refit);
    cudaEventDestroy(stop);
}

TrianglesSoADevice LBVHPlusBuilderCUDA::getDevicePtrs() {
    return {
        thrust::raw_pointer_cast(d_v0x.data()), thrust::raw_pointer_cast(d_v0y.data()), thrust::raw_pointer_cast(d_v0z.data()),
        thrust::raw_pointer_cast(d_v1x.data()), thrust::raw_pointer_cast(d_v1y.data()), thrust::raw_pointer_cast(d_v1z.data()),
        thrust::raw_pointer_cast(d_v2x.data()), thrust::raw_pointer_cast(d_v2y.data()), thrust::raw_pointer_cast(d_v2z.data())
    };
}

void LBVHPlusBuilderCUDA::prepareData(const TriangleMesh& mesh) {
    int n = mesh.size();
    d_v0x = mesh.v0x; d_v0y = mesh.v0y; d_v0z = mesh.v0z;
    d_v1x = mesh.v1x; d_v1y = mesh.v1y; d_v1z = mesh.v1z;
    d_v2x = mesh.v2x; d_v2y = mesh.v2y; d_v2z = mesh.v2z;
    
    d_triBBoxes.resize(n);  
    d_centroids.resize(n);
    d_mortonCodes.resize(n);
    d_indices.resize(n);
    d_nodes.resize(2 * n - 1);
    d_atomicCounters.resize(2 * n - 1);
    d_subtreeSize.resize(2 * n - 1);
}

void LBVHPlusBuilderCUDA::runCompute(int n) {
    if (n == 0) return;

    thrust::fill(d_atomicCounters.begin(), d_atomicCounters.end(), 0);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    int numInternalNodes = n - 1;
    int gridSizeInternal = (numInternalNodes + blockSize - 1) / blockSize;

    cudaEventRecord(start);

    kComputeBoundsAndCentroids_P<<<gridSize, blockSize>>>(getDevicePtrs(), n,
        thrust::raw_pointer_cast(d_triBBoxes.data()),
        thrust::raw_pointer_cast(d_centroids.data()));
    
    AABB_cw init; 
    AABB_cw sceneBounds = thrust::reduce(d_triBBoxes.begin(), d_triBBoxes.end(), init, AABBReduceP());
    cudaEventRecord(e_centroids);

    kComputeMortonCodes_P<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_centroids.data()),
        n, sceneBounds,
        thrust::raw_pointer_cast(d_mortonCodes.data()),
        thrust::raw_pointer_cast(d_indices.data()));
    cudaEventRecord(e_morton);

    thrust::sort_by_key(d_mortonCodes.begin(), d_mortonCodes.end(), d_indices.begin());
    cudaEventRecord(e_sort);

    kBuildInternalNodes_P<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_mortonCodes.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        thrust::raw_pointer_cast(d_nodes.data()),
        n);

    kInitLeafNodes_P<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_triBBoxes.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        n);
    cudaEventRecord(e_topology);

    kRefitAndComputeSizes<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_atomicCounters.data()),
        thrust::raw_pointer_cast(d_subtreeSize.data()),
        n);
    cudaEventRecord(e_refit);

    kOptimizeMaximalTreelets<<<gridSizeInternal, blockSize>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_subtreeSize.data()),
        numInternalNodes);
        
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&time_centroids, start, e_centroids);
    cudaEventElapsedTime(&time_morton, e_centroids, e_morton);
    cudaEventElapsedTime(&time_sort, e_morton, e_sort);
    cudaEventElapsedTime(&time_topology, e_sort, e_topology);
    float time_refit_only;
    cudaEventElapsedTime(&time_refit_only, e_topology, e_refit);
    cudaEventElapsedTime(&time_optimize, e_refit, stop);
    time_refit = time_refit_only + time_optimize;
    lastBuildTimeMs = time_centroids + time_morton + time_sort + time_topology + time_refit;
}

void LBVHPlusBuilderCUDA::downloadResults(int n) {
    std::vector<LBVHNode> lbvh_nodes(d_nodes.size());
    thrust::copy(d_nodes.begin(), d_nodes.end(), lbvh_nodes.begin());
    
    h_indices.resize(d_indices.size());
    thrust::copy(d_indices.begin(), d_indices.end(), h_indices.begin());
    
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

void LBVHPlusBuilderCUDA::build(const TriangleMesh& mesh) {
    prepareData(mesh);
    runCompute(mesh.size());
    downloadResults(mesh.size());
}

std::string LBVHPlusBuilderCUDA::getTimingBreakdown() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "  Bounds/Centroids: " << time_centroids << " ms\n";
    oss << "  Morton Codes:     " << time_morton << " ms\n";
    oss << "  Radix Sort:       " << time_sort << " ms\n";
    oss << "  Topology Build:   " << time_topology << " ms\n";
    oss << "  Refit + Optimize: " << time_refit << " ms (opt: " << time_optimize << " ms)";
    return oss.str();
}
