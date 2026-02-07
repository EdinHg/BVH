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

// -----------------------------------------------------------------------------
// 60-bit Morton Codes (20 bits per axis)
// -----------------------------------------------------------------------------

// Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
__device__ __forceinline__ uint64_t expandBits_P(uint64_t v) {
    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffffull;
    v = (v | v << 16) & 0x1f0000ff0000ffull;
    v = (v | v << 8)  & 0x100f00f00f00f00full;
    v = (v | v << 4)  & 0x10c30c30c30c30c3ull;
    v = (v | v << 2)  & 0x1249249249249249ull;
    return v;
}

__device__ __forceinline__ uint64_t morton3D_P(float x, float y, float z) {
    // Range [0, 1] -> [0, 1048575] (2^20 - 1)
    x = fminf(fmaxf(x * 1048576.0f, 0.0f), 1048575.0f);
    y = fminf(fmaxf(y * 1048576.0f, 0.0f), 1048575.0f);
    z = fminf(fmaxf(z * 1048576.0f, 0.0f), 1048575.0f);
    return (expandBits_P((uint64_t)z) << 2) | (expandBits_P((uint64_t)y) << 1) | expandBits_P((uint64_t)x);
}

__device__ __forceinline__ int clz_custom_P(uint64_t x) {
    return x == 0 ? 64 : __clzll(x);
}

__device__ int delta_P(const uint64_t* sortedMortonCodes, const uint32_t* sortedIndices, int numObjects, int i, int j) {
    if (j < 0 || j >= numObjects) return -1;
    uint64_t codeI = sortedMortonCodes[i];
    uint64_t codeJ = sortedMortonCodes[j];
    if (codeI == codeJ) return 64 + clz_custom_P((uint64_t)sortedIndices[i] ^ (uint64_t)sortedIndices[j]);
    return clz_custom_P(codeI ^ codeJ);
}

// -----------------------------------------------------------------------------
// Common Math
// -----------------------------------------------------------------------------

__host__ __device__ __forceinline__ AABB_cw unionAABB_P(const AABB_cw& a, const AABB_cw& b) {
    return a.merge(b);
}

__host__ __device__ __forceinline__ float area_P(const AABB_cw& b) {
    return b.surfaceArea();
}

struct AABBReduceP {
    __host__ __device__ AABB_cw operator()(const AABB_cw& a, const AABB_cw& b) {
        return unionAABB_P(a, b);
    }
};

// -----------------------------------------------------------------------------
// Treelet Optimization - Serial Version
// -----------------------------------------------------------------------------

__device__ void optimizeTreeletSerial(LBVHNode* nodes, int rootIdx) {
    // 1. Gather Treelet Leaves
    int leafCount = 0;
    int activeNodeIdx[MAX_TREELET_SIZE];   
    AABB_cw activeBoxes[MAX_TREELET_SIZE]; 
    
    activeNodeIdx[0] = rootIdx;
    activeBoxes[0] = nodes[rootIdx].bbox;
    leafCount = 1;
    
    // Greedy expansion to find leaves
    // This finds nodes with largest surface area to expand
    for (int step = 0; step < MAX_TREELET_SIZE - 1; ++step) {
        int bestIdx = -1;
        float maxArea = -1.0f;
        
        for (int i = 0; i < leafCount; ++i) {
            int idx = activeNodeIdx[i];
            LBVHNode node = nodes[idx];
            bool isLeaf = (node.leftChild & 0x80000000);
            
            if (!isLeaf) {
                float area = activeBoxes[i].surfaceArea();
                if (area > maxArea) {
                    maxArea = area;
                    bestIdx = i;
                }
            }
        }
        
        if (bestIdx == -1) break; 
        
        int nodeToExpand = activeNodeIdx[bestIdx];
        LBVHNode expandedNode = nodes[nodeToExpand];
        
        uint32_t left = expandedNode.leftChild;
        uint32_t right = expandedNode.rightChild;
        
        // Overwrite bestIdx with Left child
        activeNodeIdx[bestIdx] = left;
        activeBoxes[bestIdx] = nodes[left].bbox;
        
        // Append Right child
        activeNodeIdx[leafCount] = right;
        activeBoxes[leafCount] = nodes[right].bbox;
        leafCount++;
    }
    
    int n = leafCount;
    if (n < 2) return; 

    // 2. Compute Costs (DP)
    float cost[128]; // 1<<7
    uint8_t partition[128];
    
    // Precompute Areas and Base Costs
    for (int s = 1; s < (1 << n); ++s) {
        if (__popc(s) == 1) {
            // Leaf subset -> Cost 0 (for optimization purposes)
            cost[s] = 0.0f;
        } else {
            cost[s] = INF;
        }
    }

    // DP Loop by size k
    for (int k = 2; k <= n; ++k) {
        // Iterate only subsets of size k.
        // For n=7, loop 1..127. Check popcount.
        for (int s = 1; s < (1 << n); ++s) {
            if (__popc(s) == k) {
                // Compute Area of subset S
                // (Optimized: could memoize area too, but memory is tight)
                AABB_cw box = AABB_cw::empty();
                for (int i = 0; i < n; ++i) {
                    if ((s >> i) & 1) {
                        box = unionAABB_P(box, activeBoxes[i]);
                    }
                }
                float areaS = box.surfaceArea();
                
                // Find best split
                float minCost = INF;
                int bestP = -1;
                
                int delta = (s - 1) & s;
                int p = (-delta) & s;
                
                do {
                    int other = s ^ p;
                    float c = cost[p] + cost[other];
                    if (c < minCost) {
                        minCost = c;
                        bestP = p;
                    }
                    
                    p = (p - delta) & s;
                } while (p != 0);
                
                float Ci = 1.0f; // Internal node cost factor
                cost[s] = minCost + Ci * areaS;
                partition[s] = (uint8_t)bestP;
            }
        }
    }
    
    // Rebuild topology with optimal partitions
    int availableNodes[MAX_TREELET_SIZE]; 
    int availCount = 0;
    
    int stack[MAX_TREELET_SIZE];
    int top = 0;
    stack[top++] = rootIdx;
    
    // DFS to find internal nodes
    while(top > 0) {
        int curr = stack[--top];
        
        // Check if curr is a leaf of the treelet
        bool isLeafInTreelet = false;
        for(int i=0; i<n; ++i) {
            if(activeNodeIdx[i] == curr) {
                isLeafInTreelet = true;
                break;
            }
        }
        
        if(!isLeafInTreelet) {
            availableNodes[availCount++] = curr;
            LBVHNode nd = nodes[curr];
            stack[top++] = nd.leftChild; 
            stack[top++] = nd.rightChild;
        }
    }
    
    // Assert availCount == n-1 ?
    // If n=7 leaves, we need 6 internal nodes.
    // availableNodes[0] should be rootIdx.
    
    struct Task {
        int mask;
        int nodeIdx;
    };
    Task queue[MAX_TREELET_SIZE]; 
    int qHead = 0;
    int qTail = 0; 
    
    int fullMask = (1 << n) - 1;
    // availableNodes[0] is rootIdx.
    queue[qTail++] = {fullMask, availableNodes[0]};
    int usedNodes = 1;

    while(qHead < qTail) {
        Task t = queue[qHead++];
        int mask = t.mask;
        int currIdx = t.nodeIdx;
        
        int leftMask = partition[mask];
        int rightMask = mask ^ leftMask;
        
        int leftNodeIdx, rightNodeIdx;
        
        // Left Child
        if (__popc(leftMask) == 1) {
            int leafPos = __ffs(leftMask) - 1; // ffs is 1-based
            leftNodeIdx = activeNodeIdx[leafPos];
        } else {
            leftNodeIdx = availableNodes[usedNodes++];
            queue[qTail++] = {leftMask, leftNodeIdx};
        }
        
        // Right Child
        if (__popc(rightMask) == 1) {
            int leafPos = __ffs(rightMask) - 1;
            rightNodeIdx = activeNodeIdx[leafPos];
        } else {
            rightNodeIdx = availableNodes[usedNodes++];
            queue[qTail++] = {rightMask, rightNodeIdx};
        }
        
        LBVHNode& node = nodes[currIdx];
        node.leftChild = leftNodeIdx;
        node.rightChild = rightNodeIdx;
        
        // Recompute BBox for this internal node immediately
        AABB_cw box = AABB_cw::empty();
        for(int i=0; i<n; ++i) {
            if((mask >> i) & 1) {
                box = unionAABB_P(box, activeBoxes[i]);
            }
        }
        node.bbox = box;
        
        nodes[leftNodeIdx].parent = currIdx;
        nodes[rightNodeIdx].parent = currIdx;
    }
}

} // End anonymous namespace

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

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

__global__ void kComputeMortonCodes_P(const float3_cw* centroids, int n, AABB_cw sceneBounds, uint64_t* mortonCodes, uint32_t* indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3_cw c = centroids[i];
    float3_cw minB = sceneBounds.min;
    float3_cw extents = sceneBounds.max - sceneBounds.min;
    
    // Normalize to [0,1]
    float nx = (c.x - minB.x) / extents.x;
    float ny = (c.y - minB.y) / extents.y;
    float nz = (c.z - minB.z) / extents.z;
    
    mortonCodes[i] = morton3D_P(nx, ny, nz);
    indices[i] = i;
}

__global__ void kBuildInternalNodes_P(const uint64_t* sortedMortonCodes, const uint32_t* sortedIndices, LBVHNode* nodes, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects - 1) return;

    // Use 64-bit Delta
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

// -----------------------------------------------------------------------------
// Combined Refit + Optimize Kernel (Bottom-Up)
// -----------------------------------------------------------------------------

__global__ void kRefitAndOptimize(LBVHNode* nodes, int* atomicCounters, int numObjects, bool doOptimize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    uint32_t idx = (numObjects - 1) + i; 

    // Bottom-up traversal
    while (idx != 0) {
        uint32_t parent = nodes[idx].parent;
        int oldVal = atomicAdd(&atomicCounters[parent], 1);
        if (oldVal == 0) return; // First thread dies
        __threadfence();

        uint32_t left = nodes[parent].leftChild;
        uint32_t right = nodes[parent].rightChild;
        
        // Initial Refit before Optimization
        nodes[parent].bbox = unionAABB_P(nodes[left].bbox, nodes[right].bbox);
        
        if (doOptimize) {
            optimizeTreeletSerial(nodes, parent);
            // Optimization updates BBox of parent internally, but propagate up?
            // The optimization function writes `node.bbox = box;` so logic is consistent.
            // BBox IS updated.
        }
        
        idx = parent;
    }
}

// -----------------------------------------------------------------------------
// Helper Class
// -----------------------------------------------------------------------------

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

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    int numInternalNodes = n - 1;
    
    cudaEventRecord(start);

    kComputeBoundsAndCentroids_P<<<gridSize, blockSize>>>(getDevicePtrs(), n,
        thrust::raw_pointer_cast(d_triBBoxes.data()),
        thrust::raw_pointer_cast(d_centroids.data()));
    
    AABB_cw init = AABB_cw::empty(); 
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

    // Initial Refit (Correct BBoxes)
    thrust::fill(d_atomicCounters.begin(), d_atomicCounters.end(), 0);
    kRefitAndOptimize<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_atomicCounters.data()),
        n, false); // No optimize first
    cudaEventRecord(e_refit);

    // Multi-pass Optimization
    // Loop 3 times as requested
    // Pass 1
    thrust::fill(d_atomicCounters.begin(), d_atomicCounters.end(), 0);
    kRefitAndOptimize<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_atomicCounters.data()),
        n, true);

    // Pass 2
    thrust::fill(d_atomicCounters.begin(), d_atomicCounters.end(), 0);
    kRefitAndOptimize<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_atomicCounters.data()),
        n, true);

    // Pass 3 (Final)
    thrust::fill(d_atomicCounters.begin(), d_atomicCounters.end(), 0);
    kRefitAndOptimize<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_atomicCounters.data()),
        n, true);

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
