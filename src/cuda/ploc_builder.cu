#include "ploc_builder.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cub/cub.cuh>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

// --- Device Helpers ---

__device__ __forceinline__ uint32_t expandBits_ploc(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ uint32_t morton3D_ploc(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    
    uint32_t xx = expandBits_ploc((uint32_t)x);
    uint32_t yy = expandBits_ploc((uint32_t)y);
    uint32_t zz = expandBits_ploc((uint32_t)z);
    
    return xx * 4 + yy * 2 + zz;
}

// --- Kernels ---

__global__ void kComputeAABBsAndCentroids_ploc(
    TrianglesSoADevice tris, 
    int n, 
    AABB_cw* triBBoxes, 
    float3_cw* centroids) 
{
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

// Simple Block Reduction for Scene Bounds (borrowed/adapted concept)
__global__ void kReduceBounds_Step1_ploc(const AABB_cw* input, AABB_cw* output, int n) {
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
            sdata[tid] = sdata[tid].merge(sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void kReduceBounds_Step2_ploc(AABB_cw* data, int n) {
    __shared__ AABB_cw sdata[256];
    int tid = threadIdx.x;
    int i = tid;

    AABB_cw local; 
    
    if (tid < n) local = data[tid]; 
    else {
        local.min = float3_cw(FLT_MAX, FLT_MAX, FLT_MAX);
        local.max = float3_cw(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
    
    sdata[tid] = local;
    
    i += blockDim.x;
    while (i < n) {
        sdata[tid] = sdata[tid].merge(data[i]);
        i += blockDim.x;
    }

    __syncthreads();

    // Reduce shared
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid].merge(sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        data[0] = sdata[0];
    }
}

__global__ void kAssignMortonCodes_ploc(
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

    float x = (c.x - minB.x) / ((extents.x > 1e-6f) ? extents.x : 1.0f);
    float y = (c.y - minB.y) / ((extents.y > 1e-6f) ? extents.y : 1.0f);
    float z = (c.z - minB.z) / ((extents.z > 1e-6f) ? extents.z : 1.0f);

    mortonCodes[i] = morton3D_ploc(x, y, z);
    indices[i] = i;
}

__global__ void kGatherAABBs(
    const uint32_t* indices,
    const AABB_cw* source_aabbs,
    AABB_cw* dest_aabbs,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dest_aabbs[i] = source_aabbs[indices[i]];
}

__global__ void kInitializeSequence(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = i;
}

__global__ void kInitLeafNodes_ploc(
    BVHNode* nodes, 
    const AABB_cw* sortedAABBs, 
    const uint32_t* sortedIndices, 
    int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    nodes[i].bbox = sortedAABBs[i];
    nodes[i].leftChild = sortedIndices[i] | 0x80000000; 
    nodes[i].rightChild = -1;
}

__global__ void kFindNearestNeighbors_ploc(
    const AABB_cw* clusterAABBs, 
    int num_clusters,
    int radius,
    int* neighbor_indices,
    float* neighbor_dists)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_clusters) return;

    AABB_cw boxI = clusterAABBs[i];
    float min_dist = FLT_MAX;
    int best_j = -1;

    int start = max(0, i - radius);
    int end = min(num_clusters - 1, i + radius);

    for (int j = start; j <= end; ++j) {
        if (i == j) continue;
        
        AABB_cw boxJ = clusterAABBs[j];
        AABB_cw unionBox = boxI.merge(boxJ);
        float dist = unionBox.surfaceArea();

        if (dist < min_dist) {
            min_dist = dist;
            best_j = j;
        } else if (dist == min_dist) {
            if (best_j == -1 || j < best_j) {
                best_j = j;
            }
        }
    }

    neighbor_indices[i] = best_j;
    neighbor_dists[i] = min_dist;
}

__global__ void kMergeClusters_ploc(
    const int* neighbor_indices,
    const float* neighbor_dists, 
    int num_clusters,
    int* cluster_node_indices,
    AABB_cw* cluster_aabbs,
    BVHNode* nodes,
    int* next_node_idx, 
    int* is_valid) // int for CUB scan compatibility
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_clusters) return;

    int j = neighbor_indices[i];
    bool merge = false;
    
    if (j != -1) {
        if (neighbor_indices[j] == i) {
            // Mutual! Merge if i < j
            if (i < j) {
                merge = true;
            }
        }
    }

    if (merge) {
        int leftNodeIdx = cluster_node_indices[i];
        int rightNodeIdx = cluster_node_indices[j];
        
        AABB_cw leftBox = cluster_aabbs[i];
        AABB_cw rightBox = cluster_aabbs[j];
        AABB_cw unionBox = leftBox.merge(rightBox);

        // Allocate new internal node
        int newNodeIdx = atomicAdd(next_node_idx, 1);
        
        nodes[newNodeIdx].bbox = unionBox;
        nodes[newNodeIdx].leftChild = leftNodeIdx;
        nodes[newNodeIdx].rightChild = rightNodeIdx;
        
        cluster_node_indices[i] = newNodeIdx;
        cluster_aabbs[i] = unionBox;
        
        is_valid[i] = 1;
        is_valid[j] = 0; // Consumed
    } else {
        // If we were consumed (because j < i and mutual), we are invalid.
        if (j != -1 && neighbor_indices[j] == i && j < i) {
            is_valid[i] = 0;
        } else {
            is_valid[i] = 1;
        }
    }
}

// Compact kernels
__global__ void kCompactIndices(
    const int* d_in,
    int* d_out,
    const int* d_flags,
    const int* d_offsets,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (d_flags[i]) {
        d_out[d_offsets[i]] = d_in[i];
    }
}

__global__ void kCompactAABBs(
    const AABB_cw* d_in,
    AABB_cw* d_out,
    const int* d_flags,
    const int* d_offsets,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (d_flags[i]) {
        d_out[d_offsets[i]] = d_in[i];
    }
}

// --- Implementation ---

PLOCBuilderCUDA::PLOCBuilderCUDA(int r) : 
    radius(r),
    d_v0x(nullptr), d_v0y(nullptr), d_v0z(nullptr),
    d_v1x(nullptr), d_v1y(nullptr), d_v1z(nullptr),
    d_v2x(nullptr), d_v2y(nullptr), d_v2z(nullptr),
    d_triBBoxes(nullptr), d_centroids(nullptr),
    d_mortonCodes(nullptr), d_indices(nullptr),
    d_cluster_aabbs(nullptr), d_cluster_indices(nullptr),
    d_neighbors(nullptr), d_dists(nullptr),
    d_valid(nullptr), d_scan_offsets(nullptr),
    d_nodes(nullptr), d_next_node_idx(nullptr),
    d_cluster_aabbs_swap(nullptr), d_cluster_indices_swap(nullptr),
    d_temp_storage(nullptr), temp_storage_bytes(0),
    lastBuildTimeMs(0.0f),
    time_init(0.0f), time_search(0.0f), time_merge(0.0f), time_compact(0.0f), time_reorder(0.0f)
{}

PLOCBuilderCUDA::~PLOCBuilderCUDA() {
    cleanup();
}

void PLOCBuilderCUDA::reorderDFS() {
    // Reorder nodes in depth-first preorder for cache coherence during traversal.
    // Root is already at index 0. We walk the tree iteratively and build a new
    // node array where each node's children immediately follow it (or are nearby).
    
    if (h_final_nodes.empty()) return;
    
    int totalNodes = (int)h_final_nodes.size();
    std::vector<BVHNode> new_nodes;
    new_nodes.reserve(totalNodes);
    
    // old_to_new[old_idx] = new_idx
    std::vector<int> old_to_new(totalNodes, -1);
    
    // Iterative DFS using explicit stack
    std::vector<int> stack;
    stack.reserve(128); // Match traversal kernel stack size
    stack.push_back(0); // Start from root
    
    while (!stack.empty()) {
        int old_idx = stack.back();
        stack.pop_back();
        
        if (old_idx < 0 || old_idx >= totalNodes) continue;
        
        // Assign next sequential index in DFS order
        int new_idx = (int)new_nodes.size();
        old_to_new[old_idx] = new_idx;
        new_nodes.push_back(h_final_nodes[old_idx]);
        
        const BVHNode& node = h_final_nodes[old_idx];
        
        // If internal node, push children (right first so left gets lower index)
        if (!node.isLeaf()) {
            int left = node.leftChild;
            int right = node.rightChild;
            
            // Push right child first (will be processed after left)
            if (right >= 0 && right < totalNodes) {
                stack.push_back(right);
            }
            // Push left child second (will be processed first - LIFO)
            if (left >= 0 && left < totalNodes) {
                stack.push_back(left);
            }
        }
    }
    
    // Now remap all child pointers in the new array
    for (auto& node : new_nodes) {
        if (!node.isLeaf()) {
            // Remap leftChild and rightChild through old_to_new mapping
            if (node.leftChild >= 0 && node.leftChild < totalNodes) {
                node.leftChild = old_to_new[node.leftChild];
            }
            if (node.rightChild >= 0 && node.rightChild < totalNodes) {
                node.rightChild = old_to_new[node.rightChild];
            }
        }
        // Leaf nodes: leftChild contains primitive index with high bit set, no remapping needed
    }
    
    // Replace old array with reordered one
    h_final_nodes = std::move(new_nodes);
}

void PLOCBuilderCUDA::cleanup() {
    if (d_v0x) cudaFree(d_v0x); d_v0x = nullptr;
    if (d_v0y) cudaFree(d_v0y); d_v0y = nullptr;
    if (d_v0z) cudaFree(d_v0z); d_v0z = nullptr;
    if (d_v1x) cudaFree(d_v1x); d_v1x = nullptr;
    if (d_v1y) cudaFree(d_v1y); d_v1y = nullptr;
    if (d_v1z) cudaFree(d_v1z); d_v1z = nullptr;
    if (d_v2x) cudaFree(d_v2x); d_v2x = nullptr;
    if (d_v2y) cudaFree(d_v2y); d_v2y = nullptr;
    if (d_v2z) cudaFree(d_v2z); d_v2z = nullptr;
    if (d_triBBoxes) cudaFree(d_triBBoxes); d_triBBoxes = nullptr;
    if (d_centroids) cudaFree(d_centroids); d_centroids = nullptr;
    if (d_mortonCodes) cudaFree(d_mortonCodes); d_mortonCodes = nullptr;
    if (d_indices) cudaFree(d_indices); d_indices = nullptr;
    
    // PLOC buffers
    if (d_cluster_aabbs) cudaFree(d_cluster_aabbs); d_cluster_aabbs = nullptr;
    if (d_cluster_indices) cudaFree(d_cluster_indices); d_cluster_indices = nullptr;
    if (d_neighbors) cudaFree(d_neighbors); d_neighbors = nullptr;
    if (d_dists) cudaFree(d_dists); d_dists = nullptr;
    if (d_valid) cudaFree(d_valid); d_valid = nullptr;
    if (d_scan_offsets) cudaFree(d_scan_offsets); d_scan_offsets = nullptr;
    if (d_nodes) cudaFree(d_nodes); d_nodes = nullptr;
    if (d_next_node_idx) cudaFree(d_next_node_idx); d_next_node_idx = nullptr;
    
    if (d_cluster_aabbs_swap) cudaFree(d_cluster_aabbs_swap); d_cluster_aabbs_swap = nullptr;
    if (d_cluster_indices_swap) cudaFree(d_cluster_indices_swap); d_cluster_indices_swap = nullptr;
    
    if (d_temp_storage) cudaFree(d_temp_storage); d_temp_storage = nullptr;
}

void PLOCBuilderCUDA::allocate(size_t n) {
    cleanup();
    
    size_t floatSize = n * sizeof(float);
    cudaMalloc(&d_v0x, floatSize); cudaMalloc(&d_v0y, floatSize); cudaMalloc(&d_v0z, floatSize);
    cudaMalloc(&d_v1x, floatSize); cudaMalloc(&d_v1y, floatSize); cudaMalloc(&d_v1z, floatSize);
    cudaMalloc(&d_v2x, floatSize); cudaMalloc(&d_v2y, floatSize); cudaMalloc(&d_v2z, floatSize);
    
    cudaMalloc(&d_triBBoxes, n * sizeof(AABB_cw));
    cudaMalloc(&d_centroids, n * sizeof(float3_cw));
    cudaMalloc(&d_mortonCodes, n * sizeof(uint32_t));
    cudaMalloc(&d_indices, n * sizeof(uint32_t));
    
    // PLOC buffers
    cudaMalloc(&d_cluster_aabbs, n * sizeof(AABB_cw));
    cudaMalloc(&d_cluster_indices, n * sizeof(int));
    cudaMalloc(&d_cluster_aabbs_swap, n * sizeof(AABB_cw));
    cudaMalloc(&d_cluster_indices_swap, n * sizeof(int));
    
    cudaMalloc(&d_neighbors, n * sizeof(int));
    cudaMalloc(&d_dists, n * sizeof(float));
    cudaMalloc(&d_valid, n * sizeof(int));
    cudaMalloc(&d_scan_offsets, n * sizeof(int));
    
    cudaMalloc(&d_nodes, (2 * n - 1) * sizeof(BVHNode));
    cudaMalloc(&d_next_node_idx, sizeof(int));
}

std::string PLOCBuilderCUDA::getName() const { 
    return "PLOC CUDA (r=" + std::to_string(radius) + ")"; 
}

const std::vector<BVHNode>& PLOCBuilderCUDA::getNodes() const { 
    return h_final_nodes; 
}

const std::vector<uint32_t>& PLOCBuilderCUDA::getIndices() const { 
    return h_final_indices; 
}

float PLOCBuilderCUDA::getLastBuildTimeMS() const { 
    return lastBuildTimeMs; 
}

std::string PLOCBuilderCUDA::getTimingBreakdown() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "  Initialization:   " << time_init << " ms\n";
    oss << "  NN Search (Red):  " << time_search << " ms\n";
    oss << "  Merging (Green):  " << time_merge << " ms\n";
    oss << "  Compaction (Mag): " << time_compact << " ms\n";
    oss << "  DFS Reordering:   " << time_reorder << " ms";
    return oss.str();
}

void PLOCBuilderCUDA::build(const TriangleMesh& mesh) {
    int n = mesh.size();
    if (n == 0) return;

    allocate(n);
    
    // Reset timings
    time_init = 0;
    time_search = 0;
    time_merge = 0;
    time_compact = 0;
    time_reorder = 0;
    
    float temp_ms = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- INITIALIZATION PHASE ---
    cudaEventRecord(start);

    // 1. Upload Data
    cudaMemcpy(d_v0x, mesh.v0x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v0y, mesh.v0y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v0z, mesh.v0z.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1x, mesh.v1x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1y, mesh.v1y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1z, mesh.v1z.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2x, mesh.v2x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2y, mesh.v2y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2z, mesh.v2z.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    TrianglesSoADevice d_tris = {d_v0x, d_v0y, d_v0z, d_v1x, d_v1y, d_v1z, d_v2x, d_v2y, d_v2z};

    // 2. Compute Centroids & AABBs
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    kComputeAABBsAndCentroids_ploc<<<gridSize, blockSize>>>(d_tris, n, d_triBBoxes, d_centroids);

    // Global Scene Bounds via Reduction
    AABB_cw* d_blockReductions;
    cudaMalloc(&d_blockReductions, gridSize * sizeof(AABB_cw));
    kReduceBounds_Step1_ploc<<<gridSize, blockSize>>>(d_triBBoxes, d_blockReductions, n);
    if (gridSize > 1) {
        kReduceBounds_Step2_ploc<<<1, blockSize>>>(d_blockReductions, gridSize);
    }
    
    AABB_cw sceneBounds;
    cudaMemcpy(&sceneBounds, d_blockReductions, sizeof(AABB_cw), cudaMemcpyDeviceToHost);
    cudaFree(d_blockReductions);

    // 3. Morton Codes
    kAssignMortonCodes_ploc<<<gridSize, blockSize>>>(d_centroids, n, sceneBounds, d_mortonCodes, d_indices);

    // 4. Sort (Thrust allowed here)
    thrust::device_ptr<uint32_t> t_morton(d_mortonCodes);
    thrust::device_ptr<uint32_t> t_indices(d_indices);
    thrust::sort_by_key(t_morton, t_morton + n, t_indices);
    
    // Reorder AABBs to match sorted indices using custom Gather
    kGatherAABBs<<<gridSize, blockSize>>>(d_indices, d_triBBoxes, d_cluster_aabbs, n);

    // Initialize PLOC Loop Data
    kInitLeafNodes_ploc<<<gridSize, blockSize>>>(d_nodes, d_cluster_aabbs, d_indices, n);

    // Initialize d_cluster_indices to 0..n-1
    kInitializeSequence<<<gridSize, blockSize>>>(d_cluster_indices, n);

    // Reset next node index
    int initial_cnt = n;
    cudaMemcpy(d_next_node_idx, &initial_cnt, sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocate temp storage for CUB if needed
    // Look up requirements (using dummy call, doesn't execute on GPU but computes size)
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_valid, d_scan_offsets, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_ms, start, stop);
    time_init = temp_ms;

    int current_n = n;
    int radius = this->radius;

    while (current_n > 1) {
        int num_blocks = (current_n + blockSize - 1) / blockSize;

        // --- PHASE 1: SEARCH ---
        cudaEventRecord(start);
        kFindNearestNeighbors_ploc<<<num_blocks, blockSize>>>(
            d_cluster_aabbs,
            current_n,
            radius,
            d_neighbors,
            d_dists
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp_ms, start, stop);
        time_search += temp_ms;

        // --- PHASE 2: MERGE ---
        cudaEventRecord(start);
        kMergeClusters_ploc<<<num_blocks, blockSize>>>(
            d_neighbors,
            d_dists,
            current_n,
            d_cluster_indices,
            d_cluster_aabbs,
            d_nodes,
            d_next_node_idx,
            d_valid
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp_ms, start, stop);
        time_merge += temp_ms; // Accumulate merging time

        // --- PHASE 3: COMPACTION ---
        cudaEventRecord(start);
        // 1. Scan valid flags
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_valid, d_scan_offsets, current_n);

        // 2. Get new count
        int last_valid, last_offset;
        cudaMemcpy(&last_valid, d_valid + current_n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_offset, d_scan_offsets + current_n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        int new_n = last_offset + last_valid;

        // 3. Compact Indices and AABBs
        kCompactIndices<<<num_blocks, blockSize>>>(d_cluster_indices, d_cluster_indices_swap, d_valid, d_scan_offsets, current_n);
        kCompactAABBs<<<num_blocks, blockSize>>>(d_cluster_aabbs, d_cluster_aabbs_swap, d_valid, d_scan_offsets, current_n);
        
        // Swap buffers
        std::swap(d_cluster_indices, d_cluster_indices_swap);
        std::swap(d_cluster_aabbs, d_cluster_aabbs_swap);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp_ms, start, stop);
        time_compact += temp_ms;

        current_n = new_n;
    }

    int num_allocated_nodes;
    cudaMemcpy(&num_allocated_nodes, d_next_node_idx, sizeof(int), cudaMemcpyDeviceToHost);

    h_final_nodes.resize(num_allocated_nodes); 
    cudaMemcpy(h_final_nodes.data(), d_nodes, num_allocated_nodes * sizeof(BVHNode), cudaMemcpyDeviceToHost);

    h_final_indices.resize(n);
    cudaMemcpy(h_final_indices.data(), d_indices, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Fix root (last allocated) to be at index 0
    if (num_allocated_nodes > 1) {
        int root_idx = num_allocated_nodes - 1;
        if (root_idx != 0) {
            std::swap(h_final_nodes[0], h_final_nodes[root_idx]);
            for (auto& node : h_final_nodes) {
               auto fix_child = [&](int32_t& child) {
                     bool is_internal_ref = !(child & 0x80000000);
                     if (is_internal_ref) {
                         if (child == 0) child = root_idx;
                         else if (child == root_idx) child = 0;
                     }
               };
               if (!node.isLeaf()) {
                   fix_child(node.leftChild);
                   fix_child(node.rightChild);
               }
            }
        }
    }
    
    // --- DFS REORDERING PHASE ---
    // Reorder nodes for cache coherence during traversal
    auto reorder_start = std::chrono::high_resolution_clock::now();
    reorderDFS();
    auto reorder_end = std::chrono::high_resolution_clock::now();
    time_reorder = std::chrono::duration<float, std::milli>(reorder_end - reorder_start).count();
    
    // Calculate total build time including all phases
    lastBuildTimeMs = time_init + time_search + time_merge + time_compact + time_reorder;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
