#pragma once

#include "../../include/bvh_builder.h"
#include "../../include/common.h"
#include "../../include/bvh_node.h"
#include <vector>
#include <string>
#include <cuda_runtime.h>

class PLOCBuilderCUDA : public BVHBuilder {
private:
    std::vector<BVHNode> h_final_nodes;
    std::vector<uint32_t> h_final_indices;
    float lastBuildTimeMs;

    // Device Pointers (no thrust::device_vector)
    float *d_v0x, *d_v0y, *d_v0z;
    float *d_v1x, *d_v1y, *d_v1z;
    float *d_v2x, *d_v2y, *d_v2z;
    
    AABB_cw* d_triBBoxes;
    float3_cw* d_centroids;
    uint32_t* d_mortonCodes;
    uint32_t* d_indices;
    
    // PLOC specific buffers
    AABB_cw* d_cluster_aabbs;
    int* d_cluster_indices;
    int* d_neighbors;     // Nearest neighbor index
    float* d_dists;       // Nearest neighbor distance
    int* d_valid;        // Scan input (1 if valid, 0 if merged/invalid)
    int* d_scan_offsets; // Scan output
    BVHNode* d_nodes;
    int* d_next_node_idx; // Atomic counter
    
    // Double buffering pointers
    AABB_cw* d_cluster_aabbs_swap;
    int* d_cluster_indices_swap;

    // Temp storage for CUB DeviceScan
    void* d_temp_storage;
    size_t temp_storage_bytes;
    
    // Timing breakdown
    float time_init;
    float time_search;
    float time_merge;
    float time_compact;

    void cleanup();
    void allocate(size_t n);

    int radius;

public:
    PLOCBuilderCUDA(int r = 25);
    ~PLOCBuilderCUDA();

    std::string getName() const override;
    void build(const TriangleMesh& mesh) override;
    const std::vector<BVHNode>& getNodes() const override;
    const std::vector<uint32_t>& getIndices() const override;
    float getLastBuildTimeMS() const override;
    std::string getTimingBreakdown() const override;
};
