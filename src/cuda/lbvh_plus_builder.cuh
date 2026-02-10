#pragma once

#include "../../include/bvh_builder.h"
#include "../../include/common.h"
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "lbvh_builder.cuh" // Re-using LBVHNode

class LBVHPlusBuilderCUDA : public BVHBuilder {
private:
    // Device Vectors (Same as LBVH)
    thrust::device_vector<float> d_v0x, d_v0y, d_v0z;
    thrust::device_vector<float> d_v1x, d_v1y, d_v1z;
    thrust::device_vector<float> d_v2x, d_v2y, d_v2z;
    thrust::device_vector<AABB_cw> d_triBBoxes;
    thrust::device_vector<float3_cw> d_centroids;
    thrust::device_vector<uint64_t> d_mortonCodes;
    thrust::device_vector<uint32_t> d_indices;
    thrust::device_vector<LBVHNode> d_nodes;
    
    // For optimization pass
    thrust::device_vector<int> d_atomicCounters;
    thrust::device_vector<int> d_subtreeSize; // For tracking treelet size

    // Host-side results
    std::vector<BVHNode> h_nodes;
    std::vector<uint32_t> h_indices;
    float lastBuildTimeMs;
    
    // Timing breakdown
    float time_centroids;
    float time_morton;
    float time_sort;
    float time_topology;
    float time_refit;
    float time_optimize;
    
    // Profiling Events
    cudaEvent_t start, e_centroids, e_morton, e_sort, e_topology, e_refit, stop;

    TrianglesSoADevice getDevicePtrs();
    void prepareData(const TriangleMesh& mesh);
    void runCompute(int n);
    void downloadResults(int n);

public:
    LBVHPlusBuilderCUDA();
    ~LBVHPlusBuilderCUDA();

    std::string getName() const override { return "LBVH+"; }
    void build(const TriangleMesh& mesh) override;
    const std::vector<BVHNode>& getNodes() const override { return h_nodes; }
    const std::vector<uint32_t>& getIndices() const override { return h_indices; }
    float getLastBuildTimeMS() const override { return lastBuildTimeMs; }
    std::string getTimingBreakdown() const override;
    
    // Memory management
    void cleanup();
};
