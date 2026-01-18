#pragma once

#include "mesh.h"
#include "bvh_node.h"
#include <vector>
#include <string>
#include <cstdint>

// Abstract Base Class for BVH Builders
class BVHBuilder {
public:
    virtual ~BVHBuilder() = default;

    // The name of the algorithm (e.g., "LBVH", "PLOC++")
    virtual std::string getName() const = 0;

    // Main build function
    // Constructs the BVH from the input mesh
    virtual void build(const TriangleMesh& mesh) = 0;

    // Retrieve the constructed tree (flattened array)
    virtual const std::vector<BVHNode>& getNodes() const = 0;
    
    // Retrieve the reordered triangle indices after construction
    // indices[i] = original triangle index for the i-th triangle in BVH order
    // This is needed for SAH calculation and rendering
    virtual const std::vector<uint32_t>& getIndices() const = 0;
    
    // Get the last build time in milliseconds
    // This should be the pure compute time (excluding data upload)
    virtual float getLastBuildTimeMS() const = 0;
    
    // Optional: Get detailed timing breakdown
    virtual std::string getTimingBreakdown() const {
        return "";
    }
};
