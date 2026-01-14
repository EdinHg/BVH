#pragma once

#include "../math/aabb.hpp"
#include <cstdint>
#include <vector>

struct BVHNode {
    AABB bounds;
    uint32_t childOffset;   // Offset to first child in node array
    uint8_t  childCount;    // 0 = leaf, 2 = binary, 4 = quad, 8 = oct
    uint8_t  axis;          // Split axis (for traversal hints)
    uint16_t primCount;     // Number of primitives if leaf
    uint32_t primOffset;    // Index into primitive index array

    bool isLeaf() const { return childCount == 0; }
};

struct BVHResult {
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> primIndices;
    uint8_t branchingFactor;   // 2, 4, or 8

    BVHResult() : branchingFactor(2) {}
};
