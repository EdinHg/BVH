#pragma once

#include "../bvh/bvh_node.hpp"
#include <vector>

// Calculate SAH cost of a BVH
// Formula: C = C_t * sum(SA(node)/SA(root)) + C_i * sum(SA(leaf)/SA(root) * primCount)
inline float calculateSAHCost(const std::vector<BVHNode>& nodes,
                              float traversalCost = 1.0f,
                              float intersectionCost = 1.0f) {
    if (nodes.empty()) return 0.0f;

    float rootArea = nodes[0].bounds.surfaceArea();
    if (rootArea <= 0.0f) return 0.0f;

    float cost = 0.0f;

    for (const auto& node : nodes) {
        float relativeArea = node.bounds.surfaceArea() / rootArea;

        if (node.isLeaf()) {
            // Leaf: intersection cost * probability * prim count
            cost += intersectionCost * relativeArea * node.primCount;
        } else {
            // Internal: traversal cost * probability
            cost += traversalCost * relativeArea;
        }
    }

    return cost;
}
