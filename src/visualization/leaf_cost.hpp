#pragma once

#include "../bvh/bvh_node.hpp"
#include "../mesh/triangle_mesh.hpp"
#include "../math/vec3.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

// Compute per-triangle colors based on leaf cost
// Returns colors for each triangle (same order as mesh)
inline std::vector<Vec3> computeLeafCostColors(
    const TriangleMesh& mesh,
    const BVHResult& bvh,
    bool useDensityOnly = false)
{
    size_t numTris = mesh.size();
    std::vector<float> costs(numTris, 0.0f);

    size_t assignedCount = 0;

    // Traverse BVH and assign cost to each triangle
    for (const auto& node : bvh.nodes) {
        if (!node.isLeaf()) continue;

        float cost;
        if (useDensityOnly) {
            cost = static_cast<float>(node.primCount);
        } else {
            cost = node.bounds.surfaceArea() * node.primCount;
        }

        // Assign cost to all triangles in this leaf
        for (uint32_t i = 0; i < node.primCount; ++i) {
            uint32_t primIdx = bvh.primIndices[node.primOffset + i];
            if (primIdx < numTris) {
                costs[primIdx] = cost;
                assignedCount++;
            }
        }
    }

    std::cout << "Heatmap: assigned costs to " << assignedCount << "/" << numTris << " triangles\n";

    // Find min/max for normalization
    float minCost = std::numeric_limits<float>::max();
    float maxCost = 0.0f;
    size_t validCount = 0;
    
    for (float c : costs) {
        if (c > 0.0f) {
            minCost = std::min(minCost, c);
            maxCost = std::max(maxCost, c);
            validCount++;
        }
    }
    
    std::cout << "Heatmap: " << validCount << " valid costs, range [" << minCost << ", " << maxCost << "]\n";
    
    // Handle edge case where no valid costs found
    if (validCount == 0 || maxCost <= 0.0f) {
        std::cout << "Heatmap: WARNING - no valid costs, using uniform color\n";
        return std::vector<Vec3>(numTris, Vec3(0.5f, 0.5f, 0.5f));
    }
    
    // Handle edge case where all costs are the same
    if (minCost >= maxCost) {
        std::cout << "Heatmap: all costs equal, using uniform green\n";
        return std::vector<Vec3>(numTris, Vec3(0.0f, 1.0f, 0.0f));
    }
    
    // Use logarithmic scale to compress dynamic range
    float logMin = std::log(minCost);
    float logMax = std::log(maxCost);
    float div = logMax - logMin;
    if (div < 1e-6f) div = 1.0f;
    
    std::cout << "Heatmap: log range [" << logMin << ", " << logMax << "], div=" << div << "\n";

    // Map to colors: Blue (low cost) -> Green (mid) -> Red (high cost)
    std::vector<Vec3> colors(numTris);
    for (size_t i = 0; i < numTris; ++i) {
        float c = costs[i];
        
        // Handle unassigned triangles (make them magenta for visibility)
        if (c <= 0.0f) {
            colors[i] = Vec3(1.0f, 0.0f, 1.0f);  // Magenta = unassigned
            continue;
        }
        
        // Normalize using log space
        float t = (std::log(c) - logMin) / div;
        t = std::max(0.0f, std::min(1.0f, t)); // Clamp to [0, 1]

        // Reverse the mapping: Red (low cost) -> Green (mid) -> Blue (high cost)
        t = 1.0f - t;

        Vec3 color;
        if (t < 0.5f) {
            // Blue (high cost) to Green (mid)
            float s = t * 2.0f;
            color = Vec3(0.0f, s, 1.0f - s);
        } else {
            // Green to Red (low cost)
            float s = (t - 0.5f) * 2.0f;
            color = Vec3(s, 1.0f - s, 0.0f);
        }
        colors[i] = color;
    }

    return colors;
}
