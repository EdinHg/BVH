#pragma once

#include "bvh_node.hpp"
#include <fstream>
#include <string>
#include <stdexcept>
#include <limits>
#include <iomanip>
#include <iostream>

// Helper to check if a bounding box is valid (not default FLT_MAX)
inline bool isValidAABB(const AABB& b) {
    // Check for uninitialized values (FLT_MAX) or Inverted boxes
    if (b.min.x > b.max.x || b.min.y > b.max.y || b.min.z > b.max.z) return false;
    if (b.min.x == std::numeric_limits<float>::max()) return false;
    if (b.min.x == -std::numeric_limits<float>::max()) return false;
    return true;
}

inline void exportBVHToOBJ(const std::string& path,
                           const std::vector<BVHNode>& nodes,
                           bool leavesOnly = false) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // Force standard formatting (avoid scientific notation like 1.5e+02 which breaks some OBJ readers)
    file << std::fixed << std::setprecision(6);

    file << "# BVH Bounding Boxes\n";
    file << "# Total Nodes: " << nodes.size() << "\n\n";

    uint32_t vertexOffset = 1;  // OBJ is 1-indexed
    int writtenNodes = 0;

    for (size_t i = 0; i < nodes.size(); ++i) {
        const BVHNode& node = nodes[i];

        // 1. Filter: Leaves Only
        if (leavesOnly && !node.isLeaf()) continue;

        // 2. Filter: Invalid/Garbage Data
        // If the refit kernel failed, internal nodes might still be FLT_MAX. 
        // We skip them to prevent the OBJ from exploding.
        if (!isValidAABB(node.bounds)) {
            std::cout << "Warning: Node " << i << " has invalid bounds. Skipping export.\n";
            continue;
        }

        const AABB& b = node.bounds;

        // Write comment for debugging
        file << "# Node " << i << (node.isLeaf() ? " (Leaf)" : " (Internal)") << "\n";

        // 8 vertices of the bounding box
        file << "v " << b.min.x << " " << b.min.y << " " << b.max.z << "\n"; // 0
        file << "v " << b.max.x << " " << b.min.y << " " << b.max.z << "\n"; // 1
        file << "v " << b.max.x << " " << b.min.y << " " << b.min.z << "\n"; // 2
        file << "v " << b.min.x << " " << b.min.y << " " << b.min.z << "\n"; // 3
        file << "v " << b.min.x << " " << b.max.y << " " << b.max.z << "\n"; // 4
        file << "v " << b.max.x << " " << b.max.y << " " << b.max.z << "\n"; // 5
        file << "v " << b.max.x << " " << b.max.y << " " << b.min.z << "\n"; // 6
        file << "v " << b.min.x << " " << b.max.y << " " << b.min.z << "\n"; // 7

        // 12 edges (lines)
        uint32_t v = vertexOffset;
        
        // Bottom
        file << "l " << v+0 << " " << v+1 << "\n";
        file << "l " << v+1 << " " << v+2 << "\n";
        file << "l " << v+2 << " " << v+3 << "\n";
        file << "l " << v+3 << " " << v+0 << "\n";
        // Top
        file << "l " << v+4 << " " << v+5 << "\n";
        file << "l " << v+5 << " " << v+6 << "\n";
        file << "l " << v+6 << " " << v+7 << "\n";
        file << "l " << v+7 << " " << v+4 << "\n";
        // Sides
        file << "l " << v+0 << " " << v+4 << "\n";
        file << "l " << v+1 << " " << v+5 << "\n";
        file << "l " << v+2 << " " << v+6 << "\n";
        file << "l " << v+3 << " " << v+7 << "\n";
        
        file << "\n";
        vertexOffset += 8;
        writtenNodes++;
    }

    file.close();
    std::cout << "Exported " << writtenNodes << " valid nodes to " << path << std::endl;
}