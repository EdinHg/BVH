#pragma once

#include "bvh_node.hpp"
#include "../math/aabb.hpp"
#include <fstream>
#include <string>
#include <stdexcept>

// Export BVH bounding boxes as OBJ wireframe
inline void exportBVHToOBJ(const std::string& path,
                           const std::vector<BVHNode>& nodes,
                           bool leavesOnly = false) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    file << "# BVH Bounding Boxes\n";
    file << "# Nodes: " << nodes.size() << "\n\n";

    uint32_t vertexOffset = 1;  // OBJ is 1-indexed

    for (size_t i = 0; i < nodes.size(); ++i) {
        const BVHNode& node = nodes[i];

        // Skip internal nodes if leavesOnly is set
        if (leavesOnly && !node.isLeaf()) continue;

        const AABB& b = node.bounds;

        // 8 vertices of the bounding box
        //   4-----5
        //  /|    /|
        // 0-----1 |
        // | 7---|-6
        // |/    |/
        // 3-----2
        file << "# Node " << i << (node.isLeaf() ? " (leaf)" : " (internal)") << "\n";
        file << "v " << b.min.x << " " << b.min.y << " " << b.max.z << "\n";  // 0
        file << "v " << b.max.x << " " << b.min.y << " " << b.max.z << "\n";  // 1
        file << "v " << b.max.x << " " << b.min.y << " " << b.min.z << "\n";  // 2
        file << "v " << b.min.x << " " << b.min.y << " " << b.min.z << "\n";  // 3
        file << "v " << b.min.x << " " << b.max.y << " " << b.max.z << "\n";  // 4
        file << "v " << b.max.x << " " << b.max.y << " " << b.max.z << "\n";  // 5
        file << "v " << b.max.x << " " << b.max.y << " " << b.min.z << "\n";  // 6
        file << "v " << b.min.x << " " << b.max.y << " " << b.min.z << "\n";  // 7

        // 12 edges as line elements
        uint32_t v = vertexOffset;
        // Bottom face edges
        file << "l " << v+0 << " " << v+1 << "\n";
        file << "l " << v+1 << " " << v+2 << "\n";
        file << "l " << v+2 << " " << v+3 << "\n";
        file << "l " << v+3 << " " << v+0 << "\n";
        // Top face edges
        file << "l " << v+4 << " " << v+5 << "\n";
        file << "l " << v+5 << " " << v+6 << "\n";
        file << "l " << v+6 << " " << v+7 << "\n";
        file << "l " << v+7 << " " << v+4 << "\n";
        // Vertical edges
        file << "l " << v+0 << " " << v+4 << "\n";
        file << "l " << v+1 << " " << v+5 << "\n";
        file << "l " << v+2 << " " << v+6 << "\n";
        file << "l " << v+3 << " " << v+7 << "\n";
        file << "\n";

        vertexOffset += 8;
    }

    file.close();
}
