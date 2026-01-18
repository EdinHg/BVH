#include "../include/evaluator.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>

BVHStats BVHEvaluator::evaluate(BVHBuilder* builder, const TriangleMesh& mesh) {
    BVHStats stats = {};
    
    // Build the BVH and measure time
    builder->build(mesh);
    stats.buildTimeMs = builder->getLastBuildTimeMS();
    
    // Get constructed tree
    const auto& nodes = builder->getNodes();
    const auto& indices = builder->getIndices();
    
    stats.nodeCount = nodes.size();
    
    // Compute SAH cost
    stats.sahCost = computeSAH(nodes, indices, mesh);
    
    // Compute depth statistics
    std::vector<int> leafDepths;
    stats.maxDepth = 0;
    if (!nodes.empty()) {
        computeDepthStats(nodes, 0, 0, stats.maxDepth, leafDepths);
    }
    
    stats.leafCount = leafDepths.size();
    if (!leafDepths.empty()) {
        stats.avgLeafDepth = std::accumulate(leafDepths.begin(), leafDepths.end(), 0.0f) / leafDepths.size();
    }
    
    return stats;
}

float BVHEvaluator::computeSAH(const std::vector<BVHNode>& nodes,
                               const std::vector<uint32_t>& indices,
                               const TriangleMesh& mesh) {
    if (nodes.empty()) return 0.0f;
    
    // Root bounding box surface area
    float rootArea = nodes[0].bbox.surfaceArea();
    if (rootArea <= 0.0f) return 0.0f;
    
    // Standard SAH costs
    const float C_trav = 1.0f;      // Cost of traversing an internal node
    const float C_intersect = 1.0f; // Cost of intersecting a primitive
    
    return computeSAHRecursive(nodes, indices, mesh, 0, rootArea);
}

float BVHEvaluator::computeSAHRecursive(const std::vector<BVHNode>& nodes,
                                        const std::vector<uint32_t>& indices,
                                        const TriangleMesh& mesh,
                                        int nodeIdx,
                                        float rootArea) {
    if (nodeIdx < 0 || nodeIdx >= (int)nodes.size()) {
        return 0.0f;
    }
    
    const BVHNode& node = nodes[nodeIdx];
    float nodeArea = node.bbox.surfaceArea();
    float probability = nodeArea / rootArea;
    
    const float C_trav = 1.0f;
    const float C_intersect = 1.0f;
    
    if (node.isLeaf()) {
        // Leaf node: cost = probability * intersection_cost * primitive_count
        // For LBVH, each leaf typically contains 1 primitive
        return probability * C_intersect * 1.0f;
    }
    
    // Internal node: cost = probability * traversal_cost + sum(child_costs)
    float cost = probability * C_trav;
    
    // Recursively compute child costs
    if (node.leftChild >= 0 && node.leftChild < (int)nodes.size()) {
        cost += computeSAHRecursive(nodes, indices, mesh, node.leftChild, rootArea);
    }
    if (node.rightChild >= 0 && node.rightChild < (int)nodes.size()) {
        cost += computeSAHRecursive(nodes, indices, mesh, node.rightChild, rootArea);
    }
    
    return cost;
}

void BVHEvaluator::computeDepthStats(const std::vector<BVHNode>& nodes,
                                     int nodeIdx, int depth,
                                     int& maxDepth,
                                     std::vector<int>& leafDepths) {
    if (nodeIdx < 0 || nodeIdx >= (int)nodes.size()) {
        return;
    }
    
    maxDepth = std::max(maxDepth, depth);
    const BVHNode& node = nodes[nodeIdx];
    
    if (node.isLeaf()) {
        leafDepths.push_back(depth);
        return;
    }
    
    // Recurse to children
    if (node.leftChild >= 0 && node.leftChild < (int)nodes.size()) {
        computeDepthStats(nodes, node.leftChild, depth + 1, maxDepth, leafDepths);
    }
    if (node.rightChild >= 0 && node.rightChild < (int)nodes.size()) {
        computeDepthStats(nodes, node.rightChild, depth + 1, maxDepth, leafDepths);
    }
}

void BVHEvaluator::printStats(const std::string& algorithmName, const BVHStats& stats) {
    std::cout << "\n=== " << algorithmName << " Statistics ===\n";
    std::cout << "  Build Time:      " << std::setw(10) << std::fixed << std::setprecision(3) 
              << stats.buildTimeMs << " ms\n";
    std::cout << "  SAH Cost:        " << std::setw(10) << std::fixed << std::setprecision(2) 
              << stats.sahCost << "\n";
    std::cout << "  Node Count:      " << std::setw(10) << stats.nodeCount << "\n";
    std::cout << "  Leaf Count:      " << std::setw(10) << stats.leafCount << "\n";
    std::cout << "  Max Depth:       " << std::setw(10) << stats.maxDepth << "\n";
    std::cout << "  Avg Leaf Depth:  " << std::setw(10) << std::fixed << std::setprecision(2) 
              << stats.avgLeafDepth << "\n";
    
    float throughput = (stats.leafCount / 1e6f) / (stats.buildTimeMs / 1000.0f);
    std::cout << "  Throughput:      " << std::setw(10) << std::fixed << std::setprecision(2) 
              << throughput << " MTris/s\n";
}
