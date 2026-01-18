#pragma once

#include "bvh_builder.h"
#include "mesh.h"
#include <string>

// Statistics from BVH evaluation
struct BVHStats {
    float buildTimeMs;
    float sahCost;
    int nodeCount;
    int leafCount;
    int maxDepth;
    float avgLeafDepth;
};

// BVH Quality Evaluator
// Computes SAH cost and other quality metrics
class BVHEvaluator {
public:
    // Evaluate a BVH builder on the given mesh
    // Returns statistics including build time and SAH cost
    static BVHStats evaluate(BVHBuilder* builder, const TriangleMesh& mesh);
    
    // Compute SAH cost of a constructed BVH
    // Uses the standard SAH formula: C_trav + sum(P(node) * C_intersect * primCount)
    static float computeSAH(const std::vector<BVHNode>& nodes, 
                           const std::vector<uint32_t>& indices,
                           const TriangleMesh& mesh);
    
    // Print detailed statistics about the BVH
    static void printStats(const std::string& algorithmName, const BVHStats& stats);
    
private:
    // Helper: Compute surface area heuristic cost recursively
    static float computeSAHRecursive(const std::vector<BVHNode>& nodes,
                                    const std::vector<uint32_t>& indices, 
                                    const TriangleMesh& mesh,
                                    int nodeIdx,
                                    float parentArea);
    
    // Helper: Compute tree depth statistics
    static void computeDepthStats(const std::vector<BVHNode>& nodes,
                                 int nodeIdx, int depth,
                                 int& maxDepth, 
                                 std::vector<int>& leafDepths);
};
