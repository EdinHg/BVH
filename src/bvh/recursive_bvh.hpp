#pragma once

#include "bvh_builder.hpp"
#include <algorithm>
#include <numeric>

class RecursiveBVH : public BVHBuilder {
public:
    explicit RecursiveBVH(int maxLeafSize = 4) : maxLeafSize_(maxLeafSize) {}

    std::string name() const override { return "RecursiveBVH"; }

    BVHResult build(const TriangleMesh& mesh) override {
        BVHResult result;
        result.branchingFactor = 2;

        size_t n = mesh.size();
        if (n == 0) return result;

        // Initialize primitive indices
        result.primIndices.resize(n);
        std::iota(result.primIndices.begin(), result.primIndices.end(), 0);

        // Precompute centroids and bounds
        centroids_.resize(n);
        primBounds_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            centroids_[i] = mesh.getCentroid(i);
            primBounds_[i] = mesh.getBounds(i);
        }

        // Reserve space (rough estimate: 2*n nodes for binary tree)
        result.nodes.reserve(2 * n);

        // Build recursively
        buildRecursive(result, mesh, 0, n);

        centroids_.clear();
        primBounds_.clear();
        return result;
    }

private:
    int maxLeafSize_;
    std::vector<Vec3> centroids_;
    std::vector<AABB> primBounds_;

    uint32_t buildRecursive(BVHResult& result, const TriangleMesh& mesh,
                            size_t start, size_t end) {
        uint32_t nodeIdx = static_cast<uint32_t>(result.nodes.size());
        result.nodes.emplace_back();
        BVHNode& node = result.nodes[nodeIdx];

        // Compute bounds for this node
        AABB bounds;
        for (size_t i = start; i < end; ++i) {
            bounds.expand(primBounds_[result.primIndices[i]]);
        }
        node.bounds = bounds;

        size_t primCount = end - start;

        // Create leaf if few enough primitives
        if (primCount <= static_cast<size_t>(maxLeafSize_)) {
            node.childOffset = 0;
            node.childCount = 0;  // leaf
            node.axis = 0;
            node.primCount = static_cast<uint16_t>(primCount);
            node.primOffset = static_cast<uint32_t>(start);
            return nodeIdx;
        }

        // Find split axis (longest axis of centroid bounds)
        AABB centroidBounds;
        for (size_t i = start; i < end; ++i) {
            centroidBounds.expand(centroids_[result.primIndices[i]]);
        }
        int axis = centroidBounds.longestAxis();

        // Sort primitives along axis using centroid
        std::sort(result.primIndices.begin() + start,
                  result.primIndices.begin() + end,
                  [this, axis](uint32_t a, uint32_t b) {
                      return centroids_[a][axis] < centroids_[b][axis];
                  });

        // Split at median
        size_t mid = start + primCount / 2;

        // Build children
        node.axis = static_cast<uint8_t>(axis);
        node.childCount = 2;  // binary tree
        node.primCount = 0;
        node.primOffset = 0;

        uint32_t leftChild = buildRecursive(result, mesh, start, mid);
        uint32_t rightChild = buildRecursive(result, mesh, mid, end);

        // Update node after children are built (node reference may be invalid)
        result.nodes[nodeIdx].childOffset = leftChild;
        // Note: right child is at leftChild + 1 for binary trees in this layout

        return nodeIdx;
    }
};
