#pragma once

#include "bvh_builder.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

// Expand 10 bits of input into 30 bits by inserting 2 zeros between each bit
inline uint32_t expandBits(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

// Compute 30-bit Morton code for normalized coordinates [0,1]
inline uint32_t mortonCode(float x, float y, float z) {
    x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
    y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
    z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
    uint32_t xx = expandBits(static_cast<uint32_t>(x));
    uint32_t yy = expandBits(static_cast<uint32_t>(y));
    uint32_t zz = expandBits(static_cast<uint32_t>(z));
    return (xx << 2) | (yy << 1) | zz;
}

// Radix sort with 10-bit buckets (3 passes for 30-bit Morton codes)
inline void radix10Sort(std::vector<std::pair<uint32_t, uint32_t>>& mortonCodes) {
    size_t n = mortonCodes.size();
    if (n <= 1) return;
    
    std::vector<std::pair<uint32_t, uint32_t>> temp(n);
    
    for (int shift = 0; shift < 30; shift += 10) {
        int count[1024] = {0};
        
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (mortonCodes[i].first >> shift) & 0x3FF;
            count[bits]++;
        }
        
        int total = 0;
        for (int i = 0; i < 1024; i++) {
            int c = count[i];
            count[i] = total;
            total += c;
        }
        
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (mortonCodes[i].first >> shift) & 0x3FF;
            temp[count[bits]++] = mortonCodes[i];
        }
        
        mortonCodes.swap(temp);
    }
}

// Count leading zeros (portable implementation)
inline int clz32(uint32_t x) {
    if (x == 0) return 32;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(x);
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanReverse(&index, x);
    return 31 - static_cast<int>(index);
#else
    int n = 0;
    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFF) { n += 8;  x <<= 8;  }
    if (x <= 0x0FFFFFFF) { n += 4;  x <<= 4;  }
    if (x <= 0x3FFFFFFF) { n += 2;  x <<= 2;  }
    if (x <= 0x7FFFFFFF) { n += 1; }
    return n;
#endif
}

// LBVH Builder using Morton codes (Karras 2012 algorithm)
class LBVHBuilder : public BVHBuilder {
public:
    std::string name() const override { return "LBVH"; }

    BVHResult build(const TriangleMesh& mesh) override {
        BVHResult result;
        result.branchingFactor = 2;

        size_t n = mesh.size();
        if (n == 0) return result;

        // Step 1: Compute scene bounds and centroids
        AABB sceneBounds;
        std::vector<Vec3> centroids(n);
        std::vector<AABB> triBounds(n);

        for (size_t i = 0; i < n; ++i) {
            triBounds[i] = mesh.getBounds(i);
            centroids[i] = triBounds[i].center();
            sceneBounds.expand(triBounds[i]);
        }

        Vec3 sceneSize = sceneBounds.extent();

        // Step 2: Compute Morton codes
        std::vector<std::pair<uint32_t, uint32_t>> mortonCodes(n);
        for (size_t i = 0; i < n; ++i) {
            Vec3 offset = centroids[i] - sceneBounds.min;
            float nx = sceneSize.x > 0 ? offset.x / sceneSize.x : 0.5f;
            float ny = sceneSize.y > 0 ? offset.y / sceneSize.y : 0.5f;
            float nz = sceneSize.z > 0 ? offset.z / sceneSize.z : 0.5f;
            mortonCodes[i].first = mortonCode(nx, ny, nz);
            mortonCodes[i].second = static_cast<uint32_t>(i);
        }

        // Step 3: Radix sort by Morton code
        radix10Sort(mortonCodes);

        // Extract sorted indices and codes
        std::vector<uint32_t> sortedCodes(n);
        result.primIndices.resize(n);
        for (size_t i = 0; i < n; ++i) {
            result.primIndices[i] = mortonCodes[i].second;
            sortedCodes[i] = mortonCodes[i].first;
        }

        // Step 4: Build tree structure
        // Total nodes: n-1 internal + n leaves = 2n-1
        result.nodes.resize(2 * n - 1);

        // Initialize leaf nodes (indices n-1 to 2n-2)
        for (size_t i = 0; i < n; ++i) {
            BVHNode& leaf = result.nodes[n - 1 + i];
            uint32_t originalIdx = result.primIndices[i];
            leaf.bounds = triBounds[originalIdx];
            leaf.childOffset = 0;
            leaf.childCount = 0;  // leaf
            leaf.primCount = 1;
            leaf.primOffset = static_cast<uint32_t>(i);
            leaf.axis = 0;
        }

        // Build internal nodes using Karras algorithm
        std::vector<uint32_t> parents(2 * n - 1, UINT32_MAX);
        
        for (size_t i = 0; i < n - 1; ++i) {
            auto [first, last] = determineRange(sortedCodes, static_cast<int>(i), static_cast<int>(n));
            int split = findSplit(sortedCodes, first, last, static_cast<int>(n));

            uint32_t leftChild = (first == split) 
                ? static_cast<uint32_t>(n - 1 + split) 
                : static_cast<uint32_t>(split);
            uint32_t rightChild = (split + 1 == last) 
                ? static_cast<uint32_t>(n - 1 + split + 1) 
                : static_cast<uint32_t>(split + 1);

            BVHNode& node = result.nodes[i];
            node.childOffset = leftChild;
            node.childCount = 2;  // binary
            node.primCount = 0;
            node.primOffset = rightChild;  // Store right child in primOffset for internal nodes
            node.axis = 0;

            parents[leftChild] = static_cast<uint32_t>(i);
            parents[rightChild] = static_cast<uint32_t>(i);
        }

        // Step 5: Compute bounding boxes bottom-up
        std::vector<bool> visited(2 * n - 1, false);
        for (size_t i = 0; i < n; ++i) {
            uint32_t current = static_cast<uint32_t>(n - 1 + i);
            visited[current] = true;

            while (current > 0) {
                uint32_t parent = parents[current];
                if (parent == UINT32_MAX) break;

                uint32_t leftChild = result.nodes[parent].childOffset;
                uint32_t rightChild = result.nodes[parent].primOffset;
                uint32_t sibling = (leftChild == current) ? rightChild : leftChild;

                if (!visited[sibling]) break;

                result.nodes[parent].bounds.expand(result.nodes[leftChild].bounds);
                result.nodes[parent].bounds.expand(result.nodes[rightChild].bounds);

                visited[parent] = true;
                current = parent;
            }
        }

        return result;
    }

private:
    // Delta function: common prefix length between Morton codes
    int deltaNode(const std::vector<uint32_t>& codes, int i, int j, int n) const {
        if (j < 0 || j >= n) return -1;
        if (codes[i] == codes[j]) return 32 + clz32(static_cast<uint32_t>(i ^ j));
        return clz32(codes[i] ^ codes[j]);
    }

    // Determine range of leaf indices covered by internal node i
    std::pair<int, int> determineRange(const std::vector<uint32_t>& codes, int i, int n) const {
        int d = (deltaNode(codes, i, i + 1, n) - deltaNode(codes, i, i - 1, n)) > 0 ? 1 : -1;

        int deltaMin = deltaNode(codes, i, i - d, n);
        int lmax = 2;
        while (deltaNode(codes, i, i + lmax * d, n) > deltaMin) {
            lmax *= 2;
        }

        int l = 0;
        for (int t = lmax / 2; t >= 1; t /= 2) {
            if (deltaNode(codes, i, i + (l + t) * d, n) > deltaMin) {
                l += t;
            }
        }

        int j = i + l * d;
        return d > 0 ? std::make_pair(i, j) : std::make_pair(j, i);
    }

    // Binary search for split position
    int findSplit(const std::vector<uint32_t>& codes, int first, int last, int n) const {
        int deltaNode_ = deltaNode(codes, first, last, n);
        int s = 0;
        int t = last - first;

        while (t > 1) {
            t = (t + 1) >> 1;
            if (deltaNode(codes, first, first + s + t, n) > deltaNode_) {
                s += t;
            }
        }

        return first + s;
    }
};
