#pragma once

#include "common.h"
#include <cstdint>

// Unified BVH Node structure
// Works for both binary trees (LBVH) and N-ary trees (PLOC)
struct BVHNode {
    AABB_cw bbox;
    
    // For binary trees (LBVH):
    // - leftChild: index of left child (or leaf marker)
    // - rightChild: index of right child
    // For leaf nodes: leftChild = primitive index | 0x80000000
    int32_t leftChild;  
    int32_t rightChild; 
    
    // Alternative representation (if needed):
    // childOffset: start of children in node array
    // childCount: number of children (0 = leaf)
    // primOffset: start of primitives (for leaves)
    // primCount: number of primitives (for leaves)
    
    __host__ __device__ bool isLeaf() const { 
        return (leftChild < 0) || (leftChild & 0x80000000); 
    }
    
    __host__ __device__ uint32_t getPrimitiveIndex() const {
        return leftChild & 0x7FFFFFFF;
    }
};
