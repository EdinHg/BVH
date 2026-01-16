// =============================================================================
// OpenCL LBVH Kernels for AMD GCN/RDNA (Vega 8)
// Optimized for AMD Ryzen 5700U APU - Wavefront 64, 8 CUs
// =============================================================================

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// =============================================================================
// Data Structures
// =============================================================================

typedef struct {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    uint left_child;   // If bit 31 is 1, it's a leaf index
    uint right_child;  // If bit 31 is 1, it's a leaf index
    uint parent;
    uint padding;      // Align to 48 bytes (12 floats/uints)
} BVHNode;

typedef struct {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
} AABB;

typedef struct {
    float min_x, min_y, min_z;
    float extent_x, extent_y, extent_z;
} SceneBounds;

// Output node format - matches host BVHNode structure
// 40 bytes per node for tight packing
typedef struct __attribute__((packed)) {
    float min_x, min_y, min_z;    // 12 bytes
    float max_x, max_y, max_z;    // 12 bytes
    uint childOffset;              // 4 bytes
    uint primOffset;               // 4 bytes
    ushort childCount;             // 2 bytes
    ushort primCount;              // 2 bytes
    uint axis;                     // 4 bytes
} FinalBVHNode;  // Total: 40 bytes

// =============================================================================
// Utility Functions
// =============================================================================

// Expand 10 bits to 30 bits by inserting 2 zeros between each bit (Morton encoding)
inline uint expand_bits(uint v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8))  & 0x0300F00F;
    v = (v | (v << 4))  & 0x030C30C3;
    v = (v | (v << 2))  & 0x09249249;
    return v;
}

// Compute 30-bit Morton code from normalized [0,1] coordinates
inline uint morton_code_3d(float x, float y, float z) {
    // Clamp to [0, 1023] for 10-bit precision
    x = clamp(x * 1024.0f, 0.0f, 1023.0f);
    y = clamp(y * 1024.0f, 0.0f, 1023.0f);
    z = clamp(z * 1024.0f, 0.0f, 1023.0f);
    
    uint xx = expand_bits((uint)x);
    uint yy = expand_bits((uint)y);
    uint zz = expand_bits((uint)z);
    
    return (xx << 2) | (yy << 1) | zz;
}

// Count leading zeros with proper handling for zero
inline int clz_safe(uint x) {
    return (x == 0) ? 32 : clz(x);
}

// Delta function for Karras algorithm
// Returns length of longest common prefix between codes[i] and codes[j]
// Handles duplicate codes by using index as tie-breaker
inline int delta_func(__global const uint* codes, int i, int j, int n) {
    // Out of bounds check
    if (j < 0 || j >= n) {
        return -1;
    }
    
    uint code_i = codes[i];
    uint code_j = codes[j];
    
    // If codes are identical, use index XOR as tie-breaker
    // This ensures a unique ordering even for duplicates
    if (code_i == code_j) {
        // 32 bits for the code prefix (all match) + leading zeros of index XOR
        return 32 + clz_safe((uint)(i ^ j));
    }
    
    return clz_safe(code_i ^ code_j);
}

// =============================================================================
// Stage 1: Quantization & Morton Codes
// =============================================================================
// Computes Morton codes and leaf AABBs in a single pass
// Uses float4 vectorization for efficient vertex loading on GCN
// =============================================================================

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void compute_morton_codes(
    __global const float4* vertices,    // Triangle vertices (3 float4 per triangle: v0, v1, v2)
    __global uint* morton_codes,         // Output: Morton codes
    __global uint* indices,              // Output: Initial indices [0..N-1]
    __global AABB* leaf_bounds,          // Output: Per-triangle AABBs
    const SceneBounds scene_bounds,      // Scene bounding box
    const uint num_triangles             // Number of triangles
) {
    uint tid = get_global_id(0);
    
    if (tid >= num_triangles) {
        return;
    }
    
    // Load triangle vertices using float4 for vectorized memory access
    // Each triangle is stored as 3 consecutive float4 values (xyz + padding)
    uint base_idx = tid * 3;
    float4 v0 = vertices[base_idx];
    float4 v1 = vertices[base_idx + 1];
    float4 v2 = vertices[base_idx + 2];
    
    // Compute triangle AABB
    float3 aabb_min = fmin(fmin(v0.xyz, v1.xyz), v2.xyz);
    float3 aabb_max = fmax(fmax(v0.xyz, v1.xyz), v2.xyz);
    
    // Store leaf bounds for later refit
    leaf_bounds[tid].min_x = aabb_min.x;
    leaf_bounds[tid].min_y = aabb_min.y;
    leaf_bounds[tid].min_z = aabb_min.z;
    leaf_bounds[tid].max_x = aabb_max.x;
    leaf_bounds[tid].max_y = aabb_max.y;
    leaf_bounds[tid].max_z = aabb_max.z;
    
    // Compute centroid
    float3 centroid = (v0.xyz + v1.xyz + v2.xyz) * (1.0f / 3.0f);
    
    // Normalize centroid to [0, 1] relative to scene bounds
    float3 scene_min = (float3)(scene_bounds.min_x, scene_bounds.min_y, scene_bounds.min_z);
    float3 scene_extent = (float3)(scene_bounds.extent_x, scene_bounds.extent_y, scene_bounds.extent_z);
    
    // Avoid division by zero for degenerate dimensions
    float3 normalized = (centroid - scene_min) / fmax(scene_extent, (float3)(1e-10f));
    normalized = clamp(normalized, 0.0f, 1.0f);
    
    // Compute Morton code
    morton_codes[tid] = morton_code_3d(normalized.x, normalized.y, normalized.z);
    
    // Initialize index array
    indices[tid] = tid;
}

// =============================================================================
// Alternative kernel for SoA (Structure of Arrays) input format
// Matches the TriangleMesh format used in the host code
// =============================================================================

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void compute_morton_codes_soa(
    __global const float* v0x, __global const float* v0y, __global const float* v0z,
    __global const float* v1x, __global const float* v1y, __global const float* v1z,
    __global const float* v2x, __global const float* v2y, __global const float* v2z,
    __global uint* morton_codes,
    __global uint* indices,
    __global AABB* leaf_bounds,
    const SceneBounds scene_bounds,
    const uint num_triangles
) {
    uint tid = get_global_id(0);
    
    if (tid >= num_triangles) {
        return;
    }
    
    // Load triangle vertices from SoA format
    float3 v0 = (float3)(v0x[tid], v0y[tid], v0z[tid]);
    float3 v1 = (float3)(v1x[tid], v1y[tid], v1z[tid]);
    float3 v2 = (float3)(v2x[tid], v2y[tid], v2z[tid]);
    
    // Compute triangle AABB
    float3 aabb_min = fmin(fmin(v0, v1), v2);
    float3 aabb_max = fmax(fmax(v0, v1), v2);
    
    // Store leaf bounds
    leaf_bounds[tid].min_x = aabb_min.x;
    leaf_bounds[tid].min_y = aabb_min.y;
    leaf_bounds[tid].min_z = aabb_min.z;
    leaf_bounds[tid].max_x = aabb_max.x;
    leaf_bounds[tid].max_y = aabb_max.y;
    leaf_bounds[tid].max_z = aabb_max.z;
    
    // Compute centroid
    float3 centroid = (v0 + v1 + v2) * (1.0f / 3.0f);
    
    // Normalize to scene bounds
    float3 scene_min = (float3)(scene_bounds.min_x, scene_bounds.min_y, scene_bounds.min_z);
    float3 scene_extent = (float3)(scene_bounds.extent_x, scene_bounds.extent_y, scene_bounds.extent_z);
    
    float3 normalized = (centroid - scene_min) / fmax(scene_extent, (float3)(1e-10f));
    normalized = clamp(normalized, 0.0f, 1.0f);
    
    // Compute Morton code
    morton_codes[tid] = morton_code_3d(normalized.x, normalized.y, normalized.z);
    
    // Initialize index array
    indices[tid] = tid;
}

// =============================================================================
// Stage 3: Hierarchy Construction (Karras 2012 Algorithm)
// =============================================================================
// Builds BVH topology in parallel - one thread per internal node
// Implements the Karras algorithm for radix tree construction
// =============================================================================

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void build_topology(
    __global const uint* sorted_morton_codes, // Sorted Morton codes
    __global const uint* sorted_indices,      // Sorted primitive indices
    __global BVHNode* nodes,                  // Output: BVH nodes (size = N-1 internal)
    const uint num_primitives                 // N = number of primitives
) {
    uint i = get_global_id(0);
    
    // One thread per internal node (0 to N-2)
    if (i >= num_primitives - 1) {
        return;
    }
    
    int n = (int)num_primitives;
    
    // =======================================================================
    // Step 1: Determine direction of the range
    // =======================================================================
    int delta_left = delta_func(sorted_morton_codes, i, i - 1, n);
    int delta_right = delta_func(sorted_morton_codes, i, i + 1, n);
    
    // Direction is +1 if right has longer common prefix, -1 otherwise
    int d = (delta_right - delta_left >= 0) ? 1 : -1;
    
    // Minimum delta (the "edge" of our range)
    int delta_min = delta_func(sorted_morton_codes, i, i - d, n);
    
    // =======================================================================
    // Step 2: Find upper bound for the range length
    // =======================================================================
    int lmax = 2;
    while (delta_func(sorted_morton_codes, i, i + lmax * d, n) > delta_min) {
        lmax *= 2;
    }
    
    // =======================================================================
    // Step 3: Binary search to find the exact range length
    // =======================================================================
    int l = 0;
    for (int t = lmax >> 1; t >= 1; t >>= 1) {
        if (delta_func(sorted_morton_codes, i, i + (l + t) * d, n) > delta_min) {
            l += t;
        }
    }
    
    // The other end of the range
    int j = i + l * d;
    
    // Ensure first <= last (use ternary to avoid AMD OpenCL min/max ambiguity)
    int first = (i < j) ? i : j;
    int last = (i > j) ? i : j;
    
    // =======================================================================
    // Step 4: Find the split position using binary search
    // =======================================================================
    int delta_node = delta_func(sorted_morton_codes, first, last, n);
    
    int s = 0;
    int t_split = last - first;
    
    while (t_split > 1) {
        t_split = (t_split + 1) >> 1;
        
        int probe = first + s + t_split;
        if (probe < last) {
            if (delta_func(sorted_morton_codes, first, probe, n) > delta_node) {
                s += t_split;
            }
        }
    }
    
    int split = first + s;
    
    // =======================================================================
    // Step 5: Identify children
    // =======================================================================
    // Left child: if first == split, it's a leaf, otherwise internal node
    // Right child: if split+1 == last, it's a leaf, otherwise internal node
    
    uint left_child, right_child;
    
    // Left child
    if (first == split) {
        // Left child is a leaf - mark with bit 31
        left_child = (uint)split | 0x80000000u;
    } else {
        // Left child is internal node at index 'split'
        left_child = (uint)split;
    }
    
    // Right child
    if (split + 1 == last) {
        // Right child is a leaf - mark with bit 31
        right_child = (uint)(split + 1) | 0x80000000u;
    } else {
        // Right child is internal node at index 'split + 1'
        right_child = (uint)(split + 1);
    }
    
    // =======================================================================
    // Step 6: Write node data
    // =======================================================================
    nodes[i].left_child = left_child;
    nodes[i].right_child = right_child;
    
    // Initialize AABB to invalid (will be computed in refit)
    nodes[i].min_x = HUGE_VALF;
    nodes[i].min_y = HUGE_VALF;
    nodes[i].min_z = HUGE_VALF;
    nodes[i].max_x = -HUGE_VALF;
    nodes[i].max_y = -HUGE_VALF;
    nodes[i].max_z = -HUGE_VALF;
    
    // =======================================================================
    // Step 7: Set parent pointers for children
    // =======================================================================
    // We need to write the parent index to the child nodes
    // For leaves, we store the parent in a separate array (handled by host)
    // For internal nodes, we write directly to the parent field
    
    // Handle left child
    if ((left_child & 0x80000000u) == 0) {
        // Internal node
        nodes[left_child].parent = i;
    }
    
    // Handle right child
    if ((right_child & 0x80000000u) == 0) {
        // Internal node
        nodes[right_child].parent = i;
    }
}

// =============================================================================
// Kernel to set parent pointers for leaf nodes
// Must be called after build_topology
// =============================================================================

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void set_leaf_parents(
    __global const BVHNode* internal_nodes,  // Internal nodes from build_topology
    __global uint* leaf_parents,             // Output: parent index for each leaf
    const uint num_primitives
) {
    uint i = get_global_id(0);
    
    if (i >= num_primitives - 1) {
        return;
    }
    
    uint left_child = internal_nodes[i].left_child;
    uint right_child = internal_nodes[i].right_child;
    
    // If left child is a leaf, store parent
    if (left_child & 0x80000000u) {
        uint leaf_idx = left_child & 0x7FFFFFFFu;
        leaf_parents[leaf_idx] = i;
    }
    
    // If right child is a leaf, store parent
    if (right_child & 0x80000000u) {
        uint leaf_idx = right_child & 0x7FFFFFFFu;
        leaf_parents[leaf_idx] = i;
    }
}

// =============================================================================
// Stage 4: Bottom-Up Refit (Atomic Counters)
// =============================================================================
// Computes AABBs from leaves up to root using atomic synchronization
// Each leaf thread propagates upward, second visitor computes the union
// =============================================================================

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void refit_bounds(
    __global BVHNode* nodes,                 // Internal nodes (size = N-1)
    __global const AABB* leaf_bounds,        // Leaf AABBs (size = N)
    __global const uint* sorted_indices,     // Original primitive indices
    __global const uint* leaf_parents,       // Parent of each leaf
    __global int* counters,                  // Atomic counters (size = N-1)
    const uint num_primitives
) {
    uint tid = get_global_id(0);
    
    // One thread per leaf
    if (tid >= num_primitives) {
        return;
    }
    
    // Get the original triangle index for this sorted position
    uint orig_idx = sorted_indices[tid];
    
    // Load leaf bounds
    AABB bounds = leaf_bounds[orig_idx];
    float min_x = bounds.min_x;
    float min_y = bounds.min_y;
    float min_z = bounds.min_z;
    float max_x = bounds.max_x;
    float max_y = bounds.max_y;
    float max_z = bounds.max_z;
    
    // Start at the leaf's parent
    uint parent = leaf_parents[tid];
    
    // Special case: single primitive (no internal nodes)
    if (num_primitives == 1) {
        return;
    }
    
    // Traverse up to root
    while (true) {
        // Memory barrier to ensure previous writes are visible
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        
        // Atomically increment counter for this parent
        int old_count = atomic_inc(&counters[parent]);
        
        if (old_count == 0) {
            // First child to arrive - terminate this thread
            // The second child will compute the union
            return;
        }
        
        // Second child to arrive - compute union of children
        
        // Memory barrier to ensure we read updated values
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        
        // Get the sibling's bounds
        uint left_child = nodes[parent].left_child;
        uint right_child = nodes[parent].right_child;
        
        // Determine which child we came from and get sibling bounds
        float sibling_min_x, sibling_min_y, sibling_min_z;
        float sibling_max_x, sibling_max_y, sibling_max_z;
        
        // Load left child bounds
        if (left_child & 0x80000000u) {
            // Left is a leaf
            uint leaf_idx = sorted_indices[left_child & 0x7FFFFFFFu];
            AABB leaf_aabb = leaf_bounds[leaf_idx];
            sibling_min_x = leaf_aabb.min_x;
            sibling_min_y = leaf_aabb.min_y;
            sibling_min_z = leaf_aabb.min_z;
            sibling_max_x = leaf_aabb.max_x;
            sibling_max_y = leaf_aabb.max_y;
            sibling_max_z = leaf_aabb.max_z;
        } else {
            // Left is an internal node
            sibling_min_x = nodes[left_child].min_x;
            sibling_min_y = nodes[left_child].min_y;
            sibling_min_z = nodes[left_child].min_z;
            sibling_max_x = nodes[left_child].max_x;
            sibling_max_y = nodes[left_child].max_y;
            sibling_max_z = nodes[left_child].max_z;
        }
        
        // Compute union with left child
        min_x = fmin(min_x, sibling_min_x);
        min_y = fmin(min_y, sibling_min_y);
        min_z = fmin(min_z, sibling_min_z);
        max_x = fmax(max_x, sibling_max_x);
        max_y = fmax(max_y, sibling_max_y);
        max_z = fmax(max_z, sibling_max_z);
        
        // Load right child bounds
        if (right_child & 0x80000000u) {
            // Right is a leaf
            uint leaf_idx = sorted_indices[right_child & 0x7FFFFFFFu];
            AABB leaf_aabb = leaf_bounds[leaf_idx];
            sibling_min_x = leaf_aabb.min_x;
            sibling_min_y = leaf_aabb.min_y;
            sibling_min_z = leaf_aabb.min_z;
            sibling_max_x = leaf_aabb.max_x;
            sibling_max_y = leaf_aabb.max_y;
            sibling_max_z = leaf_aabb.max_z;
        } else {
            // Right is an internal node
            sibling_min_x = nodes[right_child].min_x;
            sibling_min_y = nodes[right_child].min_y;
            sibling_min_z = nodes[right_child].min_z;
            sibling_max_x = nodes[right_child].max_x;
            sibling_max_y = nodes[right_child].max_y;
            sibling_max_z = nodes[right_child].max_z;
        }
        
        // Compute union with right child
        min_x = fmin(min_x, sibling_min_x);
        min_y = fmin(min_y, sibling_min_y);
        min_z = fmin(min_z, sibling_min_z);
        max_x = fmax(max_x, sibling_max_x);
        max_y = fmax(max_y, sibling_max_y);
        max_z = fmax(max_z, sibling_max_z);
        
        // Write bounds to parent node
        nodes[parent].min_x = min_x;
        nodes[parent].min_y = min_y;
        nodes[parent].min_z = min_z;
        nodes[parent].max_x = max_x;
        nodes[parent].max_y = max_y;
        nodes[parent].max_z = max_z;
        
        // Check if we've reached the root (node 0)
        if (parent == 0) {
            return;
        }
        
        // Move to grandparent
        parent = nodes[parent].parent;
    }
}

// =============================================================================
// Utility kernel: Initialize counters to zero
// =============================================================================

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void init_counters(
    __global int* counters,
    const uint num_counters
) {
    uint tid = get_global_id(0);
    
    if (tid < num_counters) {
        counters[tid] = 0;
    }
}

// =============================================================================
// Utility kernel: Compute scene bounds reduction (Phase 1 - local reduction)
// =============================================================================

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void compute_scene_bounds_reduce(
    __global const float* v0x, __global const float* v0y, __global const float* v0z,
    __global const float* v1x, __global const float* v1y, __global const float* v1z,
    __global const float* v2x, __global const float* v2y, __global const float* v2z,
    __global float* partial_min_x, __global float* partial_min_y, __global float* partial_min_z,
    __global float* partial_max_x, __global float* partial_max_y, __global float* partial_max_z,
    __local float* local_min_x, __local float* local_min_y, __local float* local_min_z,
    __local float* local_max_x, __local float* local_max_y, __local float* local_max_z,
    const uint num_triangles
) {
    uint tid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group_id = get_group_id(0);
    uint local_size = get_local_size(0);
    
    // Initialize to extreme values
    float min_x = HUGE_VALF, min_y = HUGE_VALF, min_z = HUGE_VALF;
    float max_x = -HUGE_VALF, max_y = -HUGE_VALF, max_z = -HUGE_VALF;
    
    // Each thread processes multiple elements
    for (uint i = tid; i < num_triangles; i += get_global_size(0)) {
        // Load all three vertices
        float3 v0 = (float3)(v0x[i], v0y[i], v0z[i]);
        float3 v1 = (float3)(v1x[i], v1y[i], v1z[i]);
        float3 v2 = (float3)(v2x[i], v2y[i], v2z[i]);
        
        // Compute min/max across vertices
        float3 tri_min = fmin(fmin(v0, v1), v2);
        float3 tri_max = fmax(fmax(v0, v1), v2);
        
        // Update running min/max
        min_x = fmin(min_x, tri_min.x);
        min_y = fmin(min_y, tri_min.y);
        min_z = fmin(min_z, tri_min.z);
        max_x = fmax(max_x, tri_max.x);
        max_y = fmax(max_y, tri_max.y);
        max_z = fmax(max_z, tri_max.z);
    }
    
    // Store in local memory
    local_min_x[lid] = min_x;
    local_min_y[lid] = min_y;
    local_min_z[lid] = min_z;
    local_max_x[lid] = max_x;
    local_max_y[lid] = max_y;
    local_max_z[lid] = max_z;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel reduction in workgroup
    for (uint stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_min_x[lid] = fmin(local_min_x[lid], local_min_x[lid + stride]);
            local_min_y[lid] = fmin(local_min_y[lid], local_min_y[lid + stride]);
            local_min_z[lid] = fmin(local_min_z[lid], local_min_z[lid + stride]);
            local_max_x[lid] = fmax(local_max_x[lid], local_max_x[lid + stride]);
            local_max_y[lid] = fmax(local_max_y[lid], local_max_y[lid + stride]);
            local_max_z[lid] = fmax(local_max_z[lid], local_max_z[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (lid == 0) {
        partial_min_x[group_id] = local_min_x[0];
        partial_min_y[group_id] = local_min_y[0];
        partial_min_z[group_id] = local_min_z[0];
        partial_max_x[group_id] = local_max_x[0];
        partial_max_y[group_id] = local_max_y[0];
        partial_max_z[group_id] = local_max_z[0];
    }
}

// =============================================================================
// Kernel 7: Convert GPU BVH layout to final host format
// =============================================================================
// Converts internal BVHNode to FinalBVHNode layout and creates leaf nodes
// in a single pass on the GPU, avoiding CPU conversion loops.
//
// Output layout: [internal nodes: 0..n-2] [leaf nodes: n-1..2n-2]
// =============================================================================
__kernel void convert_to_final_layout(
    __global const BVHNode* gpu_nodes,       // Internal nodes from build
    __global const uint* sorted_indices,     // Sorted primitive indices
    __global const AABB* leaf_bounds,        // AABB per primitive (original order)
    __global FinalBVHNode* final_nodes,      // Output: 2n-1 final nodes
    __global uint* prim_indices_out,         // Output: sorted primitive indices
    const uint n                             // Number of primitives
) {
    uint tid = get_global_id(0);
    uint num_internal = n - 1;
    uint total_nodes = 2 * n - 1;
    
    // Process internal nodes (indices 0 to n-2)
    if (tid < num_internal) {
        BVHNode src = gpu_nodes[tid];
        FinalBVHNode dst;
        
        // Copy bounds
        dst.min_x = src.min_x;
        dst.min_y = src.min_y;
        dst.min_z = src.min_z;
        dst.max_x = src.max_x;
        dst.max_y = src.max_y;
        dst.max_z = src.max_z;
        
        // Convert child references
        // If child bit 31 is set, it's a leaf index - convert to final layout index
        uint left = src.left_child;
        uint right = src.right_child;
        
        if (left & 0x80000000u) {
            uint leaf_idx = left & 0x7FFFFFFFu;
            dst.childOffset = n - 1 + leaf_idx;  // Leaf starts at n-1
        } else {
            dst.childOffset = left;  // Internal node index unchanged
        }
        
        if (right & 0x80000000u) {
            uint leaf_idx = right & 0x7FFFFFFFu;
            dst.primOffset = n - 1 + leaf_idx;
        } else {
            dst.primOffset = right;
        }
        
        dst.childCount = 2;  // Binary BVH
        dst.primCount = 0;   // Internal node has no primitives
        dst.axis = 0;
        
        final_nodes[tid] = dst;
    }
    
    // Process leaf nodes (indices n-1 to 2n-2)
    // Each leaf corresponds to sorted index i, placed at position n-1+i
    if (tid < n) {
        uint final_idx = n - 1 + tid;  // Leaf position in final array
        uint orig_prim_idx = sorted_indices[tid];  // Original primitive index
        AABB bounds = leaf_bounds[orig_prim_idx];
        
        FinalBVHNode leaf;
        leaf.min_x = bounds.min_x;
        leaf.min_y = bounds.min_y;
        leaf.min_z = bounds.min_z;
        leaf.max_x = bounds.max_x;
        leaf.max_y = bounds.max_y;
        leaf.max_z = bounds.max_z;
        leaf.childOffset = 0;
        leaf.primOffset = tid;       // Index into prim_indices_out
        leaf.childCount = 0;         // Leaf marker
        leaf.primCount = 1;          // One primitive per leaf
        leaf.axis = 0;
        
        final_nodes[final_idx] = leaf;
        
        // Copy primitive index to output array
        prim_indices_out[tid] = orig_prim_idx;
    }
}
