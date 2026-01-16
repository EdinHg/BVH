#pragma once

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../math/aabb.hpp"
#include "../mesh/triangle_mesh.hpp"
#include "bvh_builder.hpp"

// =============================================================================
// Boost.Compute SHA1 Compatibility Fix
// =============================================================================
// Boost 1.85+ changed sha1::get_digest() to use unsigned char[20] instead of
// unsigned int[5]. We provide a wrapper before including Boost.Compute.
// =============================================================================
#ifndef BOOST_COMPUTE_DETAIL_SHA1_HPP
#define BOOST_COMPUTE_DETAIL_SHA1_HPP

#include <sstream>
#include <string>
#include <boost/uuid/detail/sha1.hpp>

namespace boost {
namespace compute {
namespace detail {

class sha1 {
public:
    sha1() = default;
    
    // Constructor from string - used by meta_kernel
    explicit sha1(const std::string& str) {
        process(str.c_str());
    }
    
    void process(const char* str) {
        h_.process_bytes(str, std::strlen(str));
    }
    
    operator std::string() {
        unsigned char digest[20];
        h_.get_digest(digest);
        std::ostringstream ss;
        for (int i = 0; i < 20; ++i) {
            ss << std::hex << ((digest[i] >> 4) & 0xf) << (digest[i] & 0xf);
        }
        return ss.str();
    }

private:
    boost::uuids::detail::sha1 h_;
};

}  // namespace detail
}  // namespace compute
}  // namespace boost

#endif  // BOOST_COMPUTE_DETAIL_SHA1_HPP

// =============================================================================
// Boost.Compute for radix sort only
// =============================================================================
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/compute.hpp>
namespace compute = boost::compute;

// =============================================================================
// OpenCL LBVH Builder for AMD APU (Vega 8)
// =============================================================================
// Optimized for:
//   - AMD Ryzen 5700U APU (Integrated Radeon Vega 8 Graphics)
//   - GCN 5.0 Architecture
//   - Shared System RAM (Unified Memory / Zero-Copy)
//   - Wavefront Size: 64
//   - Compute Units: 8
// =============================================================================

// GPU-side BVH node structure (must match kernel definition)
struct alignas(16) GPUBVHNode {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    uint32_t left_child;   // Bit 31 = 1 indicates leaf
    uint32_t right_child;  // Bit 31 = 1 indicates leaf
    uint32_t parent;
    uint32_t padding;
};

// Final output node structure - matches kernel FinalBVHNode
// Must be packed to match GPU output exactly (40 bytes)
#pragma pack(push, 1)
struct FinalBVHNode {
    float min_x, min_y, min_z;    // 12 bytes
    float max_x, max_y, max_z;    // 12 bytes
    uint32_t childOffset;          // 4 bytes
    uint32_t primOffset;           // 4 bytes
    uint16_t childCount;           // 2 bytes
    uint16_t primCount;            // 2 bytes
    uint32_t axis;                 // 4 bytes
};  // Total: 40 bytes
#pragma pack(pop)

// GPU-side AABB structure
struct GPUAABB {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
};

// Scene bounds for Morton code computation
struct SceneBounds {
    float min_x, min_y, min_z;
    float extent_x, extent_y, extent_z;
};

class OpenCLLBVHBuilder : public BVHBuilder {
   public:
    OpenCLLBVHBuilder() { initializeOpenCL(); }

    ~OpenCLLBVHBuilder() override = default;

    std::string name() const override { return "OpenCL LBVH (AMD APU)"; }

    BVHResult build(const TriangleMesh& mesh) override {
        BVHResult result;
        result.branchingFactor = 2;

        size_t n = mesh.size();
        if (n == 0) return result;
        if (n == 1) {
            // Handle single triangle case
            result.nodes.resize(1);
            result.primIndices = {0};
            result.nodes[0].bounds = mesh.getBounds(0);
            result.nodes[0].childOffset = 0;
            result.nodes[0].childCount = 0;
            result.nodes[0].primCount = 1;
            result.nodes[0].primOffset = 0;
            result.nodes[0].axis = 0;
            return result;
        }

        try {
            // =================================================================
            // Step 1: Compute scene bounds on GPU
            // =================================================================
            SceneBounds sceneBounds = computeSceneBoundsGPU(mesh);

            // =================================================================
            // Step 2: Compute Morton codes and leaf bounds
            // =================================================================
            auto [mortonCodes, indices, leafBounds] =
                computeMortonCodesGPU(mesh, sceneBounds);

            // =================================================================
            // Step 3: Sort by Morton code using Boost.Compute
            // =================================================================
            sortByMortonCode(mortonCodes, indices);

            // =================================================================
            // Step 4: Build BVH topology (Karras algorithm)
            // =================================================================
            auto [gpuNodes, leafParents] =
                buildTopologyGPU(mortonCodes, indices, n);

            // =================================================================
            // Step 5: Refit bounds bottom-up
            // =================================================================
            refitBoundsGPU(gpuNodes, leafBounds, indices, leafParents, n);

            // =================================================================
            // Step 6: Convert to host BVH format
            // =================================================================
            result = convertToHostFormat(gpuNodes, indices, leafBounds, n);

        } catch (const compute::opencl_error& e) {
            std::cerr << "OpenCL error: " << e.what()
                      << " (code: " << e.error_code() << ")" << std::endl;
            throw;
        }

        return result;
    }

   private:
    compute::device device_;
    compute::context context_;
    compute::command_queue queue_;
    compute::program program_;

    // Kernel handles
    compute::kernel kernel_morton_codes_soa_;
    compute::kernel kernel_build_topology_;
    compute::kernel kernel_set_leaf_parents_;
    compute::kernel kernel_refit_bounds_;
    compute::kernel kernel_init_counters_;
    compute::kernel kernel_scene_bounds_reduce_;
    compute::kernel kernel_convert_layout_;

    // Workgroup size for AMD (multiple of wavefront size 64)
    static constexpr size_t LOCAL_WORK_SIZE = 256;

    void initializeOpenCL() {
        // =================================================================
        // Device Selection: Prefer AMD GPU
        // =================================================================
        try {
            device_ = findAMDDevice();
        } catch (...) {
            std::cout << "AMD GPU not found, using default device" << std::endl;
            device_ = compute::system::default_device();
        }

        std::cout << "OpenCL Device: " << device_.name() << std::endl;
        std::cout << "  Vendor: " << device_.vendor() << std::endl;
        std::cout << "  Compute Units: " << device_.compute_units()
                  << std::endl;
        std::cout << "  Max Work Group Size: " << device_.max_work_group_size()
                  << std::endl;
        std::cout << "  Global Memory: "
                  << (device_.global_memory_size() / (1024 * 1024)) << " MB"
                  << std::endl;

        context_ = compute::context(device_);
        queue_ = compute::command_queue(context_, device_);

        // =================================================================
        // Compile OpenCL kernels
        // =================================================================
        std::string kernelSource = loadKernelSource();

        try {
            program_ =
                compute::program::create_with_source(kernelSource, context_);
            program_.build(
                "-cl-std=CL1.2 -cl-mad-enable -cl-fast-relaxed-math");
        } catch (const compute::opencl_error& e) {
            std::cerr << "OpenCL build error: " << e.what() << std::endl;
            std::cerr << "Build log:\n" << program_.build_log() << std::endl;
            throw;
        }

        // Create kernel handles
        kernel_morton_codes_soa_ =
            program_.create_kernel("compute_morton_codes_soa");
        kernel_build_topology_ = program_.create_kernel("build_topology");
        kernel_set_leaf_parents_ = program_.create_kernel("set_leaf_parents");
        kernel_refit_bounds_ = program_.create_kernel("refit_bounds");
        kernel_init_counters_ = program_.create_kernel("init_counters");
        kernel_scene_bounds_reduce_ =
            program_.create_kernel("compute_scene_bounds_reduce");
        kernel_convert_layout_ =
            program_.create_kernel("convert_to_final_layout");
    }

    compute::device findAMDDevice() {
        // Search for AMD GPU device
        for (const auto& platform : compute::system::platforms()) {
            for (const auto& dev : platform.devices()) {
                std::string vendor = dev.vendor();
                std::string name = dev.name();

                // Look for AMD GPU (not CPU)
                if ((vendor.find("AMD") != std::string::npos ||
                     vendor.find("Advanced Micro Devices") !=
                         std::string::npos) &&
                    dev.type() == CL_DEVICE_TYPE_GPU) {
                    return dev;
                }
            }
        }
        throw std::runtime_error("AMD GPU not found");
    }

    std::string loadKernelSource() {
        // Try to load from file first
        std::ifstream file("src/bvh/opencl_kernels.cl");
        if (file.is_open()) {
            std::stringstream ss;
            ss << file.rdbuf();
            return ss.str();
        }

        // Try alternative path
        file.open("opencl_kernels.cl");
        if (file.is_open()) {
            std::stringstream ss;
            ss << file.rdbuf();
            return ss.str();
        }

        // Fallback: embedded kernel source
        return getEmbeddedKernelSource();
    }

    std::string getEmbeddedKernelSource() {
        // Embedded OpenCL kernel source (subset for fallback)
        return R"(
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef struct {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    uint left_child;
    uint right_child;
    uint parent;
    uint padding;
} BVHNode;

typedef struct {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
} AABB;

typedef struct {
    float min_x, min_y, min_z;
    float extent_x, extent_y, extent_z;
} SceneBounds;

inline uint expand_bits(uint v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8))  & 0x0300F00F;
    v = (v | (v << 4))  & 0x030C30C3;
    v = (v | (v << 2))  & 0x09249249;
    return v;
}

inline uint morton_code_3d(float x, float y, float z) {
    x = clamp(x * 1024.0f, 0.0f, 1023.0f);
    y = clamp(y * 1024.0f, 0.0f, 1023.0f);
    z = clamp(z * 1024.0f, 0.0f, 1023.0f);
    uint xx = expand_bits((uint)x);
    uint yy = expand_bits((uint)y);
    uint zz = expand_bits((uint)z);
    return (xx << 2) | (yy << 1) | zz;
}

inline int clz_safe(uint x) {
    return (x == 0) ? 32 : clz(x);
}

inline int delta_func(__global const uint* codes, int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    uint code_i = codes[i];
    uint code_j = codes[j];
    if (code_i == code_j) return 32 + clz_safe((uint)(i ^ j));
    return clz_safe(code_i ^ code_j);
}

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
    if (tid >= num_triangles) return;
    
    float3 v0 = (float3)(v0x[tid], v0y[tid], v0z[tid]);
    float3 v1 = (float3)(v1x[tid], v1y[tid], v1z[tid]);
    float3 v2 = (float3)(v2x[tid], v2y[tid], v2z[tid]);
    
    float3 aabb_min = fmin(fmin(v0, v1), v2);
    float3 aabb_max = fmax(fmax(v0, v1), v2);
    
    leaf_bounds[tid].min_x = aabb_min.x;
    leaf_bounds[tid].min_y = aabb_min.y;
    leaf_bounds[tid].min_z = aabb_min.z;
    leaf_bounds[tid].max_x = aabb_max.x;
    leaf_bounds[tid].max_y = aabb_max.y;
    leaf_bounds[tid].max_z = aabb_max.z;
    
    float3 centroid = (v0 + v1 + v2) * (1.0f / 3.0f);
    float3 scene_min = (float3)(scene_bounds.min_x, scene_bounds.min_y, scene_bounds.min_z);
    float3 scene_extent = (float3)(scene_bounds.extent_x, scene_bounds.extent_y, scene_bounds.extent_z);
    float3 normalized = (centroid - scene_min) / fmax(scene_extent, (float3)(1e-10f));
    normalized = clamp(normalized, 0.0f, 1.0f);
    
    morton_codes[tid] = morton_code_3d(normalized.x, normalized.y, normalized.z);
    indices[tid] = tid;
}

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void build_topology(
    __global const uint* sorted_morton_codes,
    __global const uint* sorted_indices,
    __global BVHNode* nodes,
    const uint num_primitives
) {
    uint i = get_global_id(0);
    if (i >= num_primitives - 1) return;
    
    int n = (int)num_primitives;
    int delta_left = delta_func(sorted_morton_codes, i, i - 1, n);
    int delta_right = delta_func(sorted_morton_codes, i, i + 1, n);
    int d = (delta_right - delta_left >= 0) ? 1 : -1;
    int delta_min = delta_func(sorted_morton_codes, i, i - d, n);
    
    int lmax = 2;
    while (delta_func(sorted_morton_codes, i, i + lmax * d, n) > delta_min) lmax *= 2;
    
    int l = 0;
    for (int t = lmax >> 1; t >= 1; t >>= 1) {
        if (delta_func(sorted_morton_codes, i, i + (l + t) * d, n) > delta_min) l += t;
    }
    
    int j = i + l * d;
    int first = (i < j) ? i : j;
    int last = (i > j) ? i : j;
    
    int delta_node = delta_func(sorted_morton_codes, first, last, n);
    int s = 0, t_split = last - first;
    
    while (t_split > 1) {
        t_split = (t_split + 1) >> 1;
        int probe = first + s + t_split;
        if (probe < last && delta_func(sorted_morton_codes, first, probe, n) > delta_node) s += t_split;
    }
    
    int split = first + s;
    
    uint left_child = (first == split) ? ((uint)split | 0x80000000u) : (uint)split;
    uint right_child = (split + 1 == last) ? ((uint)(split + 1) | 0x80000000u) : (uint)(split + 1);
    
    nodes[i].left_child = left_child;
    nodes[i].right_child = right_child;
    nodes[i].min_x = HUGE_VALF;
    nodes[i].min_y = HUGE_VALF;
    nodes[i].min_z = HUGE_VALF;
    nodes[i].max_x = -HUGE_VALF;
    nodes[i].max_y = -HUGE_VALF;
    nodes[i].max_z = -HUGE_VALF;
    
    if ((left_child & 0x80000000u) == 0) nodes[left_child].parent = i;
    if ((right_child & 0x80000000u) == 0) nodes[right_child].parent = i;
}

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void set_leaf_parents(
    __global const BVHNode* internal_nodes,
    __global uint* leaf_parents,
    const uint num_primitives
) {
    uint i = get_global_id(0);
    if (i >= num_primitives - 1) return;
    
    uint left_child = internal_nodes[i].left_child;
    uint right_child = internal_nodes[i].right_child;
    
    if (left_child & 0x80000000u) leaf_parents[left_child & 0x7FFFFFFFu] = i;
    if (right_child & 0x80000000u) leaf_parents[right_child & 0x7FFFFFFFu] = i;
}

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void refit_bounds(
    __global BVHNode* nodes,
    __global const AABB* leaf_bounds,
    __global const uint* sorted_indices,
    __global const uint* leaf_parents,
    __global int* counters,
    const uint num_primitives
) {
    uint tid = get_global_id(0);
    if (tid >= num_primitives) return;
    
    uint orig_idx = sorted_indices[tid];
    AABB bounds = leaf_bounds[orig_idx];
    float min_x = bounds.min_x, min_y = bounds.min_y, min_z = bounds.min_z;
    float max_x = bounds.max_x, max_y = bounds.max_y, max_z = bounds.max_z;
    
    uint parent = leaf_parents[tid];
    if (num_primitives == 1) return;
    
    while (true) {
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        int old_count = atomic_inc(&counters[parent]);
        
        if (old_count == 0) return;
        
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        
        uint left_child = nodes[parent].left_child;
        uint right_child = nodes[parent].right_child;
        
        float sibling_min_x, sibling_min_y, sibling_min_z;
        float sibling_max_x, sibling_max_y, sibling_max_z;
        
        if (left_child & 0x80000000u) {
            uint leaf_idx = sorted_indices[left_child & 0x7FFFFFFFu];
            AABB leaf_aabb = leaf_bounds[leaf_idx];
            sibling_min_x = leaf_aabb.min_x; sibling_min_y = leaf_aabb.min_y; sibling_min_z = leaf_aabb.min_z;
            sibling_max_x = leaf_aabb.max_x; sibling_max_y = leaf_aabb.max_y; sibling_max_z = leaf_aabb.max_z;
        } else {
            sibling_min_x = nodes[left_child].min_x; sibling_min_y = nodes[left_child].min_y; sibling_min_z = nodes[left_child].min_z;
            sibling_max_x = nodes[left_child].max_x; sibling_max_y = nodes[left_child].max_y; sibling_max_z = nodes[left_child].max_z;
        }
        
        min_x = fmin(min_x, sibling_min_x); min_y = fmin(min_y, sibling_min_y); min_z = fmin(min_z, sibling_min_z);
        max_x = fmax(max_x, sibling_max_x); max_y = fmax(max_y, sibling_max_y); max_z = fmax(max_z, sibling_max_z);
        
        if (right_child & 0x80000000u) {
            uint leaf_idx = sorted_indices[right_child & 0x7FFFFFFFu];
            AABB leaf_aabb = leaf_bounds[leaf_idx];
            sibling_min_x = leaf_aabb.min_x; sibling_min_y = leaf_aabb.min_y; sibling_min_z = leaf_aabb.min_z;
            sibling_max_x = leaf_aabb.max_x; sibling_max_y = leaf_aabb.max_y; sibling_max_z = leaf_aabb.max_z;
        } else {
            sibling_min_x = nodes[right_child].min_x; sibling_min_y = nodes[right_child].min_y; sibling_min_z = nodes[right_child].min_z;
            sibling_max_x = nodes[right_child].max_x; sibling_max_y = nodes[right_child].max_y; sibling_max_z = nodes[right_child].max_z;
        }
        
        min_x = fmin(min_x, sibling_min_x); min_y = fmin(min_y, sibling_min_y); min_z = fmin(min_z, sibling_min_z);
        max_x = fmax(max_x, sibling_max_x); max_y = fmax(max_y, sibling_max_y); max_z = fmax(max_z, sibling_max_z);
        
        nodes[parent].min_x = min_x; nodes[parent].min_y = min_y; nodes[parent].min_z = min_z;
        nodes[parent].max_x = max_x; nodes[parent].max_y = max_y; nodes[parent].max_z = max_z;
        
        if (parent == 0) return;
        parent = nodes[parent].parent;
    }
}

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void init_counters(__global int* counters, const uint num_counters) {
    uint tid = get_global_id(0);
    if (tid < num_counters) counters[tid] = 0;
}

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
    
    float min_x = HUGE_VALF, min_y = HUGE_VALF, min_z = HUGE_VALF;
    float max_x = -HUGE_VALF, max_y = -HUGE_VALF, max_z = -HUGE_VALF;
    
    for (uint i = tid; i < num_triangles; i += get_global_size(0)) {
        float3 v0 = (float3)(v0x[i], v0y[i], v0z[i]);
        float3 v1 = (float3)(v1x[i], v1y[i], v1z[i]);
        float3 v2 = (float3)(v2x[i], v2y[i], v2z[i]);
        float3 tri_min = fmin(fmin(v0, v1), v2);
        float3 tri_max = fmax(fmax(v0, v1), v2);
        min_x = fmin(min_x, tri_min.x); min_y = fmin(min_y, tri_min.y); min_z = fmin(min_z, tri_min.z);
        max_x = fmax(max_x, tri_max.x); max_y = fmax(max_y, tri_max.y); max_z = fmax(max_z, tri_max.z);
    }
    
    local_min_x[lid] = min_x; local_min_y[lid] = min_y; local_min_z[lid] = min_z;
    local_max_x[lid] = max_x; local_max_y[lid] = max_y; local_max_z[lid] = max_z;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
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
    
    if (lid == 0) {
        partial_min_x[group_id] = local_min_x[0];
        partial_min_y[group_id] = local_min_y[0];
        partial_min_z[group_id] = local_min_z[0];
        partial_max_x[group_id] = local_max_x[0];
        partial_max_y[group_id] = local_max_y[0];
        partial_max_z[group_id] = local_max_z[0];
    }
}

typedef struct __attribute__((packed)) {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    uint childOffset;
    uint primOffset;
    ushort childCount;
    ushort primCount;
    uint axis;
} FinalBVHNode;

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void convert_to_final_layout(
    __global const BVHNode* gpu_nodes,
    __global const uint* sorted_indices,
    __global const AABB* leaf_bounds,
    __global FinalBVHNode* final_nodes,
    __global uint* prim_indices_out,
    const uint n
) {
    uint tid = get_global_id(0);
    uint num_internal = n - 1;
    
    if (tid < num_internal) {
        BVHNode src = gpu_nodes[tid];
        FinalBVHNode dst;
        dst.min_x = src.min_x; dst.min_y = src.min_y; dst.min_z = src.min_z;
        dst.max_x = src.max_x; dst.max_y = src.max_y; dst.max_z = src.max_z;
        
        uint left = src.left_child;
        uint right = src.right_child;
        
        if (left & 0x80000000u) { dst.childOffset = n - 1 + (left & 0x7FFFFFFFu); }
        else { dst.childOffset = left; }
        
        if (right & 0x80000000u) { dst.primOffset = n - 1 + (right & 0x7FFFFFFFu); }
        else { dst.primOffset = right; }
        
        dst.childCount = 2;
        dst.primCount = 0;
        dst.axis = 0;
        final_nodes[tid] = dst;
    }
    
    if (tid < n) {
        uint final_idx = n - 1 + tid;
        uint orig_prim_idx = sorted_indices[tid];
        AABB bounds = leaf_bounds[orig_prim_idx];
        
        FinalBVHNode leaf;
        leaf.min_x = bounds.min_x; leaf.min_y = bounds.min_y; leaf.min_z = bounds.min_z;
        leaf.max_x = bounds.max_x; leaf.max_y = bounds.max_y; leaf.max_z = bounds.max_z;
        leaf.childOffset = 0;
        leaf.primOffset = tid;
        leaf.childCount = 0;
        leaf.primCount = 1;
        leaf.axis = 0;
        final_nodes[final_idx] = leaf;
        prim_indices_out[tid] = orig_prim_idx;
    }
}
)";
    }

    // =========================================================================
    // Compute global work size (round up to multiple of local work size)
    // =========================================================================
    size_t roundUpToWorkGroupSize(size_t n) {
        return ((n + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE;
    }

    // =========================================================================
    // Stage 0: Compute scene bounds on GPU
    // =========================================================================
    SceneBounds computeSceneBoundsGPU(const TriangleMesh& mesh) {
        size_t n = mesh.size();

        // For small meshes, compute on CPU (faster than GPU launch overhead)
        if (n < 10000) {
            return computeSceneBoundsCPU(mesh);
        }

        // Create buffers using CL_MEM_USE_HOST_PTR for zero-copy
        cl_int err;

        compute::buffer buf_v0x(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v0x.data()));
        compute::buffer buf_v0y(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v0y.data()));
        compute::buffer buf_v0z(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v0z.data()));
        compute::buffer buf_v1x(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v1x.data()));
        compute::buffer buf_v1y(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v1y.data()));
        compute::buffer buf_v1z(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v1z.data()));
        compute::buffer buf_v2x(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v2x.data()));
        compute::buffer buf_v2y(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v2y.data()));
        compute::buffer buf_v2z(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v2z.data()));

        // Compute number of work groups
        size_t globalSize = roundUpToWorkGroupSize(n);
        size_t numGroups = globalSize / LOCAL_WORK_SIZE;

        // Partial results buffers
        compute::buffer buf_partial_min_x(context_, numGroups * sizeof(float),
                                          CL_MEM_READ_WRITE);
        compute::buffer buf_partial_min_y(context_, numGroups * sizeof(float),
                                          CL_MEM_READ_WRITE);
        compute::buffer buf_partial_min_z(context_, numGroups * sizeof(float),
                                          CL_MEM_READ_WRITE);
        compute::buffer buf_partial_max_x(context_, numGroups * sizeof(float),
                                          CL_MEM_READ_WRITE);
        compute::buffer buf_partial_max_y(context_, numGroups * sizeof(float),
                                          CL_MEM_READ_WRITE);
        compute::buffer buf_partial_max_z(context_, numGroups * sizeof(float),
                                          CL_MEM_READ_WRITE);

        // Set kernel arguments
        kernel_scene_bounds_reduce_.set_arg(0, buf_v0x);
        kernel_scene_bounds_reduce_.set_arg(1, buf_v0y);
        kernel_scene_bounds_reduce_.set_arg(2, buf_v0z);
        kernel_scene_bounds_reduce_.set_arg(3, buf_v1x);
        kernel_scene_bounds_reduce_.set_arg(4, buf_v1y);
        kernel_scene_bounds_reduce_.set_arg(5, buf_v1z);
        kernel_scene_bounds_reduce_.set_arg(6, buf_v2x);
        kernel_scene_bounds_reduce_.set_arg(7, buf_v2y);
        kernel_scene_bounds_reduce_.set_arg(8, buf_v2z);
        kernel_scene_bounds_reduce_.set_arg(9, buf_partial_min_x);
        kernel_scene_bounds_reduce_.set_arg(10, buf_partial_min_y);
        kernel_scene_bounds_reduce_.set_arg(11, buf_partial_min_z);
        kernel_scene_bounds_reduce_.set_arg(12, buf_partial_max_x);
        kernel_scene_bounds_reduce_.set_arg(13, buf_partial_max_y);
        kernel_scene_bounds_reduce_.set_arg(14, buf_partial_max_z);
        kernel_scene_bounds_reduce_.set_arg(
            15, compute::local_buffer<float>(LOCAL_WORK_SIZE));
        kernel_scene_bounds_reduce_.set_arg(
            16, compute::local_buffer<float>(LOCAL_WORK_SIZE));
        kernel_scene_bounds_reduce_.set_arg(
            17, compute::local_buffer<float>(LOCAL_WORK_SIZE));
        kernel_scene_bounds_reduce_.set_arg(
            18, compute::local_buffer<float>(LOCAL_WORK_SIZE));
        kernel_scene_bounds_reduce_.set_arg(
            19, compute::local_buffer<float>(LOCAL_WORK_SIZE));
        kernel_scene_bounds_reduce_.set_arg(
            20, compute::local_buffer<float>(LOCAL_WORK_SIZE));
        kernel_scene_bounds_reduce_.set_arg(21, static_cast<cl_uint>(n));

        // Execute kernel
        queue_.enqueue_1d_range_kernel(kernel_scene_bounds_reduce_, 0,
                                       globalSize, LOCAL_WORK_SIZE);

        // Read back partial results using enqueue_read_buffer
        std::vector<float> partial_min_x(numGroups), partial_min_y(numGroups),
            partial_min_z(numGroups);
        std::vector<float> partial_max_x(numGroups), partial_max_y(numGroups),
            partial_max_z(numGroups);

        queue_.enqueue_read_buffer(buf_partial_min_x, 0,
                                   numGroups * sizeof(float), partial_min_x.data());
        queue_.enqueue_read_buffer(buf_partial_min_y, 0,
                                   numGroups * sizeof(float), partial_min_y.data());
        queue_.enqueue_read_buffer(buf_partial_min_z, 0,
                                   numGroups * sizeof(float), partial_min_z.data());
        queue_.enqueue_read_buffer(buf_partial_max_x, 0,
                                   numGroups * sizeof(float), partial_max_x.data());
        queue_.enqueue_read_buffer(buf_partial_max_y, 0,
                                   numGroups * sizeof(float), partial_max_y.data());
        queue_.enqueue_read_buffer(buf_partial_max_z, 0,
                                   numGroups * sizeof(float), partial_max_z.data());
        queue_.finish();

        // Final reduction on CPU
        SceneBounds bounds;
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        float max_z = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < numGroups; ++i) {
            min_x = std::min(min_x, partial_min_x[i]);
            min_y = std::min(min_y, partial_min_y[i]);
            min_z = std::min(min_z, partial_min_z[i]);
            max_x = std::max(max_x, partial_max_x[i]);
            max_y = std::max(max_y, partial_max_y[i]);
            max_z = std::max(max_z, partial_max_z[i]);
        }

        bounds.min_x = min_x;
        bounds.min_y = min_y;
        bounds.min_z = min_z;
        bounds.extent_x = max_x - min_x;
        bounds.extent_y = max_y - min_y;
        bounds.extent_z = max_z - min_z;

        return bounds;
    }

    SceneBounds computeSceneBoundsCPU(const TriangleMesh& mesh) {
        AABB sceneBounds;
        size_t n = mesh.size();

        for (size_t i = 0; i < n; ++i) {
            sceneBounds.expand(mesh.getVertex0(i));
            sceneBounds.expand(mesh.getVertex1(i));
            sceneBounds.expand(mesh.getVertex2(i));
        }

        Vec3 extent = sceneBounds.extent();
        return SceneBounds{sceneBounds.min.x, sceneBounds.min.y,
                           sceneBounds.min.z, extent.x,
                           extent.y,          extent.z};
    }

    // =========================================================================
    // Stage 1: Compute Morton codes on GPU
    // =========================================================================
    std::tuple<compute::vector<cl_uint>, compute::vector<cl_uint>,
               compute::vector<GPUAABB>>
    computeMortonCodesGPU(const TriangleMesh& mesh,
                          const SceneBounds& sceneBounds) {
        size_t n = mesh.size();

        // Create input buffers with CL_MEM_USE_HOST_PTR (zero-copy for APU)
        compute::buffer buf_v0x(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v0x.data()));
        compute::buffer buf_v0y(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v0y.data()));
        compute::buffer buf_v0z(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v0z.data()));
        compute::buffer buf_v1x(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v1x.data()));
        compute::buffer buf_v1y(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v1y.data()));
        compute::buffer buf_v1z(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v1z.data()));
        compute::buffer buf_v2x(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v2x.data()));
        compute::buffer buf_v2y(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v2y.data()));
        compute::buffer buf_v2z(context_, n * sizeof(float),
                                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                const_cast<float*>(mesh.v2z.data()));

        // Output buffers
        compute::vector<cl_uint> mortonCodes(n, context_);
        compute::vector<cl_uint> indices(n, context_);
        compute::vector<GPUAABB> leafBounds(n, context_);

        // Set kernel arguments
        kernel_morton_codes_soa_.set_arg(0, buf_v0x);
        kernel_morton_codes_soa_.set_arg(1, buf_v0y);
        kernel_morton_codes_soa_.set_arg(2, buf_v0z);
        kernel_morton_codes_soa_.set_arg(3, buf_v1x);
        kernel_morton_codes_soa_.set_arg(4, buf_v1y);
        kernel_morton_codes_soa_.set_arg(5, buf_v1z);
        kernel_morton_codes_soa_.set_arg(6, buf_v2x);
        kernel_morton_codes_soa_.set_arg(7, buf_v2y);
        kernel_morton_codes_soa_.set_arg(8, buf_v2z);
        kernel_morton_codes_soa_.set_arg(9, mortonCodes.get_buffer());
        kernel_morton_codes_soa_.set_arg(10, indices.get_buffer());
        kernel_morton_codes_soa_.set_arg(11, leafBounds.get_buffer());
        kernel_morton_codes_soa_.set_arg(12, sizeof(SceneBounds), &sceneBounds);
        kernel_morton_codes_soa_.set_arg(13, static_cast<cl_uint>(n));

        // Execute
        size_t globalSize = roundUpToWorkGroupSize(n);
        queue_.enqueue_1d_range_kernel(kernel_morton_codes_soa_, 0, globalSize,
                                       LOCAL_WORK_SIZE);
        queue_.finish();

        return {std::move(mortonCodes), std::move(indices),
                std::move(leafBounds)};
    }

    // =========================================================================
    // Stage 2: Sort by Morton code using Boost.Compute
    // =========================================================================
    void sortByMortonCode(compute::vector<cl_uint>& mortonCodes,
                          compute::vector<cl_uint>& indices) {
        // Use Boost.Compute's radix sort (optimized for GPU)
        compute::sort_by_key(mortonCodes.begin(), mortonCodes.end(),
                             indices.begin(), queue_);
        queue_.finish();
    }

    // =========================================================================
    // Stage 3: Build BVH topology using Karras algorithm
    // =========================================================================
    std::pair<compute::vector<GPUBVHNode>, compute::vector<cl_uint>>
    buildTopologyGPU(const compute::vector<cl_uint>& mortonCodes,
                     const compute::vector<cl_uint>& indices, size_t n) {
        size_t numInternal = n - 1;

        // Allocate internal nodes
        compute::vector<GPUBVHNode> nodes(numInternal, context_);
        compute::vector<cl_uint> leafParents(n, context_);

        // Build topology
        kernel_build_topology_.set_arg(0, mortonCodes.get_buffer());
        kernel_build_topology_.set_arg(1, indices.get_buffer());
        kernel_build_topology_.set_arg(2, nodes.get_buffer());
        kernel_build_topology_.set_arg(3, static_cast<cl_uint>(n));

        size_t globalSize = roundUpToWorkGroupSize(numInternal);
        queue_.enqueue_1d_range_kernel(kernel_build_topology_, 0, globalSize,
                                       LOCAL_WORK_SIZE);

        // Set leaf parent pointers
        kernel_set_leaf_parents_.set_arg(0, nodes.get_buffer());
        kernel_set_leaf_parents_.set_arg(1, leafParents.get_buffer());
        kernel_set_leaf_parents_.set_arg(2, static_cast<cl_uint>(n));

        queue_.enqueue_1d_range_kernel(kernel_set_leaf_parents_, 0, globalSize,
                                       LOCAL_WORK_SIZE);
        queue_.finish();

        return {std::move(nodes), std::move(leafParents)};
    }

    // =========================================================================
    // Stage 4: Refit bounds bottom-up
    // =========================================================================
    void refitBoundsGPU(compute::vector<GPUBVHNode>& nodes,
                        const compute::vector<GPUAABB>& leafBounds,
                        const compute::vector<cl_uint>& indices,
                        const compute::vector<cl_uint>& leafParents, size_t n) {
        size_t numInternal = n - 1;

        // Allocate and initialize atomic counters
        compute::vector<cl_int> counters(numInternal, context_);

        kernel_init_counters_.set_arg(0, counters.get_buffer());
        kernel_init_counters_.set_arg(1, static_cast<cl_uint>(numInternal));

        size_t counterGlobalSize = roundUpToWorkGroupSize(numInternal);
        queue_.enqueue_1d_range_kernel(kernel_init_counters_, 0,
                                       counterGlobalSize, LOCAL_WORK_SIZE);

        // Refit bounds
        kernel_refit_bounds_.set_arg(0, nodes.get_buffer());
        kernel_refit_bounds_.set_arg(1, leafBounds.get_buffer());
        kernel_refit_bounds_.set_arg(2, indices.get_buffer());
        kernel_refit_bounds_.set_arg(3, leafParents.get_buffer());
        kernel_refit_bounds_.set_arg(4, counters.get_buffer());
        kernel_refit_bounds_.set_arg(5, static_cast<cl_uint>(n));

        size_t globalSize = roundUpToWorkGroupSize(n);
        queue_.enqueue_1d_range_kernel(kernel_refit_bounds_, 0, globalSize,
                                       LOCAL_WORK_SIZE);
        queue_.finish();
    }

    // =========================================================================
    // Convert GPU BVH format to host BVH format (GPU-accelerated)
    // =========================================================================
    // Uses GPU kernel to convert layout and zero-copy for efficient transfer
    // =========================================================================
    BVHResult convertToHostFormat(const compute::vector<GPUBVHNode>& gpuNodes,
                                  const compute::vector<cl_uint>& indices,
                                  const compute::vector<GPUAABB>& leafBounds,
                                  size_t n) {
        BVHResult result;
        result.branchingFactor = 2;

        size_t totalNodes = 2 * n - 1;

        // Allocate host-pinned memory for zero-copy output
        std::vector<FinalBVHNode> hostFinalNodes(totalNodes);
        std::vector<uint32_t> hostPrimIndices(n);

        // Create output buffers with USE_HOST_PTR for zero-copy on APU
        compute::buffer buf_final_nodes(
            context_,
            totalNodes * sizeof(FinalBVHNode),
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            hostFinalNodes.data());

        compute::buffer buf_prim_indices_out(
            context_,
            n * sizeof(uint32_t),
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            hostPrimIndices.data());

        // Set kernel arguments
        kernel_convert_layout_.set_arg(0, gpuNodes.get_buffer());
        kernel_convert_layout_.set_arg(1, indices.get_buffer());
        kernel_convert_layout_.set_arg(2, leafBounds.get_buffer());
        kernel_convert_layout_.set_arg(3, buf_final_nodes);
        kernel_convert_layout_.set_arg(4, buf_prim_indices_out);
        kernel_convert_layout_.set_arg(5, static_cast<cl_uint>(n));

        // Launch kernel - needs to cover max(n-1, n) = n work items
        size_t globalSize = roundUpToWorkGroupSize(n);
        queue_.enqueue_1d_range_kernel(kernel_convert_layout_, 0, globalSize,
                                       LOCAL_WORK_SIZE);

        // Map output buffers back to host (zero-copy on APU)
        queue_.enqueue_map_buffer(buf_final_nodes, CL_MAP_READ,
                                  0, totalNodes * sizeof(FinalBVHNode));
        queue_.enqueue_map_buffer(buf_prim_indices_out, CL_MAP_READ,
                                  0, n * sizeof(uint32_t));
        queue_.finish();

        // Convert FinalBVHNode to host BVHNode format
        result.nodes.resize(totalNodes);
        for (size_t i = 0; i < totalNodes; ++i) {
            const FinalBVHNode& src = hostFinalNodes[i];
            BVHNode& dst = result.nodes[i];

            dst.bounds.min = Vec3(src.min_x, src.min_y, src.min_z);
            dst.bounds.max = Vec3(src.max_x, src.max_y, src.max_z);
            dst.childOffset = src.childOffset;
            dst.primOffset = src.primOffset;
            dst.childCount = src.childCount;
            dst.primCount = src.primCount;
            dst.axis = static_cast<int>(src.axis);
        }

        // Copy primitive indices
        result.primIndices.resize(n);
        for (size_t i = 0; i < n; ++i) {
            result.primIndices[i] = hostPrimIndices[i];
        }

        return result;
    }
};
