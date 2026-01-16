#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <iomanip>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#include <cub/cub.cuh>

#include "../src/mesh/obj_loader.hpp"
#include "../src/bvh/bvh_export.hpp"

struct __align__(16) float3_cw {
    float x, y, z;
    __host__ __device__ float3_cw() : x(0), y(0), z(0) {}
    __host__ __device__ float3_cw(float a, float b, float c) : x(a), y(b), z(c) {}
    __host__ __device__ float3_cw operator+(const float3_cw& b) const { return float3_cw(x+b.x, y+b.y, z+b.z); }
    __host__ __device__ float3_cw operator-(const float3_cw& b) const { return float3_cw(x-b.x, y-b.y, z-b.z); }
    __host__ __device__ float3_cw operator*(float s) const { return float3_cw(x*s, y*s, z*s); }
};

struct __align__(16) AABB_cw {
    float3_cw min;
    float3_cw max;
    __host__ __device__ AABB_cw() {
        min = float3_cw(FLT_MAX, FLT_MAX, FLT_MAX);
        max = float3_cw(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
};

struct LBVHNode {
    AABB_cw bbox;
    uint32_t leftChild;  
    uint32_t rightChild;
    uint32_t parent;     
};

struct TrianglesSoADevice {
    float *v0x, *v0y, *v0z;
    float *v1x, *v1y, *v1z;
    float *v2x, *v2y, *v2z;
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__device__ __forceinline__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    uint32_t xx = expandBits((uint32_t)x);
    uint32_t yy = expandBits((uint32_t)y);
    uint32_t zz = expandBits((uint32_t)z);
    return xx * 4 + yy * 2 + zz;
}


__device__ __forceinline__ int clz_custom(uint32_t x) {
    return x == 0 ? 32 : __clz(x);
}

__device__ int delta(const uint32_t* sortedMortonCodes, const uint32_t* sortedIndices, int numObjects, int i, int j) {
    if (j < 0 || j >= numObjects) return -1;
    
    uint32_t codeI = sortedMortonCodes[i];
    uint32_t codeJ = sortedMortonCodes[j];
    
    if (codeI == codeJ) {
        uint32_t idxI = sortedIndices[i];
        uint32_t idxJ = sortedIndices[j];
        return 32 + clz_custom(idxI ^ idxJ);
    }
    return clz_custom(codeI ^ codeJ);
}


__global__ void kComputeBoundsAndCentroids(
    TrianglesSoADevice tris, 
    int n, 
    AABB_cw* triBBoxes, 
    float3_cw* centroids, 
    AABB_cw* globalBoundsBlock) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3_cw v0(tris.v0x[i], tris.v0y[i], tris.v0z[i]);
    float3_cw v1(tris.v1x[i], tris.v1y[i], tris.v1z[i]);
    float3_cw v2(tris.v2x[i], tris.v2y[i], tris.v2z[i]);

    float3_cw minVal, maxVal;
    minVal.x = fminf(v0.x, fminf(v1.x, v2.x));
    minVal.y = fminf(v0.y, fminf(v1.y, v2.y));
    minVal.z = fminf(v0.z, fminf(v1.z, v2.z));
    maxVal.x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    maxVal.y = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    maxVal.z = fmaxf(v0.z, fmaxf(v1.z, v2.z));

    triBBoxes[i].min = minVal;
    triBBoxes[i].max = maxVal;

    centroids[i] = (minVal + maxVal) * 0.5f;
}

__global__ void kComputeMortonCodes(
    const float3_cw* centroids, 
    int n, 
    AABB_cw sceneBounds, 
    uint32_t* mortonCodes, 
    uint32_t* indices) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3_cw c = centroids[i];
    float3_cw minB = sceneBounds.min;
    float3_cw extents = sceneBounds.max - sceneBounds.min;

    // Normalize centroid to [0,1]
    float x = (c.x - minB.x) / extents.x;
    float y = (c.y - minB.y) / extents.y;
    float z = (c.z - minB.z) / extents.z;

    mortonCodes[i] = morton3D(x, y, z);
    indices[i] = i;
}

__global__ void kBuildInternalNodes(
    const uint32_t* sortedMortonCodes,
    const uint32_t* sortedIndices,
    LBVHNode* nodes,
    int numObjects)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects - 1) return; 

    int d = (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + 1) - 
             delta(sortedMortonCodes, sortedIndices, numObjects, i, i - 1)) >= 0 ? 1 : -1;

    int min_delta = delta(sortedMortonCodes, sortedIndices, numObjects, i, i - d);
    int l_max = 2;
    while (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + l_max * d) > min_delta) {
        l_max *= 2;
    }

    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + (l + t) * d) > min_delta) {
            l += t;
        }
    }
    int j = i + l * d;

    int delta_node = delta(sortedMortonCodes, sortedIndices, numObjects, i, j);
    int s = 0;
    
    int first, last;
    if (d > 0) { first = i; last = j; }
    else       { first = j; last = i; }

    int len = last - first;
    int t = len;
    
    do {
        t = (t + 1) >> 1;
        if (delta(sortedMortonCodes, sortedIndices, numObjects, first, first + s + t) > delta_node) {
            s += t;
        }
    } while (t > 1);
    
    int split = first + s;

    uint32_t leftIdx, rightIdx;
    
    if (split == first) leftIdx = (numObjects - 1) + split; 
    else                leftIdx = split;                   

    if (split + 1 == last) rightIdx = (numObjects - 1) + split + 1; 
    else                   rightIdx = split + 1;                   

    nodes[i].leftChild = leftIdx;
    nodes[i].rightChild = rightIdx;
    
    
    nodes[leftIdx].parent = i;
    nodes[rightIdx].parent = i;
}


__global__ void kInitLeafNodes(
    LBVHNode* nodes, 
    const AABB_cw* triBBoxes, 
    const uint32_t* sortedIndices, 
    int numObjects) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    int leafIdx = (numObjects - 1) + i; 
    uint32_t originalIndex = sortedIndices[i];

    
    nodes[leafIdx].bbox = triBBoxes[originalIndex];
    
    nodes[leafIdx].leftChild = originalIndex | 0x80000000; 
    nodes[leafIdx].rightChild = 0xFFFFFFFF;
}


__global__ void kRefitHierarchy(
    LBVHNode* nodes, 
    int* atomicCounters, 
    int numObjects) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numObjects) return;

    
    uint32_t idx = (numObjects - 1) + i;
    
    
    while (idx != 0) { 
        uint32_t parent = nodes[idx].parent;
        
       int oldVal = atomicAdd(&atomicCounters[parent], 1);
        
        if (oldVal == 0) {
           return;
        }

        
        uint32_t left = nodes[parent].leftChild;
        uint32_t right = nodes[parent].rightChild;

        AABB_cw leftBox = nodes[left].bbox;
        AABB_cw rightBox = nodes[right].bbox;

        AABB_cw unionBox;
        unionBox.min.x = fminf(leftBox.min.x, rightBox.min.x);
        unionBox.min.y = fminf(leftBox.min.y, rightBox.min.y);
        unionBox.min.z = fminf(leftBox.min.z, rightBox.min.z);
        
        unionBox.max.x = fmaxf(leftBox.max.x, rightBox.max.x);
        unionBox.max.y = fmaxf(leftBox.max.y, rightBox.max.y);
        unionBox.max.z = fmaxf(leftBox.max.z, rightBox.max.z);

        
        nodes[parent].bbox = unionBox;

        
        idx = parent;
    }
}

struct AABBReduce {
    __host__ __device__ AABB_cw operator()(const AABB_cw& a, const AABB_cw& b) {
        AABB_cw res;
        res.min.x = fminf(a.min.x, b.min.x);
        res.min.y = fminf(a.min.y, b.min.y);
        res.min.z = fminf(a.min.z, b.min.z);
        res.max.x = fmaxf(a.max.x, b.max.x);
        res.max.y = fmaxf(a.max.y, b.max.y);
        res.max.z = fmaxf(a.max.z, b.max.z);
        return res;
    }
};


class LBVHBuilderCUDA {
private:
    thrust::device_vector<float> d_v0x, d_v0y, d_v0z;
    thrust::device_vector<float> d_v1x, d_v1y, d_v1z;
    thrust::device_vector<float> d_v2x, d_v2y, d_v2z;
    
    thrust::device_vector<AABB_cw> d_triBBoxes;
    thrust::device_vector<float3_cw> d_centroids;
    thrust::device_vector<uint32_t> d_mortonCodes;
    thrust::device_vector<uint32_t> d_indices;
    
    thrust::device_vector<uint32_t> d_mortonCodes_alt; 
    thrust::device_vector<uint32_t> d_indices_alt;     
    thrust::device_vector<uint8_t>  d_cub_temp_storage; 
    
    thrust::device_vector<LBVHNode> d_nodes;
    thrust::device_vector<int> d_atomicFlags;

    TrianglesSoADevice getDevicePtrs() {
        return {
            thrust::raw_pointer_cast(d_v0x.data()), thrust::raw_pointer_cast(d_v0y.data()), thrust::raw_pointer_cast(d_v0z.data()),
            thrust::raw_pointer_cast(d_v1x.data()), thrust::raw_pointer_cast(d_v1y.data()), thrust::raw_pointer_cast(d_v1z.data()),
            thrust::raw_pointer_cast(d_v2x.data()), thrust::raw_pointer_cast(d_v2y.data()), thrust::raw_pointer_cast(d_v2z.data())
        };
    }

public:
    void build(const TriangleMesh& tris) {
        int n = tris.size();
        if (n == 0) return;

        d_v0x = tris.v0x; d_v0y = tris.v0y; d_v0z = tris.v0z;
        d_v1x = tris.v1x; d_v1y = tris.v1y; d_v1z = tris.v1z;
        d_v2x = tris.v2x; d_v2y = tris.v2y; d_v2z = tris.v2z;

        d_triBBoxes.resize(n);
        d_centroids.resize(n);
        d_mortonCodes.resize(n);
        d_indices.resize(n);
        d_nodes.resize(2 * n - 1);
        d_atomicFlags.resize(2 * n - 1);
        d_mortonCodes_alt.resize(n);
        d_indices_alt.resize(n);
        
        thrust::fill(d_atomicFlags.begin(), d_atomicFlags.end(), 0);

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        kComputeBoundsAndCentroids<<<gridSize, blockSize>>>(getDevicePtrs(), n, 
            thrust::raw_pointer_cast(d_triBBoxes.data()), 
            thrust::raw_pointer_cast(d_centroids.data()), 
            nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        
        AABB_cw init; 
        init.min = float3_cw(FLT_MAX, FLT_MAX, FLT_MAX);
        init.max = float3_cw(-FLT_MAX, -FLT_MAX, -FLT_MAX);


        AABB_cw sceneBounds = thrust::reduce(d_triBBoxes.begin(), d_triBBoxes.end(), init, AABBReduce());

        kComputeMortonCodes<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_centroids.data()), 
            n, sceneBounds, 
            thrust::raw_pointer_cast(d_mortonCodes.data()), 
            thrust::raw_pointer_cast(d_indices.data()));
        CUDA_CHECK(cudaDeviceSynchronize());

        // 1. Setup pointers for CUB
        uint32_t* d_keys_in   = thrust::raw_pointer_cast(d_mortonCodes.data());
        uint32_t* d_keys_out  = thrust::raw_pointer_cast(d_mortonCodes_alt.data());
        uint32_t* d_values_in = thrust::raw_pointer_cast(d_indices.data());
        uint32_t* d_values_out= thrust::raw_pointer_cast(d_indices_alt.data());

        // 2. Determine temporary storage size needed
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        // First call just asks "How much memory do you need?"
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            d_values_in, d_values_out,
            n
        );

        // 3. Allocate that temp storage (using our cached vector)
        if (d_cub_temp_storage.size() < temp_storage_bytes) {
            d_cub_temp_storage.resize(temp_storage_bytes);
        }
        d_temp_storage = thrust::raw_pointer_cast(d_cub_temp_storage.data());

        // 4. Run the actual sort
        // Notice: bit_start=0, bit_end=30 (since Morton codes are 30 bits)
        // Restricting the bit range makes it even faster!
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            d_values_in, d_values_out,
            n, 0, 30 
        );

        // 5. Swap the vectors
        // The sorted data is now in the "_alt" vectors.
        // We swap them so d_mortonCodes and d_indices hold the valid sorted data
        // for the rest of your pipeline.
        d_mortonCodes.swap(d_mortonCodes_alt);
        d_indices.swap(d_indices_alt);

        kBuildInternalNodes<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_mortonCodes.data()),
            thrust::raw_pointer_cast(d_indices.data()),
            thrust::raw_pointer_cast(d_nodes.data()),
            n);
        CUDA_CHECK(cudaDeviceSynchronize());

        kInitLeafNodes<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_triBBoxes.data()),
            thrust::raw_pointer_cast(d_indices.data()),
            n);
        CUDA_CHECK(cudaDeviceSynchronize());

        kRefitHierarchy<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_atomicFlags.data()),
            n);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::cout << "BVH Build Complete on GPU. Nodes: " << (2*n-1) << std::endl;
    }

    void verify() {
        if(d_nodes.empty()) return;
        LBVHNode root = d_nodes[0];
        std::cout << "Root Bounds: " 
                  << "[" << root.bbox.min.x << ", " << root.bbox.min.y << ", " << root.bbox.min.z << "] - "
                  << "[" << root.bbox.max.x << ", " << root.bbox.max.y << ", " << root.bbox.max.z << "]\n";
    }

    std::vector<BVHNode> getNodes() const {
        std::vector<LBVHNode> hostNodes(d_nodes.begin(), d_nodes.end());
        std::vector<BVHNode> result;
        result.reserve(hostNodes.size());
        
        for (size_t i = 0; i < hostNodes.size(); ++i) {
            const auto& node = hostNodes[i];
            BVHNode bvhNode;
            bvhNode.bounds.min = Vec3(node.bbox.min.x, node.bbox.min.y, node.bbox.min.z);
            bvhNode.bounds.max = Vec3(node.bbox.max.x, node.bbox.max.y, node.bbox.max.z);
            
            if (node.leftChild & 0x80000000) {
                bvhNode.childCount = 0;  
                bvhNode.primOffset = node.leftChild & 0x7FFFFFFF;  
                bvhNode.primCount = 1;
            } else {
                bvhNode.childCount = 2;  
                bvhNode.childOffset = node.leftChild;  
                bvhNode.primCount = 0;
            }
            bvhNode.axis = 0;
            result.push_back(bvhNode);
        }
        return result;
    }
};


void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  -i, --input <file.obj>    Input OBJ file to load\n"
              << "  -o, --output <file.obj>   Output OBJ file for BVH export\n"
              << "  -n, --triangles <count>   Number of random triangles (default: 10000000)\n"
              << "  -l, --leaves-only         Export only leaf node bounding boxes\n"
              << "  -h, --help                Show this help message\n"
              << "\nExamples:\n"
              << "  " << programName << " -i model.obj -o bvh.obj\n"
              << "  " << programName << " -n 1000000 -o bvh.obj\n"
              << "  " << programName << " -i model.obj -o bvh.obj -l\n";
}


TriangleMesh generateRandomTriangles(int n) {
    TriangleMesh mesh;
    mesh.reserve(n);
    
    for (int i = 0; i < n; ++i) {
        float x = (rand() % 1000) / 10.0f;
        float y = (rand() % 1000) / 10.0f;
        float z = (rand() % 1000) / 10.0f;
        
        mesh.addTriangle(
            Vec3(x, y, z),
            Vec3(x + 1, y, z),
            Vec3(x, y + 1, z)
        );
    }
    
    return mesh;
}


int main(int argc, char* argv[]) {
    std::string inputFile;
    std::string outputFile;
    int numTriangles = 10000000;  
    bool leavesOnly = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                inputFile = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a file path\n";
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                outputFile = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a file path\n";
                return 1;
            }
        } else if (arg == "-n" || arg == "--triangles") {
            if (i + 1 < argc) {
                numTriangles = std::atoi(argv[++i]);
                if (numTriangles <= 0) {
                    std::cerr << "Error: Invalid triangle count\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: " << arg << " requires a number\n";
                return 1;
            }
        } else if (arg == "-l" || arg == "--leaves-only") {
            leavesOnly = true;
        } else {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    TriangleMesh mesh;
    
    if (!inputFile.empty()) {
        std::cout << "Loading OBJ file: " << inputFile << std::endl;
        try {
            mesh = loadOBJ(inputFile);
            std::cout << "Loaded " << mesh.size() << " triangles from OBJ file\n";
        } catch (const std::exception& e) {
            std::cerr << "Error loading OBJ: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "Generating " << numTriangles << " random triangles..." << std::endl;
        mesh = generateRandomTriangles(numTriangles);
    }

    if (mesh.size() == 0) {
        std::cerr << "Error: No triangles to process\n";
        return 1;
    }

    LBVHBuilderCUDA builder;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    builder.build(mesh);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Build Time: " << milliseconds << " ms" << std::endl;
    builder.verify();

    if (!outputFile.empty()) {
        std::cout << "Exporting BVH to: " << outputFile << std::endl;
        try {
            std::vector<BVHNode> nodes = builder.getNodes();
            exportBVHToOBJ(outputFile, nodes, leavesOnly);
            std::cout << "Exported " << nodes.size() << " BVH nodes"
                      << (leavesOnly ? " (leaves only)" : "") << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error exporting BVH: " << e.what() << std::endl;
            return 1;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}