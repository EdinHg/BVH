#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

// --- GEOMETRY & STRUCTS ---

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

using BVHNode = LBVHNode;

struct TriangleMesh {
    std::vector<float> v0x, v0y, v0z;
    std::vector<float> v1x, v1y, v1z;
    std::vector<float> v2x, v2y, v2z;

    size_t size() const { return v0x.size(); }
    
    void resize(size_t n) {
        v0x.resize(n); v0y.resize(n); v0z.resize(n);
        v1x.resize(n); v1y.resize(n); v1z.resize(n);
        v2x.resize(n); v2y.resize(n); v2z.resize(n);
    }
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

// --- DEVICE FUNCTIONS ---

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

// --- KERNELS ---

__global__ void kComputeBoundsAndCentroids(TrianglesSoADevice tris, int n, AABB_cw* triBBoxes, float3_cw* centroids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    float3_cw v0(tris.v0x[i], tris.v0y[i], tris.v0z[i]);
    float3_cw v1(tris.v1x[i], tris.v1y[i], tris.v1z[i]);
    float3_cw v2(tris.v2x[i], tris.v2y[i], tris.v2z[i]);

    float3_cw minVal, maxVal;

    minVal.x = fminf(v0.x, fminf(v1.x, v2.x)); minVal.y = fminf(v0.y, fminf(v1.y, v2.y)); minVal.z = fminf(v0.z, fminf(v1.z, v2.z));
    maxVal.x = fmaxf(v0.x, fmaxf(v1.x, v2.x)); maxVal.y = fmaxf(v0.y, fmaxf(v1.y, v2.y)); maxVal.z = fmaxf(v0.z, fmaxf(v1.z, v2.z));

    triBBoxes[i].min = minVal; triBBoxes[i].max = maxVal;

    centroids[i] = (minVal + maxVal) * 0.5f;
}

__global__ void kComputeMortonCodes(const float3_cw* centroids, int n, AABB_cw sceneBounds, uint32_t* mortonCodes, uint32_t* indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    float3_cw c = centroids[i];
    float3_cw minB = sceneBounds.min;
    float3_cw extents = sceneBounds.max - sceneBounds.min;

    float x = (c.x - minB.x) / extents.x;
    float y = (c.y - minB.y) / extents.y;
    float z = (c.z - minB.z) / extents.z;

    mortonCodes[i] = morton3D(x, y, z);
    indices[i] = i;
}

__global__ void kBuildInternalNodes(const uint32_t* sortedMortonCodes, const uint32_t* sortedIndices, LBVHNode* nodes, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numObjects - 1) return;

    int d = 
        (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + 1) - 
         delta(sortedMortonCodes, sortedIndices, numObjects, i, i - 1)) >= 0 ? 1 : -1;

    int min_delta = delta(sortedMortonCodes, sortedIndices, numObjects, i, i - d);
    int l_max = 2;

    while (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + l_max * d) > min_delta) l_max *= 2;

    int l = 0;

    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta(sortedMortonCodes, sortedIndices, numObjects, i, i + (l + t) * d) > min_delta) l += t;
    }

    int j = i + l * d;
    int delta_node = delta(sortedMortonCodes, sortedIndices, numObjects, i, j);

    int s = 0;

    int first = (d > 0) ? i : j;
    int last = (d > 0) ? j : i;
    int len = last - first;
    int t = len;

    do {
        t = (t + 1) >> 1;
        if (delta(sortedMortonCodes, sortedIndices, numObjects, first, first + s + t) > delta_node) s += t;
    } while (t > 1);

    int split = first + s;

    uint32_t leftIdx = (split == first) ? (numObjects - 1) + split : split;
    uint32_t rightIdx = (split + 1 == last) ? (numObjects - 1) + split + 1 : split + 1;

    nodes[i].leftChild = leftIdx;
    nodes[i].rightChild = rightIdx;
    nodes[leftIdx].parent = i;
    nodes[rightIdx].parent = i;
}

__global__ void kInitLeafNodes(LBVHNode* nodes, const AABB_cw* triBBoxes, const uint32_t* sortedIndices, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numObjects) return;

    int leafIdx = (numObjects - 1) + i;
    uint32_t originalIndex = sortedIndices[i];

    nodes[leafIdx].bbox = triBBoxes[originalIndex];
    nodes[leafIdx].leftChild = originalIndex | 0x80000000;
    nodes[leafIdx].rightChild = 0xFFFFFFFF;
}

__global__ void kRefitHierarchy(LBVHNode* nodes, int* atomicCounters, int numObjects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numObjects) return;

    uint32_t idx = (numObjects - 1) + i;

    while (idx != 0) {
        uint32_t parent = nodes[idx].parent;

        int oldVal = atomicAdd(&atomicCounters[parent], 1);

        if (oldVal == 0) return;

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
        res.min.x = fminf(a.min.x, b.min.x); res.min.y = fminf(a.min.y, b.min.y); res.min.z = fminf(a.min.z, b.min.z);
        res.max.x = fmaxf(a.max.x, b.max.x); res.max.y = fmaxf(a.max.y, b.max.y); res.max.z = fmaxf(a.max.z, b.max.z);
        return res;
    }
};

// --- BUILDER CLASS ---

class LBVHBuilderCUDA {
private:
    // Device Vectors
    thrust::device_vector<float> d_v0x, d_v0y, d_v0z;
    thrust::device_vector<float> d_v1x, d_v1y, d_v1z;
    thrust::device_vector<float> d_v2x, d_v2y, d_v2z;
    thrust::device_vector<AABB_cw> d_triBBoxes;
    thrust::device_vector<float3_cw> d_centroids;
    thrust::device_vector<uint32_t> d_mortonCodes;
    thrust::device_vector<uint32_t> d_indices;
    thrust::device_vector<LBVHNode> d_nodes;
    thrust::device_vector<int> d_atomicFlags;

    // Profiling Events
    cudaEvent_t start, e_centroids, e_morton, e_sort, e_topology, stop;

    TrianglesSoADevice getDevicePtrs() {
        return {
            thrust::raw_pointer_cast(d_v0x.data()), thrust::raw_pointer_cast(d_v0y.data()), thrust::raw_pointer_cast(d_v0z.data()),
            thrust::raw_pointer_cast(d_v1x.data()), thrust::raw_pointer_cast(d_v1y.data()), thrust::raw_pointer_cast(d_v1z.data()),
            thrust::raw_pointer_cast(d_v2x.data()), thrust::raw_pointer_cast(d_v2y.data()), thrust::raw_pointer_cast(d_v2z.data())
        };
    }

public:
    LBVHBuilderCUDA() {
        cudaEventCreate(&start);
        cudaEventCreate(&e_centroids);
        cudaEventCreate(&e_morton);
        cudaEventCreate(&e_sort);
        cudaEventCreate(&e_topology);
        cudaEventCreate(&stop);
    }

    ~LBVHBuilderCUDA() {
        cudaEventDestroy(start);
        cudaEventDestroy(e_centroids);
        cudaEventDestroy(e_morton);
        cudaEventDestroy(e_sort);
        cudaEventDestroy(e_topology);
        cudaEventDestroy(stop);
    }

    void prepareData(const TriangleMesh& mesh) {
        int n = mesh.size();
        d_v0x = mesh.v0x; d_v0y = mesh.v0y; d_v0z = mesh.v0z;
        d_v1x = mesh.v1x; d_v1y = mesh.v1y; d_v1z = mesh.v1z;
        d_v2x = mesh.v2x; d_v2y = mesh.v2y; d_v2z = mesh.v2z;
        
        d_centroids.resize(n);
        d_mortonCodes.resize(n);
        d_indices.resize(n);
        d_nodes.resize(2 * n - 1);
        d_atomicFlags.resize(2 * n - 1);
    }

    void runCompute(int n) {
        if (n == 0) return;

        thrust::fill(d_atomicFlags.begin(), d_atomicFlags.end(), 0);

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        cudaEventRecord(start);

        // 1. Centroids
        kComputeBoundsAndCentroids<<<gridSize, blockSize>>>(getDevicePtrs(), n,
            thrust::raw_pointer_cast(d_triBBoxes.data()),
            thrust::raw_pointer_cast(d_centroids.data()));
        
        AABB_cw init; 
        AABB_cw sceneBounds = thrust::reduce(d_triBBoxes.begin(), d_triBBoxes.end(), init, AABBReduce());
        
        cudaEventRecord(e_centroids);

        // 2. Morton Codes
        kComputeMortonCodes<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_centroids.data()),
            n, sceneBounds,
            thrust::raw_pointer_cast(d_mortonCodes.data()),
            thrust::raw_pointer_cast(d_indices.data()));
            
        cudaEventRecord(e_morton);

        // 3. Sort
        thrust::sort_by_key(d_mortonCodes.begin(), d_mortonCodes.end(), d_indices.begin());
        
        cudaEventRecord(e_sort);

        // 4. Topology
        kBuildInternalNodes<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_mortonCodes.data()),
            thrust::raw_pointer_cast(d_indices.data()),
            thrust::raw_pointer_cast(d_nodes.data()),
            n);

        kInitLeafNodes<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_triBBoxes.data()),
            thrust::raw_pointer_cast(d_indices.data()),
            n);
            
        cudaEventRecord(e_topology);

        // 5. Refit
        kRefitHierarchy<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_atomicFlags.data()),
            n);
            
        cudaEventRecord(stop);
        
        // Print Breakdown
        cudaEventSynchronize(stop);
        float t_c, t_m, t_s, t_t, t_r;
        cudaEventElapsedTime(&t_c, start, e_centroids);
        cudaEventElapsedTime(&t_m, e_centroids, e_morton);
        cudaEventElapsedTime(&t_s, e_morton, e_sort);
        cudaEventElapsedTime(&t_t, e_sort, e_topology);
        cudaEventElapsedTime(&t_r, e_topology, stop);
        float total = t_c + t_m + t_s + t_t + t_r;

        std::cout << "\n   [GPU Phase Breakdown]\n";
        std::cout << "   Bounds/Centroids: " << std::setw(6) << t_c << " ms\n";
        std::cout << "   Morton Codes:     " << std::setw(6) << t_m << " ms\n";
        std::cout << "   Radix Sort:       " << std::setw(6) << t_s << " ms\n";
        std::cout << "   Topology Build:   " << std::setw(6) << t_t << " ms\n";
        std::cout << "   Bottom-Up Refit:  " << std::setw(6) << t_r << " ms\n";
        std::cout << "   Total Kernel Time:" << std::setw(6) << total << " ms\n\n";
    }

    void verify() {
        if(d_nodes.empty()) return;
        LBVHNode hostRoot = d_nodes[0]; 
        AABB_cw root = hostRoot.bbox;
        std::cout << "Verification Root Bounds:\n";
        std::cout << "  Min: " << root.min.x << ", " << root.min.y << ", " << root.min.z << "\n";
        std::cout << "  Max: " << root.max.x << ", " << root.max.y << ", " << root.max.z << "\n";
    }

    std::vector<LBVHNode> getRawNodes() {
        std::vector<LBVHNode> h_nodes(d_nodes.size());
        thrust::copy(d_nodes.begin(), d_nodes.end(), h_nodes.begin());
        return h_nodes;
    }

    std::vector<BVHNode> getNodes() {
        return getRawNodes(); 
    }
};

// --- DATA HELPERS ---

void printUsage(const char* name) {
    std::cout << "Usage: " << name << " [options]\n"
              << "Options:\n"
              << "  -i, --input <file>       Input OBJ file\n"
              << "  -o, --output <file>      Output file\n"
              << "  -n, --triangles <num>    Number of random triangles (default 10M)\n"
              << "  -l, --leaves-only        Export leaves only to OBJ\n"
              << "  -c, --colab-export       Export binary dump\n"
              << "  -h, --help               Show help\n";
}

TriangleMesh generateRandomTriangles(int n) {
    TriangleMesh mesh;
    mesh.resize(n);

    for (int i = 0; i < n; ++i) {
        float x = (rand() % 1000) / 10.0f;
        float y = (rand() % 1000) / 10.0f;
        float z = (rand() % 1000) / 10.0f;
        mesh.v0x[i] = x; mesh.v0y[i] = y; mesh.v0z[i] = z;
        mesh.v1x[i] = x+1; mesh.v1y[i] = y; mesh.v1z[i] = z;
        mesh.v2x[i] = x; mesh.v2y[i] = y+1; mesh.v2z[i] = z;
    }
    
    return mesh;
}

TriangleMesh loadOBJ(const std::string& filename) {
    TriangleMesh mesh;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file");

    std::vector<float3_cw> verts;
    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 2) == "v ") {
            std::istringstream s(line.substr(2));
            float3_cw v;
            s >> v.x >> v.y >> v.z;
            verts.push_back(v);
        } else if (line.substr(0, 2) == "f ") {
            std::istringstream s(line.substr(2));
            int idx[3];
            std::string tmp;

            for(int i=0; i<3; ++i) {
                s >> tmp;
                size_t slash = tmp.find('/');
                idx[i] = std::stoi(tmp.substr(0, slash)) - 1;
            }

            if(idx[0] < verts.size() && idx[1] < verts.size() && idx[2] < verts.size()) {
                mesh.v0x.push_back(verts[idx[0]].x); mesh.v0y.push_back(verts[idx[0]].y); mesh.v0z.push_back(verts[idx[0]].z);
                mesh.v1x.push_back(verts[idx[1]].x); mesh.v1y.push_back(verts[idx[1]].y); mesh.v1z.push_back(verts[idx[1]].z);
                mesh.v2x.push_back(verts[idx[2]].x); mesh.v2y.push_back(verts[idx[2]].y); mesh.v2z.push_back(verts[idx[2]].z);
            }
        }
    }
    return mesh;
}

struct VizNode {
    float min[3];
    float max[3];
    int leftIdx;
    int rightIdx;
};

void exportBVHToBinary(const std::string& filename, const std::vector<LBVHNode>& nodes) {
    std::ofstream file(filename, std::ios::binary);
    
    std::vector<VizNode> exportNodes;
    exportNodes.reserve(nodes.size());

    for(const auto& node : nodes) {
        VizNode vn;
        vn.min[0] = node.bbox.min.x; 
        vn.min[1] = node.bbox.min.y; 
        vn.min[2] = node.bbox.min.z;
        
        vn.max[0] = node.bbox.max.x; 
        vn.max[1] = node.bbox.max.y; 
        vn.max[2] = node.bbox.max.z;

        vn.leftIdx = static_cast<int>(node.leftChild);
        vn.rightIdx = static_cast<int>(node.rightChild);
        
        exportNodes.push_back(vn);
    }

    file.write(reinterpret_cast<const char*>(exportNodes.data()), exportNodes.size() * sizeof(VizNode));
    std::cout << "Exported " << exportNodes.size() << " nodes to " << filename << " (" << (exportNodes.size() * sizeof(VizNode))/1024/1024 << " MB)\n";
}

void exportBVHToOBJ(const std::string& filename, const std::vector<BVHNode>& nodes, bool leavesOnly) {
    std::ofstream file(filename);
    int v_offset = 1;

    auto writeBox = [&](const AABB_cw& b) {
        float x0 = b.min.x, y0 = b.min.y, z0 = b.min.z;
        float x1 = b.max.x, y1 = b.max.y, z1 = b.max.z;

        file << "v " << x0 << " " << y0 << " " << z0 << "\n";
        file << "v " << x1 << " " << y0 << " " << z0 << "\n";
        file << "v " << x1 << " " << y1 << " " << z0 << "\n";
        file << "v " << x0 << " " << y1 << " " << z0 << "\n";
        file << "v " << x0 << " " << y0 << " " << z1 << "\n";
        file << "v " << x1 << " " << y0 << " " << z1 << "\n";
        file << "v " << x1 << " " << y1 << " " << z1 << "\n";
        file << "v " << x0 << " " << y1 << " " << z1 << "\n";

        int o = v_offset;
        file << "l " << o << " " << o+1 << " " << o+2 << " " << o+3 << " " << o << "\n"; 
        file << "l " << o+4 << " " << o+5 << " " << o+6 << " " << o+7 << " " << o+4 << "\n"; 
        file << "l " << o << " " << o+4 << "\n";
        file << "l " << o+1 << " " << o+5 << "\n";
        file << "l " << o+2 << " " << o+6 << "\n";
        file << "l " << o+3 << " " << o+7 << "\n";
        v_offset += 8;
    };

    int numLeafs = (nodes.size() + 1) / 2;
    for (size_t i = 0; i < nodes.size(); ++i) {
        bool isLeaf = (i >= nodes.size() - numLeafs);
        
        bool nodeIsLeaf = (i >= (nodes.size() + 1) / 2 - 1); // Rough check based on array layout
        if (leavesOnly && !nodeIsLeaf) continue;
        writeBox(nodes[i].bbox);
    }
}

// --- MAIN ---

int main(int argc, char* argv[]) {
    std::string inputFile;
    std::string outputFile;
    int numTriangles = 10000000;  
    bool leavesOnly = false;
    bool colabExport = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) inputFile = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) outputFile = argv[++i];
        } else if (arg == "-n" || arg == "--triangles") {
            if (i + 1 < argc) numTriangles = std::atoi(argv[++i]);
        } else if (arg == "-l" || arg == "--leaves-only") {
            leavesOnly = true;
        } else if (arg == "-c" || arg == "--colab-export") {
            colabExport = true;
        }
    }

    TriangleMesh mesh;
    if (!inputFile.empty()) {
        std::cout << "Loading OBJ file: " << inputFile << std::endl;
        try {
            mesh = loadOBJ(inputFile);
            std::cout << "Loaded " << mesh.size() << " triangles\n";
        } catch (const std::exception& e) {
            std::cerr << "Error loading OBJ: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "Generating " << numTriangles << " random triangles..." << std::endl;
        mesh = generateRandomTriangles(numTriangles);
    }

    if (mesh.size() == 0) return 1;

    LBVHBuilderCUDA builder;
    
    // 1. Prepare Data (Allocation + Copy) - Ne tajmira se
    std::cout << "Uploading data to GPU and allocating memory..." << std::endl;
    builder.prepareData(mesh);

    // Opcionlni warm-up run
    // builder.runCompute(mesh.size());
    // CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Starting optimized build..." << std::endl;
    cudaEventRecord(start);
    
    // 2. Run Compute 
    builder.runCompute(mesh.size());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Triangles: " << mesh.size() << std::endl;
    std::cout << "Compute Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Throughput: " << (mesh.size() / 1000000.0f) / (milliseconds / 1000.0f) << " MTris/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    
    builder.verify();

    if (!outputFile.empty()) {
        if (colabExport) {
            std::cout << "Exporting BVH to Colab binary format: " << outputFile << std::endl;
            std::vector<LBVHNode> nodes = builder.getRawNodes();
            exportBVHToBinary(outputFile, nodes);
        } else {
            std::cout << "Exporting BVH to OBJ format: " << outputFile << std::endl;
            std::vector<BVHNode> bvhNodes = builder.getNodes();
            exportBVHToOBJ(outputFile, bvhNodes, leavesOnly);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}