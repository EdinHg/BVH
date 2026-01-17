// OpenMP-optimized LBVH Builder with Random Triangle Generation
// Standalone implementation for benchmarking
// Compile: g++ -O3 -fopenmp -std=c++17 ompLBVH.cpp -o ompLBVH
// Usage:
//   Random triangles: ./ompLBVH <num_triangles> [num_threads] [--export] [--export-leaves]
//   From OBJ file:    ./ompLBVH <model.obj> [num_threads] [--export] [--export-leaves]

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <random>
#include <chrono>
#include <cstring>
#include <string>
#include <omp.h>

// ============================================================================
// Math Structures
// ============================================================================

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }

    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }

    float length() const { return std::sqrt(x * x + y * y + z * z); }

    static Vec3 min(const Vec3& a, const Vec3& b) {
        return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
    }
    static Vec3 max(const Vec3& a, const Vec3& b) {
        return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
    }
};

struct AABB {
    Vec3 min, max;

    AABB() : min(std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()),
             max(std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest()) {}

    AABB(const Vec3& min, const Vec3& max) : min(min), max(max) {}

    void expand(const Vec3& p) {
        min = Vec3::min(min, p);
        max = Vec3::max(max, p);
    }

    void expand(const AABB& other) {
        min = Vec3::min(min, other.min);
        max = Vec3::max(max, other.max);
    }

    Vec3 extent() const { return max - min; }
    Vec3 center() const { return (min + max) * 0.5f; }

    float surfaceArea() const {
        Vec3 d = extent();
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }

    bool valid() const {
        return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }
};

// ============================================================================
// BVH Structures
// ============================================================================

struct BVHNode {
    AABB bounds;
    uint32_t childOffset;   // Offset to first child in node array
    uint8_t  childCount;    // 0 = leaf, 2 = binary
    uint8_t  axis;
    uint16_t primCount;
    uint32_t primOffset;

    bool isLeaf() const { return childCount == 0; }
};

// ============================================================================
// Triangle Mesh (SoA layout for cache efficiency)
// ============================================================================

struct TriangleMesh {
    std::vector<float> v0x, v0y, v0z;
    std::vector<float> v1x, v1y, v1z;
    std::vector<float> v2x, v2y, v2z;

    size_t size() const { return v0x.size(); }

    void reserve(size_t n) {
        v0x.reserve(n); v0y.reserve(n); v0z.reserve(n);
        v1x.reserve(n); v1y.reserve(n); v1z.reserve(n);
        v2x.reserve(n); v2y.reserve(n); v2z.reserve(n);
    }

    void addTriangle(const Vec3& a, const Vec3& b, const Vec3& c) {
        v0x.push_back(a.x); v0y.push_back(a.y); v0z.push_back(a.z);
        v1x.push_back(b.x); v1y.push_back(b.y); v1z.push_back(b.z);
        v2x.push_back(c.x); v2y.push_back(c.y); v2z.push_back(c.z);
    }

    Vec3 getVertex0(size_t i) const { return {v0x[i], v0y[i], v0z[i]}; }
    Vec3 getVertex1(size_t i) const { return {v1x[i], v1y[i], v1z[i]}; }
    Vec3 getVertex2(size_t i) const { return {v2x[i], v2y[i], v2z[i]}; }

    Vec3 getCentroid(size_t i) const {
        return (getVertex0(i) + getVertex1(i) + getVertex2(i)) / 3.0f;
    }

    AABB getBounds(size_t i) const {
        AABB box;
        box.expand(getVertex0(i));
        box.expand(getVertex1(i));
        box.expand(getVertex2(i));
        return box;
    }
};

// ============================================================================
// Random Triangle Generation
// ============================================================================

TriangleMesh generateRandomTriangles(size_t count, uint32_t seed = 42) {
    TriangleMesh mesh;
    mesh.reserve(count);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < count; ++i) {
        Vec3 v0(dist(rng), dist(rng), dist(rng));
        Vec3 v1(dist(rng), dist(rng), dist(rng));
        Vec3 v2(dist(rng), dist(rng), dist(rng));
        mesh.addTriangle(v0, v1, v2);
    }

    return mesh;
}

// ============================================================================
// OBJ File Loader
// ============================================================================

TriangleMesh loadOBJ(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open OBJ file: " + path);
    }

    std::vector<Vec3> vertices;
    TriangleMesh mesh;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back({x, y, z});
        }
        else if (prefix == "f") {
            std::vector<int> faceIndices;
            std::string token;
            while (iss >> token) {
                // Handle formats: "v", "v/vt", "v/vt/vn", "v//vn"
                int idx = std::stoi(token.substr(0, token.find('/')));
                // OBJ indices are 1-based, negative means relative
                if (idx < 0) idx = static_cast<int>(vertices.size()) + idx + 1;
                faceIndices.push_back(idx - 1);
            }
            // Triangulate face (fan triangulation)
            for (size_t i = 1; i + 1 < faceIndices.size(); ++i) {
                mesh.addTriangle(
                    vertices[faceIndices[0]],
                    vertices[faceIndices[i]],
                    vertices[faceIndices[i + 1]]
                );
            }
        }
    }

    return mesh;
}

// ============================================================================
// Morton Code Functions
// ============================================================================

inline uint32_t expandBits(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

inline uint32_t mortonCode(float x, float y, float z) {
    x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
    y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
    z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
    uint32_t xx = expandBits(static_cast<uint32_t>(x));
    uint32_t yy = expandBits(static_cast<uint32_t>(y));
    uint32_t zz = expandBits(static_cast<uint32_t>(z));
    return (xx << 2) | (yy << 1) | zz;
}

// ============================================================================
// Parallel Radix Sort (OpenMP optimized)
// ============================================================================

void parallelRadixSort(std::vector<std::pair<uint32_t, uint32_t>>& mortonCodes) {
    size_t n = mortonCodes.size();
    if (n <= 1) return;

    std::vector<std::pair<uint32_t, uint32_t>> temp(n);

    for (int shift = 0; shift < 30; shift += 10) {
        int count[1024] = {0};

        // Parallel counting
        #pragma omp parallel
        {
            int localCount[1024] = {0};

            #pragma omp for nowait
            for (size_t i = 0; i < n; i++) {
                uint32_t bits = (mortonCodes[i].first >> shift) & 0x3FF;
                localCount[bits]++;
            }

            // Merge local counts
            #pragma omp critical
            {
                for (int i = 0; i < 1024; i++) {
                    count[i] += localCount[i];
                }
            }
        }

        // Prefix sum (sequential - small overhead)
        int total = 0;
        for (int i = 0; i < 1024; i++) {
            int c = count[i];
            count[i] = total;
            total += c;
        }

        // Parallel scatter
        #pragma omp parallel
        {
            int localCount[1024];
            std::memcpy(localCount, count, sizeof(count));

            #pragma omp for
            for (size_t i = 0; i < n; i++) {
                uint32_t bits = (mortonCodes[i].first >> shift) & 0x3FF;
                int pos = localCount[bits];
                localCount[bits]++;
                temp[pos] = mortonCodes[i];
            }
        }

        mortonCodes.swap(temp);
    }
}

// ============================================================================
// Count Leading Zeros
// ============================================================================

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

// ============================================================================
// OpenMP LBVH Builder
// ============================================================================

class ompLBVHBuilder {
public:
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> primIndices;

    void build(const TriangleMesh& mesh) {
        size_t n = mesh.size();
        if (n == 0) return;

        // Step 1: Compute scene bounds and centroids (exact algorithm from lbvh_builder.hpp)
        AABB sceneBounds;
        std::vector<Vec3> centroids(n);
        std::vector<AABB> triBounds(n);

        for (size_t i = 0; i < n; ++i) {
            triBounds[i] = mesh.getBounds(i);
            centroids[i] = triBounds[i].center();
            sceneBounds.expand(triBounds[i]);
        }

        Vec3 sceneSize = sceneBounds.extent();

        // Step 2: Compute Morton codes (parallel)
        std::vector<std::pair<uint32_t, uint32_t>> mortonCodes(n);

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            Vec3 offset = centroids[i] - sceneBounds.min;
            float nx = sceneSize.x > 0 ? offset.x / sceneSize.x : 0.5f;
            float ny = sceneSize.y > 0 ? offset.y / sceneSize.y : 0.5f;
            float nz = sceneSize.z > 0 ? offset.z / sceneSize.z : 0.5f;
            mortonCodes[i].first = mortonCode(nx, ny, nz);
            mortonCodes[i].second = static_cast<uint32_t>(i);
        }

        // Step 3: Parallel radix sort
        parallelRadixSort(mortonCodes);

        // Extract sorted data
        std::vector<uint32_t> sortedCodes(n);
        primIndices.resize(n);

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            primIndices[i] = mortonCodes[i].second;
            sortedCodes[i] = mortonCodes[i].first;
        }

        // Step 4: Build tree structure
        nodes.resize(2 * n - 1);

        // Initialize leaf nodes (parallel)
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            BVHNode& leaf = nodes[n - 1 + i];
            uint32_t originalIdx = primIndices[i];
            leaf.bounds = triBounds[originalIdx];
            leaf.childOffset = 0;
            leaf.childCount = 0;  // leaf
            leaf.primCount = 1;
            leaf.primOffset = static_cast<uint32_t>(i);
            leaf.axis = 0;
        }

        // Build internal nodes (parallel using Karras algorithm)
        std::vector<uint32_t> parents(2 * n - 1, UINT32_MAX);

        #pragma omp parallel for
        for (size_t i = 0; i < n - 1; ++i) {
            auto [first, last] = determineRange(sortedCodes, static_cast<int>(i), static_cast<int>(n));
            int split = findSplit(sortedCodes, first, last, static_cast<int>(n));

            uint32_t leftChild = (first == split)
                ? static_cast<uint32_t>(n - 1 + split)
                : static_cast<uint32_t>(split);
            uint32_t rightChild = (split + 1 == last)
                ? static_cast<uint32_t>(n - 1 + split + 1)
                : static_cast<uint32_t>(split + 1);

            BVHNode& node = nodes[i];
            node.childOffset = leftChild;
            node.childCount = 2;  // binary
            node.primCount = 0;
            node.primOffset = rightChild;
            node.axis = 0;

            parents[leftChild] = static_cast<uint32_t>(i);
            parents[rightChild] = static_cast<uint32_t>(i);
        }

        // Step 5: Compute bounding boxes bottom-up (exact algorithm from lbvh_builder.hpp)
        std::vector<bool> visited(2 * n - 1, false);

        for (size_t i = 0; i < n; ++i) {
            uint32_t current = static_cast<uint32_t>(n - 1 + i);
            visited[current] = true;

            while (current > 0) {
                uint32_t parent = parents[current];
                if (parent == UINT32_MAX) break;

                uint32_t leftChild = nodes[parent].childOffset;
                uint32_t rightChild = nodes[parent].primOffset;
                uint32_t sibling = (leftChild == current) ? rightChild : leftChild;

                if (!visited[sibling]) break;  // Wait for sibling

                nodes[parent].bounds.expand(nodes[leftChild].bounds);
                nodes[parent].bounds.expand(nodes[rightChild].bounds);

                visited[parent] = true;
                current = parent;
            }
        }
    }

    float calculateSAHCost(float traversalCost = 1.0f, float intersectionCost = 1.0f) const {
        if (nodes.empty()) return 0.0f;

        float rootArea = nodes[0].bounds.surfaceArea();
        if (rootArea <= 0.0f) return 0.0f;

        float cost = 0.0f;

        #pragma omp parallel for reduction(+:cost)
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            float relativeArea = node.bounds.surfaceArea() / rootArea;

            if (node.isLeaf()) {
                cost += intersectionCost * relativeArea * node.primCount;
            } else {
                cost += traversalCost * relativeArea;
            }
        }

        return cost;
    }

private:
    int deltaNode(const std::vector<uint32_t>& codes, int i, int j, int n) const {
        if (j < 0 || j >= n) return -1;
        if (codes[i] == codes[j]) return 32 + clz32(static_cast<uint32_t>(i ^ j));
        return clz32(codes[i] ^ codes[j]);
    }

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

// ============================================================================
// BVH Export to OBJ
// ============================================================================

void exportBVHToOBJ(const std::string& path,
                    const std::vector<BVHNode>& nodes,
                    bool leavesOnly = false) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << "\n";
        return;
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

// ============================================================================
// Timer Utility
// ============================================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { end_ = std::chrono::high_resolution_clock::now(); }
    double elapsedMs() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_);
        return duration.count() / 1000.0;
    }
private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "  Random: " << argv[0] << " <num_triangles> [num_threads] [--export] [--export-leaves]\n";
        std::cerr << "  OBJ:    " << argv[0] << " <model.obj> [num_threads] [--export] [--export-leaves]\n";
        return 1;
    }

    std::string firstArg = argv[1];
    bool loadFromOBJ = (firstArg.find(".obj") != std::string::npos);

    size_t numTriangles = 100000;
    int numThreads = omp_get_max_threads();
    bool doExport = false;
    bool leavesOnly = false;
    std::string objPath;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--export") {
            doExport = true;
        } else if (arg == "--export-leaves") {
            doExport = true;
            leavesOnly = true;
        } else if (i == 1) {
            if (loadFromOBJ) {
                objPath = arg;
            } else {
                numTriangles = std::stoull(argv[1]);
            }
        } else if (i == 2 && arg.find("--") == std::string::npos) {
            numThreads = std::stoi(argv[2]);
        }
    }

    omp_set_num_threads(numThreads);

    std::cout << "OpenMP LBVH Builder Benchmark\n";
    std::cout << "============================\n";
    if (loadFromOBJ) {
        std::cout << "Input: " << objPath << "\n";
    } else {
        std::cout << "Triangles: " << numTriangles << " (random)\n";
    }
    std::cout << "Threads: " << numThreads << "\n";
    if (doExport) {
        std::cout << "Export: " << (leavesOnly ? "leaves only" : "all nodes") << "\n";
    }
    std::cout << "\n";

    // Load or generate triangles
    Timer timer;
    TriangleMesh mesh;

    timer.start();
    if (loadFromOBJ) {
        try {
            mesh = loadOBJ(objPath);
            timer.stop();
            std::cout << "OBJ loading: " << timer.elapsedMs() << " ms\n";
            std::cout << "Loaded " << mesh.size() << " triangles\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    } else {
        mesh = generateRandomTriangles(numTriangles);
        timer.stop();
        std::cout << "Triangle generation: " << timer.elapsedMs() << " ms\n";
    }

    // Build BVH
    ompLBVHBuilder builder;
    timer.start();
    builder.build(mesh);
    timer.stop();
    double buildTime = timer.elapsedMs();

    std::cout << "BVH construction: " << buildTime << " ms\n";
    std::cout << "Triangles/sec: " << (numTriangles / buildTime * 1000.0) << "\n";
    std::cout << "Total nodes: " << builder.nodes.size() << "\n";

    // Calculate SAH cost
    timer.start();
    float sahCost = builder.calculateSAHCost();
    timer.stop();

    std::cout << "SAH cost calculation: " << timer.elapsedMs() << " ms\n";
    std::cout << "SAH cost: " << sahCost << "\n";

    // Export BVH if requested
    if (doExport) {
        std::cout << "\nExporting BVH...\n";
        exportBVHToOBJ("omp_bvh_output.obj", builder.nodes, leavesOnly);
        std::cout << "Exported to: omp_bvh_output.obj\n";
    }

    return 0;
}
