#pragma once
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <queue>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <atomic>
#include <cstring>

#include "geometry.cpp"
#include "ray.cpp"

struct BVHBuildStats {
    double boundsComputationTime = 0.0;
    double mortonCodeComputationTime = 0.0;
    double radixSortTime = 0.0;
    double leafInitializationTime = 0.0;
    double internalNodeConstructionTime = 0.0;
    double bboxComputationTime = 0.0;
    double totalTime = 0.0;
    
    void print() const {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "=== BVH Build Statistics (sorted by duration) ===\n";
        std::cout << std::string(70, '=') << "\n";
        
        auto sorted = getSortedTimes();
        std::cout << std::fixed << std::setprecision(6);
        
        for (const auto& [stage, time] : sorted) {
            double percentage = (time / totalTime) * 100.0;
            std::cout << std::setw(30) << std::left << stage << ": "
                      << std::setw(12) << std::right << (time * 1000) << " ms  "
                      << "(" << std::setprecision(1) << std::setw(5) << percentage << "%)\n";
        }
        
        std::cout << std::string(70, '-') << "\n";
        std::cout << std::setw(30) << std::left << "Total time" << ": "
                  << std::setprecision(6) << std::setw(12) << std::right << (totalTime * 1000) 
                  << " ms  (100.0%)\n";
        std::cout << std::string(70, '=') << "\n";
    }
    
    std::vector<std::pair<std::string, double>> getSortedTimes() const {
        std::vector<std::pair<std::string, double>> times = {
            {"Bounds computation", boundsComputationTime},
            {"Morton code computation", mortonCodeComputationTime},
            {"Radix sort", radixSortTime},
            {"Leaf initialization", leafInitializationTime},
            {"Internal node construction", internalNodeConstructionTime},
            {"BBox computation", bboxComputationTime}
        };
        std::sort(times.begin(), times.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        return times;
    }
};

struct alignas(32) LBVHNode {
    AABB bbox; // 24 bytes
    
    union {
        struct {
            uint32_t leftChild;
            uint32_t rightChild;
        }; // Internal uses 8 bytes
        struct {
            uint32_t triangleIndex;
            uint32_t padding; 
        }; // Leaf uses 8 bytes
    };
    
    LBVHNode() {} 
    
    bool isLeaf() const { return (leftChild & 0x80000000) != 0; }
};

// Expand 10 bits of input into 30 bits by inserting 2 zeros between each bit
// Example: AB CDEF GHIJ -> 0000 A00B 00C0 0D00 E00F 00G0 0H00 I00J
uint32_t expandBits(uint32_t v) {           
//  0x030000FF = 0000 0011 0000 0000 0000 0000 1111 1111 
//           v = 0000 00AB 0000 0000 0000 0000 CDEF GHIJ
    v = (v | (v << 16)) & 0x030000FF; 

//  0x0300F00F = 0000 0011 0000 0000 1111 0000 0000 1111
//           v = 0000 00AB 0000 0000 CDEF 0000 0000 GHIJ 
    v = (v | (v << 8)) & 0x0300F00F;

//  0x030C30C3 = 0000 0011 0000 1100 0011 0000 1100 0011
//           v = 0000 00AB 0000 CD00 00EF 0000 GH00 00IJ 
    v = (v | (v << 4)) & 0x030C30C3;

//  0x09249249 = 0000 1001 0010 0100 1001 0010 0100 1001
//           v = 0000 A00B 00C0 0D00 E00F 00G0 0H00 I00J
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

inline __attribute__((always_inline)) __m256i expandBitsAVX2(__m256i v) {
    __m256i mask1 = _mm256_set1_epi32(0x030000FF);
    __m256i mask2 = _mm256_set1_epi32(0x0300F00F);
    __m256i mask3 = _mm256_set1_epi32(0x030C30C3);
    __m256i mask4 = _mm256_set1_epi32(0x09249249);

    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 16)), mask1);
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 8)), mask2);
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 4)), mask3);
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 2)), mask4);
    
    return v;
}

// Compute 30-bit Morton code, interleaving like so: xyzxyzxyz...
uint32_t mortonCode(float x, float y, float z) {
    x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
    y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
    z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
    uint32_t xx = expandBits((uint32_t)x);
    uint32_t yy = expandBits((uint32_t)y);
    uint32_t zz = expandBits((uint32_t)z);
    return (xx << 2) | (yy << 1) | zz;
}

// AVX2 version of mortonCode for 8 points
inline __attribute__((always_inline)) __m256i mortonCodeAVX2(__m256 x, __m256 y, __m256 z) {
    __m256 scale = _mm256_set1_ps(1024.0f);
    __m256 zero = _mm256_set1_ps(0.0f);
    __m256 maxVal = _mm256_set1_ps(1023.0f);

    x = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x, scale), zero), maxVal);
    y = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(y, scale), zero), maxVal);
    z = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(z, scale), zero), maxVal);

    __m256i xx = _mm256_cvttps_epi32(x);
    __m256i yy = _mm256_cvttps_epi32(y);
    __m256i zz = _mm256_cvttps_epi32(z);

    xx = expandBitsAVX2(xx);
    yy = expandBitsAVX2(yy);
    zz = expandBitsAVX2(zz);

    return _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(xx, 2), _mm256_slli_epi32(yy, 1)), zz);
}

// Count leading zeros
int clz(uint32_t x) {
    if (x == 0) return 32;
    return __builtin_clz(x); 
}

// Custom aligned buffer that doesn't zero-initialize
template<typename T, size_t Alignment = 32>
class AlignedBuffer {
private:
    T* ptr = nullptr;
    size_t sz = 0;
    size_t cap = 0;

public:
    AlignedBuffer() = default;
    
    ~AlignedBuffer() {
        if (ptr) _mm_free(ptr);
    }
    
    // Disable copy
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    
    // Enable move
    AlignedBuffer(AlignedBuffer&& other) noexcept : ptr(other.ptr), sz(other.sz), cap(other.cap) {
        other.ptr = nullptr;
        other.sz = 0;
        other.cap = 0;
    }
    
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) _mm_free(ptr);
            ptr = other.ptr;
            sz = other.sz;
            cap = other.cap;
            other.ptr = nullptr;
            other.sz = 0;
            other.cap = 0;
        }
        return *this;
    }
    
    void resize(size_t n) {
        if (n > cap) {
            size_t newCap = n + (n >> 1); 
            if (newCap < 32) newCap = 32;
            
            T* newPtr = (T*)_mm_malloc(newCap * sizeof(T), Alignment);
            if (ptr) _mm_free(ptr);
            ptr = newPtr;
            cap = newCap;
        }
        sz = n;
    }
    
    T* data() { return ptr; }
    const T* data() const { return ptr; }
    size_t size() const { return sz; }
    T& operator[](size_t i) { return ptr[i]; }
    const T& operator[](size_t i) const { return ptr[i]; }
    
    void clear() { sz = 0; }
};

class LBVHBuilder {
private:
    TrianglesSoA triangles;
    AlignedBuffer<LBVHNode> nodes;
    
    // Persistent buffers
    AlignedBuffer<std::pair<uint32_t, uint32_t>> mortonCodes;
    AlignedBuffer<std::pair<uint32_t, uint32_t>> sortTemp;
    AlignedBuffer<int> atomicFlags;
    AlignedBuffer<AABB> BBoxes;
    AlignedBuffer<Vector3> centroids;
    AlignedBuffer<uint32_t> parents; 
    
    // Radix sort buffers
    AlignedBuffer<int> rs_localCounts;
    AlignedBuffer<int> rs_threadOffsets;
    
    BVHBuildStats lastBuildStats;
    bool enableBenchmark = false;

    void radixSort(AlignedBuffer<std::pair<uint32_t, uint32_t>>& data) {
        int n = data.size();
        if (n == 0) return;
        
        const int numThreads = omp_get_max_threads();
        const int RADIX = 1024;
        const int CACHE_LINE = 64;
        const int INTS_PER_CACHE = CACHE_LINE / sizeof(int);
        
        // Padded to prevent false sharing
        const int paddedSize = ((RADIX + INTS_PER_CACHE - 1) / INTS_PER_CACHE) * INTS_PER_CACHE;
        
        size_t requiredCountSize = (size_t)numThreads * paddedSize;
        if (rs_localCounts.size() < requiredCountSize) rs_localCounts.resize(requiredCountSize);
        if (rs_threadOffsets.size() < requiredCountSize) rs_threadOffsets.resize(requiredCountSize);
        if (sortTemp.size() < n) sortTemp.resize(n);
        
        std::pair<uint32_t, uint32_t>* src = data.data();
        std::pair<uint32_t, uint32_t>* dst = sortTemp.data();
        
        for (int shift = 0; shift < 30; shift += 10) {
            // Histogram
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                int* myCount = &rs_localCounts[tid * paddedSize];
                std::memset(myCount, 0, RADIX * sizeof(int));
                #pragma omp for schedule(static) nowait
                for (int i = 0; i < n; i++) {
                    uint32_t key = (src[i].first >> shift) & 0x3FF;
                    myCount[key]++;
                }
            }
            
            // Prefix sum
            int globalCount[RADIX] = {0};
            for (int t = 0; t < numThreads; t++) {
                int* myCount = &rs_localCounts[t * paddedSize];
                for (int i = 0; i < RADIX; i++) globalCount[i] += myCount[i];
            }
            
            int globalOffsets[RADIX];
            globalOffsets[0] = 0;
            for (int i = 1; i < RADIX; i++) globalOffsets[i] = globalOffsets[i-1] + globalCount[i-1];
            
            // Offsets
            for (int i = 0; i < RADIX; i++) {
                int offset = globalOffsets[i];
                for (int t = 0; t < numThreads; t++) {
                    rs_threadOffsets[t * paddedSize + i] = offset;
                    offset += rs_localCounts[t * paddedSize + i];
                }
            }
            
            // Scatter
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                int* myOffset = &rs_threadOffsets[tid * paddedSize];
                #pragma omp for schedule(static) nowait
                for (int i = 0; i < n; i++) {
                    uint32_t key = (src[i].first >> shift) & 0x3FF;
                    int pos = myOffset[key]++;
                    dst[pos] = src[i];
                }
            }
            
            std::swap(src, dst);
        }
        
        if (src != data.data()) {
            #pragma omp parallel for
            for (int i = 0; i < n; i++) data[i] = src[i];
        }
    }
    
    void ComputeBoundsAndCentroidsAVX(const TrianglesSoA& triangles, 
                                      AlignedBuffer<AABB>& triBBoxes, 
                                      AlignedBuffer<Vector3>& centroids,
                                      AABB& sceneBounds) 
    {
        size_t n = triangles.size();
        const __m256 half = _mm256_set1_ps(0.5f);
        const float INF = std::numeric_limits<float>::infinity();
        
        sceneBounds.min = Vector3(INF, INF, INF);
        sceneBounds.max = Vector3(-INF, -INF, -INF);

        #pragma omp parallel
        {
            __m256 locMinX = _mm256_set1_ps(INF);
            __m256 locMinY = _mm256_set1_ps(INF);
            __m256 locMinZ = _mm256_set1_ps(INF);
            __m256 locMaxX = _mm256_set1_ps(-INF);
            __m256 locMaxY = _mm256_set1_ps(-INF);
            __m256 locMaxZ = _mm256_set1_ps(-INF);

            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; i += 8) {
                size_t remaining = n - i;
                if (remaining >= 8) {
                    __m256 r_v0x = _mm256_loadu_ps(&triangles.v0x[i]);
                    __m256 r_v0y = _mm256_loadu_ps(&triangles.v0y[i]);
                    __m256 r_v0z = _mm256_loadu_ps(&triangles.v0z[i]);
                    __m256 r_v1x = _mm256_loadu_ps(&triangles.v1x[i]);
                    __m256 r_v1y = _mm256_loadu_ps(&triangles.v1y[i]);
                    __m256 r_v1z = _mm256_loadu_ps(&triangles.v1z[i]);
                    __m256 r_v2x = _mm256_loadu_ps(&triangles.v2x[i]);
                    __m256 r_v2y = _mm256_loadu_ps(&triangles.v2y[i]);
                    __m256 r_v2z = _mm256_loadu_ps(&triangles.v2z[i]);

                    __m256 minX = _mm256_min_ps(r_v0x, _mm256_min_ps(r_v1x, r_v2x));
                    __m256 maxX = _mm256_max_ps(r_v0x, _mm256_max_ps(r_v1x, r_v2x));
                    __m256 minY = _mm256_min_ps(r_v0y, _mm256_min_ps(r_v1y, r_v2y));
                    __m256 maxY = _mm256_max_ps(r_v0y, _mm256_max_ps(r_v1y, r_v2y));
                    __m256 minZ = _mm256_min_ps(r_v0z, _mm256_min_ps(r_v1z, r_v2z));
                    __m256 maxZ = _mm256_max_ps(r_v0z, _mm256_max_ps(r_v1z, r_v2z));

                    locMinX = _mm256_min_ps(locMinX, minX);
                    locMaxX = _mm256_max_ps(locMaxX, maxX);
                    locMinY = _mm256_min_ps(locMinY, minY);
                    locMaxY = _mm256_max_ps(locMaxY, maxY);
                    locMinZ = _mm256_min_ps(locMinZ, minZ);
                    locMaxZ = _mm256_max_ps(locMaxZ, maxZ);

                    float tmp[48];
                    _mm256_storeu_ps(tmp, minX);
                    _mm256_storeu_ps(tmp+8, minY);
                    _mm256_storeu_ps(tmp+16, minZ);
                    _mm256_storeu_ps(tmp+24, maxX);
                    _mm256_storeu_ps(tmp+32, maxY);
                    _mm256_storeu_ps(tmp+40, maxZ);
                    
                    for(int k=0; k<8; ++k) {
                        triBBoxes[i+k].min.x = tmp[k];
                        triBBoxes[i+k].min.y = tmp[k+8];
                        triBBoxes[i+k].min.z = tmp[k+16];
                        triBBoxes[i+k].max.x = tmp[k+24];
                        triBBoxes[i+k].max.y = tmp[k+32];
                        triBBoxes[i+k].max.z = tmp[k+40];
                    }

                    __m256 centX = _mm256_mul_ps(_mm256_add_ps(minX, maxX), half);
                    __m256 centY = _mm256_mul_ps(_mm256_add_ps(minY, maxY), half);
                    __m256 centZ = _mm256_mul_ps(_mm256_add_ps(minZ, maxZ), half);

                    float cTmp[24];
                    _mm256_storeu_ps(cTmp, centX);
                    _mm256_storeu_ps(cTmp+8, centY);
                    _mm256_storeu_ps(cTmp+16, centZ);
                    
                    for(int k=0; k<8; ++k) {
                        centroids[i+k].x = cTmp[k];
                        centroids[i+k].y = cTmp[k+8];
                        centroids[i+k].z = cTmp[k+16];
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        Triangle tri;
                        triangles.getTriangle(i + j, tri.v0, tri.v1, tri.v2);
                        AABB& triBox = triBBoxes[i+j];
                        
                        triBox.min.x = std::min({tri.v0.x, tri.v1.x, tri.v2.x});
                        triBox.min.y = std::min({tri.v0.y, tri.v1.y, tri.v2.y});
                        triBox.min.z = std::min({tri.v0.z, tri.v1.z, tri.v2.z});
                        triBox.max.x = std::max({tri.v0.x, tri.v1.x, tri.v2.x});
                        triBox.max.y = std::max({tri.v0.y, tri.v1.y, tri.v2.y});
                        triBox.max.z = std::max({tri.v0.z, tri.v1.z, tri.v2.z});
                        
                        centroids[i+j].x = (triBox.min.x + triBox.max.x) * 0.5f;
                        centroids[i+j].y = (triBox.min.y + triBox.max.y) * 0.5f;
                        centroids[i+j].z = (triBox.min.z + triBox.max.z) * 0.5f;
                    }
                }
            }
            
            float minXarr[8], minYarr[8], minZarr[8];
            float maxXarr[8], maxYarr[8], maxZarr[8];
            _mm256_storeu_ps(minXarr, locMinX);
            _mm256_storeu_ps(minYarr, locMinY);
            _mm256_storeu_ps(minZarr, locMinZ);
            _mm256_storeu_ps(maxXarr, locMaxX);
            _mm256_storeu_ps(maxYarr, locMaxY);
            _mm256_storeu_ps(maxZarr, locMaxZ);

            float tMinX = INF, tMinY = INF, tMinZ = INF;
            float tMaxX = -INF, tMaxY = -INF, tMaxZ = -INF;

            for(int k=0; k<8; ++k) {
                tMinX = std::min(tMinX, minXarr[k]);
                tMinY = std::min(tMinY, minYarr[k]);
                tMinZ = std::min(tMinZ, minZarr[k]);
                tMaxX = std::max(tMaxX, maxXarr[k]);
                tMaxY = std::max(tMaxY, maxYarr[k]);
                tMaxZ = std::max(tMaxZ, maxZarr[k]);
            }
            
            size_t startFallback = (n / 8) * 8;
            for(size_t j = startFallback; j < n; ++j) {
                tMinX = std::min(tMinX, triBBoxes[j].min.x);
                tMinY = std::min(tMinY, triBBoxes[j].min.y);
                tMinZ = std::min(tMinZ, triBBoxes[j].min.z);
                tMaxX = std::max(tMaxX, triBBoxes[j].max.x);
                tMaxY = std::max(tMaxY, triBBoxes[j].max.y);
                tMaxZ = std::max(tMaxZ, triBBoxes[j].max.z);
            }

            #pragma omp critical
            {
                sceneBounds.min.x = std::min(sceneBounds.min.x, tMinX);
                sceneBounds.min.y = std::min(sceneBounds.min.y, tMinY);
                sceneBounds.min.z = std::min(sceneBounds.min.z, tMinZ);
                sceneBounds.max.x = std::max(sceneBounds.max.x, tMaxX);
                sceneBounds.max.y = std::max(sceneBounds.max.y, tMaxY);
                sceneBounds.max.z = std::max(sceneBounds.max.z, tMaxZ);
            }
        }
    }
    
public:
    LBVHBuilder() = default;
    
    // Legacy support
    LBVHBuilder(const std::vector<Triangle>& inputTriangles) {
        triangles = convertToSoA(inputTriangles);
        buildBVH();
    }
    
    LBVHBuilder(const TrianglesSoA& inputTriangles) : triangles(inputTriangles) {
        buildBVH();
    }

    void buildBVH(const TrianglesSoA& inputTriangles, bool benchmark = false) {
        enableBenchmark = benchmark;
        triangles = inputTriangles;
        buildBVH();
    }
    
    // Legacy support
    void buildBVH(const std::vector<Triangle>& inputTriangles, bool benchmark = false) {
        triangles = convertToSoA(inputTriangles);
        enableBenchmark = benchmark;
        buildBVH();
    }

    void buildBVH() {
        auto buildStart = std::chrono::high_resolution_clock::now();
        
        int n = triangles.size();
        if (n == 0) {
            std::cerr << "No triangles to build BVH from\n";
            return;
        }
        
        if (mortonCodes.size() < n) mortonCodes.resize(n);
        if (nodes.size() < 2 * n - 1) nodes.resize(2 * n - 1);
        if (atomicFlags.size() < 2 * n - 1) atomicFlags.resize(2 * n - 1);
        if (BBoxes.size() < n) BBoxes.resize(n);
        if (centroids.size() < n) centroids.resize(n);
        if (parents.size() < 2 * n - 1) parents.resize(2 * n - 1);
        
        #pragma omp parallel for
        for(int i=0; i < 2*n-1; ++i) parents[i] = static_cast<uint32_t>(-1);
        
        auto t1 = std::chrono::high_resolution_clock::now();
        
        AABB sceneBounds;
        ComputeBoundsAndCentroidsAVX(triangles, BBoxes, centroids, sceneBounds);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        
        Vector3 sceneSize = sceneBounds.max - sceneBounds.min;
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += 8) {
            int remaining = n - i;
            if (remaining >= 8) {
                float x[8], y[8], z[8];
                for (int j = 0; j < 8; j++) {
                    Vector3 centroid = centroids[i+j];
                    Vector3 offset = centroid - sceneBounds.min;
                    x[j] = sceneSize.x > 0 ? offset.x / sceneSize.x : 0.5f;
                    y[j] = sceneSize.y > 0 ? offset.y / sceneSize.y : 0.5f;
                    z[j] = sceneSize.z > 0 ? offset.z / sceneSize.z : 0.5f;
                }
                
                __m256 vx = _mm256_loadu_ps(x);
                __m256 vy = _mm256_loadu_ps(y);
                __m256 vz = _mm256_loadu_ps(z);
                
                __m256i codes = mortonCodeAVX2(vx, vy, vz);
                uint32_t results[8];
                _mm256_storeu_si256((__m256i*)results, codes);
                
                for (int j = 0; j < 8; j++) mortonCodes[i+j] = {results[j], i+j};
            } else {
                for (int j = 0; j < remaining; j++) {
                    Vector3 centroid = centroids[i+j];
                    Vector3 offset = centroid - sceneBounds.min;
                    float nx = sceneSize.x > 0 ? offset.x / sceneSize.x : 0.5f;
                    float ny = sceneSize.y > 0 ? offset.y / sceneSize.y : 0.5f;
                    float nz = sceneSize.z > 0 ? offset.z / sceneSize.z : 0.5f;
                    mortonCodes[i+j] = {mortonCode(nx, ny, nz), i+j};
                }
            }
        }
        
        auto t3 = std::chrono::high_resolution_clock::now();
        
        radixSort(mortonCodes);
        
        auto t4 = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            LBVHNode& leaf = nodes[n - 1 + i];
            uint32_t originalIdx = mortonCodes[i].second;
            leaf.bbox = BBoxes[originalIdx];
            leaf.leftChild = originalIdx | 0x80000000;
            leaf.rightChild = static_cast<uint32_t>(-1);
        }
        
        auto t5 = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n - 1; i++) {
            auto range = determineRangeMC(mortonCodes, i, n);
            int split = findSplitMC(mortonCodes, range.first, range.second);
            
            uint32_t leftChild = (range.first == split) ? (n - 1 + split) : split;
            uint32_t rightChild = (split + 1 == range.second) ? (n - 1 + split + 1) : (split + 1);
            
            LBVHNode& internalNode = nodes[i];
            internalNode.leftChild = leftChild;
            internalNode.rightChild = rightChild;
            
            parents[leftChild] = i;
            parents[rightChild] = i;
        }
        
        auto t6 = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < 2 * n - 1; i++) atomicFlags[i] = 0;
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            uint32_t current = n - 1 + i;
            while (current > 0) {
                uint32_t parent = parents[current];
                if (parent == static_cast<uint32_t>(-1)) break;
                
                uint8_t oldVal = __atomic_fetch_add(&atomicFlags[parent], 1, __ATOMIC_ACQ_REL);
                if (oldVal == 0) break;
                
                uint32_t left = nodes[parent].leftChild;
                uint32_t right = nodes[parent].rightChild;
                
                __m256 leftBox = _mm256_loadu_ps(&nodes[left].bbox.min.x);
                __m256 rightBox = _mm256_loadu_ps(&nodes[right].bbox.min.x);
                
                __m256 minRes = _mm256_min_ps(leftBox, rightBox);
                __m256 maxRes = _mm256_max_ps(leftBox, rightBox);
                
                __m256 result = _mm256_blend_ps(minRes, maxRes, 0x38);
                
                _mm_storeu_ps(&nodes[parent].bbox.min.x, _mm256_castps256_ps128(result));
                
                __m128 high = _mm256_extractf128_ps(result, 1);
                _mm_storel_pi((__m64*)&nodes[parent].bbox.max.y, high);
                
                current = parent;
            }
        }
        
        auto t7 = std::chrono::high_resolution_clock::now();
        
        if (enableBenchmark) {
            lastBuildStats.boundsComputationTime = std::chrono::duration<double>(t2 - t1).count();
            lastBuildStats.mortonCodeComputationTime = std::chrono::duration<double>(t3 - t2).count();
            lastBuildStats.radixSortTime = std::chrono::duration<double>(t4 - t3).count();
            lastBuildStats.leafInitializationTime = std::chrono::duration<double>(t5 - t4).count();
            lastBuildStats.internalNodeConstructionTime = std::chrono::duration<double>(t6 - t5).count();
            lastBuildStats.bboxComputationTime = std::chrono::duration<double>(t7 - t6).count();
            lastBuildStats.totalTime = std::chrono::duration<double>(t7 - buildStart).count();
        } else {
            std::cout << "LBVH built: " << n << " triangles, " << nodes.size() 
                      << " nodes (node size: " << sizeof(LBVHNode) << " bytes)\n";
        }
    }
    
    int deltaNodeMC(const AlignedBuffer<std::pair<uint32_t, uint32_t>>& codes, int i, int j, int n) {
        if (j < 0 || j >= n) return -1;
        uint32_t codeI = codes[i].first;
        uint32_t codeJ = codes[j].first;
        if (codeI == codeJ) return 32 + clz(i ^ j);
        return clz(codeI ^ codeJ);
    }

    std::pair<int, int> determineRangeMC(const AlignedBuffer<std::pair<uint32_t, uint32_t>>& codes, int i, int n) {
        int d = (deltaNodeMC(codes, i, i + 1, n) - deltaNodeMC(codes, i, i - 1, n)) > 0 ? 1 : -1;
        int delta_min = deltaNodeMC(codes, i, i - d, n);
        int lmax = 2;
        while (deltaNodeMC(codes, i, i + lmax * d, n) > delta_min) lmax *= 2;
        int l = 0;
        for (int t = lmax / 2; t >= 1; t /= 2) {
            if (deltaNodeMC(codes, i, i + (l + t) * d, n) > delta_min) l += t;
        }
        int j = i + l * d;
        return d > 0 ? std::make_pair(i, j) : std::make_pair(j, i);
    }
    
    int findSplitMC(const AlignedBuffer<std::pair<uint32_t, uint32_t>>& codes, int first, int last) {
        uint32_t firstCode = codes[first].first;
        uint32_t lastCode = codes[last].first;
        int delta_node = clz(firstCode ^ lastCode);
        if (firstCode == lastCode) delta_node = 32 + clz(first ^ last);
        
        int s = 0;
        int t = last - first;
        while (t > 1) {
            t = (t + 1) >> 1;
            int mid = first + s + t;
            int delta_current;
            if (mid < 0 || mid >= codes.size()) delta_current = -1;
            else {
                uint32_t midCode = codes[mid].first;
                if (firstCode == midCode) delta_current = 32 + clz(first ^ mid);
                else delta_current = clz(firstCode ^ midCode);
            }
            if (delta_current > delta_node) s += t;
        }
        return first + s;
    }
    
    const std::vector<LBVHNode> getNodes() const { 
        const LBVHNode* p = nodes.data();
        return std::vector<LBVHNode>(p, p + nodes.size());
    }
    
    const LBVHNode* getNodesPtr() const { return nodes.data(); }
    size_t getNodesCount() const { return nodes.size(); }
    
    const BVHBuildStats& getLastBuildStats() const { return lastBuildStats; }
    const TrianglesSoA& getTriangles() const { return triangles; }

    void castRay(Ray& ray) {
        if (nodes.size() == 0) return;
        IntersectLBVHRecursive(ray, 0);
    }
    
    std::set<uint32_t> castRayWithTracking(Ray& ray) {
        std::set<uint32_t> visitedNodes;
        if (nodes.size() > 0) IntersectLBVHWithTracking(ray, 0, visitedNodes);
        return visitedNodes;
    }

private:
    void IntersectLBVHRecursive(Ray& ray, uint32_t nodeIdx) {
        const LBVHNode& node = nodes[nodeIdx]; 
        if (!IntersectAABBLBVH(ray, node.bbox)) return;
        if (node.isLeaf()) {
            uint32_t idx = node.leftChild & 0x7FFFFFFF;
            Vector3 v0(triangles.v0x[idx], triangles.v0y[idx], triangles.v0z[idx]);
            Vector3 v1(triangles.v1x[idx], triangles.v1y[idx], triangles.v1z[idx]);
            Vector3 v2(triangles.v2x[idx], triangles.v2y[idx], triangles.v2z[idx]);
            Triangle tri(v0, v1, v2);
            IntersectTri(ray, tri, idx);
        } else {
            if (node.leftChild != -1) IntersectLBVHRecursive(ray, node.leftChild);
            if (node.rightChild != -1) IntersectLBVHRecursive(ray, node.rightChild);
        }
    }
    
    void IntersectLBVHWithTracking(Ray& ray, uint32_t nodeIdx, std::set<uint32_t>& visitedNodes) {
        const LBVHNode& node = nodes[nodeIdx];
        if (!IntersectAABBLBVH(ray, node.bbox)) return;
        visitedNodes.insert(nodeIdx);
        if (node.isLeaf()) {
            uint32_t idx = node.leftChild & 0x7FFFFFFF;
            Vector3 v0(triangles.v0x[idx], triangles.v0y[idx], triangles.v0z[idx]);
            Vector3 v1(triangles.v1x[idx], triangles.v1y[idx], triangles.v1z[idx]);
            Vector3 v2(triangles.v2x[idx], triangles.v2y[idx], triangles.v2z[idx]);
            Triangle tri(v0, v1, v2);
            IntersectTri(ray, tri, idx);
        } else {
            if (node.leftChild != -1) IntersectLBVHWithTracking(ray, node.leftChild, visitedNodes);
            if (node.rightChild != -1) IntersectLBVHWithTracking(ray, node.rightChild, visitedNodes);
        }
    }
};

void IntersectLBVHIterative(const std::vector<LBVHNode>& nodes, const TrianglesSoA& triangles, Ray& ray) {
    if (nodes.empty()) return;
    uint32_t stack[128];
    int stackPtr = 0;
    stack[stackPtr++] = 0;
    while (stackPtr > 0) {
        uint32_t nodeIdx = stack[--stackPtr];
        if (nodeIdx >= nodes.size()) continue;
        const LBVHNode& node = nodes[nodeIdx];
        if (!IntersectAABBLBVH(ray, node.bbox)) continue;
        if (node.isLeaf()) {
            uint32_t idx = node.leftChild & 0x7FFFFFFF;
            Vector3 v0(triangles.v0x[idx], triangles.v0y[idx], triangles.v0z[idx]);
            Vector3 v1(triangles.v1x[idx], triangles.v1y[idx], triangles.v1z[idx]);
            Vector3 v2(triangles.v2x[idx], triangles.v2y[idx], triangles.v2z[idx]);
            Triangle tri(v0, v1, v2);
            IntersectTri(ray, tri, idx);
        } else {
            if (node.rightChild != -1) stack[stackPtr++] = node.rightChild;
            if (node.leftChild != -1) stack[stackPtr++] = node.leftChild;
        }
    }
}