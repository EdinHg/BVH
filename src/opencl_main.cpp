// =============================================================================
// OpenCL LBVH Builder Benchmark
// =============================================================================
// Tests the OpenCL implementation of LBVH against the CPU version
// Target: AMD Ryzen 5700U APU (Vega 8)
// =============================================================================

#include "bvh/opencl_lbvh_builder.hpp"
#include "bvh/lbvh_builder.hpp"
#include "mesh/obj_loader.hpp"
#include "mesh/triangle_mesh.hpp"
#include "benchmark/sah_cost.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

// Simple timer class for benchmarking
class BenchTimer {
public:
    BenchTimer() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsedMs() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_);
        return duration.count() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Compute SAH cost wrapper
inline float computeSAHCost(const BVHResult& result) {
    return calculateSAHCost(result.nodes);
}

// Generate a test mesh with random triangles
TriangleMesh generateRandomMesh(size_t numTriangles, float sceneSize = 100.0f) {
    TriangleMesh mesh;
    mesh.reserve(numTriangles);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> posDist(-sceneSize/2, sceneSize/2);
    std::uniform_real_distribution<float> sizeDist(0.1f, 2.0f);

    for (size_t i = 0; i < numTriangles; ++i) {
        float cx = posDist(rng);
        float cy = posDist(rng);
        float cz = posDist(rng);
        float size = sizeDist(rng);

        Vec3 v0(cx, cy, cz);
        Vec3 v1(cx + size, cy, cz);
        Vec3 v2(cx + size * 0.5f, cy + size, cz + size * 0.5f);

        mesh.addTriangle(v0, v1, v2);
    }

    return mesh;
}

// Generate a test mesh with spatially coherent triangles (clustered)
TriangleMesh generateClusteredMesh(size_t numTriangles, size_t numClusters = 16) {
    TriangleMesh mesh;
    mesh.reserve(numTriangles);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> clusterDist(-50.0f, 50.0f);
    std::uniform_real_distribution<float> offsetDist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> sizeDist(0.1f, 0.5f);

    size_t trianglesPerCluster = numTriangles / numClusters;

    for (size_t c = 0; c < numClusters; ++c) {
        float clusterX = clusterDist(rng);
        float clusterY = clusterDist(rng);
        float clusterZ = clusterDist(rng);

        for (size_t i = 0; i < trianglesPerCluster; ++i) {
            float cx = clusterX + offsetDist(rng);
            float cy = clusterY + offsetDist(rng);
            float cz = clusterZ + offsetDist(rng);
            float size = sizeDist(rng);

            Vec3 v0(cx, cy, cz);
            Vec3 v1(cx + size, cy, cz);
            Vec3 v2(cx + size * 0.5f, cy + size, cz + size * 0.5f);

            mesh.addTriangle(v0, v1, v2);
        }
    }

    return mesh;
}

void runBenchmark(const std::string& name, const TriangleMesh& mesh) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Benchmark: " << name << std::endl;
    std::cout << "Triangles: " << mesh.size() << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // CPU LBVH Builder
    LBVHBuilder cpuBuilder;
    BVHResult cpuResult;
    double cpuTime = 0;

    {
        BenchTimer timer;
        cpuResult = cpuBuilder.build(mesh);
        cpuTime = timer.elapsedMs();
    }

    std::cout << "\nCPU LBVH Builder:" << std::endl;
    std::cout << "  Build time:    " << std::fixed << std::setprecision(2) 
              << cpuTime << " ms" << std::endl;
    std::cout << "  Nodes:         " << cpuResult.nodes.size() << std::endl;
    std::cout << "  SAH Cost:      " << std::setprecision(4) 
              << computeSAHCost(cpuResult) << std::endl;

    // OpenCL LBVH Builder
    try {
        OpenCLLBVHBuilder gpuBuilder;
        BVHResult gpuResult;
        double gpuTime = 0;

        // Warmup run
        gpuResult = gpuBuilder.build(mesh);

        // Timed run
        {
            BenchTimer timer;
            gpuResult = gpuBuilder.build(mesh);
            gpuTime = timer.elapsedMs();
        }

        std::cout << "\nOpenCL LBVH Builder (AMD APU):" << std::endl;
        std::cout << "  Build time:    " << std::fixed << std::setprecision(2) 
                  << gpuTime << " ms" << std::endl;
        std::cout << "  Nodes:         " << gpuResult.nodes.size() << std::endl;
        std::cout << "  SAH Cost:      " << std::setprecision(4) 
                  << computeSAHCost(gpuResult) << std::endl;

        // Speedup
        double speedup = cpuTime / gpuTime;
        std::cout << "\nSpeedup: " << std::setprecision(2) << speedup << "x" << std::endl;

        // Verify results
        bool valid = true;
        
        // Check node count
        if (cpuResult.nodes.size() != gpuResult.nodes.size()) {
            std::cout << "\nWARNING: Node count mismatch!" << std::endl;
            std::cout << "  CPU: " << cpuResult.nodes.size() << std::endl;
            std::cout << "  GPU: " << gpuResult.nodes.size() << std::endl;
            valid = false;
        }

        // Check root bounds (should be similar)
        if (valid && !gpuResult.nodes.empty() && !cpuResult.nodes.empty()) {
            const AABB& cpuRoot = cpuResult.nodes[0].bounds;
            const AABB& gpuRoot = gpuResult.nodes[0].bounds;
            
            float eps = 1e-3f;
            bool boundsMatch = 
                std::abs(cpuRoot.min.x - gpuRoot.min.x) < eps &&
                std::abs(cpuRoot.min.y - gpuRoot.min.y) < eps &&
                std::abs(cpuRoot.min.z - gpuRoot.min.z) < eps &&
                std::abs(cpuRoot.max.x - gpuRoot.max.x) < eps &&
                std::abs(cpuRoot.max.y - gpuRoot.max.y) < eps &&
                std::abs(cpuRoot.max.z - gpuRoot.max.z) < eps;

            if (!boundsMatch) {
                std::cout << "\nWARNING: Root bounds mismatch!" << std::endl;
                std::cout << "  CPU: [" << cpuRoot.min.x << "," << cpuRoot.min.y << "," << cpuRoot.min.z 
                          << "] - [" << cpuRoot.max.x << "," << cpuRoot.max.y << "," << cpuRoot.max.z << "]" << std::endl;
                std::cout << "  GPU: [" << gpuRoot.min.x << "," << gpuRoot.min.y << "," << gpuRoot.min.z 
                          << "] - [" << gpuRoot.max.x << "," << gpuRoot.max.y << "," << gpuRoot.max.z << "]" << std::endl;
                valid = false;
            }
        }

        if (valid) {
            std::cout << "\nValidation: PASSED" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "\nOpenCL Error: " << e.what() << std::endl;
    }
}

void runScalingTest() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "SCALING TEST" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::vector<size_t> sizes = {1000, 10000, 100000, 500000, 1000000, 10000000, 30000000};

    std::cout << std::setw(12) << "Triangles" 
              << std::setw(15) << "CPU (ms)"
              << std::setw(15) << "GPU (ms)"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(54, '-') << std::endl;

    OpenCLLBVHBuilder gpuBuilder;
    LBVHBuilder cpuBuilder;

    for (size_t size : sizes) {
        TriangleMesh mesh = generateClusteredMesh(size);

        // Warmup
        gpuBuilder.build(mesh);

        double cpuTime, gpuTime;

        {
            BenchTimer timer;
            cpuBuilder.build(mesh);
            cpuTime = timer.elapsedMs();
        }

        {
            BenchTimer timer;
            gpuBuilder.build(mesh);
            gpuTime = timer.elapsedMs();
        }

        std::cout << std::setw(12) << size 
                  << std::setw(15) << std::fixed << std::setprecision(2) << cpuTime
                  << std::setw(15) << gpuTime
                  << std::setw(12) << std::setprecision(2) << (cpuTime / gpuTime) << "x"
                  << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "OpenCL LBVH Builder Benchmark" << std::endl;
    std::cout << "Target: AMD Ryzen 5700U APU (Vega 8)" << std::endl;
    std::cout << std::endl;

    // Check if a model file was provided
    if (argc > 1) {
        std::string modelPath = argv[1];
        std::cout << "Loading model: " << modelPath << std::endl;

        try {
            TriangleMesh mesh = loadOBJ(modelPath);
            runBenchmark("Loaded Model", mesh);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load model: " << e.what() << std::endl;
        }
    } else {
        // Run synthetic benchmarks
        std::cout << "No model file provided. Running synthetic benchmarks." << std::endl;
        std::cout << "Usage: " << argv[0] << " [model.obj]" << std::endl;

        // // Small test
        // {
        //     TriangleMesh mesh = generateRandomMesh(10000);
        //     runBenchmark("Random 10K Triangles", mesh);
        // }

        // // Medium test
        // {
        //     TriangleMesh mesh = generateClusteredMesh(100000);
        //     runBenchmark("Clustered 100K Triangles", mesh);
        // }

        // Large test
        // {
        //     TriangleMesh mesh = generateClusteredMesh(30000000);
        //     runBenchmark("Clustered 30M Triangles", mesh);
        // }

        // Scaling test
        runScalingTest();
    }

    std::cout << "\nBenchmark complete." << std::endl;
    return 0;
}
