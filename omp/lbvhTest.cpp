#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <functional>

#include "IOzaTesting.cpp"
#include "lbvh.cpp"


void exportBVHToOBJ(const std::string& filename, 
                    const TrianglesSoA& triangles,
                    const LBVHNode* bvhNodes,
                    int nodeCount,
                    int maxDepth = -1) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    file << "# BVH Visualization\n";
    int vertexCount = 1;

    file << "# Original Triangles\n";
    for (size_t i = 0; i < triangles.size(); i++) {
        Vector3 v0(triangles.v0x[i], triangles.v0y[i], triangles.v0z[i]);
        Vector3 v1(triangles.v1x[i], triangles.v1y[i], triangles.v1z[i]);
        Vector3 v2(triangles.v2x[i], triangles.v2y[i], triangles.v2z[i]);
        
        file << "v " << v0.x << " " << v0.y << " " << v0.z << "\n";
        file << "v " << v1.x << " " << v1.y << " " << v1.z << "\n";
        file << "v " << v2.x << " " << v2.y << " " << v2.z << "\n";
        
        file << "f " << vertexCount << " " << (vertexCount + 1) << " " 
             << (vertexCount + 2) << "\n";
        vertexCount += 3;
    }

    // Export bounding boxes as wireframe cubes
    file << "\n# Bounding Boxes\n";
    
    std::function<void(int, int)> exportNode = [&](int nodeIdx, int depth) {
        if (nodeIdx < 0 || nodeIdx >= nodeCount) return;
        if (maxDepth != -1 && depth > maxDepth) return;

        const LBVHNode& node = bvhNodes[nodeIdx];
        const AABB& box = node.bbox;

        // 8 corners of the bounding box
        Vector3 corners[8] = {
            Vector3(box.min.x, box.min.y, box.min.z),
            Vector3(box.max.x, box.min.y, box.min.z),
            Vector3(box.max.x, box.max.y, box.min.z),
            Vector3(box.min.x, box.max.y, box.min.z),
            Vector3(box.min.x, box.min.y, box.max.z),
            Vector3(box.max.x, box.min.y, box.max.z),
            Vector3(box.max.x, box.max.y, box.max.z),
            Vector3(box.min.x, box.max.y, box.max.z),
        };

        int baseVertex = vertexCount;
        for (int i = 0; i < 8; i++) {
            file << "v " << corners[i].x << " " << corners[i].y << " " << corners[i].z << "\n";
        }
        vertexCount += 8;

        // Define the 12 edges of the box as lines
        int edges[12][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, // bottom face
            {4, 5}, {5, 6}, {6, 7}, {7, 4}, // top face
            {0, 4}, {1, 5}, {2, 6}, {3, 7}  // vertical edges
        };

        for (int i = 0; i < 12; i++) {
            file << "l " << (baseVertex + edges[i][0]) << " " 
                 << (baseVertex + edges[i][1]) << "\n";
        }

        // Recursively export child nodes
        if (!node.isLeaf()) {
            if (node.leftChild != -1) {
                exportNode(node.leftChild, depth + 1);
            }
            if (node.rightChild != -1) {
                exportNode(node.rightChild, depth + 1);
            }
        }
    };

    // Start export from root node (index 0)
    if (nodeCount > 0) {
        exportNode(0, 0);
    }

    file.close();
    std::cout << "Exported BVH to " << filename << " with " << vertexCount - 1 << " vertices\n";
}

void runBenchmark(const TrianglesSoA& triangles, int NUM_RUNS = 5, int maxThreads = -1) {
    // If maxThreads not specified, use system maximum
    if (maxThreads <= 0) {
        maxThreads = omp_get_max_threads();
    }
    
    // Build thread count list, capping at maxThreads
    std::vector<int> threadCounts = {1};
    for (int t : {2, 4, 8, 16, 32}) {
        if (t <= maxThreads && (threadCounts.empty() || t != threadCounts.back())) {
            threadCounts.push_back(t);
        }
    }
    // Always include the specified maximum
    if (threadCounts.back() != maxThreads) {
        threadCounts.push_back(maxThreads);
    }
    
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "=== BENCHMARK RESULTS (averaged over " << NUM_RUNS << " runs) ===\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << "Triangles: " << triangles.size() << "\n";
    std::cout << "Max threads available: " << omp_get_max_threads() << "\n";
    std::cout << "Testing up to: " << maxThreads << " threads\n\n";
    
    struct PhaseResult {
        std::string name;
        double time;
        double baselineTime;
    };
    
    struct ConfigResult {
        int threads;
        double totalTime;
        std::vector<PhaseResult> phases;
    };
    
    std::vector<ConfigResult> allResults;
    
    for (int numThreads : threadCounts) {
        omp_set_num_threads(numThreads);
        
        std::cout << "Testing with " << numThreads << " thread(s)... ";
        std::cout.flush();
        
        std::map<std::string, std::vector<double>> stageTimes;
        std::vector<double> totalTimes;
        
        LBVHBuilder builder; // Instantiate once to reuse buffers
        
        for (int run = 0; run < NUM_RUNS; run++) {
            TrianglesSoA triCopy = triangles;  // Changed from Triangle vector
            builder.buildBVH(triCopy, true);
            
            const BVHBuildStats& stats = builder.getLastBuildStats();
            
            stageTimes["Bounds computation"].push_back(stats.boundsComputationTime);
            stageTimes["Morton code computation"].push_back(stats.mortonCodeComputationTime);
            stageTimes["Radix sort"].push_back(stats.radixSortTime);
            stageTimes["Leaf initialization"].push_back(stats.leafInitializationTime);
            stageTimes["Internal node construction"].push_back(stats.internalNodeConstructionTime);
            stageTimes["BBox computation"].push_back(stats.bboxComputationTime);
            totalTimes.push_back(stats.totalTime);
        }
        
        ConfigResult result;
        result.threads = numThreads;
        
        // Calculate averages
        double avgTotal = 0.0;
        for (double t : totalTimes) avgTotal += t;
        result.totalTime = avgTotal / NUM_RUNS;
        
        for (const auto& [stage, times] : stageTimes) {
            double sum = 0.0;
            for (double t : times) sum += t;
            double avg = sum / NUM_RUNS;
            
            PhaseResult phase;
            phase.name = stage;
            phase.time = avg;
            phase.baselineTime = 0.0;
            result.phases.push_back(phase);
        }
        
        allResults.push_back(result);
        std::cout << "Done (avg: " << std::fixed << std::setprecision(4) << result.totalTime << "s)\n";
    }
    
    // Store baseline times
    double baselineTotal = allResults[0].totalTime;
    for (auto& phase : allResults[0].phases) {
        phase.baselineTime = phase.time;
    }
    for (size_t i = 1; i < allResults.size(); i++) {
        for (size_t p = 0; p < allResults[i].phases.size(); p++) {
            allResults[i].phases[p].baselineTime = allResults[0].phases[p].time;
        }
    }
    
    // Print total times
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "Total Construction Time:\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::setw(10) << "Threads" << std::setw(18) << "Time (s)" 
              << std::setw(18) << "Speedup" << std::setw(18) << "Efficiency\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (const auto& result : allResults) {
        double speedup = baselineTotal / result.totalTime;
        double efficiency = (speedup / result.threads) * 100.0;
        std::cout << std::setw(10) << result.threads
                  << std::setw(18) << std::fixed << std::setprecision(6) << result.totalTime
                  << std::setw(15) << std::setprecision(2) << speedup << "x"
                  << std::setw(16) << std::setprecision(1) << efficiency << "%\n";
    }
    
    // Print phase-by-phase breakdown
    std::cout << "\n\n" << std::string(100, '=') << "\n";
    std::cout << "Phase-by-Phase Breakdown:\n";
    std::cout << std::string(100, '=') << "\n";
    
    for (size_t phaseIdx = 0; phaseIdx < allResults[0].phases.size(); phaseIdx++) {
        std::cout << "\n" << allResults[0].phases[phaseIdx].name << ":\n";
        std::cout << std::string(90, '-') << "\n";
        std::cout << std::setw(10) << "Threads" << std::setw(15) << "Time (s)" 
                  << std::setw(15) << "% of Total" << std::setw(15) << "Speedup" 
                  << std::setw(15) << "Efficiency\n";
        std::cout << std::string(90, '-') << "\n";
        
        for (const auto& result : allResults) {
            const auto& phase = result.phases[phaseIdx];
            double speedup = phase.baselineTime / phase.time;
            double efficiency = (speedup / result.threads) * 100.0;
            double percentage = (phase.time / result.totalTime) * 100.0;
            
            std::cout << std::setw(10) << result.threads
                      << std::setw(15) << std::fixed << std::setprecision(6) << phase.time
                      << std::setw(14) << std::setprecision(1) << percentage << "%"
                      << std::setw(13) << std::setprecision(2) << speedup << "x"
                      << std::setw(14) << std::setprecision(1) << efficiency << "%\n";
        }
    }
    
    std::cout << "\n" << std::string(100, '=') << "\n\n";
}

int main(int argc, char* argv[]) {
    std::string inputFile;
    std::string outputFile = "output_bvh.obj";
    int randomCount = 0;
    bool verbose = false;
    bool benchmark = false;
    bool exportBVH = false;
    int threadCount = omp_get_max_threads();
    int benchmarkRuns = 5;
    int maxBenchmarkThreads = -1; // -1 means use system max

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            inputFile = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            exportBVH = true;
            outputFile = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            threadCount = std::atoi(argv[++i]);
        } else if (arg == "-r" && i + 1 < argc) {
            randomCount = std::atoi(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-b" || arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "-n" && i + 1 < argc) {
            benchmarkRuns = std::atoi(argv[++i]);
            if (benchmarkRuns < 1) benchmarkRuns = 1;
        } else if (arg == "-m" && i + 1 < argc) {
            maxBenchmarkThreads = std::atoi(argv[++i]);
            if (maxBenchmarkThreads < 1) maxBenchmarkThreads = -1;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "BVH Construction using Morton Codes\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  -i <file>      Input OBJ file\n";
            std::cout << "  -r <count>     Generate random triangles\n";
            std::cout << "  -o <file>      Output OBJ file (default: output_bvh.obj)\n";
            std::cout << "  -t <threads>   Number of threads for single run (default: max available)\n";
            std::cout << "  -v, --verbose  Show detailed timing for each phase\n";
            std::cout << "  -b, --benchmark Run benchmark mode (test multiple thread counts)\n";
            std::cout << "  -n <runs>      Number of runs per thread config in benchmark (default: 5)\n";
            std::cout << "  -m <threads>   Maximum threads to test in benchmark (default: system max)\n";
            std::cout << "  -h, --help     Show this help\n\n";
            std::cout << "Examples:\n";
            std::cout << "  " << argv[0] << " -i model.obj -o output.obj\n";
            std::cout << "  " << argv[0] << " -r 100000 -v\n";
            std::cout << "  " << argv[0] << " -r 1000000 -b -n 10\n";
            std::cout << "  " << argv[0] << " -r 1000000 -b -m 8 -n 5  (test up to 8 threads)\n";
            return 0;
        }
    }

    TrianglesSoA triangles;  // Changed from std::vector<Triangle>

    if (!inputFile.empty()) {
        triangles = OBJLoader::loadOBJSOA(inputFile);
    } else if (randomCount > 0) {
        triangles = generateRandomTrianglesSOA(randomCount);
    } else {
        std::cout << "No input specified. Generating 10 random triangles.\n";
        triangles = generateRandomTrianglesSOA(10);
    }

    if (triangles.size() == 0) {
        std::cerr << "No triangles to process!\n";
        return 1;
    }

    if (benchmark) {
        runBenchmark(triangles, benchmarkRuns, maxBenchmarkThreads);
    } else {
        omp_set_num_threads(threadCount);
        std::cout << "OpenMP enabled with " << threadCount << " threads\n";
        std::cout << "Building BVH using Morton codes...\n";
        
        LBVHBuilder bvhBuilder;
        auto start = std::chrono::high_resolution_clock::now();
        bvhBuilder.buildBVH(triangles, verbose);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> duration = end - start;
        std::cout << "BVH construction complete.\n";
        std::cout << "Construction time: " << duration.count() << " seconds\n";
        
        if (verbose) {
            bvhBuilder.getLastBuildStats().print();
        }
        
        if (exportBVH) {
            const std::vector<LBVHNode>& nodes = bvhBuilder.getNodes();
            exportBVHToOBJ(outputFile, 
                           triangles,  // Now TrianglesSoA
                           nodes.data(),
                           static_cast<int>(nodes.size()),
                           -1);
        }
    }

    return 0;
}