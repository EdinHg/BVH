#include "../include/bvh_builder.h"
#include "../include/evaluator.h"
#include "../include/mesh.h"
#include "cuda/lbvh_builder.cuh"
// #include "cuda/ploc_builder.cuh"  // Uncomment when implemented

#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <string>

// Forward declarations from loader.cpp
TriangleMesh loadMesh(int argc, char** argv);

void printUsage(const char* programName) {
    std::cout << "BVH Construction Algorithm Comparison Tool\n\n";
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input <file>       Load OBJ file\n";
    std::cout << "  -n, --triangles <count>  Generate N random triangles (default: 1000000)\n";
    std::cout << "  -h, --help               Show this help\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << programName << " -i bunny.obj\n";
    std::cout << "  " << programName << " -n 10000000\n";
}

int main(int argc, char** argv) {
    // Check for help flag
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    std::cout << "========================================\n";
    std::cout << "  BVH Construction Benchmark\n";
    std::cout << "========================================\n\n";

    // 1. Load Mesh
    TriangleMesh mesh;
    try {
        mesh = loadMesh(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error loading mesh: " << e.what() << "\n";
        return 1;
    }

    if (mesh.size() == 0) {
        std::cerr << "Error: No triangles loaded\n";
        return 1;
    }

    std::cout << "\n";

    // 2. Register Algorithms
    std::vector<std::unique_ptr<BVHBuilder>> builders;
    builders.push_back(std::make_unique<LBVHBuilderCUDA>());
    // builders.push_back(std::make_unique<PLOCBuilderCUDA>()); // Uncomment when implemented

    // 3. Print table header
    std::cout << "┌─────────────┬─────────────────┬──────────────┬──────────────────────┐\n";
    std::cout << "│ Algorithm   │ Build Time (ms) │ SAH Cost     │ Throughput (MTris/s) │\n";
    std::cout << "├─────────────┼─────────────────┼──────────────┼──────────────────────┤\n";

    // 4. Unified Testing Loop
    for (auto& builder : builders) {
        try {
            // Optional warmup
            // builder->build(mesh);
            
            // Run benchmark
            BVHStats stats = BVHEvaluator::evaluate(builder.get(), mesh);
            
            float throughput = (mesh.size() / 1e6f) / (stats.buildTimeMs / 1000.0f);
            
            std::cout << "│ " << std::setw(11) << std::left << builder->getName() 
                      << " │ " << std::setw(15) << std::right << std::fixed << std::setprecision(3) << stats.buildTimeMs 
                      << " │ " << std::setw(12) << std::fixed << std::setprecision(2) << stats.sahCost 
                      << " │ " << std::setw(20) << std::fixed << std::setprecision(2) << throughput << " │\n";
            
            // Print detailed timing breakdown if available
            std::string breakdown = builder->getTimingBreakdown();
            if (!breakdown.empty()) {
                std::cout << "│             │ Breakdown:      │              │                      │\n";
                std::istringstream iss(breakdown);
                std::string line;
                while (std::getline(iss, line)) {
                    std::cout << "│             │ " << std::setw(15) << std::left << line;
                    // Pad to complete the row
                    int padding = 15 - line.length();
                    for (int i = 0; i < padding; ++i) std::cout << " ";
                    std::cout << " │              │                      │\n";
                }
            }
            
            std::cout << "├─────────────┼─────────────────┼──────────────┼──────────────────────┤\n";
            
        } catch (const std::exception& e) {
            std::cerr << "Error evaluating " << builder->getName() << ": " << e.what() << "\n";
            std::cout << "├─────────────┼─────────────────┼──────────────┼──────────────────────┤\n";
        }
    }
    
    std::cout << "└─────────────┴─────────────────┴──────────────┴──────────────────────┘\n\n";

    // 5. Print detailed statistics for each algorithm
    std::cout << "========================================\n";
    std::cout << "  Detailed Statistics\n";
    std::cout << "========================================\n";
    
    for (auto& builder : builders) {
        try {
            // Build again to get fresh stats (or cache from previous run)
            BVHStats stats = BVHEvaluator::evaluate(builder.get(), mesh);
            BVHEvaluator::printStats(builder->getName(), stats);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "  Benchmark Complete\n";
    std::cout << "========================================\n";

    return 0;
}
