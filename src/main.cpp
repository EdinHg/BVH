#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <string>
#include <cstdlib>
#include <ctime>

#include "mesh/obj_loader.hpp"
#include "bvh/bvh_builder.hpp"
#include "bvh/recursive_bvh.hpp"
#include "bvh/lbvh_builder.hpp"
#include "bvh/bvh_export.hpp"
#include "benchmark/timer.hpp"
#include "benchmark/sah_cost.hpp"

#ifdef USE_CUDA
#include "bvh/lbvh_cuda_builder.hpp"
#endif

TriangleMesh generateRandomTriangles(int n) {
    TriangleMesh mesh;
    mesh.reserve(n);
    
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    for (int i = 0; i < n; ++i) {
        float x = (std::rand() % 1000) / 10.0f;
        float y = (std::rand() % 1000) / 10.0f;
        float z = (std::rand() % 1000) / 10.0f;
        
        mesh.addTriangle(
            Vec3(x, y, z),
            Vec3(x + 1.0f, y, z),
            Vec3(x, y + 1.0f, z)
        );
    }
    
    return mesh;
}

void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " <model.obj> [options]\n";
    std::cerr << "   or: " << program << " --random <count> [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  --random <n>     Generate n random triangles instead of loading OBJ\n";
    std::cerr << "  --lbvh           Use CPU LBVH (Morton code) builder\n";
#ifdef USE_CUDA
    std::cerr << "  --cuda           Use CUDA LBVH builder\n";
    std::cerr << "  --compare        Compare CPU LBVH vs CUDA LBVH\n";
#endif
    std::cerr << "  --export         Export BVH bounding boxes to <model>_bvh.obj\n";
    std::cerr << "  --export-leaves  Export only leaf bounding boxes\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string modelPath;
    bool doExport = false;
    bool leavesOnly = false;
    bool useLBVH = false;
    bool useCUDA = false;
    bool compareMode = false;
    int randomTriangles = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--export") doExport = true;
        else if (arg == "--export-leaves") { doExport = true; leavesOnly = true; }
        else if (arg == "--lbvh") useLBVH = true;
        else if (arg == "--random" && i + 1 < argc) {
            randomTriangles = std::atoi(argv[++i]);
        }
#ifdef USE_CUDA
        else if (arg == "--cuda") useCUDA = true;
        else if (arg == "--compare") compareMode = true;
#endif
        else if (arg[0] != '-' && modelPath.empty()) {
            modelPath = arg;
        }
    }

    // Load or generate mesh
    TriangleMesh mesh;
    if (randomTriangles > 0) {
        std::cout << "Generating " << randomTriangles << " random triangles...\n";
        mesh = generateRandomTriangles(randomTriangles);
        std::cout << "Generated " << mesh.size() << " triangles\n\n";
    } else if (!modelPath.empty()) {
        try {
            mesh = loadOBJ(modelPath);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
        std::cout << "Loaded " << mesh.size() << " triangles from " << modelPath << "\n\n";
    } else {
        printUsage(argv[0]);
        return 1;
    }

    if (mesh.size() == 0) {
        std::cerr << "Error: No triangles in mesh\n";
        return 1;
    }

    // Register builders
    std::vector<std::unique_ptr<BVHBuilder>> builders;
    
    if (compareMode) {
        // Compare mode: run both CPU and CUDA LBVH
        builders.push_back(std::make_unique<LBVHBuilder>());
#ifdef USE_CUDA
        builders.push_back(std::make_unique<LBVHBuilderCUDA>());
#endif
    } else if (useCUDA) {
#ifdef USE_CUDA
        builders.push_back(std::make_unique<LBVHBuilderCUDA>());
#else
        std::cerr << "Error: CUDA support not compiled in\n";
        return 1;
#endif
    } else if (useLBVH) {
        builders.push_back(std::make_unique<LBVHBuilder>());
    } else {
        builders.push_back(std::make_unique<RecursiveBVH>(4));
    }

    // Print header
    std::cout << std::left << std::setw(20) << "Builder"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(12) << "SAH Cost"
              << std::setw(10) << "Nodes"
              << "\n";
    std::cout << std::string(54, '-') << "\n";

    // Run each builder
    for (auto& builder : builders) {
        Timer timer;

        timer.start();
        BVHResult result = builder->build(mesh);
        timer.stop();

        float sahCost = calculateSAHCost(result.nodes);

        std::cout << std::left << std::setw(20) << builder->name()
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << timer.elapsedMs()
                  << std::setw(12) << std::fixed << std::setprecision(2) << sahCost
                  << std::setw(10) << result.nodes.size()
                  << "\n";

        // Export BVH if requested
        if (doExport) {
            std::string exportPath = modelPath;
            size_t dotPos = exportPath.rfind('.');
            if (dotPos != std::string::npos) {
                exportPath = exportPath.substr(0, dotPos);
            }
            exportPath += "_bvh.obj";

            try {
                exportBVHToOBJ(exportPath, result.nodes, leavesOnly);
                std::cout << "Exported BVH to: " << exportPath << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Export error: " << e.what() << "\n";
            }
        }
    }

    return 0;
}
