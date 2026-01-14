#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <string>

#include "mesh/obj_loader.hpp"
#include "bvh/bvh_builder.hpp"
#include "bvh/recursive_bvh.hpp"
#include "bvh/lbvh_builder.hpp"
#include "bvh/bvh_export.hpp"
#include "benchmark/timer.hpp"
#include "benchmark/sah_cost.hpp"

void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " <model.obj> [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  --lbvh           Use LBVH (Morton code) builder\n";
    std::cerr << "  --export         Export BVH bounding boxes to <model>_bvh.obj\n";
    std::cerr << "  --export-leaves  Export only leaf bounding boxes\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string modelPath = argv[1];
    bool doExport = false;
    bool leavesOnly = false;
    bool useLBVH = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--export") doExport = true;
        else if (arg == "--export-leaves") { doExport = true; leavesOnly = true; }
        else if (arg == "--lbvh") useLBVH = true;
    }

    // Load mesh
    TriangleMesh mesh;
    try {
        mesh = loadOBJ(modelPath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Loaded " << mesh.size() << " triangles from " << modelPath << "\n\n";

    if (mesh.size() == 0) {
        std::cerr << "Error: No triangles in mesh\n";
        return 1;
    }

    // Register builders
    std::vector<std::unique_ptr<BVHBuilder>> builders;
    if (useLBVH) {
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
