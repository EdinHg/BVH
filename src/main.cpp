#include "../include/bvh_builder.h"
#include "../include/evaluator.h"
#include "../include/mesh.h"
#include "../include/render_engine.h"
#include "cuda/lbvh_builder.cuh"
#include "cuda/lbvh_plus_builder.cuh"
#include "cuda/ploc_builder.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <cctype>

// Forward declarations from loader.cpp
TriangleMesh loadMesh(int argc, char** argv);

// Export helpers
struct VizNode {
    float min[3];
    float max[3];
    int leftIdx;
    int rightIdx;
};

void exportBVHToBinary(const std::string& filename, const std::vector<BVHNode>& nodes) {
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
    std::cout << "Exported " << exportNodes.size() << " nodes to " << filename
              << " (" << (exportNodes.size() * sizeof(VizNode))/1024/1024 << " MB)\n";
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
        bool nodeIsLeaf = (i >= (nodes.size() + 1) / 2 - 1);
        if (leavesOnly && !nodeIsLeaf) continue;
        writeBox(nodes[i].bbox);
    }
}

// Sanitize algorithm name for use as a filename component
static std::string sanitizeName(const std::string& name) {
    std::string result;
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '+') {
            result += c;
        } else if (c == ' ' || c == '(' || c == ')' || c == '=') {
            result += '_';
        }
    }
    // Remove trailing underscores
    while (!result.empty() && result.back() == '_') result.pop_back();
    return result;
}

void printUsage(const char* programName) {
    std::cout << "BVH Construction Algorithm Comparison Tool\n\n";
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input <file>        Load OBJ file\n";
    std::cout << "  -n, --triangles <count>   Generate N random triangles (default: 1000000)\n";
    std::cout << "  -a, --algorithm <name>    Run specific algorithm (lbvh, ploc, all)\n";
    std::cout << "  -o, --output <file>       Export BVH to file\n";
    std::cout << "  -c, --colab-export        Export as binary (for Colab visualization)\n";
    std::cout << "  -l, --leaves-only         Export only leaf bounding boxes\n";
    std::cout << "  -r, --radius <value>      Set search radius for PLOC (default: 25)\n";
    std::cout << "  -h, --help                Show this help\n\n";
    std::cout << "Render options:\n";
    std::cout << "  --render <prefix>         Render each BVH to <prefix>_<algo>.ppm\n";
    std::cout << "  --render-size <WxH>       Render resolution (default: 1024x768)\n";
    std::cout << "  --shading <mode>          normal|depth|diffuse|heatmap (default: normal)\n";
    std::cout << "  --camera <ex,ey,ez,lx,ly,lz>  Camera eye + look-at (default: auto-fit)\n";
    std::cout << "  --camera-up <ux,uy,uz>    Camera up vector (default: 0,1,0)\n";
    std::cout << "  --fov <degrees>           Vertical FOV (default: 60)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " -i bunny.obj\n";
    std::cout << "  " << programName << " -n 10000000 -a lbvh\n";
    std::cout << "  " << programName << " -n 1000000 -o bvh.bin -c\n";
    std::cout << "  " << programName << " -i bunny.obj --render output --shading heatmap\n";
    std::cout << "  " << programName << " -i bunny.obj --render img --camera 0,1,3,0,0,0 --fov 45\n";
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    std::string selectedAlgo = "all";
    std::string outputFile;
    bool colabExport = false;
    bool leavesOnly = false;
    int radius = 0; // 0 means default/not set

    // Render options
    std::string renderPrefix;
    int renderWidth = 1024, renderHeight = 768;
    std::string shadingStr = "normal";
    std::string cameraStr;
    std::string cameraUpStr;
    float fov = 0.0f; // 0 means use default (60)

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if ((arg == "-a" || arg == "--algorithm") && i + 1 < argc) {
            selectedAlgo = argv[++i];
        }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputFile = argv[++i];
        }
        else if (arg == "-c" || arg == "--colab-export") {
            colabExport = true;
        }
        else if (arg == "-l" || arg == "--leaves-only") {
            leavesOnly = true;
        }
        else if ((arg == "-r" || arg == "--radius") && i + 1 < argc) {
            radius = std::stoi(argv[++i]);
        }
        else if (arg == "--render" && i + 1 < argc) {
            renderPrefix = argv[++i];
        }
        else if (arg == "--render-size" && i + 1 < argc) {
            std::string sizeStr = argv[++i];
            // Parse WxH format
            size_t xpos = sizeStr.find('x');
            if (xpos == std::string::npos) xpos = sizeStr.find('X');
            if (xpos != std::string::npos) {
                renderWidth  = std::stoi(sizeStr.substr(0, xpos));
                renderHeight = std::stoi(sizeStr.substr(xpos + 1));
            } else {
                std::cerr << "Warning: --render-size expects WxH format. Using default.\n";
            }
        }
        else if (arg == "--shading" && i + 1 < argc) {
            shadingStr = argv[++i];
        }
        else if (arg == "--camera" && i + 1 < argc) {
            cameraStr = argv[++i];
        }
        else if (arg == "--camera-up" && i + 1 < argc) {
            cameraUpStr = argv[++i];
        }
        else if (arg == "--fov" && i + 1 < argc) {
            fov = std::stof(argv[++i]);
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
    
    if (selectedAlgo == "all") {
        builders.push_back(std::make_unique<LBVHBuilderCUDA>());
        builders.push_back(std::make_unique<LBVHPlusBuilderCUDA>());
        builders.push_back(std::make_unique<PLOCBuilderCUDA>(10));
        builders.push_back(std::make_unique<PLOCBuilderCUDA>(25));
        builders.push_back(std::make_unique<PLOCBuilderCUDA>(100));
    }
    else if (selectedAlgo == "lbvh") {
        builders.push_back(std::make_unique<LBVHBuilderCUDA>());
    }
    else if (selectedAlgo == "lbvh+") {
        builders.push_back(std::make_unique<LBVHPlusBuilderCUDA>());
    }
    else if (selectedAlgo == "ploc") {
        if (radius > 0) {
            builders.push_back(std::make_unique<PLOCBuilderCUDA>(radius));
        } else {
             // If manual run without radius, simulate benchmark mode or default? 
             // "utilize 3 different values... during benchmark"
             // I'll add all 3 for consistency if no radius is specified, so user sees the comparison.
            builders.push_back(std::make_unique<PLOCBuilderCUDA>(10));
            builders.push_back(std::make_unique<PLOCBuilderCUDA>(25));
            builders.push_back(std::make_unique<PLOCBuilderCUDA>(100));
        }
    }
    else {
        std::cerr << "Unknown algorithm: " << selectedAlgo << "\n";
        std::cerr << "Available: lbvh, lbvh+, ploc, all\n";
        return 1;
    }

    // 3. Print table header
    std::cout << "┌──────────────────┬─────────────────────────────────────────────────────────┬──────────────┬──────────────────────────┐\n";
    std::cout << "│ Algorithm        │ Build Time (ms)                                         │ SAH Cost     │ Throughput (MTris/s)     │\n";
    std::cout << "├──────────────────┼─────────────────────────────────────────────────────────┼──────────────┼──────────────────────────┤\n";

    // 4. Unified Testing Loop
    for (auto& builder : builders) {
        try {
            // Optional warmup
            // builder->build(mesh);
            
            // Run benchmark
            BVHStats stats = BVHEvaluator::evaluate(builder.get(), mesh);
            
            float throughput = (mesh.size() / 1e6f) / (stats.buildTimeMs / 1000.0f);
            
            std::cout << "│ " << std::setw(16) << std::left << builder->getName() 
                      << " │ " << std::setw(55) << std::right << std::fixed << std::setprecision(3) << stats.buildTimeMs 
                      << " │ " << std::setw(12) << std::fixed << std::setprecision(2) << stats.sahCost 
                      << " │ " << std::setw(24) << std::fixed << std::setprecision(2) << throughput << " │\n";
            
            // Print detailed timing breakdown if available
            std::string breakdown = builder->getTimingBreakdown();
            if (!breakdown.empty()) {
                std::cout << "│                  │ Breakdown:                                              │              │                          │\n";
                std::istringstream iss(breakdown);
                std::string line;
                while (std::getline(iss, line)) {
                    std::cout << "│                  │ " << line;
                    // Pad to complete the row (column width is 55)
                    int padding = 55 - static_cast<int>(line.length());
                    for (int i = 0; i < padding; ++i) std::cout << " ";
                    std::cout << " │              │                          │\n";
                }
            }
            
            std::cout << "├──────────────────┼─────────────────────────────────────────────────────────┼──────────────┼──────────────────────────┤\n";
            
        } catch (const std::exception& e) {
            std::cerr << "Error evaluating " << builder->getName() << ": " << e.what() << "\n";
            std::cout << "├──────────────────┼─────────────────────────────────────────────────────────┼──────────────┼──────────────────────────┤\n";
        }
    }
    
    std::cout << "└──────────────────┴─────────────────────────────────────────────────────────┴──────────────┴──────────────────────────┘\n\n";

    // 4. Export BVH if requested
    if (!outputFile.empty() && !builders.empty()) {
        // Default to the first builder
        BVHBuilder* exporter = builders[0].get();

        // Try to find PLOC r=25 specifically as requested
        for (auto& b : builders) {
            if (b->getName() == "PLOC CUDA (r=25)") {
                exporter = b.get();
                break;
            }
        }

        std::cout << "Exporting BVH from " << exporter->getName() << "...\n";
        const auto& nodes = exporter->getNodes();
        
        if (colabExport) {
            std::cout << "Exporting to binary format: " << outputFile << "\n";
            exportBVHToBinary(outputFile, nodes);
        } else {
            std::cout << "Exporting to OBJ format: " << outputFile << "\n";
            exportBVHToOBJ(outputFile, nodes, leavesOnly);
            std::cout << "Exported " << nodes.size() << " nodes\n";
        }
    }

    // 5. Render images if requested
    if (!renderPrefix.empty() && !builders.empty()) {
        std::cout << "\n========================================\n";
        std::cout << "  Rendering\n";
        std::cout << "========================================\n";

        ShadingMode shadingMode = parseShadingMode(shadingStr);

        // Use the first builder's BVH for auto-camera computation
        Camera camera = parseCameraString(cameraStr, cameraUpStr, fov,
                                          builders[0]->getNodes());

        // ---- NEW: Compute global heatmap max if using heatmap mode ----
        float globalHeatmapMax = 0.0f;
        if (shadingMode == ShadingMode::HEATMAP) {
            std::cout << "Pre-rendering to compute global heatmap normalization...\n";
            for (auto& builder : builders) {
                const auto& nodes = builder->getNodes();
                if (nodes.empty()) continue;

                RenderStats preStats = renderImage(
                    nodes, mesh, renderWidth, renderHeight,
                    camera, shadingMode, "", 0.0f // Empty output, auto-normalize
                );
                globalHeatmapMax = std::max(globalHeatmapMax, preStats.maxNodesVisited);
            }
            std::cout << "Global heatmap max (p99.9): " << globalHeatmapMax << "\n\n";
        }

        std::cout << "Camera: eye=("
                  << camera.eye.x << ", " << camera.eye.y << ", " << camera.eye.z
                  << ") lookAt=("
                  << camera.lookAt.x << ", " << camera.lookAt.y << ", " << camera.lookAt.z
                  << ") fov=" << camera.fovY << "\n";
        std::cout << "Shading: " << shadingStr << "  Resolution: "
                  << renderWidth << "x" << renderHeight << "\n\n";

        for (auto& builder : builders) {
            try {
                const auto& nodes = builder->getNodes();
                if (nodes.empty()) {
                    std::cerr << "Skipping render for " << builder->getName()
                              << ": no BVH nodes\n";
                    continue;
                }

                std::string sanitized = sanitizeName(builder->getName());
                std::string outFile = renderPrefix + "_" + sanitized + ".ppm";

                RenderStats rStats = renderImage(
                    nodes, mesh,
                    renderWidth, renderHeight,
                    camera, shadingMode, outFile,
                    globalHeatmapMax  // ← Use fixed max
                );

                printRenderStats(builder->getName(), rStats);
                std::cout << "\n";

            } catch (const std::exception& e) {
                std::cerr << "Error rendering " << builder->getName()
                          << ": " << e.what() << "\n";
            }
        }
    }

    // 6. Print detailed statistics for each algorithm
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
