#pragma once

#include "batch_config.h"
#include "mesh.h"
#include "bvh_builder.h"
#include "render_engine.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

// Statistics for a single test run
struct TestResult {
    std::string modelName;
    int modelTriangles;
    std::string algorithmName;
    int iteration;
    float buildTimeMs;
    float sahCost;
    int nodeCount;
    int leafCount;
    int maxDepth;
    float avgLeafDepth;
    float throughput;
    float renderTimeMs;
    float avgNodesVisited;
    uint64_t totalRays;
    bool success;
    std::string errorMsg;
};

// Aggregated statistics across multiple iterations
struct AggregatedStats {
    std::string modelName;
    int modelTriangles;
    std::string algorithmName;
    int numIterations;
    
    // Build time stats
    float meanBuildTime;
    float stdDevBuildTime;
    float minBuildTime;
    float maxBuildTime;
    
    // SAH cost stats
    float meanSAHCost;
    float stdDevSAHCost;
    
    // Throughput stats
    float meanThroughput;
    float stdDevThroughput;
    
    // Tree quality (same across iterations)
    int nodeCount;
    int leafCount;
    int maxDepth;
    float avgLeafDepth;
    
    // Render stats (if available)
    bool hasRenderStats;
    float meanRenderTime;
    float meanNodesVisited;
    float meanRaysPerSec;  // Million rays per second
};

// Batch test runner with progress tracking and result export
class BatchRunner {
public:
    struct BuilderPair {
        std::string name;
        BVHBuilder* builder;
    };
    
    BatchRunner();
    
    // Main entry point: run batch tests according to config
    void run(const BatchConfig& config);
    
private:
    std::vector<TestResult> results;
    
    // Load a single model from config
    TriangleMesh loadModel(const ModelConfig& model);
    
     // Test a single builder on a mesh for multiple iterations
     void testAlgorithm(BVHBuilder* builder,
                        const TriangleMesh& mesh,
                        const std::string& modelName,
                        int modelTriangles,
                        int iterations,
                        bool warmup,
                        bool quiet,
                        const BatchRenderConfig& renderConfig);
    
    // Create all builders from algorithm list and PLOC radius values
    std::vector<BuilderPair> createBuilders(const BatchConfig& config);
    
    // Export all results to CSV
    void exportToCSV(const std::string& filename, const BatchConfig& config);
    
    // Export aggregated statistics to summary CSV
    void exportSummaryCSV(const std::string& filename, const BatchConfig& config);
    
    // Print progress line (minimal output)
    void printProgress(const std::string& modelName,
                       const std::string& algoName,
                       int iteration,
                       int totalIterations,
                       bool success,
                       const std::string& status);
    
    // Calculate statistics from results
    std::vector<AggregatedStats> calculateAggregatedStats(const BatchConfig& config);
};
