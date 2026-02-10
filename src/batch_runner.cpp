#include "../include/batch_runner.h"
#include "../include/evaluator.h"
#include "cuda/lbvh_builder.cuh"
#include "cuda/lbvh_plus_builder.cuh"
#include "cuda/ploc_builder.cuh"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>

// Forward declarations from loader.cpp
TriangleMesh loadOBJ(const std::string& filename);
TriangleMesh generateRandomTriangles(int count);

BatchRunner::BatchRunner() = default;

TriangleMesh BatchRunner::loadModel(const ModelConfig& model) {
    if (model.type == "obj") {
        return loadOBJ(model.path);
    } else if (model.type == "random") {
        return generateRandomTriangles(model.triangles);
    }
    throw std::runtime_error("Unknown model type: " + model.type);
}

std::vector<BatchRunner::BuilderPair> BatchRunner::createBuilders(const BatchConfig& config) {
    std::vector<BuilderPair> builders;
    
    for (const auto& algo : config.algorithms) {
        if (algo == "lbvh") {
            builders.push_back({"LBVH", new LBVHBuilderCUDA()});
        } else if (algo == "lbvh+") {
            builders.push_back({"LBVH+", new LBVHPlusBuilderCUDA()});
        } else if (algo == "ploc") {
            for (int radius : config.plocRadius) {
                std::string name = "PLOC (r=" + std::to_string(radius) + ")";
                builders.push_back({name, new PLOCBuilderCUDA(radius)});
            }
        } else if (algo == "all") {
            builders.push_back({"LBVH", new LBVHBuilderCUDA()});
            builders.push_back({"LBVH+", new LBVHPlusBuilderCUDA()});
            for (int radius : config.plocRadius) {
                std::string name = "PLOC (r=" + std::to_string(radius) + ")";
                builders.push_back({name, new PLOCBuilderCUDA(radius)});
            }
        }
    }
    
    return builders;
}

void BatchRunner::testAlgorithm(BVHBuilder* builder,
                                 const TriangleMesh& mesh,
                                 const std::string& modelName,
                                 int modelTriangles,
                                 int iterations,
                                 bool warmup,
                                 bool quiet) {
    // Warmup iteration (not recorded)
    if (warmup) {
        try {
            builder->build(mesh);
            
            // Clean up GPU memory after warmup to free space before timed iterations
            auto* lbvh = dynamic_cast<LBVHBuilderCUDA*>(builder);
            if (lbvh) lbvh->cleanup();
            
            auto* lbvhPlus = dynamic_cast<LBVHPlusBuilderCUDA*>(builder);
            if (lbvhPlus) lbvhPlus->cleanup();
            
            auto* ploc = dynamic_cast<PLOCBuilderCUDA*>(builder);
            if (ploc) ploc->cleanup();
        } catch (...) {
            // Ignore warmup errors
        }
    }
    
    // Timed iterations
    for (int i = 1; i <= iterations; ++i) {
        TestResult result;
        result.modelName = modelName;
        result.modelTriangles = modelTriangles;
        result.algorithmName = builder->getName();
        result.iteration = i;
        result.success = false;
        
        try {
            // Build and evaluate
            BVHStats stats = BVHEvaluator::evaluate(builder, mesh);
            
            result.buildTimeMs = stats.buildTimeMs;
            result.sahCost = stats.sahCost;
            result.nodeCount = stats.nodeCount;
            result.leafCount = stats.leafCount;
            result.maxDepth = stats.maxDepth;
            result.avgLeafDepth = stats.avgLeafDepth;
            result.throughput = (mesh.size() / 1e6f) / (stats.buildTimeMs / 1000.0f);
            result.renderTimeMs = 0.0f;
            result.avgNodesVisited = 0.0f;
            result.success = true;
            
            if (!quiet) {
                printProgress(modelName, builder->getName(), i, iterations, true, "OK");
            }
        } catch (const std::exception& e) {
            result.errorMsg = e.what();
            result.success = false;
            
            if (!quiet) {
                printProgress(modelName, builder->getName(), i, iterations, false, "FAILED");
            }
        }
        
        results.push_back(result);
    }
}

void BatchRunner::printProgress(const std::string& modelName,
                                const std::string& algoName,
                                int iteration,
                                int totalIterations,
                                bool success,
                                const std::string& status) {
    // Progress bar visualization
    int barWidth = 20;
    float progress = static_cast<float>(iteration) / totalIterations;
    int filledWidth = static_cast<int>(progress * barWidth);
    
    std::cout << "  [" << std::setw(25) << std::left << algoName << "] ";
    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        std::cout << (i < filledWidth ? "█" : " ");
    }
    std::cout << "] " << std::setw(2) << std::right << iteration << "/" << totalIterations 
              << " [" << status << "]\n";
}

void BatchRunner::exportToCSV(const std::string& filename, const BatchConfig& config) {
    std::ofstream csv(filename);
    if (!csv.is_open()) {
        std::cerr << "Error: Could not open CSV file for writing: " << filename << "\n";
        return;
    }
    
    // Header
    csv << "Model,ModelType,TriangleCount,Algorithm,Iteration,BuildTimeMs,SAHCost,"
        << "NodeCount,LeafCount,MaxDepth,AvgLeafDepth,Throughput_MTris_s,"
        << "RenderTimeMs,AvgNodesVisited,Status\n";
    
    // Data rows
    for (const auto& result : results) {
        csv << result.modelName << ","
            << "mixed" << ","
            << result.modelTriangles << ","
            << result.algorithmName << ","
            << result.iteration << ","
            << std::fixed << std::setprecision(3) << result.buildTimeMs << ","
            << std::fixed << std::setprecision(2) << result.sahCost << ","
            << result.nodeCount << ","
            << result.leafCount << ","
            << result.maxDepth << ","
            << std::fixed << std::setprecision(2) << result.avgLeafDepth << ","
            << std::fixed << std::setprecision(2) << result.throughput << ","
            << (result.renderTimeMs > 0.0f ? std::to_string(result.renderTimeMs) : "N/A") << ","
            << (result.avgNodesVisited > 0.0f ? std::to_string(result.avgNodesVisited) : "N/A") << ","
            << (result.success ? "OK" : "FAILED") << "\n";
    }
    
    csv.close();
}

std::vector<AggregatedStats> BatchRunner::calculateAggregatedStats(const BatchConfig& config) {
    std::map<std::string, std::vector<TestResult>> groupedResults;
    
    // Group results by model + algorithm
    for (const auto& result : results) {
        if (result.success) {
            std::string key = result.modelName + "|" + result.algorithmName;
            groupedResults[key].push_back(result);
        }
    }
    
    std::vector<AggregatedStats> stats;
    
    for (const auto& [key, testResults] : groupedResults) {
        if (testResults.empty()) continue;
        
        AggregatedStats agg;
        agg.modelName = testResults[0].modelName;
        agg.modelTriangles = testResults[0].modelTriangles;
        agg.algorithmName = testResults[0].algorithmName;
        agg.numIterations = testResults.size();
        
        // Collect metrics
        std::vector<float> buildTimes, sahCosts, throughputs;
        for (const auto& res : testResults) {
            buildTimes.push_back(res.buildTimeMs);
            sahCosts.push_back(res.sahCost);
            throughputs.push_back(res.throughput);
        }
        
        // Calculate statistics
        auto calcStats = [](const std::vector<float>& values, float& mean, float& stdDev, float& minVal, float& maxVal) {
            if (values.empty()) return;
            mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
            minVal = *std::min_element(values.begin(), values.end());
            maxVal = *std::max_element(values.begin(), values.end());
            
            float sumSqDiff = 0.0f;
            for (float v : values) {
                sumSqDiff += (v - mean) * (v - mean);
            }
            stdDev = std::sqrt(sumSqDiff / values.size());
        };
        
        float buildStdDev;
        calcStats(buildTimes, agg.meanBuildTime, buildStdDev, agg.minBuildTime, agg.maxBuildTime);
        agg.stdDevBuildTime = buildStdDev;
        
        calcStats(sahCosts, agg.meanSAHCost, agg.stdDevSAHCost, agg.minBuildTime, agg.maxBuildTime);
        calcStats(throughputs, agg.meanThroughput, agg.stdDevThroughput, agg.minBuildTime, agg.maxBuildTime);
        
        // Tree structure stats (same across iterations)
        agg.nodeCount = testResults[0].nodeCount;
        agg.leafCount = testResults[0].leafCount;
        agg.maxDepth = testResults[0].maxDepth;
        agg.avgLeafDepth = testResults[0].avgLeafDepth;
        agg.hasRenderStats = testResults[0].renderTimeMs > 0.0f;
        
        if (agg.hasRenderStats) {
            std::vector<float> renderTimes, nodesVisited;
            for (const auto& res : testResults) {
                renderTimes.push_back(res.renderTimeMs);
                nodesVisited.push_back(res.avgNodesVisited);
            }
            float dummy1, dummy2, dummy3;
            calcStats(renderTimes, agg.meanRenderTime, dummy1, dummy2, dummy3);
            calcStats(nodesVisited, agg.meanNodesVisited, dummy1, dummy2, dummy3);
        }
        
        stats.push_back(agg);
    }
    
    return stats;
}

void BatchRunner::exportSummaryCSV(const std::string& filename, const BatchConfig& config) {
    auto aggStats = calculateAggregatedStats(config);
    
    std::string summaryFile = filename;
    size_t dotPos = summaryFile.find_last_of('.');
    if (dotPos != std::string::npos) {
        summaryFile.insert(dotPos, "_summary");
    } else {
        summaryFile += "_summary";
    }
    
    std::ofstream csv(summaryFile);
    if (!csv.is_open()) {
        std::cerr << "Error: Could not open summary CSV file for writing: " << summaryFile << "\n";
        return;
    }
    
    // Header
    csv << "Model,TriangleCount,Algorithm,Iterations,"
        << "MeanBuildTime_ms,StdDevBuildTime_ms,MinBuildTime_ms,MaxBuildTime_ms,"
        << "MeanSAHCost,StdDevSAHCost,"
        << "MeanThroughput_MTris_s,StdDevThroughput_MTris_s,"
        << "NodeCount,LeafCount,MaxDepth,AvgLeafDepth\n";
    
    // Data rows
    for (const auto& agg : aggStats) {
        csv << agg.modelName << ","
            << agg.modelTriangles << ","
            << agg.algorithmName << ","
            << agg.numIterations << ","
            << std::fixed << std::setprecision(3) << agg.meanBuildTime << ","
            << std::fixed << std::setprecision(3) << agg.stdDevBuildTime << ","
            << std::fixed << std::setprecision(3) << agg.minBuildTime << ","
            << std::fixed << std::setprecision(3) << agg.maxBuildTime << ","
            << std::fixed << std::setprecision(2) << agg.meanSAHCost << ","
            << std::fixed << std::setprecision(2) << agg.stdDevSAHCost << ","
            << std::fixed << std::setprecision(2) << agg.meanThroughput << ","
            << std::fixed << std::setprecision(2) << agg.stdDevThroughput << ","
            << agg.nodeCount << ","
            << agg.leafCount << ","
            << agg.maxDepth << ","
            << std::fixed << std::setprecision(2) << agg.avgLeafDepth << "\n";
    }
    
    csv.close();
}

void BatchRunner::run(const BatchConfig& config) {
    // Validate configuration
    std::string errorMsg;
    if (!validateBatchConfig(config, errorMsg)) {
        std::cerr << "Configuration error: " << errorMsg << "\n";
        return;
    }
    
    if (!config.quiet) {
        std::cout << "\n========================================\n";
        std::cout << "  BVH Batch Testing Mode\n";
        std::cout << "========================================\n\n";
    }
    
    // Create builders
    auto builders = createBuilders(config);
    
    // Calculate total tests
    int totalTests = config.models.size() * builders.size() * config.iterations;
    
    if (!config.quiet) {
        std::cout << "Configuration: " << config.outputFile << "\n";
        std::cout << "Models: " << config.models.size();
        std::cout << ", Algorithms: " << builders.size();
        std::cout << ", Iterations: " << config.iterations;
        if (config.warmup) std::cout << " (+ 1 warmup)";
        std::cout << "\n";
        std::cout << "Total tests: " << totalTests << "\n\n";
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    int completedTests = 0;
    
    // Test each model
    for (size_t modelIdx = 0; modelIdx < config.models.size(); ++modelIdx) {
        const auto& modelConfig = config.models[modelIdx];
        
        if (!config.quiet) {
            std::cout << "[Model " << (modelIdx + 1) << "/" << config.models.size() << "] "
                      << modelConfig.name;
        }
        
        TriangleMesh mesh;
        try {
            mesh = loadModel(modelConfig);
            if (!config.quiet) {
                std::cout << " (" << mesh.size() << " triangles)\n";
            }
        } catch (const std::exception& e) {
            std::cerr << " [ERROR: " << e.what() << "]\n";
            continue;
        }
        
        // Recreate builders for each model
        auto modelBuilders = createBuilders(config);
        
        // Test each algorithm
        for (size_t algoIdx = 0; algoIdx < modelBuilders.size(); ++algoIdx) {
            const auto& builderPair = modelBuilders[algoIdx];
            
            try {
                testAlgorithm(builderPair.builder, mesh, modelConfig.name, mesh.size(),
                              config.iterations, config.warmup, config.quiet);
                completedTests += config.iterations;
            } catch (const std::exception& e) {
                if (!config.quiet) {
                    std::cerr << "    Error testing " << builderPair.name << ": " << e.what() << "\n";
                }
            }
            
            // Clean up builder
            delete builderPair.builder;
            
            // Print ETA
            if (!config.quiet && completedTests < totalTests) {
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    currentTime - startTime).count() / 1000.0f;
                float rate = completedTests / elapsed;
                int remainingTests = totalTests - completedTests;
                float etaSeconds = remainingTests / rate;
                
                std::cout << "    ETA: " << std::fixed << std::setprecision(1) << etaSeconds << "s\n";
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime).count() / 1000.0f;
    
    // Export results
    exportToCSV(config.outputFile, config);
    exportSummaryCSV(config.outputFile, config);
    
    if (!config.quiet) {
        std::cout << "\n========================================\n";
        std::cout << "Batch complete! Results exported to:\n";
        std::cout << "  - " << config.outputFile << "\n";
        
        std::string summaryFile = config.outputFile;
        size_t dotPos = summaryFile.find_last_of('.');
        if (dotPos != std::string::npos) {
            summaryFile.insert(dotPos, "_summary");
        } else {
            summaryFile += "_summary";
        }
        std::cout << "  - " << summaryFile << "\n";
        
        std::cout << "Total time: " << std::fixed << std::setprecision(1) << totalTime << "s\n";
        std::cout << "========================================\n\n";
    }
}
