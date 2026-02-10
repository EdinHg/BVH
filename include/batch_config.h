#pragma once

#include <string>
#include <vector>
#include <map>

// Configuration for a single model to test
struct ModelConfig {
    std::string type;        // "obj" or "random"
    std::string path;        // Path to OBJ file (for type="obj")
    int triangles;           // Number of triangles (for type="random")
    std::string name;        // Display name (auto-generated if empty)
};

// Configuration for rendering in batch mode
struct BatchRenderConfig {
    bool enabled;
    std::string prefix;      // Output file prefix for renders
    std::string size;        // Format: "WIDTHxHEIGHT"
    std::string shading;     // "normal", "depth", "diffuse", "heatmap"
    std::string camera;      // Camera specification string
    std::string cameraUp;    // Camera up vector
    float fov;               // Field of view
    
    BatchRenderConfig() : enabled(false), size("1024x768"), shading("normal"), fov(0.0f) {}
};

// Complete batch configuration
struct BatchConfig {
    int iterations;          // Number of iterations per test
    bool warmup;             // Whether to run a warmup iteration
    std::string outputFile;  // CSV output file path
    std::vector<std::string> algorithms;  // Which algorithms to test: "lbvh", "lbvh+", "ploc", or "all"
    std::vector<int> plocRadius;          // PLOC radius values to test
    std::vector<ModelConfig> models;      // Models to test
    BatchRenderConfig render;             // Rendering configuration
    bool quiet;              // Suppress console output (except errors)
    
    BatchConfig() : iterations(5), warmup(true), quiet(false) {}
};

// Parse configuration from JSON file
BatchConfig loadBatchConfig(const std::string& filename);

// Validate configuration
bool validateBatchConfig(const BatchConfig& config, std::string& errorMsg);
