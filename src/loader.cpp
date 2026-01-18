#include "../include/mesh.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>

// Load OBJ file into TriangleMesh
TriangleMesh loadOBJ(const std::string& filename) {
    TriangleMesh mesh;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open OBJ file: " + filename);
    }

    std::vector<float3_cw> vertices;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;
        
        if (prefix == "v") {
            // Vertex position
            float3_cw v;
            ss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        }
        else if (prefix == "f") {
            // Face (triangle)
            int indices[3];
            std::string token;
            
            for (int i = 0; i < 3; ++i) {
                ss >> token;
                size_t slash = token.find('/');
                std::string indexStr = (slash != std::string::npos) ? token.substr(0, slash) : token;
                indices[i] = std::stoi(indexStr) - 1; // OBJ indices are 1-based
            }
            
            // Validate indices
            bool valid = true;
            for (int i = 0; i < 3; ++i) {
                if (indices[i] < 0 || indices[i] >= (int)vertices.size()) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                mesh.addTriangle(vertices[indices[0]], 
                               vertices[indices[1]], 
                               vertices[indices[2]]);
            }
        }
    }
    
    file.close();
    
    std::cout << "Loaded " << mesh.size() << " triangles from " << filename << "\n";
    return mesh;
}

// Generate random triangles for testing
TriangleMesh generateRandomTriangles(int count) {
    TriangleMesh mesh;
    mesh.resize(count);
    
    // Initialize random seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    for (int i = 0; i < count; ++i) {
        // Random position in [0, 100] cube
        float x = (std::rand() % 1000) / 10.0f;
        float y = (std::rand() % 1000) / 10.0f;
        float z = (std::rand() % 1000) / 10.0f;
        
        // Create small triangle
        mesh.v0x[i] = x;       mesh.v0y[i] = y;       mesh.v0z[i] = z;
        mesh.v1x[i] = x + 1.0f; mesh.v1y[i] = y;       mesh.v1z[i] = z;
        mesh.v2x[i] = x;       mesh.v2y[i] = y + 1.0f; mesh.v2z[i] = z;
    }
    
    std::cout << "Generated " << count << " random triangles\n";
    return mesh;
}

// Load mesh from command-line arguments
TriangleMesh loadMesh(int argc, char** argv) {
    std::string inputFile;
    int numTriangles = 1000000; // Default: 1M triangles
    
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            inputFile = argv[++i];
        }
        else if ((arg == "-n" || arg == "--triangles") && i + 1 < argc) {
            numTriangles = std::atoi(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -i, --input <file>       Load OBJ file\n"
                      << "  -n, --triangles <count>  Generate N random triangles (default: 1000000)\n"
                      << "  -h, --help               Show this help\n";
            exit(0);
        }
    }
    
    // Load mesh
    if (!inputFile.empty()) {
        return loadOBJ(inputFile);
    } else {
        return generateRandomTriangles(numTriangles);
    }
}
