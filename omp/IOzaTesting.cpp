#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include "geometry.cpp"

class OBJLoader {
public:
    static TrianglesSoA loadOBJSOA(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return TrianglesSoA();
        }

        std::vector<Vector3> vertices;
        TrianglesSoA triangles;
        
        vertices.reserve(10000);
        triangles.reserve(10000);

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "v") {
                float x, y, z;
                iss >> x >> y >> z;
                vertices.emplace_back(x, y, z);
            } else if (prefix == "f") {
                std::vector<int> indices;
                std::string vertex;
                while (iss >> vertex) {
                    size_t slash = vertex.find('/');
                    int idx = std::stoi(slash != std::string::npos 
                                       ? vertex.substr(0, slash) 
                                       : vertex);
                    indices.push_back(idx - 1); 
                }

                if (indices.size() >= 3) {
                    for (size_t i = 1; i + 1 < indices.size(); i++) {
                        if (indices[0] >= 0 && indices[0] < vertices.size() &&
                            indices[i] >= 0 && indices[i] < vertices.size() &&
                            indices[i+1] >= 0 && indices[i+1] < vertices.size()) {
                            triangles.push_back(
                                vertices[indices[0]], 
                                vertices[indices[i]], 
                                vertices[indices[i+1]]
                            );
                        }
                    }
                }
            }
        }

        file.close();
        std::cout << "Loaded " << triangles.size() << " triangles from " << filename << "\n";
        return triangles;
    }

    static std::vector<Triangle> loadOBJ(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << "\n";
            return {};
        }
        
        std::vector<Vector3> vertices;
        std::vector<Triangle> triangles;
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;
            
            if (type == "v") {
                float x, y, z;
                iss >> x >> y >> z;
                vertices.push_back(Vector3(x, y, z));
            } else if (type == "f") {
                std::vector<int> indices;
                std::string vertex;
                
                while (iss >> vertex) {
                    size_t pos = vertex.find('/');
                    int idx = std::stoi(pos == std::string::npos ? 
                                       vertex : vertex.substr(0, pos));
                    indices.push_back(idx - 1); 
                }
                
                for (size_t i = 1; i + 1 < indices.size(); i++) {
                    triangles.emplace_back(
                        vertices[indices[0]],
                        vertices[indices[i]],
                        vertices[indices[i + 1]]
                    );
                }
            }
        }
        
        std::cout << "Loaded " << vertices.size() << " vertices and " 
                  << triangles.size() << " triangles\n";
        
        return triangles;
    }
};

 

inline TrianglesSoA generateRandomTrianglesSOA(int count) {
    TrianglesSoA triangles;
    triangles.reserve(count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);  

    for (int i = 0; i < count; i++) {
        Vector3 v0(dis(gen), dis(gen), dis(gen));
        Vector3 v1(dis(gen), dis(gen), dis(gen));
        Vector3 v2(dis(gen), dis(gen), dis(gen));
        triangles.push_back(v0, v1, v2);
    }

    std::cout << "Generated " << count << " random triangles\n";
    return triangles;
}

std::vector<Triangle> generateRandomTriangles(int count) {
    std::vector<Triangle> triangles;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dis(-100.0f, 100.0f);  
    std::uniform_real_distribution<float> size_dis(0.5f, 5.0f);      

    for (int i = 0; i < count; i++) {
        Triangle tri;
        float cx = pos_dis(gen);
        float cy = pos_dis(gen);
        float cz = pos_dis(gen);
        float size = size_dis(gen);

        tri.v0 = Vector3(cx, cy, cz);
        tri.v1 = Vector3(cx + size, cy, cz);
        tri.v2 = Vector3(cx, cy + size, cz);
        triangles.push_back(tri);
    }

    std::cout << "Generated " << count << " random triangles" << std::endl;
    return triangles;
}