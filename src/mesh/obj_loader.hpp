#pragma once

#include "triangle_mesh.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

inline TriangleMesh loadOBJ(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open OBJ file: " + path);
    }

    std::vector<Vec3> vertices;
    TriangleMesh mesh;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back({x, y, z});
        }
        else if (prefix == "f") {
            std::vector<int> faceIndices;
            std::string token;
            while (iss >> token) {
                // Handle formats: "v", "v/vt", "v/vt/vn", "v//vn"
                int idx = std::stoi(token.substr(0, token.find('/')));
                // OBJ indices are 1-based, negative means relative
                if (idx < 0) idx = static_cast<int>(vertices.size()) + idx + 1;
                faceIndices.push_back(idx - 1);
            }
            // Triangulate face (fan triangulation)
            for (size_t i = 1; i + 1 < faceIndices.size(); ++i) {
                mesh.addTriangle(
                    vertices[faceIndices[0]],
                    vertices[faceIndices[i]],
                    vertices[faceIndices[i + 1]]
                );
            }
        }
    }

    return mesh;
}
