#pragma once

#include <glad/glad.h>
#include "../mesh/triangle_mesh.hpp"
#include "../math/vec3.hpp"
#include <vector>

class MeshRenderer {
public:
    GLuint vao = 0, vbo = 0;
    size_t vertexCount = 0;

    void upload(const TriangleMesh& mesh, const std::vector<Vec3>& colors) {
        // Interleaved: pos(3) + color(3) per vertex, 3 vertices per triangle
        size_t numTris = mesh.size();
        vertexCount = numTris * 3;

        std::vector<float> data;
        data.reserve(vertexCount * 6);

        for (size_t i = 0; i < numTris; ++i) {
            Vec3 v0 = mesh.getVertex0(i);
            Vec3 v1 = mesh.getVertex1(i);
            Vec3 v2 = mesh.getVertex2(i);
            Vec3 c = colors[i];

            // Vertex 0
            data.push_back(v0.x); data.push_back(v0.y); data.push_back(v0.z);
            data.push_back(c.x);  data.push_back(c.y);  data.push_back(c.z);
            // Vertex 1
            data.push_back(v1.x); data.push_back(v1.y); data.push_back(v1.z);
            data.push_back(c.x);  data.push_back(c.y);  data.push_back(c.z);
            // Vertex 2
            data.push_back(v2.x); data.push_back(v2.y); data.push_back(v2.z);
            data.push_back(c.x);  data.push_back(c.y);  data.push_back(c.z);
        }

        if (!vao) glGenVertexArrays(1, &vao);
        if (!vbo) glGenBuffers(1, &vbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);

        // Position attribute (location 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // Color attribute (location 1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    void draw() const {
        if (vao && vertexCount > 0) {
            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertexCount));
            glBindVertexArray(0);
        }
    }

    ~MeshRenderer() {
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
    }
};
