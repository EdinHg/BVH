#pragma once

#include "../math/vec3.hpp"
#include "../math/aabb.hpp"
#include <vector>

// Structure of Arrays for GPU/SIMD-friendly access
struct TriangleMesh {
    // Vertex 0
    std::vector<float> v0x, v0y, v0z;
    // Vertex 1
    std::vector<float> v1x, v1y, v1z;
    // Vertex 2
    std::vector<float> v2x, v2y, v2z;

    size_t size() const { return v0x.size(); }

    void reserve(size_t n) {
        v0x.reserve(n); v0y.reserve(n); v0z.reserve(n);
        v1x.reserve(n); v1y.reserve(n); v1z.reserve(n);
        v2x.reserve(n); v2y.reserve(n); v2z.reserve(n);
    }

    void addTriangle(const Vec3& a, const Vec3& b, const Vec3& c) {
        v0x.push_back(a.x); v0y.push_back(a.y); v0z.push_back(a.z);
        v1x.push_back(b.x); v1y.push_back(b.y); v1z.push_back(b.z);
        v2x.push_back(c.x); v2y.push_back(c.y); v2z.push_back(c.z);
    }

    Vec3 getVertex0(size_t i) const { return {v0x[i], v0y[i], v0z[i]}; }
    Vec3 getVertex1(size_t i) const { return {v1x[i], v1y[i], v1z[i]}; }
    Vec3 getVertex2(size_t i) const { return {v2x[i], v2y[i], v2z[i]}; }

    Vec3 getCentroid(size_t i) const {
        return (getVertex0(i) + getVertex1(i) + getVertex2(i)) / 3.0f;
    }

    AABB getBounds(size_t i) const {
        AABB box;
        box.expand(getVertex0(i));
        box.expand(getVertex1(i));
        box.expand(getVertex2(i));
        return box;
    }
};
