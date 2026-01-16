#pragma once
#include <cmath>
#include <algorithm>
#include <vector>

struct Vector3 {
    float x, y, z;
    
    Vector3() : x(0), y(0), z(0) {}
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
    
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    
    Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }
    
    Vector3 operator/(float scalar) const {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }
    
    float dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    Vector3 cross(const Vector3& other) const {
        return Vector3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    Vector3 normalize() const {
        float len = length();
        return len > 0 ? (*this) / len : Vector3(0, 0, 0);
    }
};

struct AABB {
    Vector3 min;
    Vector3 max;
    
    AABB() : min(Vector3(INFINITY, INFINITY, INFINITY)), 
             max(Vector3(-INFINITY, -INFINITY, -INFINITY)) {}
    
    AABB(const Vector3& min, const Vector3& max) : min(min), max(max) {}
    
    void expand(const Vector3& point) {
        min.x = std::min(min.x, point.x);
        min.y = std::min(min.y, point.y);
        min.z = std::min(min.z, point.z);
        max.x = std::max(max.x, point.x);
        max.y = std::max(max.y, point.y);
        max.z = std::max(max.z, point.z);
    }
    
    void expand(const AABB& other) {
        expand(other.min);
        expand(other.max);
    }
    
    Vector3 center() const {
        return (min + max) * 0.5f;
    }
    
    Vector3 size() const {
        return max - min;
    }
};

struct TrianglesSoA {
    std::vector<float> v0x, v0y, v0z;
    std::vector<float> v1x, v1y, v1z;
    std::vector<float> v2x, v2y, v2z;
    
    size_t size() const { return v0x.size(); }
    
    void resize(size_t n) {
        v0x.resize(n); v0y.resize(n); v0z.resize(n);
        v1x.resize(n); v1y.resize(n); v1z.resize(n);
        v2x.resize(n); v2y.resize(n); v2z.resize(n);
    }
    
    void reserve(size_t n) {
        v0x.reserve(n); v0y.reserve(n); v0z.reserve(n);
        v1x.reserve(n); v1y.reserve(n); v1z.reserve(n);
        v2x.reserve(n); v2y.reserve(n); v2z.reserve(n);
    }
    
    void push_back(const Vector3& a, const Vector3& b, const Vector3& c) {
        v0x.push_back(a.x); v0y.push_back(a.y); v0z.push_back(a.z);
        v1x.push_back(b.x); v1y.push_back(b.y); v1z.push_back(b.z);
        v2x.push_back(c.x); v2y.push_back(c.y); v2z.push_back(c.z);
    }
    
    void getTriangle(size_t i, Vector3& a, Vector3& b, Vector3& c) const {
        a = Vector3(v0x[i], v0y[i], v0z[i]);
        b = Vector3(v1x[i], v1y[i], v1z[i]);
        c = Vector3(v2x[i], v2y[i], v2z[i]);
    }
};

struct Triangle {
    Vector3 v0, v1, v2;
    
    Triangle() {}
    Triangle(const Vector3& a, const Vector3& b, const Vector3& c) 
        : v0(a), v1(b), v2(c) {}
};

inline TrianglesSoA convertToSoA(const std::vector<Triangle>& triangles) {
    TrianglesSoA soa;
    soa.reserve(triangles.size());
    for (const auto& tri : triangles) {
        soa.push_back(tri.v0, tri.v1, tri.v2);
    }
    return soa;
}

inline std::vector<Triangle> convertToAoS(const TrianglesSoA& soa) {
    std::vector<Triangle> triangles;
    triangles.reserve(soa.size());
    for (size_t i = 0; i < soa.size(); i++) {
        Vector3 v0, v1, v2;
        soa.getTriangle(i, v0, v1, v2);
        triangles.emplace_back(v0, v1, v2);
    }
    return triangles;
}


