#pragma once

#include "vec3.hpp"
#include <limits>

struct AABB {
    Vec3 min, max;

    AABB() : min(std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()),
             max(std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest()) {}

    AABB(const Vec3& min, const Vec3& max) : min(min), max(max) {}

    void expand(const Vec3& p) {
        min = Vec3::min(min, p);
        max = Vec3::max(max, p);
    }

    void expand(const AABB& other) {
        min = Vec3::min(min, other.min);
        max = Vec3::max(max, other.max);
    }

    Vec3 extent() const { return max - min; }
    Vec3 center() const { return (min + max) * 0.5f; }

    float surfaceArea() const {
        Vec3 d = extent();
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }

    int longestAxis() const {
        Vec3 d = extent();
        if (d.x >= d.y && d.x >= d.z) return 0;
        if (d.y >= d.z) return 1;
        return 2;
    }

    bool valid() const {
        return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }
};
