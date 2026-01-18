#pragma once

#include "common.h"
#include <vector>
#include <cstddef>
#include <algorithm>

// Structure-of-Arrays triangle mesh for efficient GPU access
struct TriangleMesh {
    std::vector<float> v0x, v0y, v0z;
    std::vector<float> v1x, v1y, v1z;
    std::vector<float> v2x, v2y, v2z;

    size_t size() const { 
        return v0x.size(); 
    }
    
    void resize(size_t n) {
        v0x.resize(n); v0y.resize(n); v0z.resize(n);
        v1x.resize(n); v1y.resize(n); v1z.resize(n);
        v2x.resize(n); v2y.resize(n); v2z.resize(n);
    }
    
    void clear() {
        v0x.clear(); v0y.clear(); v0z.clear();
        v1x.clear(); v1y.clear(); v1z.clear();
        v2x.clear(); v2y.clear(); v2z.clear();
    }
    
    void addTriangle(const float3_cw& v0, const float3_cw& v1, const float3_cw& v2) {
        v0x.push_back(v0.x); v0y.push_back(v0.y); v0z.push_back(v0.z);
        v1x.push_back(v1.x); v1y.push_back(v1.y); v1z.push_back(v1.z);
        v2x.push_back(v2.x); v2y.push_back(v2.y); v2z.push_back(v2.z);
    }
    
    AABB_cw getTriangleBounds(size_t i) const {
        float3_cw v0(v0x[i], v0y[i], v0z[i]);
        float3_cw v1(v1x[i], v1y[i], v1z[i]);
        float3_cw v2(v2x[i], v2y[i], v2z[i]);
        
        AABB_cw bbox;
        bbox.min.x = std::min(v0.x, std::min(v1.x, v2.x));
        bbox.min.y = std::min(v0.y, std::min(v1.y, v2.y));
        bbox.min.z = std::min(v0.z, std::min(v1.z, v2.z));
        bbox.max.x = std::max(v0.x, std::max(v1.x, v2.x));
        bbox.max.y = std::max(v0.y, std::max(v1.y, v2.y));
        bbox.max.z = std::max(v0.z, std::max(v1.z, v2.z));
        return bbox;
    }
};

// Device-side pointers for kernel access
struct TrianglesSoADevice {
    float *v0x, *v0y, *v0z;
    float *v1x, *v1y, *v1z;
    float *v2x, *v2y, *v2z;
};
