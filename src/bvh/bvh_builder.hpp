#pragma once

#include "bvh_node.hpp"
#include "../mesh/triangle_mesh.hpp"
#include <string>

class BVHBuilder {
public:
    virtual ~BVHBuilder() = default;
    virtual std::string name() const = 0;
    virtual BVHResult build(const TriangleMesh& mesh) = 0;
};
