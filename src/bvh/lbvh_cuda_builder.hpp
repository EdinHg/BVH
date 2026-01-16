#pragma once

#include "bvh_builder.hpp"

// Forward declaration - actual implementation in cuda/lbvh_cuda.cu
class LBVHBuilderCUDA : public BVHBuilder {
public:
    LBVHBuilderCUDA();
    ~LBVHBuilderCUDA() override;
    
    std::string name() const override { return "LBVH-CUDA"; }
    BVHResult build(const TriangleMesh& mesh) override;

private:
    class Impl;
    Impl* pImpl;
};
