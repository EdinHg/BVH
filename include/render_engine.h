#pragma once

#include "common.h"
#include "bvh_node.h"
#include "mesh.h"

#include <string>
#include <vector>
#include <cstdint>

// ============================================================================
// Shading Modes
// ============================================================================
enum class ShadingMode {
    NORMAL,     // Surface normal mapped to RGB
    DEPTH,      // Grayscale depth buffer
    DIFFUSE,    // Simple directional light + ambient
    HEATMAP     // BVH traversal cost visualization (nodes visited per ray)
};

// ============================================================================
// Camera
// ============================================================================
struct Camera {
    float3_cw eye;
    float3_cw lookAt;
    float3_cw up;
    float fovY;     // Vertical field of view in degrees

    Camera() : eye(0, 0, 5), lookAt(0, 0, 0), up(0, 1, 0), fovY(60.0f) {}
    Camera(float3_cw e, float3_cw l, float3_cw u, float fov)
        : eye(e), lookAt(l), up(u), fovY(fov) {}
};

// ============================================================================
// Render Statistics
// ============================================================================
struct RenderStats {
    float renderTimeMs;
    int width;
    int height;
    uint64_t totalRays;
    float avgNodesVisited;
    float avgAABBTests;
    float avgTriTests;
    float maxNodesVisited;
};

// ============================================================================
// Functions
// ============================================================================

// Auto-compute a camera that frames the entire scene from a 3/4 angle view.
// Uses the root node's AABB to determine scene bounds.
Camera autoCamera(const std::vector<BVHNode>& nodes);

// Parse camera from a CLI string: "ex,ey,ez,lx,ly,lz"
// Falls back to autoCamera if the string is empty or malformed.
Camera parseCameraString(const std::string& cameraStr,
                         const std::string& upStr,
                         float fov,
                         const std::vector<BVHNode>& nodes);

// Parse shading mode from string. Returns NORMAL for unrecognized input.
ShadingMode parseShadingMode(const std::string& str);

// Render an image of the scene using GPU-accelerated BVH ray tracing.
// Writes a PPM image to outputFile and returns traversal statistics.
RenderStats renderImage(const std::vector<BVHNode>& nodes,
                        const TriangleMesh& mesh,
                        int width, int height,
                        const Camera& camera,
                        ShadingMode mode,
                        const std::string& outputFile,
                        float fixedHeatmapMax = 0.0f);  // ← NEW: 0 = auto

// Print render statistics to stdout in a formatted block.
void printRenderStats(const std::string& algorithmName, const RenderStats& stats);
