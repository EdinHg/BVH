// ============================================================================
// GPU Ray Tracing Render Engine for BVH Quality Comparison
// ============================================================================
//
// Implements a CUDA ray tracer that traverses the unified BVHNode structure
// and supports multiple shading modes for visual comparison of BVH quality.
//
// Key shading mode for BVH comparison: HEATMAP -- colors pixels by the number
// of BVH nodes visited per ray, directly visualizing traversal efficiency.
// ============================================================================

#include "../include/render_engine.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <numeric>

// ============================================================================
// Constants
// ============================================================================

#define BVH_STACK_SIZE 64
#define RENDER_BLOCK_SIZE 16
#define EPSILON 1e-6f

// Background color (dark gray)
#define BG_R 0.15f
#define BG_G 0.15f
#define BG_B 0.18f

// ============================================================================
// Device Vector Math Helpers
// ============================================================================

__device__ __host__ inline float3_cw d_cross(const float3_cw& a, const float3_cw& b) {
    return float3_cw(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __host__ inline float d_dot(const float3_cw& a, const float3_cw& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float d_length(const float3_cw& v) {
    return sqrtf(d_dot(v, v));
}

__device__ __host__ inline float3_cw d_normalize(const float3_cw& v) {
    float len = d_length(v);
    if (len < 1e-12f) return float3_cw(0, 0, 0);
    float inv = 1.0f / len;
    return float3_cw(v.x * inv, v.y * inv, v.z * inv);
}

__device__ __host__ inline float3_cw d_negate(const float3_cw& v) {
    return float3_cw(-v.x, -v.y, -v.z);
}

// ============================================================================
// Device Ray Structure
// ============================================================================

struct Ray {
    float3_cw origin;
    float3_cw direction;
    float3_cw invDirection; // Precomputed 1/dir for slab AABB test
};

// ============================================================================
// Device: Ray Generation (Pinhole Camera)
// ============================================================================

__device__ Ray generateRay(const Camera& cam, int x, int y, int width, int height) {
    // Build orthonormal camera basis
    float3_cw forward = d_normalize(cam.lookAt - cam.eye);
    float3_cw right   = d_normalize(d_cross(forward, cam.up));
    float3_cw camUp   = d_cross(right, forward);

    // Image plane dimensions from FOV
    float fovRad = cam.fovY * 3.14159265358979f / 180.0f;
    float halfH  = tanf(fovRad * 0.5f);
    float aspect = (float)width / (float)height;
    float halfW  = halfH * aspect;

    // Normalized device coordinates [-1, 1] with 0.5 pixel center offset
    float u = (2.0f * (x + 0.5f) / (float)width  - 1.0f) * halfW;
    float v = (2.0f * (y + 0.5f) / (float)height - 1.0f) * halfH;

    // Ray direction: forward + u*right - v*up (v negated so y points up)
    float3_cw dir;
    dir.x = forward.x + u * right.x - v * camUp.x;
    dir.y = forward.y + u * right.y - v * camUp.y;
    dir.z = forward.z + u * right.z - v * camUp.z;
    dir = d_normalize(dir);

    Ray ray;
    ray.origin    = cam.eye;
    ray.direction = dir;
    ray.invDirection = float3_cw(
        fabsf(dir.x) > EPSILON ? 1.0f / dir.x : (dir.x >= 0 ? 1e30f : -1e30f),
        fabsf(dir.y) > EPSILON ? 1.0f / dir.y : (dir.y >= 0 ? 1e30f : -1e30f),
        fabsf(dir.z) > EPSILON ? 1.0f / dir.z : (dir.z >= 0 ? 1e30f : -1e30f)
    );

    return ray;
}

// ============================================================================
// Device: Ray-AABB Intersection (Slab Method)
// ============================================================================

__device__ bool intersectAABB(const Ray& ray, const AABB_cw& bbox, float tmax) {
    float tx1 = (bbox.min.x - ray.origin.x) * ray.invDirection.x;
    float tx2 = (bbox.max.x - ray.origin.x) * ray.invDirection.x;
    float tmin_x = fminf(tx1, tx2);
    float tmax_x = fmaxf(tx1, tx2);

    float ty1 = (bbox.min.y - ray.origin.y) * ray.invDirection.y;
    float ty2 = (bbox.max.y - ray.origin.y) * ray.invDirection.y;
    float tmin_y = fminf(ty1, ty2);
    float tmax_y = fmaxf(ty1, ty2);

    float tz1 = (bbox.min.z - ray.origin.z) * ray.invDirection.z;
    float tz2 = (bbox.max.z - ray.origin.z) * ray.invDirection.z;
    float tmin_z = fminf(tz1, tz2);
    float tmax_z = fmaxf(tz1, tz2);

    float tenter = fmaxf(fmaxf(tmin_x, tmin_y), tmin_z);
    float texit  = fminf(fminf(tmax_x, tmax_y), tmax_z);

    return texit >= fmaxf(tenter, 0.0f) && tenter < tmax;
}

// ============================================================================
// Device: Ray-Triangle Intersection (Moller-Trumbore)
// ============================================================================

__device__ bool intersectTriangle(const Ray& ray,
                                  float v0x, float v0y, float v0z,
                                  float v1x, float v1y, float v1z,
                                  float v2x, float v2y, float v2z,
                                  float& out_t,
                                  float3_cw& out_normal) {
    // Edge vectors
    float e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    float e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

    // P = dir x e2
    float px = ray.direction.y * e2z - ray.direction.z * e2y;
    float py = ray.direction.z * e2x - ray.direction.x * e2z;
    float pz = ray.direction.x * e2y - ray.direction.y * e2x;

    float det = e1x * px + e1y * py + e1z * pz;
    if (fabsf(det) < EPSILON) return false;

    float inv_det = 1.0f / det;

    // T = origin - v0
    float tx = ray.origin.x - v0x;
    float ty = ray.origin.y - v0y;
    float tz = ray.origin.z - v0z;

    // Barycentric u
    float u = (tx * px + ty * py + tz * pz) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    // Q = T x e1
    float qx = ty * e1z - tz * e1y;
    float qy = tz * e1x - tx * e1z;
    float qz = tx * e1y - ty * e1x;

    // Barycentric v
    float v = (ray.direction.x * qx + ray.direction.y * qy + ray.direction.z * qz) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    // Intersection distance
    float t = (e2x * qx + e2y * qy + e2z * qz) * inv_det;
    if (t < EPSILON) return false;

    out_t = t;

    // Geometric normal (unnormalized cross product of edges, then normalize)
    out_normal.x = e1y * e2z - e1z * e2y;
    out_normal.y = e1z * e2x - e1x * e2z;
    out_normal.z = e1x * e2y - e1y * e2x;
    out_normal = d_normalize(out_normal);

    return true;
}

// ============================================================================
// Device: Iterative BVH Traversal with Explicit Stack
// ============================================================================

struct HitInfo {
    float t;
    float3_cw normal;
    uint32_t primIdx;
    bool hit;
    int nodesVisited;
    int aabbTests;
    int triTests;
};

__device__ HitInfo traverseBVH(const Ray& ray,
                               const BVHNode* nodes,
                               int numNodes,
                               const TrianglesSoADevice& tris) {
    HitInfo result;
    result.t = 1e30f;
    result.hit = false;
    result.nodesVisited = 0;
    result.aabbTests = 0;
    result.triTests = 0;

    // Explicit stack for iterative traversal
    int stack[BVH_STACK_SIZE];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // Push root node index

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];

        if (nodeIdx < 0 || nodeIdx >= numNodes) continue;

        const BVHNode& node = nodes[nodeIdx];
        result.nodesVisited++;

        // Test ray against node AABB
        result.aabbTests++;
        if (!intersectAABB(ray, node.bbox, result.t)) {
            continue;
        }

        if (node.isLeaf()) {
            // Leaf node: test the triangle primitive.
            // getPrimitiveIndex() returns the ORIGINAL triangle index
            // directly usable as an index into the mesh SoA arrays.
            uint32_t primIdx = node.getPrimitiveIndex();
            result.triTests++;

            float t;
            float3_cw normal;
            if (intersectTriangle(ray,
                    tris.v0x[primIdx], tris.v0y[primIdx], tris.v0z[primIdx],
                    tris.v1x[primIdx], tris.v1y[primIdx], tris.v1z[primIdx],
                    tris.v2x[primIdx], tris.v2y[primIdx], tris.v2z[primIdx],
                    t, normal)) {
                if (t < result.t) {
                    result.t = t;
                    result.normal = normal;
                    result.primIdx = primIdx;
                    result.hit = true;
                }
            }
        } else {
            // Internal node: push both children onto the stack.
            // Push right first so left is processed first (LIFO).
            int left  = node.leftChild;
            int right = node.rightChild;

            if (right >= 0 && right < numNodes && stackPtr < BVH_STACK_SIZE) {
                stack[stackPtr++] = right;
            }
            if (left >= 0 && left < numNodes && stackPtr < BVH_STACK_SIZE) {
                stack[stackPtr++] = left;
            }
        }
    }

    return result;
}

// ============================================================================
// Render Kernel
// ============================================================================

__global__ void kRenderKernel(
    const BVHNode* nodes,
    int numNodes,
    TrianglesSoADevice tris,
    Camera camera,
    int width,
    int height,
    int shadingMode,    // Cast from ShadingMode enum
    float sceneDiag,    // Scene AABB diagonal (for depth normalization)
    float3_cw lightDir, // Directional light direction (normalized)
    float* outputRGB,   // [width * height * 3]
    int* nodeVisitCounts,  // [width * height]
    int* aabbTestCounts,   // [width * height]
    int* triTestCounts)    // [width * height]
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIdx = y * width + x;

    // Generate primary ray for this pixel
    Ray ray = generateRay(camera, x, y, width, height);

    // Traverse BVH to find closest intersection
    HitInfo hit = traverseBVH(ray, nodes, numNodes, tris);

    // Store per-pixel traversal statistics
    nodeVisitCounts[pixelIdx] = hit.nodesVisited;
    aabbTestCounts[pixelIdx]  = hit.aabbTests;
    triTestCounts[pixelIdx]   = hit.triTests;

    // Shade pixel based on selected mode
    float r, g, b;

    if (!hit.hit) {
        // Miss: background color
        r = BG_R; g = BG_G; b = BG_B;
    } else {
        switch (shadingMode) {
            case 0: { // NORMAL -- map surface normal to RGB
                r = (hit.normal.x + 1.0f) * 0.5f;
                g = (hit.normal.y + 1.0f) * 0.5f;
                b = (hit.normal.z + 1.0f) * 0.5f;
                break;
            }
            case 1: { // DEPTH -- grayscale, closer = brighter
                float depth = hit.t / sceneDiag;
                depth = fmaxf(0.0f, fminf(1.0f, depth));
                float val = 1.0f - depth;
                r = val; g = val; b = val;
                break;
            }
            case 2: { // DIFFUSE -- directional light + ambient
                float3_cw n = hit.normal;
                // Flip normal to face the camera if needed
                if (d_dot(n, ray.direction) > 0.0f) {
                    n = d_negate(n);
                }
                float diffuse = fmaxf(0.0f, d_dot(n, lightDir));
                float ambient = 0.15f;
                float shade = fminf(1.0f, ambient + diffuse * 0.85f);
                // Subtle material color variation based on normal
                r = shade * (0.7f + 0.3f * fabsf(n.x));
                g = shade * (0.7f + 0.3f * fabsf(n.y));
                b = shade * (0.7f + 0.3f * fabsf(n.z));
                break;
            }
            case 3: { // HEATMAP -- store raw traversal count; CPU applies colormap
                float val = (float)hit.nodesVisited;
                r = val; g = val; b = val;
                break;
            }
            default: {
                r = 1.0f; g = 0.0f; b = 1.0f; // Magenta = error
                break;
            }
        }
    }

    outputRGB[pixelIdx * 3 + 0] = r;
    outputRGB[pixelIdx * 3 + 1] = g;
    outputRGB[pixelIdx * 3 + 2] = b;
}

// ============================================================================
// Host: Upload triangle mesh to device (SoA)
// ============================================================================

static TrianglesSoADevice uploadMesh(const TriangleMesh& mesh) {
    TrianglesSoADevice d_tris;
    size_t n = mesh.size();
    size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_tris.v0x, bytes));
    CUDA_CHECK(cudaMalloc(&d_tris.v0y, bytes));
    CUDA_CHECK(cudaMalloc(&d_tris.v0z, bytes));
    CUDA_CHECK(cudaMalloc(&d_tris.v1x, bytes));
    CUDA_CHECK(cudaMalloc(&d_tris.v1y, bytes));
    CUDA_CHECK(cudaMalloc(&d_tris.v1z, bytes));
    CUDA_CHECK(cudaMalloc(&d_tris.v2x, bytes));
    CUDA_CHECK(cudaMalloc(&d_tris.v2y, bytes));
    CUDA_CHECK(cudaMalloc(&d_tris.v2z, bytes));

    CUDA_CHECK(cudaMemcpy(d_tris.v0x, mesh.v0x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tris.v0y, mesh.v0y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tris.v0z, mesh.v0z.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tris.v1x, mesh.v1x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tris.v1y, mesh.v1y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tris.v1z, mesh.v1z.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tris.v2x, mesh.v2x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tris.v2y, mesh.v2y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tris.v2z, mesh.v2z.data(), bytes, cudaMemcpyHostToDevice));

    return d_tris;
}

static void freeMeshDevice(TrianglesSoADevice& d_tris) {
    cudaFree(d_tris.v0x); cudaFree(d_tris.v0y); cudaFree(d_tris.v0z);
    cudaFree(d_tris.v1x); cudaFree(d_tris.v1y); cudaFree(d_tris.v1z);
    cudaFree(d_tris.v2x); cudaFree(d_tris.v2y); cudaFree(d_tris.v2z);
}

// ============================================================================
// Host: Save image as PPM (P6 binary)
// ============================================================================

static bool savePPM(const std::string& filename,
                    const float* rgb, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filename << " for writing\n";
        return false;
    }

    // PPM P6 header
    file << "P6\n" << width << " " << height << "\n255\n";

    // Convert float [0,1] to uint8 [0,255]
    std::vector<uint8_t> pixels(width * height * 3);
    for (int i = 0; i < width * height * 3; ++i) {
        float v = rgb[i];
        v = std::max(0.0f, std::min(1.0f, v));
        pixels[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }

    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    return true;
}

// ============================================================================
// Host: Apply heatmap colormap (CPU post-process)
// ============================================================================

static void applyHeatmapColormap(float* rgb, const int* nodeVisitCounts,
                                 int width, int height) {
    int numPixels = width * height;

    // Collect non-zero (hit) counts for adaptive normalization
    std::vector<int> hit_counts;
    hit_counts.reserve(numPixels);
    for (int i = 0; i < numPixels; ++i) {
        if (nodeVisitCounts[i] > 0) {
            hit_counts.push_back(nodeVisitCounts[i]);
        }
    }

    // Use 99th percentile for normalization to avoid outlier skew
    float maxVal = 128.0f; // Fallback
    if (!hit_counts.empty()) {
        std::sort(hit_counts.begin(), hit_counts.end());
        size_t p99_idx = std::min(hit_counts.size() - 1,
                                   (size_t)(hit_counts.size() * 0.99));
        maxVal = std::max(1.0f, (float)hit_counts[p99_idx]);
    }

    // Apply blue-cyan-green-yellow-red colormap
    for (int i = 0; i < numPixels; ++i) {
        if (nodeVisitCounts[i] == 0) {
            // Background/miss pixel
            rgb[i * 3 + 0] = BG_R;
            rgb[i * 3 + 1] = BG_G;
            rgb[i * 3 + 2] = BG_B;
        } else {
            float t = (float)nodeVisitCounts[i] / maxVal;
            t = std::max(0.0f, std::min(1.0f, t));

            float r, g, b;
            if (t < 0.25f) {
                float s = t / 0.25f;
                r = 0.0f; g = s; b = 1.0f;
            } else if (t < 0.5f) {
                float s = (t - 0.25f) / 0.25f;
                r = 0.0f; g = 1.0f; b = 1.0f - s;
            } else if (t < 0.75f) {
                float s = (t - 0.5f) / 0.25f;
                r = s; g = 1.0f; b = 0.0f;
            } else {
                float s = (t - 0.75f) / 0.25f;
                r = 1.0f; g = 1.0f - s; b = 0.0f;
            }

            rgb[i * 3 + 0] = r;
            rgb[i * 3 + 1] = g;
            rgb[i * 3 + 2] = b;
        }
    }
}

// ============================================================================
// Host: Auto-Camera from Scene AABB
// ============================================================================

Camera autoCamera(const std::vector<BVHNode>& nodes) {
    Camera cam;
    cam.fovY = 60.0f;
    cam.up = float3_cw(0, 1, 0);

    if (nodes.empty()) {
        cam.eye    = float3_cw(0, 0, 5);
        cam.lookAt = float3_cw(0, 0, 0);
        return cam;
    }

    // Scene bounds from root node AABB
    const AABB_cw& bounds = nodes[0].bbox;
    float3_cw center;
    center.x = (bounds.min.x + bounds.max.x) * 0.5f;
    center.y = (bounds.min.y + bounds.max.y) * 0.5f;
    center.z = (bounds.min.z + bounds.max.z) * 0.5f;

    float3_cw extent;
    extent.x = bounds.max.x - bounds.min.x;
    extent.y = bounds.max.y - bounds.min.y;
    extent.z = bounds.max.z - bounds.min.z;

    float diagonal = d_length(extent);
    if (diagonal < 1e-6f) diagonal = 2.0f;

    // Place camera to frame the entire scene at a 3/4 view angle
    float fovRad = cam.fovY * 3.14159265358979f / 180.0f;
    float dist = (diagonal * 0.5f) / tanf(fovRad * 0.5f) * 1.2f;

    cam.eye.x = center.x + dist * 0.6f;
    cam.eye.y = center.y + dist * 0.3f;
    cam.eye.z = center.z + dist * 0.7f;
    cam.lookAt = center;

    return cam;
}

// ============================================================================
// Host: Parse Camera String
// ============================================================================

static std::vector<float> parseFloatList(const std::string& str) {
    std::vector<float> values;
    std::string current;
    for (char c : str) {
        if (c == ',') {
            if (!current.empty()) {
                values.push_back(std::stof(current));
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        values.push_back(std::stof(current));
    }
    return values;
}

Camera parseCameraString(const std::string& cameraStr,
                         const std::string& upStr,
                         float fov,
                         const std::vector<BVHNode>& nodes) {
    Camera cam = autoCamera(nodes);

    if (!cameraStr.empty()) {
        auto vals = parseFloatList(cameraStr);
        if (vals.size() >= 6) {
            cam.eye    = float3_cw(vals[0], vals[1], vals[2]);
            cam.lookAt = float3_cw(vals[3], vals[4], vals[5]);
        } else {
            std::cerr << "Warning: --camera expects 6 values (ex,ey,ez,lx,ly,lz). "
                      << "Using auto-camera.\n";
        }
    }

    if (!upStr.empty()) {
        auto vals = parseFloatList(upStr);
        if (vals.size() >= 3) {
            cam.up = float3_cw(vals[0], vals[1], vals[2]);
        }
    }

    if (fov > 0.0f) {
        cam.fovY = fov;
    }

    return cam;
}

// ============================================================================
// Host: Parse Shading Mode
// ============================================================================

ShadingMode parseShadingMode(const std::string& str) {
    if (str == "normal")  return ShadingMode::NORMAL;
    if (str == "depth")   return ShadingMode::DEPTH;
    if (str == "diffuse") return ShadingMode::DIFFUSE;
    if (str == "heatmap") return ShadingMode::HEATMAP;
    std::cerr << "Warning: Unknown shading mode '" << str
              << "'. Using 'normal'.\n";
    return ShadingMode::NORMAL;
}

// ============================================================================
// Host: Main Render Function
// ============================================================================

RenderStats renderImage(const std::vector<BVHNode>& nodes,
                        const TriangleMesh& mesh,
                        int width, int height,
                        const Camera& camera,
                        ShadingMode mode,
                        const std::string& outputFile) {
    RenderStats stats = {};
    stats.width  = width;
    stats.height = height;
    stats.totalRays = (uint64_t)width * height;

    if (nodes.empty() || mesh.size() == 0) {
        std::cerr << "Error: Empty BVH or mesh, cannot render\n";
        return stats;
    }

    int numPixels = width * height;
    int numNodes  = (int)nodes.size();

    // ---- Compute scene diagonal for depth normalization ----
    const AABB_cw& bounds = nodes[0].bbox;
    float3_cw extent;
    extent.x = bounds.max.x - bounds.min.x;
    extent.y = bounds.max.y - bounds.min.y;
    extent.z = bounds.max.z - bounds.min.z;
    float sceneDiag = d_length(extent);
    if (sceneDiag < 1e-6f) sceneDiag = 1.0f;

    // ---- Fixed directional light from upper-right-front ----
    float3_cw lightDir = d_normalize(float3_cw(0.5f, 0.8f, 0.6f));

    // ---- Allocate and upload BVH nodes ----
    BVHNode* d_nodes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nodes, numNodes * sizeof(BVHNode)));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), numNodes * sizeof(BVHNode),
                          cudaMemcpyHostToDevice));

    // ---- Upload triangle mesh ----
    TrianglesSoADevice d_tris = uploadMesh(mesh);

    // ---- Allocate output buffers ----
    float* d_rgb = nullptr;
    int*   d_nodeVisits = nullptr;
    int*   d_aabbTests  = nullptr;
    int*   d_triTests   = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rgb,        numPixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nodeVisits, numPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_aabbTests,  numPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_triTests,   numPixels * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_rgb,        0, numPixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_nodeVisits, 0, numPixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_aabbTests,  0, numPixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_triTests,   0, numPixels * sizeof(int)));

    // ---- Launch render kernel ----
    dim3 blockDim(RENDER_BLOCK_SIZE, RENDER_BLOCK_SIZE);
    dim3 gridDim((width  + RENDER_BLOCK_SIZE - 1) / RENDER_BLOCK_SIZE,
                 (height + RENDER_BLOCK_SIZE - 1) / RENDER_BLOCK_SIZE);

    int shadingModeInt = static_cast<int>(mode);

    cudaEvent_t startEvt, stopEvt;
    CUDA_CHECK(cudaEventCreate(&startEvt));
    CUDA_CHECK(cudaEventCreate(&stopEvt));

    CUDA_CHECK(cudaEventRecord(startEvt));

    kRenderKernel<<<gridDim, blockDim>>>(
        d_nodes, numNodes, d_tris,
        camera, width, height,
        shadingModeInt, sceneDiag, lightDir,
        d_rgb, d_nodeVisits, d_aabbTests, d_triTests
    );

    CUDA_CHECK(cudaEventRecord(stopEvt));
    CUDA_CHECK(cudaEventSynchronize(stopEvt));

    float renderMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&renderMs, startEvt, stopEvt));
    stats.renderTimeMs = renderMs;

    CUDA_CHECK(cudaEventDestroy(startEvt));
    CUDA_CHECK(cudaEventDestroy(stopEvt));

    // ---- Download results to host ----
    std::vector<float> h_rgb(numPixels * 3);
    std::vector<int>   h_nodeVisits(numPixels);
    std::vector<int>   h_aabbTests(numPixels);
    std::vector<int>   h_triTests(numPixels);

    CUDA_CHECK(cudaMemcpy(h_rgb.data(),        d_rgb,        numPixels * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_nodeVisits.data(), d_nodeVisits, numPixels * sizeof(int),       cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_aabbTests.data(),  d_aabbTests,  numPixels * sizeof(int),       cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_triTests.data(),   d_triTests,   numPixels * sizeof(int),       cudaMemcpyDeviceToHost));

    // ---- Free device memory ----
    cudaFree(d_nodes);
    freeMeshDevice(d_tris);
    cudaFree(d_rgb);
    cudaFree(d_nodeVisits);
    cudaFree(d_aabbTests);
    cudaFree(d_triTests);

    // ---- Compute aggregate statistics (over hit pixels only) ----
    double sumNodes = 0, sumAABB = 0, sumTri = 0;
    int maxNodes = 0;
    uint64_t hitCount = 0;

    for (int i = 0; i < numPixels; ++i) {
        if (h_nodeVisits[i] > 0) {
            sumNodes += h_nodeVisits[i];
            sumAABB  += h_aabbTests[i];
            sumTri   += h_triTests[i];
            maxNodes = std::max(maxNodes, h_nodeVisits[i]);
            hitCount++;
        }
    }

    if (hitCount > 0) {
        stats.avgNodesVisited = (float)(sumNodes / hitCount);
        stats.avgAABBTests    = (float)(sumAABB  / hitCount);
        stats.avgTriTests     = (float)(sumTri   / hitCount);
    }
    stats.maxNodesVisited = (float)maxNodes;

    // ---- Apply heatmap colormap on CPU if needed ----
    if (mode == ShadingMode::HEATMAP) {
        applyHeatmapColormap(h_rgb.data(), h_nodeVisits.data(), width, height);
    }

    // ---- Save PPM image ----
    if (!outputFile.empty()) {
        if (savePPM(outputFile, h_rgb.data(), width, height)) {
            std::cout << "  Saved: " << outputFile
                      << " (" << width << "x" << height << ")\n";
        }
    }

    return stats;
}

// ============================================================================
// Host: Print Render Statistics
// ============================================================================

void printRenderStats(const std::string& algorithmName, const RenderStats& stats) {
    std::cout << "  --- Render: " << algorithmName << " ---\n";
    std::cout << "    Resolution:         "
              << stats.width << " x " << stats.height << "\n";
    std::cout << "    Render Time:        " << std::fixed << std::setprecision(3)
              << stats.renderTimeMs << " ms\n";
    std::cout << "    Total Rays:         " << stats.totalRays << "\n";

    if (stats.renderTimeMs > 0.0f) {
        double raysPerSec = (double)stats.totalRays / (stats.renderTimeMs / 1000.0);
        std::cout << "    Rays/sec:           " << std::fixed << std::setprecision(2)
                  << raysPerSec / 1e6 << " M\n";
    }

    std::cout << "    Avg Nodes Visited:  " << std::fixed << std::setprecision(1)
              << stats.avgNodesVisited << "\n";
    std::cout << "    Avg AABB Tests:     " << std::fixed << std::setprecision(1)
              << stats.avgAABBTests << "\n";
    std::cout << "    Avg Tri Tests:      " << std::fixed << std::setprecision(1)
              << stats.avgTriTests << "\n";
    std::cout << "    Max Nodes Visited:  " << std::fixed << std::setprecision(0)
              << stats.maxNodesVisited << "\n";
}
