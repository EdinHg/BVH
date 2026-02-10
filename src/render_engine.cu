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

#define BVH_STACK_SIZE 128
#define RENDER_BLOCK_SIZE 16
#define EPSILON 1e-6f

// Background color (dark gray)
#define BG_R 0.15f
#define BG_G 0.15f
#define BG_B 0.18f

// ============================================================================
// Turbo Colormap Lookup Table (Google Research)
// ============================================================================
// Source: https://research.google/blog/turbo-an-improved-rainbow-colormap-for-visualization/

static const float kTurboColormap[256][3] = {
    {0.18995f,0.07176f,0.23217f},{0.19483f,0.08339f,0.26149f},{0.19956f,0.09498f,0.29024f},{0.20415f,0.10652f,0.31844f},
    {0.20860f,0.11802f,0.34607f},{0.21291f,0.12947f,0.37314f},{0.21708f,0.14087f,0.39964f},{0.22111f,0.15223f,0.42558f},
    {0.22500f,0.16354f,0.45096f},{0.22875f,0.17481f,0.47578f},{0.23236f,0.18603f,0.50004f},{0.23582f,0.19720f,0.52373f},
    {0.23915f,0.20833f,0.54686f},{0.24234f,0.21941f,0.56942f},{0.24539f,0.23044f,0.59142f},{0.24830f,0.24143f,0.61286f},
    {0.25107f,0.25237f,0.63374f},{0.25369f,0.26327f,0.65406f},{0.25618f,0.27412f,0.67381f},{0.25853f,0.28492f,0.69300f},
    {0.26074f,0.29568f,0.71162f},{0.26280f,0.30639f,0.72968f},{0.26473f,0.31706f,0.74718f},{0.26652f,0.32768f,0.76412f},
    {0.26816f,0.33825f,0.78050f},{0.26967f,0.34878f,0.79631f},{0.27103f,0.35926f,0.81156f},{0.27226f,0.36970f,0.82624f},
    {0.27334f,0.38008f,0.84037f},{0.27429f,0.39043f,0.85393f},{0.27509f,0.40072f,0.86692f},{0.27576f,0.41097f,0.87936f},
    {0.27628f,0.42118f,0.89123f},{0.27667f,0.43134f,0.90254f},{0.27691f,0.44145f,0.91328f},{0.27701f,0.45152f,0.92347f},
    {0.27698f,0.46153f,0.93309f},{0.27680f,0.47151f,0.94214f},{0.27648f,0.48144f,0.95064f},{0.27603f,0.49132f,0.95857f},
    {0.27543f,0.50115f,0.96594f},{0.27469f,0.51094f,0.97275f},{0.27381f,0.52069f,0.97899f},{0.27273f,0.53040f,0.98461f},
    {0.27106f,0.54015f,0.98930f},{0.26878f,0.54995f,0.99303f},{0.26592f,0.55979f,0.99583f},{0.26252f,0.56967f,0.99773f},
    {0.25862f,0.57958f,0.99876f},{0.25425f,0.58950f,0.99896f},{0.24946f,0.59943f,0.99835f},{0.24427f,0.60937f,0.99697f},
    {0.23874f,0.61931f,0.99485f},{0.23288f,0.62923f,0.99202f},{0.22676f,0.63913f,0.98851f},{0.22039f,0.64901f,0.98436f},
    {0.21382f,0.65886f,0.97959f},{0.20708f,0.66866f,0.97423f},{0.20021f,0.67842f,0.96833f},{0.19326f,0.68812f,0.96190f},
    {0.18625f,0.69775f,0.95498f},{0.17923f,0.70732f,0.94761f},{0.17223f,0.71680f,0.93981f},{0.16529f,0.72620f,0.93161f},
    {0.15844f,0.73551f,0.92305f},{0.15173f,0.74472f,0.91416f},{0.14519f,0.75381f,0.90496f},{0.13886f,0.76279f,0.89550f},
    {0.13278f,0.77165f,0.88580f},{0.12698f,0.78037f,0.87590f},{0.12151f,0.78896f,0.86581f},{0.11639f,0.79740f,0.85559f},
    {0.11167f,0.80569f,0.84525f},{0.10738f,0.81381f,0.83484f},{0.10357f,0.82177f,0.82437f},{0.10026f,0.82955f,0.81389f},
    {0.09750f,0.83714f,0.80342f},{0.09532f,0.84455f,0.79299f},{0.09377f,0.85175f,0.78264f},{0.09287f,0.85875f,0.77240f},
    {0.09267f,0.86554f,0.76230f},{0.09320f,0.87211f,0.75237f},{0.09451f,0.87844f,0.74265f},{0.09662f,0.88454f,0.73316f},
    {0.09958f,0.89040f,0.72393f},{0.10342f,0.89600f,0.71500f},{0.10815f,0.90142f,0.70599f},{0.11374f,0.90673f,0.69651f},
    {0.12014f,0.91193f,0.68660f},{0.12733f,0.91701f,0.67627f},{0.13526f,0.92197f,0.66556f},{0.14391f,0.92680f,0.65448f},
    {0.15323f,0.93151f,0.64308f},{0.16319f,0.93609f,0.63137f},{0.17377f,0.94053f,0.61938f},{0.18491f,0.94484f,0.60713f},
    {0.19659f,0.94901f,0.59466f},{0.20877f,0.95304f,0.58199f},{0.22142f,0.95692f,0.56914f},{0.23449f,0.96065f,0.55614f},
    {0.24797f,0.96423f,0.54303f},{0.26180f,0.96765f,0.52981f},{0.27597f,0.97092f,0.51653f},{0.29042f,0.97403f,0.50321f},
    {0.30513f,0.97697f,0.48987f},{0.32006f,0.97974f,0.47654f},{0.33517f,0.98234f,0.46325f},{0.35043f,0.98477f,0.45002f},
    {0.36581f,0.98702f,0.43688f},{0.38127f,0.98909f,0.42386f},{0.39678f,0.99098f,0.41098f},{0.41229f,0.99268f,0.39826f},
    {0.42778f,0.99419f,0.38575f},{0.44321f,0.99551f,0.37345f},{0.45854f,0.99663f,0.36140f},{0.47375f,0.99755f,0.34963f},
    {0.48879f,0.99828f,0.33816f},{0.50362f,0.99879f,0.32701f},{0.51822f,0.99910f,0.31622f},{0.53255f,0.99919f,0.30581f},
    {0.54658f,0.99907f,0.29581f},{0.56026f,0.99873f,0.28623f},{0.57357f,0.99817f,0.27712f},{0.58646f,0.99739f,0.26849f},
    {0.59891f,0.99638f,0.26038f},{0.61088f,0.99514f,0.25280f},{0.62233f,0.99366f,0.24579f},{0.63323f,0.99195f,0.23937f},
    {0.64362f,0.98999f,0.23356f},{0.65394f,0.98775f,0.22835f},{0.66428f,0.98524f,0.22370f},{0.67462f,0.98246f,0.21960f},
    {0.68494f,0.97941f,0.21602f},{0.69525f,0.97610f,0.21294f},{0.70553f,0.97255f,0.21032f},{0.71577f,0.96875f,0.20815f},
    {0.72596f,0.96470f,0.20640f},{0.73610f,0.96043f,0.20504f},{0.74617f,0.95593f,0.20406f},{0.75617f,0.95121f,0.20343f},
    {0.76608f,0.94627f,0.20311f},{0.77591f,0.94113f,0.20310f},{0.78563f,0.93579f,0.20336f},{0.79524f,0.93025f,0.20386f},
    {0.80473f,0.92452f,0.20459f},{0.81410f,0.91861f,0.20552f},{0.82333f,0.91253f,0.20663f},{0.83241f,0.90627f,0.20788f},
    {0.84133f,0.89986f,0.20926f},{0.85010f,0.89328f,0.21074f},{0.85868f,0.88655f,0.21230f},{0.86709f,0.87968f,0.21391f},
    {0.87530f,0.87267f,0.21555f},{0.88331f,0.86553f,0.21719f},{0.89112f,0.85826f,0.21880f},{0.89870f,0.85087f,0.22038f},
    {0.90605f,0.84337f,0.22188f},{0.91317f,0.83576f,0.22328f},{0.92004f,0.82806f,0.22456f},{0.92666f,0.82025f,0.22570f},
    {0.93301f,0.81236f,0.22667f},{0.93909f,0.80439f,0.22744f},{0.94489f,0.79634f,0.22800f},{0.95039f,0.78823f,0.22831f},
    {0.95560f,0.78005f,0.22836f},{0.96049f,0.77181f,0.22811f},{0.96507f,0.76352f,0.22754f},{0.96931f,0.75519f,0.22663f},
    {0.97323f,0.74682f,0.22536f},{0.97679f,0.73842f,0.22369f},{0.98000f,0.73000f,0.22161f},{0.98289f,0.72140f,0.21918f},
    {0.98549f,0.71250f,0.21650f},{0.98781f,0.70330f,0.21358f},{0.98986f,0.69382f,0.21043f},{0.99163f,0.68408f,0.20706f},
    {0.99314f,0.67408f,0.20348f},{0.99438f,0.66386f,0.19971f},{0.99535f,0.65341f,0.19577f},{0.99607f,0.64277f,0.19165f},
    {0.99654f,0.63193f,0.18738f},{0.99675f,0.62093f,0.18297f},{0.99672f,0.60977f,0.17842f},{0.99644f,0.59846f,0.17376f},
    {0.99593f,0.58703f,0.16899f},{0.99517f,0.57549f,0.16412f},{0.99419f,0.56386f,0.15918f},{0.99297f,0.55214f,0.15417f},
    {0.99153f,0.54036f,0.14910f},{0.98987f,0.52854f,0.14398f},{0.98799f,0.51667f,0.13883f},{0.98590f,0.50479f,0.13367f},
    {0.98360f,0.49291f,0.12849f},{0.98108f,0.48104f,0.12332f},{0.97837f,0.46920f,0.11817f},{0.97545f,0.45740f,0.11305f},
    {0.97234f,0.44565f,0.10797f},{0.96904f,0.43399f,0.10294f},{0.96555f,0.42241f,0.09798f},{0.96187f,0.41093f,0.09310f},
    {0.95801f,0.39958f,0.08831f},{0.95398f,0.38836f,0.08362f},{0.94977f,0.37729f,0.07905f},{0.94538f,0.36638f,0.07461f},
    {0.94084f,0.35566f,0.07031f},{0.93612f,0.34513f,0.06616f},{0.93125f,0.33482f,0.06218f},{0.92623f,0.32473f,0.05837f},
    {0.92105f,0.31489f,0.05475f},{0.91572f,0.30530f,0.05134f},{0.91024f,0.29599f,0.04814f},{0.90463f,0.28696f,0.04516f},
    {0.89888f,0.27824f,0.04243f},{0.89298f,0.26981f,0.03993f},{0.88691f,0.26152f,0.03753f},{0.88066f,0.25334f,0.03521f},
    {0.87422f,0.24526f,0.03297f},{0.86760f,0.23730f,0.03082f},{0.86079f,0.22945f,0.02875f},{0.85380f,0.22170f,0.02677f},
    {0.84662f,0.21407f,0.02487f},{0.83926f,0.20654f,0.02305f},{0.83172f,0.19912f,0.02131f},{0.82399f,0.19182f,0.01966f},
    {0.81608f,0.18462f,0.01809f},{0.80799f,0.17753f,0.01660f},{0.79971f,0.17055f,0.01520f},{0.79125f,0.16368f,0.01387f},
    {0.78260f,0.15693f,0.01264f},{0.77377f,0.15028f,0.01148f},{0.76476f,0.14374f,0.01041f},{0.75556f,0.13731f,0.00942f},
    {0.74617f,0.13098f,0.00851f},{0.73661f,0.12477f,0.00769f},{0.72686f,0.11867f,0.00695f},{0.71692f,0.11268f,0.00629f},
    {0.70680f,0.10680f,0.00571f},{0.69650f,0.10102f,0.00522f},{0.68602f,0.09536f,0.00481f},{0.67535f,0.08980f,0.00449f},
    {0.66449f,0.08436f,0.00424f},{0.65345f,0.07902f,0.00408f},{0.64223f,0.07380f,0.00401f},{0.63082f,0.06868f,0.00401f},
    {0.61923f,0.06367f,0.00410f},{0.60746f,0.05878f,0.00427f},{0.59550f,0.05399f,0.00453f},{0.58336f,0.04931f,0.00486f},
    {0.57103f,0.04474f,0.00529f},{0.55852f,0.04028f,0.00579f},{0.54583f,0.03593f,0.00638f},{0.53295f,0.03169f,0.00705f},
    {0.51989f,0.02756f,0.00780f},{0.50664f,0.02354f,0.00863f},{0.49321f,0.01963f,0.00955f},{0.47960f,0.01583f,0.01055f}
};

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
    float3_cw invDirection; 
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
    // if (u < 0.0f || u > 1.0f) return false;
    if (u < -1e-5f || u > 1.00001f) return false;

    // Q = T x e1
    float qx = ty * e1z - tz * e1y;
    float qy = tz * e1x - tx * e1z;
    float qz = tx * e1y - ty * e1x;

    // Barycentric v
    float v = (ray.direction.x * qx + ray.direction.y * qy + ray.direction.z * qz) * inv_det;
    // if (v < 0.0f || u + v > 1.0f) return false;
    if (v < -1e-5f || u + v > 1.00001f) return false;

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
// Host: Turbo Colormap Helper Function
// ============================================================================

// Get Turbo colormap RGB value for normalized input t in [0, 1]
// Uses linear interpolation between lookup table entries for smooth gradients
static inline void getTurboColor(float t, float& r, float& g, float& b) {
    // Clamp t to valid range
    t = std::max(0.0f, std::min(1.0f, t));
    
    // Map t to the 256-entry table
    float scaled = t * 255.0f;
    int idx0 = static_cast<int>(scaled);
    int idx1 = std::min(idx0 + 1, 255);
    
    // Linear interpolation weight
    float frac = scaled - static_cast<float>(idx0);
    
    // Interpolate between two adjacent entries
    r = kTurboColormap[idx0][0] * (1.0f - frac) + kTurboColormap[idx1][0] * frac;
    g = kTurboColormap[idx0][1] * (1.0f - frac) + kTurboColormap[idx1][1] * frac;
    b = kTurboColormap[idx0][2] * (1.0f - frac) + kTurboColormap[idx1][2] * frac;
}

// ============================================================================
// Host: Apply heatmap colormap (CPU post-process)
// ============================================================================

static void applyHeatmapColormap(float* rgb, const int* nodeVisitCounts,
                                 int width, int height, float fixedMax = 0.0f) {
    int numPixels = width * height;

    // Determine normalization value
    float maxVal = fixedMax;
    
    if (maxVal <= 0.0f) {
        // Collect all hit pixel counts
        std::vector<int> hit_counts;
        hit_counts.reserve(numPixels);
        for (int i = 0; i < numPixels; ++i) {
            if (nodeVisitCounts[i] > 0) {
                hit_counts.push_back(nodeVisitCounts[i]);
            }
        }
        
        maxVal = 128.0f; // Fallback
        if (!hit_counts.empty()) {
            std::sort(hit_counts.begin(), hit_counts.end());
            
            // Use 99.9th percentile instead of 99th to show more red hotspots
            size_t p999_idx = std::min(hit_counts.size() - 1,
                                       (size_t)(hit_counts.size() * 0.999));
            maxVal = std::max(1.0f, (float)hit_counts[p999_idx]);
            
            // Optional: print diagnostics
            int actualMax = hit_counts.back();
            std::cout << "    Heatmap normalization: p99.9=" << maxVal 
                      << " (actual max=" << actualMax << ")\n";
        }
    }

    // Apply colormap using maxVal (anything >= maxVal becomes pure red)
    for (int i = 0; i < numPixels; ++i) {
        if (nodeVisitCounts[i] == 0) {
            // Background/miss pixel
            rgb[i * 3 + 0] = BG_R;
            rgb[i * 3 + 1] = BG_G;
            rgb[i * 3 + 2] = BG_B;
        } else {
            // Normalize node visit count to [0, 1]
            float t = (float)nodeVisitCounts[i] / maxVal;
            t = std::max(0.0f, std::min(1.0f, t));  // Clamp to [0,1]

            // Apply Turbo colormap
            float r, g, b;
            getTurboColor(t, r, g, b);

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
                        const std::string& outputFile,
                        float fixedHeatmapMax) {
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
        applyHeatmapColormap(h_rgb.data(), h_nodeVisits.data(), width, height, fixedHeatmapMax);
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
