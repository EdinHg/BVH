# GPU Render Engine -- Implementation Details

This document describes the implementation of the CUDA ray tracing render engine
used to visually compare BVH traversal quality across different construction
algorithms. The engine lives in `src/render_engine.cu` (~780 lines) with its
public interface declared in `include/render_engine.h`.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Structures on the GPU](#2-data-structures-on-the-gpu)
3. [Ray Generation -- Pinhole Camera Model](#3-ray-generation----pinhole-camera-model)
4. [Ray-AABB Intersection -- Slab Method](#4-ray-aabb-intersection----slab-method)
5. [Ray-Triangle Intersection -- Moller-Trumbore](#5-ray-triangle-intersection----moller-trumbore)
6. [BVH Traversal -- Iterative Stack-Based](#6-bvh-traversal----iterative-stack-based)
7. [Shading Modes](#7-shading-modes)
   - [7.1 Normal Mapping](#71-normal-mapping)
   - [7.2 Depth](#72-depth)
   - [7.3 Diffuse Lighting](#73-diffuse-lighting)
   - [7.4 Heatmap (Traversal Cost)](#74-heatmap-traversal-cost)
8. [Host Orchestration and Output](#8-host-orchestration-and-output)
9. [Statistics Collection](#9-statistics-collection)
10. [Auto-Camera](#10-auto-camera)

---

## 1. Architecture Overview

The render engine follows a standard GPU ray tracing pipeline:

```
Host (CPU)                              Device (GPU)
───────────                             ────────────
renderImage()
  ├─ Upload BVH nodes   ──cudaMemcpy──►  d_nodes[]
  ├─ Upload mesh SoA     ──cudaMemcpy──►  d_tris (9 float arrays)
  ├─ Allocate output buffers            ►  d_rgb[], d_nodeVisits[], ...
  │
  ├─ Launch kRenderKernel ─────────────►  1 thread per pixel
  │                                       ├─ generateRay()
  │                                       ├─ traverseBVH()
  │                                       │   ├─ intersectAABB()   (per node)
  │                                       │   └─ intersectTriangle() (per leaf)
  │                                       └─ shade pixel (switch on mode)
  │
  ├─ Download d_rgb, stat buffers ◄────  cudaMemcpy D2H
  ├─ [Heatmap only] CPU colormap
  └─ Save PPM (P6 binary)
```

The kernel is launched as a 2D grid of 16x16 thread blocks over the image
dimensions. Each thread processes exactly one pixel: it generates a primary ray,
traverses the BVH to find the closest triangle intersection, and writes an RGB
value plus per-pixel statistics to separate output buffers.

**Source references:**
- Kernel launch: `render_engine.cu:668-686`
- Kernel function: `render_engine.cu:288-372`

---

## 2. Data Structures on the GPU

### BVHNode (unified format, 40 bytes)

```cpp
struct BVHNode {
    AABB_cw bbox;          // 32 bytes: min(x,y,z,pad) + max(x,y,z,pad)
    int32_t leftChild;     // 4 bytes
    int32_t rightChild;    // 4 bytes
};
```

- **Internal node:** `leftChild` and `rightChild` are indices into the nodes
  array.
- **Leaf node:** `leftChild` has the high bit set (`leftChild | 0x80000000`).
  The original triangle index is extracted by masking: `leftChild & 0x7FFFFFFF`.
- Detection: `isLeaf()` checks `(leftChild < 0) || (leftChild & 0x80000000)`.

Both `isLeaf()` and `getPrimitiveIndex()` are marked `__host__ __device__` so
they work in both CPU evaluation code and GPU traversal kernels.

### TrianglesSoADevice (Structure-of-Arrays)

```cpp
struct TrianglesSoADevice {
    float *v0x, *v0y, *v0z;   // Vertex 0 of each triangle
    float *v1x, *v1y, *v1z;   // Vertex 1
    float *v2x, *v2y, *v2z;   // Vertex 2
};
```

Nine separate `float*` arrays, one per coordinate per vertex. This SoA layout
is better for GPU memory coalescing than an AoS (Array-of-Structures) layout
because neighboring threads accessing neighboring triangles read contiguous
memory addresses.

### float3_cw (16-byte aligned)

```cpp
struct __align__(16) float3_cw {
    float x, y, z;    // 12 bytes data + 4 bytes padding
};
```

The 16-byte alignment ensures efficient GPU loads/stores (aligned to a
128-bit boundary). Arithmetic operators `+`, `-`, `*` are defined as
`__host__ __device__` for use on both CPU and GPU.

---

## 3. Ray Generation -- Pinhole Camera Model

**Source:** `render_engine.cu:83-116` (`generateRay`)

Each CUDA thread computes a ray for pixel `(x, y)` using a pinhole camera:

### Step 1: Build orthonormal basis

```
forward = normalize(lookAt - eye)
right   = normalize(forward x up)
camUp   = right x forward
```

`forward` points from the eye toward the look-at point. `right` is computed
via cross product with the world-up vector. `camUp` is recomputed (not the
raw `up` input) to guarantee an orthonormal basis even if the user-provided
up vector is not exactly perpendicular to `forward`.

### Step 2: Image plane from FOV

```
halfH  = tan(fovY / 2)
aspect = width / height
halfW  = halfH * aspect
```

The vertical half-angle of the FOV determines how tall the virtual image plane
is at unit distance from the eye. The horizontal extent is scaled by the aspect
ratio.

### Step 3: Pixel to ray direction

```
u = (2 * (x + 0.5) / width  - 1) * halfW
v = (2 * (y + 0.5) / height - 1) * halfH

direction = normalize(forward + u * right - v * camUp)
```

The `+0.5` offset places the sample at the center of each pixel. The `v`
coordinate is negated so that `y=0` corresponds to the top of the image
(screen-space convention) while the camera's up vector points upward in
world space.

### Step 4: Precompute inverse direction

```
invDirection = (1/dir.x, 1/dir.y, 1/dir.z)
```

For near-zero components, a large signed value (`+/-1e30`) is substituted
to avoid division by zero while preserving the sign for the slab test.
This is computed once per ray and reused for every AABB intersection test.

---

## 4. Ray-AABB Intersection -- Slab Method

**Source:** `render_engine.cu:122-142` (`intersectAABB`)

Tests whether a ray intersects an axis-aligned bounding box using the
Kay-Kajiya slab method. The AABB is treated as the intersection of three
axis-aligned slabs (x, y, z).

```
For each axis a in {x, y, z}:
    t1_a = (bbox.min.a - ray.origin.a) * ray.invDirection.a
    t2_a = (bbox.max.a - ray.origin.a) * ray.invDirection.a
    tmin_a = min(t1_a, t2_a)
    tmax_a = max(t1_a, t2_a)

tenter = max(tmin_x, tmin_y, tmin_z)     // latest entry
texit  = min(tmax_x, tmax_y, tmax_z)     // earliest exit

hit = (texit >= max(tenter, 0)) AND (tenter < current_tmax)
```

The `current_tmax` parameter is the distance to the closest intersection found
so far (`result.t`). This early rejection eliminates boxes that are entirely
behind an already-found hit, which is critical for performance -- it prunes
large portions of the BVH tree for secondary hits.

Using the precomputed `invDirection` avoids 6 divisions per AABB test. The
`min`/`max` swap handles negative ray directions without branching.

---

## 5. Ray-Triangle Intersection -- Moller-Trumbore

**Source:** `render_engine.cu:148-199` (`intersectTriangle`)

Implements the Moller-Trumbore algorithm for ray-triangle intersection.
This is a standard algorithm that computes the intersection directly in
barycentric coordinates without first computing the plane equation.

### Algorithm

Given triangle vertices `V0, V1, V2` and ray `O + t*D`:

```
e1 = V1 - V0                    // Edge vectors
e2 = V2 - V0

P  = D x e2                     // Cross products for Cramer's rule
det = dot(e1, P)
if |det| < epsilon: return false // Ray parallel to triangle

inv_det = 1 / det
T = O - V0

u = dot(T, P) * inv_det         // First barycentric coordinate
if u < 0 or u > 1: return false

Q = T x e1
v = dot(D, Q) * inv_det         // Second barycentric coordinate
if v < 0 or u + v > 1: return false

t = dot(e2, Q) * inv_det        // Ray parameter at intersection
if t < epsilon: return false     // Intersection behind ray origin
```

### Normal computation

The geometric face normal is computed as the normalized cross product of the
two edge vectors:

```
normal = normalize(e1 x e2)
```

This is a flat normal (constant across the triangle face). No smooth
(per-vertex) normals are computed because the OBJ loader only stores vertex
positions in SoA format, not normals. The flat normal is sufficient for all
four shading modes.

**Key detail:** The triangle vertex data is passed as 9 individual `float`
parameters (not a struct) to match the SoA memory layout. The kernel reads
directly from the per-coordinate arrays using the primitive index:
`tris.v0x[primIdx], tris.v0y[primIdx], ...`

---

## 6. BVH Traversal -- Iterative Stack-Based

**Source:** `render_engine.cu:215-282` (`traverseBVH`)

CUDA kernels cannot use recursion efficiently (limited stack, poor performance),
so the traversal uses an explicit stack allocated in thread-local registers/local
memory.

### Algorithm

```
stack = [root_node_index]    // stack of size 64
closest_t = infinity

while stack is not empty:
    nodeIdx = stack.pop()
    node = nodes[nodeIdx]
    nodesVisited++

    // Test ray against this node's bounding box
    aabbTests++
    if not intersectAABB(ray, node.bbox, closest_t):
        continue                 // Prune: box missed or behind closer hit

    if node.isLeaf():
        primIdx = node.getPrimitiveIndex()
        triTests++
        if intersectTriangle(ray, triangle[primIdx]):
            if t < closest_t:
                closest_t = t    // Update closest hit
                record normal, primIdx
    else:
        // Push both children (right first, so left is popped first)
        stack.push(node.rightChild)
        stack.push(node.leftChild)
```

### Design decisions

- **Stack size of 64:** Supports BVH trees up to depth 64, which is far deeper
  than any practical BVH (typical depth is 20-35 for millions of triangles).

- **Left-first traversal:** Right child is pushed first so the left child sits
  on top and is processed next. This is arbitrary without ordered traversal --
  a future optimization would test the nearer child first based on ray direction.

- **Bounds checking:** `nodeIdx < 0 || nodeIdx >= numNodes` guards against
  corrupt BVH data. `stackPtr < BVH_STACK_SIZE` prevents stack overflow.

- **Early termination via tmax:** The AABB test receives `result.t` (the closest
  hit found so far) as `tmax`. Any box whose entry point is beyond this distance
  is skipped entirely. This is the most important optimization -- as closer hits
  are found, more of the BVH tree gets pruned.

### Per-ray statistics

The `HitInfo` struct tracks three counters per ray:
- `nodesVisited`: total BVH nodes popped from the stack
- `aabbTests`: total AABB intersection tests performed (same as nodesVisited)
- `triTests`: total triangle intersection tests at leaf nodes

These are written to per-pixel GPU buffers and downloaded for aggregate
statistics and heatmap visualization.

---

## 7. Shading Modes

After traversal, each thread determines an RGB color based on the selected
shading mode. The mode is passed as an `int` to the kernel (cast from the
`ShadingMode` enum) and selected via a `switch` statement.

If the ray misses all geometry, the background color `(0.15, 0.15, 0.18)` (dark
gray) is used regardless of shading mode.

### 7.1 Normal Mapping

**Source:** `render_engine.cu:329-333` (case 0)

```cpp
r = (hit.normal.x + 1.0) * 0.5
g = (hit.normal.y + 1.0) * 0.5
b = (hit.normal.z + 1.0) * 0.5
```

Maps the unit surface normal vector `[-1, +1]` to RGB `[0, 1]`:

| Normal component | Range    | Color mapping      |
|------------------|----------|--------------------|
| `nx = +1` (right)  | R = 1.0  | Red               |
| `nx = -1` (left)   | R = 0.0  | No red            |
| `ny = +1` (up)     | G = 1.0  | Green             |
| `ny = -1` (down)   | G = 0.0  | No green          |
| `nz = +1` (toward) | B = 1.0  | Blue              |
| `nz = -1` (away)   | B = 0.0  | No blue           |

**Purpose:** Verifies that:
- Triangle geometry loads correctly (no scrambled vertices)
- The BVH traversal finds the correct closest triangle (not a farther one)
- Normals are computed and oriented correctly
- The image should look identical for all BVH algorithms since they index the
  same triangles -- any difference indicates a bug

**Typical appearance:** A colorful rendering where surfaces facing right are
red, surfaces facing up are green, etc. It looks like a "rainbow" version of
the model.

### 7.2 Depth

**Source:** `render_engine.cu:335-340` (case 1)

```cpp
depth = hit.t / sceneDiag            // Normalize by scene diagonal
depth = clamp(depth, 0, 1)
val   = 1.0 - depth                  // Invert: closer = brighter
r = g = b = val                      // Grayscale
```

The intersection distance `hit.t` is normalized by the scene's AABB diagonal
length (computed from `nodes[0].bbox` on the host before kernel launch). This
maps the entire scene depth range roughly to `[0, 1]`.

The value is inverted so closer surfaces appear **white/bright** and distant
surfaces appear **dark/black**. The result is a grayscale image.

**Purpose:** Verifies that:
- The closest-hit logic works (closer triangles should occlude farther ones)
- The depth ordering is consistent across the model
- All algorithms produce the same depth image (since they contain the same
  geometry)

**Typical appearance:** A white-to-black gradient across the model, like a
Z-buffer visualization.

### 7.3 Diffuse Lighting

**Source:** `render_engine.cu:342-355` (case 2)

Implements a simple Lambertian shading model with a fixed directional light.

#### Step 1: Normal orientation

```cpp
if dot(normal, ray.direction) > 0:
    normal = -normal     // Flip to face the camera
```

The geometric normal from Moller-Trumbore may point either way (depending on
triangle winding order). If it points away from the camera (same direction as
the incoming ray), it is flipped. This ensures correct shading for
single-sided meshes that have inconsistent winding.

#### Step 2: Lambertian diffuse

```cpp
diffuse = max(0, dot(normal, lightDir))    // Cosine of angle to light
ambient = 0.15                              // Constant ambient fill
shade   = min(1, ambient + diffuse * 0.85)
```

The light direction is fixed at `normalize(0.5, 0.8, 0.6)` -- coming from the
upper-right-front. The dot product gives the cosine of the angle between the
surface normal and the light, which is the Lambertian diffuse term. Surfaces
facing the light are bright; surfaces facing away receive only ambient light.

The `0.85` factor scales the diffuse contribution so that fully lit surfaces
reach `ambient + 0.85 = 1.0`.

#### Step 3: Normal-based color variation

```cpp
r = shade * (0.7 + 0.3 * |nx|)
g = shade * (0.7 + 0.3 * |ny|)
b = shade * (0.7 + 0.3 * |nz|)
```

Instead of a flat gray material, a subtle color is derived from the absolute
normal components. Surfaces facing along the X axis get a slight red tint,
Y-facing surfaces get green, Z-facing get blue. The variation is mild (70% base
+ 30% normal-dependent) so the image looks like a naturally lit clay/ceramic
model with slight directional coloring.

**Purpose:** Produces the most visually readable rendering. Good for:
- Verifying overall scene geometry
- Screenshots and presentations
- Camera angle adjustments

**Typical appearance:** A soft, naturally lit 3D model with subtle warm/cool
color shifts based on surface orientation.

### 7.4 Heatmap (Traversal Cost)

**Source:**
- GPU side: `render_engine.cu:357-360` (case 3)
- CPU colormap: `render_engine.cu:443-496` (`applyHeatmapColormap`)

This is the most important mode for BVH quality comparison. It visualizes how
many BVH nodes each ray had to visit, directly showing traversal efficiency.

#### GPU side (simple)

```cpp
// Store raw count as float; CPU will apply the colormap
r = g = b = (float)hit.nodesVisited
```

The kernel writes the raw integer node visit count to all three RGB channels.
This is **not** a valid color -- it is a scalar value (possibly hundreds) stored
temporarily. The actual coloring happens on the CPU after download.

#### CPU colormap (`applyHeatmapColormap`)

##### Adaptive normalization (99th percentile)

```cpp
// Collect all non-zero visit counts (hit pixels only)
for each pixel:
    if nodeVisitCounts[i] > 0:
        hit_counts.push_back(nodeVisitCounts[i])

sort(hit_counts)
maxVal = hit_counts[size * 0.99]     // 99th percentile
```

Using the 99th percentile instead of the absolute maximum prevents a few
extreme outlier rays (e.g., rays grazing many overlapping boxes) from
compressing the rest of the color range. The top 1% of costly rays are clamped
to red.

##### Five-band colormap

The normalized value `t = count / maxVal` (clamped to `[0, 1]`) is mapped
through a blue-to-red colormap divided into 4 interpolation bands:

| Range t       | Color transition      | Meaning              |
|---------------|-----------------------|----------------------|
| `0.00 - 0.25` | Blue -> Cyan          | Very cheap traversal |
| `0.25 - 0.50` | Cyan -> Green         | Below average cost   |
| `0.50 - 0.75` | Green -> Yellow       | Above average cost   |
| `0.75 - 1.00` | Yellow -> Red         | Expensive traversal  |

Each band linearly interpolates between two colors:

```
t in [0, 0.25):     s = t / 0.25
                     R=0,     G=s,   B=1       (blue to cyan)

t in [0.25, 0.50):   s = (t - 0.25) / 0.25
                     R=0,     G=1,   B=1-s     (cyan to green)

t in [0.50, 0.75):   s = (t - 0.50) / 0.25
                     R=s,     G=1,   B=0       (green to yellow)

t in [0.75, 1.00]:   s = (t - 0.75) / 0.25
                     R=1,     G=1-s, B=0       (yellow to red)
```

Miss pixels (background) retain the standard background color `(0.15, 0.15, 0.18)`.

**Purpose:** This is the primary BVH quality diagnostic:

- A better BVH (lower SAH cost) should show more blue/cyan (fewer nodes per ray)
- A worse BVH should show more yellow/red (more nodes per ray)
- PLOC (especially with higher radius) should generally produce cooler
  (bluer) heatmaps than LBVH, reflecting its better tree quality at the
  cost of longer build time
- Edge regions and thin geometry tend to be more expensive (redder) because
  rays pass through many overlapping bounding boxes
- Side-by-side heatmap comparison across algorithms directly visualizes the
  BVH quality trade-off

---

## 8. Host Orchestration and Output

**Source:** `render_engine.cu:612-753` (`renderImage`)

The `renderImage()` function is the main entry point, called from `main.cpp`
for each BVH algorithm. It performs:

1. **Compute scene parameters:**
   - Scene diagonal from `nodes[0].bbox` (root AABB) for depth normalization
   - Fixed light direction `normalize(0.5, 0.8, 0.6)`

2. **Upload to GPU:**
   - BVH nodes array: `cudaMalloc` + `cudaMemcpy` (host-to-device)
   - Triangle mesh: 9 separate `cudaMalloc`/`cudaMemcpy` calls (one per SoA
     component)

3. **Allocate output buffers:**
   - `d_rgb`: `float[width * height * 3]` -- RGB color per pixel
   - `d_nodeVisits`: `int[width * height]` -- nodes visited per pixel
   - `d_aabbTests`: `int[width * height]` -- AABB tests per pixel
   - `d_triTests`: `int[width * height]` -- triangle tests per pixel

4. **Launch kernel** with CUDA event timing (sub-millisecond precision)

5. **Download results** from GPU to host vectors

6. **Compute statistics** (averages over hit pixels only)

7. **Apply heatmap colormap** on CPU (only for heatmap mode)

8. **Save PPM image** (P6 binary format)

### PPM Output Format

The image is saved as PPM P6 (binary):

```
P6\n
<width> <height>\n
255\n
<width * height * 3 bytes of RGB data>
```

Float RGB values `[0.0, 1.0]` are converted to `uint8` `[0, 255]` with
rounding: `byte = (uint8_t)(value * 255.0 + 0.5)`. Values are clamped to
`[0, 1]` before conversion.

PPM was chosen because it requires no external image library dependency and is
readable by PIL/Pillow, matplotlib, GIMP, and most image viewers.

---

## 9. Statistics Collection

**Source:** `render_engine.cu:317-319` (GPU write), `render_engine.cu:717-737` (CPU aggregate)

Each thread writes its per-ray counters to dedicated integer buffers (one entry
per pixel). After download, the host iterates over all pixels and computes:

| Statistic         | Computation                            |
|-------------------|----------------------------------------|
| `avgNodesVisited` | sum(nodeVisits) / hitCount             |
| `avgAABBTests`    | sum(aabbTests) / hitCount              |
| `avgTriTests`     | sum(triTests) / hitCount               |
| `maxNodesVisited` | max(nodeVisits) across all pixels      |

Only **hit pixels** (where `nodeVisits > 0`) are included in the averages.
Background/miss pixels are excluded because they still traverse some nodes
before determining no hit exists, but including them would dilute the metric.

These statistics are printed via `printRenderStats()` and also returned in the
`RenderStats` struct for programmatic use (e.g., the Colab notebook parses them
for bar charts).

---

## 10. Auto-Camera

**Source:** `render_engine.cu:502-538` (`autoCamera`)

When no `--camera` CLI argument is provided, the camera is automatically
positioned to frame the entire scene:

1. Read the root node's AABB (`nodes[0].bbox`) to get scene bounds
2. Compute the center and diagonal length of the AABB
3. Calculate the distance needed to fit the diagonal within the FOV:
   ```
   dist = (diagonal / 2) / tan(fovY / 2) * 1.2
   ```
   The `1.2` factor adds 20% margin so the model doesn't touch the image edges.
4. Place the eye at a 3/4 angle offset from center:
   ```
   eye.x = center.x + dist * 0.6
   eye.y = center.y + dist * 0.3
   eye.z = center.z + dist * 0.7
   ```
   This produces a slightly elevated, off-axis view that shows three faces of
   the model (similar to an isometric-like perspective).
5. Set `lookAt = center`

The auto-camera uses the first builder's BVH (all builders produce the same
root AABB since they contain the same triangles), so the camera is consistent
across all algorithm renders.
