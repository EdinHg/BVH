# BVH Builder & Visualizer

A C++17 implementation of Bounding Volume Hierarchy (BVH) construction algorithms with an interactive OpenGL visualizer for analyzing tree quality via heatmap visualization.

## Features

- **BVH Construction Algorithms**
  - **RecursiveBVH** - Top-down SAH-based (Surface Area Heuristic) binary BVH builder
  - **LBVH** - Linear BVH using Morton codes with radix sort for fast construction

- **CLI Benchmark Tool** (`bvh_test`)
  - Build time measurement
  - SAH cost calculation
  - Node count statistics
  - BVH export to OBJ format

- **Interactive Visualizer** (`bvh_visualizer`)
  - OpenGL-based mesh rendering
  - Heatmap visualization of BVH leaf costs
  - Free-fly camera controls

## Building

### Requirements

- CMake 3.16+
- C++17 compatible compiler
- OpenGL support

### Build Steps

```bash
mkdir build
cd build
cmake ..
make
```

On Windows with MinGW:
```bash
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
```

GLFW is automatically fetched via CMake's FetchContent.

## Usage

### BVH Test (CLI Benchmark)

```bash
./bvh_test <model.obj> [options]
```

**Options:**
- `--lbvh` - Use LBVH (Morton code) builder instead of recursive SAH
- `--export` - Export BVH bounding boxes to `<model>_bvh.obj`
- `--export-leaves` - Export only leaf bounding boxes

**Example:**
```bash
./bvh_test ../models/buddha.obj --lbvh
```

### BVH Visualizer

```bash
./bvh_visualizer <model.obj> [--lbvh]
```

**Controls:**
| Key | Action |
|-----|--------|
| WASD | Move camera |
| Space/Shift | Move up/down |
| Mouse | Look around |
| Middle-drag | Pan |
| Scroll | Zoom in/out |
| H | Toggle heatmap |
| ESC | Exit |

**Example:**
```bash
./bvh_visualizer ../models/buddha.obj --lbvh
```

## Project Structure

```
├── src/
│   ├── main.cpp              # CLI benchmark entry point
│   ├── visualizer_main.cpp   # OpenGL visualizer entry point
│   ├── bvh/
│   │   ├── bvh_builder.hpp   # BVH builder interface
│   │   ├── bvh_node.hpp      # BVH node structure
│   │   ├── recursive_bvh.hpp # SAH-based recursive builder
│   │   ├── lbvh_builder.hpp  # Morton code LBVH builder
│   │   └── bvh_export.hpp    # OBJ export functionality
│   ├── mesh/
│   │   ├── obj_loader.hpp    # OBJ file loader
│   │   └── triangle_mesh.hpp # Triangle mesh representation
│   ├── math/
│   │   ├── aabb.hpp          # Axis-aligned bounding box
│   │   └── vec3.hpp          # 3D vector math
│   ├── engine/
│   │   ├── camera.hpp        # Camera controls
│   │   ├── shader.hpp        # OpenGL shader utilities
│   │   └── mesh_renderer.hpp # OpenGL mesh rendering
│   ├── visualization/
│   │   └── leaf_cost.hpp     # Heatmap cost calculation
│   └── benchmark/
│       ├── timer.hpp         # High-resolution timer
│       └── sah_cost.hpp      # SAH cost computation
├── models/                   # OBJ model files
└── extern/
    └── glad/                 # OpenGL loader
```

## Models

Test models can be downloaded from the [McGuire Computer Graphics Archive](https://casual-effects.com/data/).

## License

This project was developed as part of academic coursework.
