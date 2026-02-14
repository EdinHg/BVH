# GPU BVH Construction Benchmark

A comprehensive CUDA-based benchmark comparing multiple Bounding Volume Hierarchy (BVH) construction algorithms with integrated GPU ray tracing for visualization and quality evaluation.

## Overview

This project implements and benchmarks three GPU-accelerated BVH construction algorithms:

1. **LBVH (Linear BVH)** 
2. **LBVH+ (Enhanced LBVH)** 
3. **PLOC (Parallel Locally-Ordered Clustering)** 

All implementations are optimized for NVIDIA GPUs using CUDA and provide detailed performance metrics including build times, SAH cost analysis, and traversal statistics.

## Features

### BVH Construction
- **Multiple Construction Algorithms**: Compare LBVH, LBVH+, and PLOC with configurable parameters
- **CUB Radix Sort**: High-performance radix sort for Morton code ordering (LBVH)
- **Treelet Optimization**: Surface area heuristic-based optimization pass (LBVH+)
- **Configurable PLOC Radius**: Test different search radii (e.g., 10, 25, 100) for PLOC

### Input/Output
- **OBJ File Support**: Load and process triangle meshes from OBJ files (triangles, quads, n-gons)
- **Synthetic Mesh Generation**: Create random triangle meshes for scalability testing
- **BVH Export**: Export constructed BVH to binary or OBJ format for external visualization
- **CSV Export**: Export detailed statistics for analysis

### Performance Analysis
- **Detailed Timing Breakdown**: Per-stage timing for each construction phase
- **SAH Cost Evaluation**: Surface Area Heuristic cost calculation
- **Tree Statistics**: Node count, leaf count, max depth, average leaf depth
- **Throughput Metrics**: Million triangles per second

### GPU Ray Tracing Engine
- **Four Shading Modes**:
  - **Normal**: Surface normals mapped to RGB colors
  - **Depth**: Grayscale depth buffer visualization
  - **Diffuse**: Simple directional lighting with ambient occlusion
  - **Heatmap**: BVH traversal cost visualization (nodes visited per ray)
- **Flexible Camera Control**: Automatic scene framing or manual camera positioning
- **Traversal Statistics**: Average/max nodes visited, AABB tests, triangle intersections
- **PPM Image Output**: Standard image format for easy viewing

### Batch Testing System
- **JSON Configuration**: Define test suites with multiple models and algorithms
- **Statistical Aggregation**: Mean, standard deviation, min, max across iterations
- **Warmup Iterations**: Optional warmup pass before timing measurements
- **Progress Tracking**: Monitor batch execution progress
- **Integrated Rendering**: Optionally render images during batch tests

## Project Structure

```
BVH/
├── include/                    # Header files
│   ├── bvh_builder.h          # Abstract BVH builder interface
│   ├── bvh_node.h             # Unified BVH node structure
│   ├── common.h               # CUDA utilities and error checking
│   ├── mesh.h                 # Structure-of-Arrays triangle mesh
│   ├── evaluator.h            # SAH calculation and quality metrics
│   ├── render_engine.h        # GPU ray tracing engine interface
│   ├── batch_config.h         # Batch configuration structures
│   └── batch_runner.h         # Batch test runner
├── src/
│   ├── cuda/                  # CUDA implementations
│   │   ├── lbvh_builder.cu           # LBVH with CUB radix sort
│   │   ├── lbvh_builder.cuh
│   │   ├── lbvh_plus_builder.cu      # LBVH+ with Thrust and optimization
│   │   ├── lbvh_plus_builder.cuh
│   │   ├── ploc_builder.cu           # PLOC algorithm
│   │   └── ploc_builder.cuh
│   ├── main.cpp               # Main benchmark driver
│   ├── loader.cpp             # OBJ loading and mesh generation
│   ├── evaluator.cpp          # BVH quality evaluation
│   ├── batch_config.cpp       # JSON configuration parsing
│   ├── batch_runner.cpp       # Batch testing orchestration
│   └── render_engine.cu       # GPU ray tracing implementation
├── models/                    # OBJ test models
│   ├── buddha.obj
│   ├── conference.obj
│   ├── dragon.obj
│   ├── gallery.obj
│   ├── hairball.obj
│   ├── monkey.obj
│   ├── powerplant.obj
│   ├── sibnek.obj
│   └── sponza.obj
├── docs/                      # Documentation
│   └── render_engine.md
├── batch_config.json          # Example batch configuration
├── CMakeLists.txt            # Build configuration
└── README.md                 # This file
```

## Requirements

- **CUDA Toolkit** 11.0 or later
- **CMake** 3.18 or later
- **C++ Compiler** with C++17 support
- **NVIDIA GPU** with compute capability 6.0 or higher
- **CUB** (included with CUDA Toolkit)
- **Thrust** (included with CUDA Toolkit)

## Building

### 1. Clone the repository

```bash
git clone <repository-url>
cd BVH
```

### 2. Create build directory

```bash
mkdir build
cd build
```

### 3. Configure with CMake

```bash
cmake ..
```

**Note**: CMake will automatically detect your GPU architecture. To specify manually:

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..  # For RTX 30-series
cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..  # For RTX 40-series
```

### 4. Build the project

```bash
cmake --build . --config Release
```

Or on Linux/macOS:

```bash
make -j$(nproc)
```

### 5. Run the benchmark

```bash
./bvh_benchmark [options]
```

## Usage

### Command-Line Options

#### Input Options
```
  -i, --input <file>          Load OBJ file
  -n, --triangles <count>     Generate N random triangles (default: 1000000)
```

#### Algorithm Selection
```
  -a, --algorithm <name>      Run specific algorithm: lbvh, lbvh+, ploc, all
  -r, --radius <value>        Set search radius for PLOC (default: 25)
```

#### Export Options
```
  -o, --output <file>         Export BVH to file (OBJ or binary)
  -c, --colab-export          Export as binary format
  -l, --leaves-only           Export only leaf bounding boxes
  --csv-export <file>         Export statistics to CSV file
```

#### Rendering Options
```
  --render <prefix>           Render each BVH to <prefix>_<algo>.ppm
  --render-size <WxH>         Image resolution (default: 1024x768)
  --shading <mode>            Shading mode: normal|depth|diffuse|heatmap (default: normal)
  --camera <ex,ey,ez,lx,ly,lz>   Camera position and look-at point
  --camera-up <ux,uy,uz>      Camera up vector (default: 0,1,0)
  --fov <degrees>             Vertical field of view (default: 60)
```

#### Batch Testing
```
  --batch <config.json>       Run batch tests from JSON configuration
  --batch-quiet               Suppress output except errors
```

#### Other
```
  -h, --help                  Show help message
```

### Examples

#### Interactive Mode

**Run all algorithms on 10 million random triangles:**
```bash
./bvh_benchmark -n 10000000
```

**Load OBJ file and test specific algorithm:**
```bash
./bvh_benchmark -i models/dragon.obj -a lbvh+
```

**Generate BVH and export for visualization:**
```bash
./bvh_benchmark -n 1000000 -o bvh.bin -c
```

**Compare PLOC with custom radius:**
```bash
./bvh_benchmark -n 5000000 -a ploc -r 30
```

**Render with heatmap shading to visualize BVH quality:**
```bash
./bvh_benchmark -i models/sponza.obj -a all --render output --shading heatmap
```

**Test with custom camera and export statistics:**
```bash
./bvh_benchmark -i models/buddha.obj -a lbvh+ \
  --render buddha --camera 0,2,5,0,1,0 --fov 45 \
  --csv-export results.csv
```

#### Batch Mode

**Run comprehensive batch tests:**
```bash
./bvh_benchmark --batch batch_config.json
```

**Run batch tests quietly (errors only):**
```bash
./bvh_benchmark --batch batch_config.json --batch-quiet
```

**Example batch_config.json:**
```json
{
  "iterations": 5,
  "warmup": true,
  "quiet": false,
  "output": "benchmark_results.csv",
  "algorithms": ["lbvh", "lbvh+", "ploc"],
  "ploc_radius": [10, 25, 100],
  "models": [
    {
      "type": "obj",
      "path": "models/dragon.obj",
      "name": "Dragon"
    },
    {
      "type": "random",
      "triangles": 1000000,
      "name": "Random 1M"
    }
  ],
  "render": {
    "enabled": true,
    "prefix": "batch_render",
    "size": "1920x1080",
    "shading": "heatmap"
  }
}
```

## Output

### Console Output

The benchmark provides detailed output for each algorithm:

```
Building BVH with LBVH...
  Build time: 12.34 ms
  Morton codes: 1.23 ms
  Sorting: 4.56 ms
  Hierarchy: 3.45 ms
  AABB: 2.10 ms

Tree Statistics:
  Total nodes: 1999998
  Leaf nodes: 1000000
  Internal nodes: 999998
  Max depth: 42
  Avg leaf depth: 28.3

SAH Cost: 2.45e+06
Throughput: 81.04 Mtris/s
```

### Rendering Output

Rendered images are saved as PPM files:
- `output_lbvh.ppm` - LBVH render
- `output_lbvh+.ppm` - LBVH+ render
- `output_ploc.ppm` - PLOC render

**Heatmap mode** shows BVH traversal efficiency:
- Blue/Cool colors: Efficient (fewer nodes traversed)
- Red/Hot colors: Inefficient (many nodes traversed)

### CSV Export

Statistics are exported in CSV format with columns:
- Model name
- Triangle count
- Algorithm name
- Build time (ms)
- SAH cost
- Total nodes
- Leaf nodes
- Max depth
- Average leaf depth
- Throughput (Mtris/s)

## Test Models

The project includes several test models in the `models/` directory:
- **buddha.obj** - Stanford Buddha
- **conference.obj** - Conference room scene
- **dragon.obj** - Stanford Dragon
- **gallery.obj** - Art gallery interior
- **hairball.obj** - Complex hairball geometry
- **monkey.obj** - Suzanne monkey head
- **powerplant.obj** - Industrial powerplant scene
- **sibnek.obj** - Sibenik Cathedral
- **sponza.obj** - Crytek Sponza Atrium

Additional models can be downloaded from:

**[McGuire Computer Graphics Archive](https://casual-effects.com/data/)**

This archive contains a wide variety of high-quality 3D models suitable for BVH testing, including San Miguel, Bistro, and many more.

Place downloaded OBJ files in the `models/` directory.

## Algorithm Comparison

| Algorithm | Sorting Library | Optimization | Typical Speed | Quality (SAH) |
|-----------|----------------|--------------|---------------|---------------|
| LBVH      | CUB           | None         | Fastest       | Good          |
| LBVH+     | Thrust        | Treelet      | Fast          | Better        |
| PLOC      | Thrust        | Clustering   | Moderate      | Best          |

**LBVH** is the fastest construction method, suitable for dynamic scenes requiring frequent BVH rebuilds.

**LBVH+** adds a post-processing optimization pass that restructures small subtrees (treelets) using SAH, improving quality with minimal performance overhead.

**PLOC** produces the highest quality BVHs by clustering nearby primitives during construction. The search radius parameter controls the trade-off between build time and quality.

## Performance Tips

1. **Use LBVH for dynamic scenes** where BVH needs frequent rebuilding
2. **Use LBVH+ for balanced performance** when you need better quality without much slowdown
3. **Use PLOC for static scenes** where build time is less critical than render performance
4. **Test different PLOC radii** (e.g., 10, 25, 100) to find optimal trade-offs
5. **Use batch mode** for comprehensive benchmarking across multiple models
6. **Use heatmap rendering** to visualize BVH quality and identify problem areas

## License

This project is for academic and research purposes.

## References

This project implements algorithms from the following papers:

1. **Tero Karras.** "Maximizing parallelism in the construction of BVHs, octrees, and k-d trees." *Proceedings of the Fourth ACM SIGGRAPH/Eurographics Conference on High-Performance Graphics*, 2012, pp. 33-37.  
   [https://doi.org/10.2312/EGGH/HPG12/033-037](https://doi.org/10.2312/EGGH/HPG12/033-037)

2. **Tero Karras and Timo Aila.** "Fast parallel construction of high-quality bounding volume hierarchies." *Proceedings of the 5th High-Performance Graphics Conference*, 2013, pp. 89-99.  
   [https://doi.org/10.1145/2492045.2492055](https://doi.org/10.1145/2492045.2492055)

3. **Daniel Meister and Jiří Bittner.** "Parallel locally-ordered clustering for bounding volume hierarchy construction." *IEEE Transactions on Visualization and Computer Graphics*, vol. 24, no. 3, 2017, pp. 1345-1353.  
   [https://doi.org/10.1109/TVCG.2017.2669983](https://doi.org/10.1109/TVCG.2017.2669983)
