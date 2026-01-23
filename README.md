# GPU BVH Construction Benchmark

A comprehensive CUDA-based benchmark comparing multiple Bounding Volume Hierarchy (BVH) construction algorithms for ray tracing acceleration structures.

## Overview

This project implements and benchmarks three GPU-accelerated BVH construction algorithms:

1. **LBVH (Linear BVH)** - Uses Thrust library for sorting Morton codes
2. **LBVH-NoThrust** - Custom implementation with integrated radix sort (no Thrust dependency for sorting)
3. **PLOC (Parallel Locally-Ordered Clustering)** - Advanced parallel construction algorithm

All implementations are optimized for NVIDIA GPUs using CUDA and provide detailed performance metrics including build times, traversal quality, and memory usage.

## Features

- **Multiple BVH Construction Algorithms**: Compare LBVH, LBVH-NoThrust, and PLOC
- **Custom Radix Sort**: 30-bit key-value radix sort optimized for Morton codes
- **OBJ File Support**: Load and process triangle meshes from OBJ files
- **Synthetic Mesh Generation**: Create random triangle meshes for testing
- **Performance Metrics**: Detailed timing breakdown for each construction stage
- **BVH Export**: Export constructed BVH to binary or OBJ format for visualization
- **Traversal Quality Evaluation**: Measure BVH quality through ray casting tests

## Project Structure

```
BVH/
├── include/                    # Header files
│   ├── bvh_builder.h          # Abstract BVH builder interface
│   ├── bvh_node.h             # BVH node structure
│   ├── common.h               # Common utilities and CUDA error checking
│   ├── evaluator.h            # BVH quality evaluation
│   └── mesh.h                 # Triangle mesh data structures
├── src/
│   ├── cuda/                  # CUDA implementations
│   │   ├── lbvh_builder.cu           # LBVH with Thrust
│   │   ├── lbvh_builder.cuh
│   │   ├── lbvh_builder_nothrust.cu  # LBVH with custom radix sort
│   │   ├── lbvh_builder_nothrust.cuh
│   │   ├── ploc_builder.cu           # PLOC algorithm
│   │   ├── ploc_builder.cuh
│   │   ├── radix_sort_kv.cu          # Custom key-value radix sort
│   │   └── radix_sort_kv.cuh
│   ├── main.cpp               # Main benchmark driver
│   ├── loader.cpp             # Mesh loading utilities
│   ├── evaluator.cpp          # BVH quality evaluation
│   └── radix_sort_test.cu     # Standalone radix sort test
├── models/                    # Place OBJ files here
├── CMakeLists.txt            # Build configuration
└── README.md                 # This file
```

## Requirements

- **CUDA Toolkit** 11.0 or later
- **CMake** 3.18 or later
- **C++ Compiler** with C++17 support
- **NVIDIA GPU** with compute capability 6.0 or higher

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

```
Options:
  -i, --input <file>        Load OBJ file
  -n, --triangles <count>   Generate N random triangles (default: 1000000)
  -a, --algorithm <name>    Run specific algorithm (lbvh, lbvh-nothrust, ploc, all)
  -o, --output <file>       Export BVH to file
  -c, --colab-export        Export as binary (for Colab visualization)
  -l, --leaves-only         Export only leaf bounding boxes
  -r, --radius <value>      Set search radius for PLOC (default: 25)
  -h, --help                Show help message
```

### Examples

**Run all algorithms on 10 million random triangles:**
```bash
./bvh_benchmark -n 10000000
```

**Load OBJ file and test specific algorithm:**
```bash
./bvh_benchmark -i models/bunny.obj -a lbvh-nothrust
```

**Generate BVH and export for visualization:**
```bash
./bvh_benchmark -n 1000000 -o bvh.bin -c
```

**Compare PLOC with custom radius:**
```bash
./bvh_benchmark -n 5000000 -a ploc -r 30
```

## Test Models

Download test models from the McGuire Computer Graphics Archive:

**[McGuire Computer Graphics Archive](https://casual-effects.com/data/)**

This archive contains a wide variety of high-quality 3D models suitable for BVH testing, including:
- San Miguel
- Sponza
- Crytek Sponza
- Sibenik Cathedral
- Conference Room
- And many more...

Place downloaded OBJ files in the `models/` directory.

## License

This project is for academic and research purposes.

## References

- **LBVH**: [T. Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (2012)](https://www.researchgate.net/publication/262394649_Maximizing_Parallelism_in_the_Construction_of_BVHs_Octrees_and_k-d_Trees)
- **PLOC**: [Meister & Bittner, "Parallel Locally-Ordered Clustering for BVH Construction" (2018)](https://ieeexplore.ieee.org/document/7857089)
- **Radix Sort**: [D. Merrill & A. Grimshaw, "High performance and scalable radix sorting: a case study of implementing parallelism for gpu computing" (2011)](https://www.researchgate.net/publication/220440139_High_Performance_and_Scalable_Radix_Sorting_a_Case_Study_of_Implementing_Dynamic_Parallelism_for_GPU_Computing)