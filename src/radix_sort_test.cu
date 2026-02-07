#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// --- CUSTOM CUDA KERNELS ---

// Kernel 1: Compute Predicates (0 if bit is 0, 1 if bit is 1)
// We also store the inverted predicate for the scan (to calculate 0-bin destinations)
__global__ void compute_predicates(const uint32_t* input, int n, int bit, int* predicates, int* address_scan) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t val = input[idx];
        // Extract the specific bit (0 or 1)
        int bit_val = (val >> bit) & 1;

        predicates[idx] = bit_val;

        // For the 0-bin, we need to scan the *inverse* (where bit is 0)
        // This helps us find the destination for elements that have a 0 at this position
        address_scan[idx] = 1 - bit_val;
    }
}

// Kernel 2: Scatter
// Uses the scanned addresses to place elements into the Output array
__global__ void scatter(const uint32_t* input, uint32_t* output, int n, int bit,
                        const int* predicates, const int* zero_scan, int total_zeros) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t val = input[idx];
        int bit_val = predicates[idx];

        int dst_idx = 0;

        if (bit_val == 0) {
            // For 0s: Destination is the scan value - 1 (converting 1-based count to 0-based index)
            dst_idx = zero_scan[idx] - 1;
        } else {
            // For 1s: We need to know how many 1s were before us.
            // Total items before us = idx
            // Total 0s before us = zero_scan[idx] (Since our bit is 1, it didn't add to the zero count)
            // Therefore: Ones Before Us = idx - zero_scan[idx]

            int ones_before = idx - zero_scan[idx]; // <--- FIXED LINE (Removed the "- 1")
            dst_idx = total_zeros + ones_before;
        }

        output[dst_idx] = val;
    }
}

// Host Wrapper for Custom Radix Sort
void custom_radix_sort_30bit(thrust::device_vector<uint32_t>& d_vec) {
    int n = d_vec.size();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Allocate auxiliary memory
    thrust::device_vector<uint32_t> d_temp(n); // Double buffer
    thrust::device_vector<int> d_predicates(n);
    thrust::device_vector<int> d_scan(n);

    // Raw pointers for kernels
    uint32_t* d_in_ptr = thrust::raw_pointer_cast(d_vec.data());
    uint32_t* d_out_ptr = thrust::raw_pointer_cast(d_temp.data());
    int* d_pred_ptr = thrust::raw_pointer_cast(d_predicates.data());
    int* d_scan_ptr = thrust::raw_pointer_cast(d_scan.data());

    // Loop over 30 bits (0 to 29)
    for (int bit = 0; bit < 30; ++bit) {
        // 1. Compute Predicates and prepare for Scan
        compute_predicates<<<blocks, threads>>>(d_in_ptr, n, bit, d_pred_ptr, d_scan_ptr);
        cudaDeviceSynchronize();

        // 2. Exclusive Scan (Prefix Sum) to find offsets for the 0-bin
        // We use Thrust's highly optimized scan here to keep the "Custom" part focused on Radix logic
        // In-place inclusive scan for simpler index math in scatter
        thrust::inclusive_scan(thrust::device, d_scan.begin(), d_scan.end(), d_scan.begin());

        // Get total zeros (last element of scan)
        int total_zeros = d_scan[n - 1];

        // 3. Scatter elements to new position
        scatter<<<blocks, threads>>>(d_in_ptr, d_out_ptr, n, bit, d_pred_ptr, d_scan_ptr, total_zeros);
        cudaDeviceSynchronize();

        // Swap pointers (Ping-Pong)
        std::swap(d_in_ptr, d_out_ptr);
    }

    // If the final result ended up in d_temp (odd number of swaps?), copy back.
    // We swapped pointers. If d_in_ptr is now pointing to d_temp's data, we need to ensure
    // d_vec actually holds that data.
    // Since we did 30 iterations (even), d_in_ptr should point to the original d_vec memory
    // IF we started with d_in_ptr = d_vec.
    // Let's trace:
    // Start: In=A, Out=B
    // Loop 0: Read A, Write B. Swap -> In=B, Out=A
    // ...
    // Loop 29 (30th pass): Read B, Write A. Swap -> In=A, Out=B
    // Correct, the valid data is in A (d_vec).
}

// --- BENCHMARKING UTILS ---

void verify_sorted(const thrust::device_vector<uint32_t>& d_vec) {
    bool is_sorted = thrust::is_sorted(d_vec.begin(), d_vec.end());
    if (is_sorted) {
        std::cout << "VALIDITY: Passed" << std::endl;
    } else {
        std::cout << "VALIDITY: FAILED!" << std::endl;
    }
}

int main() {
    const size_t N = 1 << 24; // 16 Million elements
    std::cout << "Sorting " << N << " 30-bit integers..." << std::endl;

    // Generate Random Data (0 to 2^30)
    thrust::host_vector<uint32_t> h_data(N);
    for (size_t i = 0; i < N; i++) {
        h_data[i] = rand() & 0x3FFFFFFF; // Mask to 30 bits
    }

    // --- 1. BENCHMARK CUSTOM SORT ---
    thrust::device_vector<uint32_t> d_custom = h_data;

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    custom_radix_sort_30bit(d_custom);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "\n--- Custom Radix-2 Implementation ---" << std::endl;
    std::cout << "Time: " << diff.count() << " s" << std::endl;
    std::cout << "Throughput: " << (N / diff.count()) / 1e6 << " MKeys/s" << std::endl;
    verify_sorted(d_custom);

    // --- 2. BENCHMARK THRUST SORT ---
    thrust::device_vector<uint32_t> d_thrust = h_data;

    cudaDeviceSynchronize();
    start = std::chrono::high_resolution_clock::now();

    thrust::sort(d_thrust.begin(), d_thrust.end());

    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;

    std::cout << "\n--- Thrust (Merrill Optimized) ---" << std::endl;
    std::cout << "Time: " << diff.count() << " s" << std::endl;
    std::cout << "Throughput: " << (N / diff.count()) / 1e6 << " MKeys/s" << std::endl;
    verify_sorted(d_thrust);

    return 0;
}