#include "radix_sort_kv.cuh"
#include "../../include/common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// Compute bit positions for prefix sum
__global__ void compute_predicates_kv(
    const uint32_t* input_keys,
    int n,
    int bit,
    int* predicates,
    int* address_scan)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t key = input_keys[idx];
        int bit_val = (key >> bit) & 1;

        predicates[idx] = bit_val;

        address_scan[idx] = 1 - bit_val;
    }
}

// Scatter kernel for placing keys and values into their new positions
__global__ void scatter_kv(
    const uint32_t* input_keys,
    uint32_t* output_keys,
    const uint32_t* input_values,
    uint32_t* output_values,
    int n,
    int bit,
    const int* predicates,
    const int* zero_scan,
    const int* d_total_zeros_ptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int total_zeros = *d_total_zeros_ptr;
        uint32_t key = input_keys[idx];
        uint32_t value = input_values[idx];
        int bit_val = predicates[idx];

        int dst_idx = 0;

        if (bit_val == 0) {
            dst_idx = zero_scan[idx] - 1;
        } else {
            int ones_before = idx - zero_scan[idx];
            dst_idx = total_zeros + ones_before;
        }

        output_keys[dst_idx] = key;
        output_values[dst_idx] = value;
    }
}

// Main Radix Sort Function for 30-bit keys
void radixSortKeyValue30bit(uint32_t* d_keys, uint32_t* d_values, int n) {
    if (n <= 0) return;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    uint32_t* d_temp_keys;
    uint32_t* d_temp_values;
    int* d_predicates;
    int* d_scan;

    CUDA_CHECK(cudaMalloc(&d_temp_keys, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_temp_values, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_predicates, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan, n * sizeof(int)));

    // ping-pong buffers
    uint32_t* d_in_keys = d_keys;
    uint32_t* d_out_keys = d_temp_keys;
    uint32_t* d_in_values = d_values;
    uint32_t* d_out_values = d_temp_values;

    // Using 30 bits for sorting
    for (int bit = 0; bit < 30; ++bit) {
        // Predicates - prepare for Scan
        compute_predicates_kv<<<blocks, threads>>>(d_in_keys, n, bit, d_predicates, d_scan);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Inclusive Scan (Prefix Sum) to find offsets for the 0-bin, koristimo thrust :(
        thrust::inclusive_scan(thrust::device, d_scan, d_scan + n, d_scan);

        // Scatter elements to new positions
        scatter_kv<<<blocks, threads>>>(
            d_in_keys, d_out_keys,
            d_in_values, d_out_values,
            n, bit, d_predicates, d_scan, d_scan + n - 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers (Ping-Pong)
        uint32_t* temp_keys = d_in_keys;
        d_in_keys = d_out_keys;
        d_out_keys = temp_keys;

        uint32_t* temp_values = d_in_values;
        d_in_values = d_out_values;
        d_out_values = temp_values;
    }

    CUDA_CHECK(cudaFree(d_temp_keys));
    CUDA_CHECK(cudaFree(d_temp_values));
    CUDA_CHECK(cudaFree(d_predicates));
    CUDA_CHECK(cudaFree(d_scan));
}