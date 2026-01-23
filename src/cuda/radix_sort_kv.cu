#include "radix_sort_kv.cuh"
#include "../../include/common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// --- CUSTOM CUDA KERNELS FOR KEY-VALUE RADIX SORT ---

/**
 * Kernel 1: Compute Predicates
 * 
 * For each element, extracts the bit at position 'bit' from the key (Morton code).
 * Stores:
 * - predicates[idx] = bit value (0 or 1)
 * - address_scan[idx] = inverse (1 - bit_val), used for counting zeros
 * 
 * The address_scan array will be prefix-summed to determine destination
 * indices for elements with bit=0.
 */
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
        // Extract the specific bit (0 or 1)
        int bit_val = (key >> bit) & 1;

        predicates[idx] = bit_val;

        // For the 0-bin, we need to scan the *inverse* (where bit is 0)
        // This helps us find the destination for elements that have a 0 at this position
        address_scan[idx] = 1 - bit_val;
    }
}

/**
 * Kernel 2: Scatter (Key-Value Version)
 * 
 * Uses the scanned addresses to place both keys and values into their
 * sorted positions for this bit position.
 * 
 * Elements with bit=0 go to the front (indices 0 to total_zeros-1)
 * Elements with bit=1 go to the back (indices total_zeros to n-1)
 * 
 * Both the key (Morton code) and value (triangle index) are moved together,
 * maintaining their association.
 */
__global__ void scatter_kv(
    const uint32_t* input_keys,
    uint32_t* output_keys,
    const uint32_t* input_values,
    uint32_t* output_values,
    int n,
    int bit,
    const int* predicates,
    const int* zero_scan,
    int total_zeros)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t key = input_keys[idx];
        uint32_t value = input_values[idx];
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
            int ones_before = idx - zero_scan[idx];
            dst_idx = total_zeros + ones_before;
        }

        // Move both key and value to maintain association
        output_keys[dst_idx] = key;
        output_values[dst_idx] = value;
    }
}

/**
 * Host Wrapper for Custom Key-Value Radix Sort
 * 
 * Performs a 30-bit radix sort on key-value pairs where:
 * - Keys are Morton codes (30-bit unsigned integers)
 * - Values are triangle indices (associated data)
 * 
 * The sort is stable and maintains the key-value association throughout.
 * Results are stored back in the original d_keys and d_values arrays.
 * 
 * Memory Management:
 * - Allocates temporary buffers internally
 * - Uses double-buffering (ping-pong) between original and temp arrays
 * - Frees all temporary memory before returning
 * 
 * Algorithm:
 * For each of 30 bits (LSB to MSB):
 *   1. Compute predicates (which bin each element belongs to)
 *   2. Prefix sum to calculate destination offsets
 *   3. Scatter elements to their new positions
 *   4. Swap input/output pointers
 * 
 * After 30 iterations (even number), final result is in original arrays.
 */
void radixSortKeyValue30bit(uint32_t* d_keys, uint32_t* d_values, int n) {
    if (n <= 0) return;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Allocate auxiliary memory
    uint32_t* d_temp_keys;
    uint32_t* d_temp_values;
    int* d_predicates;
    int* d_scan;

    CUDA_CHECK(cudaMalloc(&d_temp_keys, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_temp_values, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_predicates, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan, n * sizeof(int)));

    // Pointers for ping-pong buffering
    uint32_t* d_in_keys = d_keys;
    uint32_t* d_out_keys = d_temp_keys;
    uint32_t* d_in_values = d_values;
    uint32_t* d_out_values = d_temp_values;

    // Loop over 30 bits (0 to 29)
    for (int bit = 0; bit < 30; ++bit) {
        // 1. Compute Predicates and prepare for Scan
        compute_predicates_kv<<<blocks, threads>>>(d_in_keys, n, bit, d_predicates, d_scan);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. Inclusive Scan (Prefix Sum) to find offsets for the 0-bin
        // We use Thrust's highly optimized scan for the prefix sum computation
        thrust::inclusive_scan(thrust::device, d_scan, d_scan + n, d_scan);

        // Get total zeros (last element of scan)
        int total_zeros;
        CUDA_CHECK(cudaMemcpy(&total_zeros, d_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost));

        // 3. Scatter elements (both keys and values) to new positions
        scatter_kv<<<blocks, threads>>>(
            d_in_keys, d_out_keys,
            d_in_values, d_out_values,
            n, bit, d_predicates, d_scan, total_zeros);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. Swap pointers (Ping-Pong)
        // After swap, output becomes input for next iteration
        uint32_t* temp_keys = d_in_keys;
        d_in_keys = d_out_keys;
        d_out_keys = temp_keys;

        uint32_t* temp_values = d_in_values;
        d_in_values = d_out_values;
        d_out_values = temp_values;
    }

    // After 30 iterations (even number), the sorted data is back in the original arrays
    // d_in_keys now points to d_keys, d_in_values points to d_values
    // No copy-back needed!

    // Free temporary memory
    CUDA_CHECK(cudaFree(d_temp_keys));
    CUDA_CHECK(cudaFree(d_temp_values));
    CUDA_CHECK(cudaFree(d_predicates));
    CUDA_CHECK(cudaFree(d_scan));
}
