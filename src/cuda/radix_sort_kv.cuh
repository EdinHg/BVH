#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * Custom 30-bit Radix Sort for Key-Value pairs
 * 
 * Sorts Morton codes (keys) and their associated triangle indices (values)
 * using a bit-by-bit radix sort algorithm (LSB first, 30 bits).
 * 
 * This implementation uses:
 * - Radix-2 (bit-by-bit) sorting
 * - Double buffering (ping-pong between input and temporary buffers)
 * - Thrust inclusive_scan for prefix sum computation
 * - In-place sorting (results stored back in d_keys/d_values)
 * 
 * @param d_keys    Device pointer to keys (Morton codes) - sorted in-place
 * @param d_values  Device pointer to values (triangle indices) - reordered to match sorted keys
 * @param n         Number of elements to sort
 */
void radixSortKeyValue30bit(uint32_t* d_keys, uint32_t* d_values, int n);
