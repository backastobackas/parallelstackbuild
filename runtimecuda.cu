#include "cuda_runtime.h"
#include <iostream>

#define N 100
#define THREADS_PER_BLOCK 10
#define START_VALUE 0

__global__ void printArrayAndFindMax(int* d_array, int* d_max) {
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += THREADS_PER_BLOCK) {
        int value = d_array[i] + START_VALUE;
        int square = value * value;
        printf("Thread %d: Value: %d, Square: %d\n", tid, value, square);

        // Use atomicMax to find the maximum square value
        atomicMax(d_max, square);
    }
}

int main() {
    int h_array[N];
    int* d_array;
    int* d_max;

    // Allocate device memory
    cudaMalloc((void**)&d_array, N * sizeof(int));
    cudaMalloc((void**)&d_max, sizeof(int));

    // Initialize host array with values 0 to 99
    for (int i = 0; i < N; ++i) {
        h_array[i] = i;
    }

    // Copy host array to device
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize max to 0 on the device
    cudaMemset(d_max, 0, sizeof(int));

    // Launch kernel to print array values, squares, and find the maximum square value
    printArrayAndFindMax << <1, THREADS_PER_BLOCK >>> (d_array, d_max);

    // Synchronize to wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the maximum square value back to the host
    int max_square = 0;
    cudaMemcpy(&max_square, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Maximum Square Value: " << max_square << std::endl;

    // Free allocated memory
    cudaFree(d_array);
    cudaFree(d_max);

    return 0;
}
