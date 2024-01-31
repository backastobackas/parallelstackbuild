#include <stdio.h>
#include <iostream>
#define THREADS_PER_BLOCK 10
#define N 100

__global__ void printNumbers(const int* h_array, int* h_array_sqrt, int* max) {
    int tid = threadIdx.x;
    int localMax = 0;
    for (int i = tid; i < N; i += 10) {
        printf("Thread %d: %d\n", tid, i);
        h_array_sqrt[i] = i * i;
        if (localMax < i * i) {
            localMax = i * i;
        }
        atomicMax(max, localMax);
    }
}

int main() {
    int h_array[N];
    int h_array_sqrt[N];
    int* d_array;
    int* d_array_sqrt;
    int* d_max;
    cudaMalloc((void**)&d_array, N * sizeof(int));
    cudaMalloc((void**)&d_array_sqrt, N * sizeof(int));
    cudaMalloc((void**)&d_max, sizeof(int));

    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    printNumbers << <1, 10 >> > (d_array, d_array_sqrt, d_max);

    int max = 0;

    cudaDeviceSynchronize();
    cudaMemcpy(h_array_sqrt, d_array_sqrt, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_array_sqrt);

    for (int i = 0; i < N; i++) {
        std::cout << "squared " << h_array_sqrt[i] << std::endl;
    }
    std::cout << "MAX " << max;
    return 0;
}