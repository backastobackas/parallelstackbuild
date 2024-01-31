#include "cuda_runtime.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>

using namespace std;

__global__ void primes(int* result);

int main() {
    int* doubled_matrix;
    // result will be held in this array
    int* doubled_matrix_host = new int[1000];
    // fill array with random values

    cudaMalloc((void**)&doubled_matrix, 1000 * sizeof(int));

    primes << <1, 25 >> > (doubled_matrix);
    cudaDeviceSynchronize();

    cudaMemcpy(doubled_matrix_host, doubled_matrix, 1000 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1000; i++) {
        if (doubled_matrix_host[i] != 0) {
            cout << doubled_matrix_host[i] << " ";
        }
    }

    return 0;
}
// Function that is run on GPU as many times as there elements in our matrix. One thread computes one element in the 
// result matrix.
__global__ void primes(int* result) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int start = index * 40;

    bool prime;

    for (int i = 0; i < 40; i++) {
        prime = true;
        int value = 1000 + start + i;
        for (int j = 2; j * j <= value; j++) {
            if (value % j == 0) {
                prime = false;
                break;
            }
        }
        if (prime) {
            result[start + i] = value;
        }
    }
}

