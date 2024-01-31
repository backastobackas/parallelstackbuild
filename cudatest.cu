#include "cuda_runtime.h"
#include <iostream>

#define N 100
#define THREADS_PER_BLOCK 10
#define START_VALUE 0

__device__ bool isPrime(int num)
{
    if (num < 2)
    {
        return false;
    }

    for (int i = 2; i * i <= num; ++i)
    {
        if (num % i == 0)
        {
            return false;
        }
    }

    return true;
}
__global__ void printArrayAndFindPrimes(int* d_array, int* d_primes, int* d_prime_count)
{
    int tid = threadIdx.x;
    int prime_count = 0;

    for (int i = tid; i < N; i += THREADS_PER_BLOCK)
    {
        int value = d_array[i] + START_VALUE;
        int square = value * value;
        printf("Threads %d: Value %d: Square %d: \n ", tid, value, square);

        if (isPrime(square))
        {
            d_primes[prime_count++] = square;
        }
    }

    // Store the count of primes in the provided variable
    if (tid == 0)
    {
        *d_prime_count = prime_count;
    }
}

__global__ void primes(int* result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int start = index * 40;

    for (int i = 0; i < 40; i++)
    {
        result[start + i] = 0;  // Initialize result array to 0
        bool prime = true;
        int value = 1000 + start + i;
        for (int j = 2; j * j <= value; j++)
        {
            if (value % j == 0)
            {
                prime = false;
                break;
            }
        }
        if (prime)
        {
            result[start + i] = value;
        }
    }
}

int main()
{
    int h_array[N];
    int* d_array;
    int* d_primes;
    int* d_prime_count;
    int h_prime_count;

    // Allocate device memory
    cudaMalloc((void**)&d_array, N * sizeof(int));
    cudaMalloc((void**)&d_primes, N * sizeof(int));
    cudaMalloc((void**)&d_prime_count, sizeof(int));

    // Initialize host array with values 0 to 99
    for (int i = 0; i < N; ++i)
    {
        h_array[i] = i;
    }

    // Copy host array to device
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to print array values, find primes, and store them in d_primes array
    printArrayAndFindPrimes << <1, THREADS_PER_BLOCK >> > (d_array, d_primes, d_prime_count);

    // Synchronize to wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the count of primes back to the host
    cudaMemcpy(&h_prime_count, d_prime_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy the primes back to the host
    int* h_primes = new int[h_prime_count];
    cudaMemcpy(h_primes, d_primes, h_prime_count * sizeof(int), cudaMemcpyDeviceToHost);

    // Print primes
    std::cout << "Prime Numbers: ";
    for (int i = 0; i < h_prime_count; ++i)
    {
        std::cout << h_primes[i] << " ";
    }
    std::cout << std::endl;

    // Free allocated memory
    cudaFree(d_array);
    cudaFree(d_primes);
    cudaFree(d_prime_count);
    delete[] h_primes;

    return 0;
}
