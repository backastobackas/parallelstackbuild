#include <iostream>
#include <fstream>
#include <omp.h>

#define ARRAY_SIZE 1000
#define NUM_THREADS 10

int main() {
    int data[ARRAY_SIZE];
    float averages[NUM_THREADS];
    float total = 0.0;

    // Initialize the array
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        data[i] = i;
    }

    // Task 1: Calculate averages and total
#pragma omp parallel num_threads(NUM_THREADS)
    {
        int id = omp_get_thread_num();
        int start = id * (ARRAY_SIZE / NUM_THREADS);
        int end = (id + 1) * (ARRAY_SIZE / NUM_THREADS);

        float sum = 0.0;
        for (int i = start; i < end; ++i) {
            sum += data[i];
        }

        averages[id] = sum / (ARRAY_SIZE / NUM_THREADS);

#pragma omp critical
        {
            total += sum; // Safely accumulate to total within a critical section
        }
    }

    // Write averages to a file
    std::ofstream resultsFile("results.txt");
    for (int i = 0; i < NUM_THREADS; ++i) {
        resultsFile << "Average for thread " << i << ": " << averages[i] << "\n";
    }
    resultsFile.close();

    // Print total
    std::cout << "Total: " << total << std::endl;

    return 0;
}