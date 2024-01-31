#include <iostream>
#include <omp.h>
#include <vector>
#include <algorithm>

#define Array_size 100
#define Num_threads 5
#define array_start 0

int main() {
    int data[Array_size];
    int squares[Array_size];
    int maxval = 0;
    int sum = 0;
    int mashed[Array_size];
    int maxIndexes[Num_threads];

#pragma omp parallel for
    for (int i = 0; i < Array_size; i++) {
        data[i] = i + array_start;
    }

#pragma omp parallel num_threads(Num_threads - 1)
    {
        int id = omp_get_thread_num();
        int start = id * (Array_size / (Num_threads - 1));
        int end = (id + 1) * (Array_size / (Num_threads - 1));

        maxIndexes[id] = end;

        for (int i = start; i < end; i++) {
            squares[i] = data[i] * data[i];

            if (data[i] < end + array_start) {
                maxIndexes[id] = i;
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < Array_size; i++) {
        mashed[i] = data[i] * squares[i];
    }

    // Step 1: Create an array of indices
    std::vector<int> indices(Array_size);
    for (int i = 0; i < Array_size; ++i) {
        indices[i] = i;
    }

    // Step 2: Sort indices based on the corresponding values in the mashed array
    std::sort(indices.begin(), indices.end(),
        [&](int a, int b) { return mashed[a] > mashed[b]; });

    // Step 3: Create a new array with mashed elements in descending order
    int filterArray[Array_size];
#pragma omp parallel for
    for (int i = 0; i < Array_size; i++) {
        filterArray[i] = mashed[indices[i]];
    }

    // Print the results
    std::cout << "Mashed array: ";
    for (int i = 0; i < Array_size; i++) {
        std::cout << mashed[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Filter array in descending order: ";
    for (int i = 0; i < Array_size; i++) {
        std::cout << filterArray[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
