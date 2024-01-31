#include <iostream>
#include <omp.h>

int main() {

    int array[25][20];

    for (int i = 0; i < 25; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            array[i][j] = j;
        }
    }

    int sum = 0;
    int sortedSums[25] = { 0 };


#pragma omp parallel
    {
        int localSums[25] = { 0 };

#pragma omp parallel for num_threads(4) reduction(+:sum)
        for (int i = 0; i < 25; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                sum += array[i][j];
                localSums[i] += array[i][j];
            }

#pragma omp critical
            {
                // Insert the sum into the sortedSums array in descending order
                int k = i;
                while (k > 0 && localSums[i] > sortedSums[k - 1]) {
                    sortedSums[k] = sortedSums[k - 1];
                    k--;
                }
                sortedSums[k] = localSums[i];
            }

        }
    }
    std::cout << sum;

    // Print sorted partial sums
    std::cout << "Sorted Partial Sums:\n";
    for (int i = 0; i < 25; i++) {
        std::cout << sortedSums[i] << " ";
    }


}