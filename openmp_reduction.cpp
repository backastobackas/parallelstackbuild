#include <iostream>
#include <omp.h>

int main() {
    const int arraySize = 10;
    int array1[arraySize] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int array2[arraySize] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

    int result[arraySize];
    int sum = 0;  // Variable for summing the result array

    // Dalis A: Sudauginame masyvus pagal indeksà
#pragma omp parallel for
    for (int i = 0; i < arraySize; ++i) {
        result[i] = array1[i] * array2[i];
    }

    // Susumuojame rezultatø masyvà
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < arraySize; ++i) {
        sum += result[i];
    }

    // Dalis B: Kiekvieno threado max indekso radimas ir atspausdinimas
    int maxIndex = 0;

#pragma omp parallel private(maxIndex)
    {
#pragma omp for
        for (int i = 0; i < arraySize; ++i) {
            if (i > maxIndex) {
                maxIndex = i;
            }
        }

#pragma omp single
        {
            std::cout << "Max indeksas: " << maxIndex << std::endl;
        }
    }

    // Atspausdiname rezultatus ir suma
    std::cout << "Sudaugintø masyvø rezultatai: ";
    for (int i = 0; i < arraySize; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Suma: " << sum << std::endl;

    return 0;
}
