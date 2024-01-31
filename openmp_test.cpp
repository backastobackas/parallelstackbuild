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
	for (int i = 0; i < Array_size; i++)
	{
		data[i] = i + array_start;
	}

#pragma omp parallel num_threads(Num_threads - 1)
	{
		int id = omp_get_thread_num();
		int start = id * (Array_size / (Num_threads - 1));
		int end = (id + 1) * (Array_size / (Num_threads - 1));

		maxIndexes[id] = end;

		for (int i = start; i < end; i++)
		{
			squares[i] = data[i] * data[i];

			if (data[i] < end + array_start)
			{
				maxIndexes[id] = i;
			}
		}
	}
#pragma omp parallel for
		for (int i = 0; i < Array_size; i++)
		{
			 mashed[i] = data[i] * squares[i];

		}

#pragma omp parallel num_threads(1) reduction(+:sum)
	{
		int id = omp_get_thread_num();

		if (id == 0)
		{
			std::cout << "Max Indexes for each thread: ";
			for (int i = 0; i < Num_threads - 1; ++i) {
				std::cout << "Thread " << i << ": " << maxIndexes[i] << " ";
			}
			std::cout << std::endl;

			std::cout << "Orgas: ";
			for (int i = 0; i < Array_size; i++)
			{
				std::cout << data[i] << " ";
				sum += data[i];
			}
			std::cout << std::endl;

			std::cout << "kvardratas: ";
			for (int i = 0; i < Array_size; i++)
			{
				std::cout << squares[i] << " ";
			}
			std::cout << std::endl;

			for (int i = 0; i < Array_size; i++)
			{
				if (squares[i] > maxval)
				{
					maxval = squares[i];
				}
			}
			std::cout << "max: " << maxval << std::endl;
			std::cout << "sum: " << sum << std::endl;
			std::cout << "mashed: ";
			for (int i = 0; i < Array_size; i++)
			{
				std::cout << mashed[i] << " ";
			}
			std::cout << std::endl;
		}
	}
	return 0;
}