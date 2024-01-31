#include <iostream>
#include <stack>
#include <thread>
#include <mutex>

std::stack<int> orgstack;
std::stack<int> squaresstack;
std::mutex myMutex;

const int N = 100;
const int Num_threads = 10;
const int start_value = 0;

void squarenumbers()
{
	std::unique_lock<std::mutex > lock(myMutex);

	while (!orgstack.empty())
	{
		int value = orgstack.top();
		orgstack.pop();
		squaresstack.push(value*value);
	}
}

int Findmax()
{
	std::unique_lock<std::mutex > lock(myMutex);
	int maxval = 0;

	while (!squaresstack.empty())
	{
		int max = squaresstack.top();
		squaresstack.pop();
		if (max > maxval)
		{
			maxval = max;
		}
	}
	return maxval;
}
int main() {
	for (int i = 0; i < N; i++)
	{
		orgstack.push(i + start_value);
	}
	std::thread threads[Num_threads];
	for (int i = 0; i < Num_threads; i++)
	{
		threads[i] = std::thread(squarenumbers);
	}
	for (int i = 0; i < Num_threads; i++)
	{
		threads[i].join();
	}
	std::stack<int> maxstack = squaresstack;
	int maxval = Findmax();
	std::cout << "squared: ";
	while (!maxstack.empty())
	{
		std::cout << maxstack.top() << " ";
		maxstack.pop();
	}
	std::cout << std::endl;

	
	std::cout << "max: " << maxval;
	return 0;
}