#include <iostream>
#include <stack>
#include <thread>
#include <mutex>
#include <vector>

std::stack<int> orgstack;
std::stack<int> squaresstack;
std::mutex myMutex;

const int N = 100;
const int Num_threads = 10;
const int start_value = 0;

void squarenumbers()
{
    std::unique_lock<std::mutex> lock(myMutex);

    while (!orgstack.empty())
    {
        int value = orgstack.top();
        squaresstack.push(value * value);
        orgstack.pop();
    }
}

int Findmax()
{
    std::unique_lock<std::mutex> lock(myMutex);
    int maxval = 0;

    while (!squaresstack.empty())
    {
        int max = squaresstack.top();
        if (max > maxval)
        {
            maxval = max;
        }
        squaresstack.pop();
    }
    return maxval;
}

int main()
{
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

    // Create copies of the stacks for printing
    std::stack<int> orgstackCopy = orgstack;
    std::stack<int> squaresstackCopy = squaresstack;

    std::cout << "Original Stack: ";
    while (!orgstackCopy.empty())
    {
        std::cout << orgstackCopy.top() << " ";
        orgstackCopy.pop();
    }
    std::cout << std::endl;

    std::cout << "Squared Stack: ";
    while (!squaresstackCopy.empty())
    {
        std::cout << squaresstackCopy.top() << " ";
        squaresstackCopy.pop();
    }
    std::cout << std::endl;

    int maxval = Findmax();
    std::cout << "Max Value: " << maxval << std::endl;

    return 0;
}
