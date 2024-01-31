#include <iostream>
#include <stack>
#include <thread>
#include <mutex>

std::stack<int> myStack;  // Stack to store values
std::mutex myMutex;       // Mutex for protecting concurrent access to the stack

const int THREAD_COUNT = 10;    // Number of threads
const int ITERATION_COUNT = 5;  // Number of iterations for each thread

// Function executed by each thread to push values to the stack
void pushToStack(int id) {
    for (int i = 0; i < ITERATION_COUNT; ++i) {
        std::unique_lock<std::mutex> lock(myMutex);  // Lock the mutex for exclusive access
        myStack.push(id * ITERATION_COUNT + i);     // Push a value to the stack
        lock.unlock();                              // Unlock the mutex
        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Simulate some work
    }
}

// Function executed by the background thread to pop values from the stack
void popFromStack() {
    for (int i = 0; i < THREAD_COUNT * ITERATION_COUNT; ++i) {
        std::unique_lock<std::mutex> lock(myMutex);  // Lock the mutex for exclusive access
        if (!myStack.empty()) {
            int maxValue = myStack.top();  // Get the top value without popping it
            myStack.pop();                 // Pop the top value from the stack
            lock.unlock();                 // Unlock the mutex
            std::cout << "Background Thread: " << std::this_thread::get_id() << " Popped Max Value: " << maxValue << std::endl;
        }
        else {
            lock.unlock();  // If the stack is empty, unlock the mutex and wait
            std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Simulate some work
        }
    }
}

int main() {
    std::thread threads[THREAD_COUNT];
    std::thread backgroundThread;

    // Create threads
    for (int i = 0; i < THREAD_COUNT; ++i) {
        threads[i] = std::thread(pushToStack, i + 1);
    }

    // Create background thread
    backgroundThread = std::thread(popFromStack);

    // Join threads
    for (int i = 0; i < THREAD_COUNT; ++i) {
        threads[i].join();
    }

    // Join background thread
    backgroundThread.join();

    return 0;
}
