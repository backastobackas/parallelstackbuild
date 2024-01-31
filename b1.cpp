#include <mutex>
#include <vector>
#include <thread>
#include <iostream>
#include <condition_variable>

class Stack{
    private:
        int size = 0;
        int data[10];
        int max = 0;
        std::mutex mutex;
        std::condition_variable convar;
    public:
        void push(int value){
            std::unique_lock<std::mutex> lock(mutex);
            
            convar.wait(lock, [&] { return size < 10;});
            
            data[size] = value;
            size++;

            if (value > max){
                max = value;
            }

            convar.notify_one();
        }

        void pop(){
            std::unique_lock<std::mutex> lock(mutex);
            convar.wait(lock, [&] { return size > 0;});

            data[size] = 0;
            size--;
            convar.notify_all();
        }

        int getSize(){
            std::unique_lock<std::mutex> lock(mutex);
            return size;
        }

        int getMax(){
            std::unique_lock<std::mutex> lock(mutex);
            return max;
        }
};

void worker(Stack &stack, int id){
    
    for (int i = 0; i < 5; i++)
    {
        stack.push(id);
        std::cout << "added: " << id << std::endl;
    }
}

void background(Stack &stack){
    for (size_t i = 0; i < 40; i++)
    {
        stack.pop();
        std::cout << "popped: " << i << " " << std::endl;
    }
}

int main(){

    Stack stack;

    std::vector<std::thread> threads;

    for (size_t i = 0; i < 10; i++)
    {
        threads.push_back(std::thread(worker, std::ref(stack), i));
    }
    std::thread thread = std::thread(background, std::ref(stack));

    for (int i = 0; i < 10; i++)
    {
        threads[i].join();
    }
    thread.join();
    
    return 0;
}