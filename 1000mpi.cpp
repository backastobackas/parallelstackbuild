#include <iostream>
#include <mpi.h>
#include <random>

using namespace std;
using namespace MPI;

void calculate_full_average();
void calculate_partial_average();

const int MAIN_PROCESS = 0;
const int TAG_PARTIAL_ARRAY = 2;
const int TAG_PARTIAL_AVERAGE = 1;

int main() {
    Init();
    auto rank = MPI::COMM_WORLD.Get_rank();
    if (rank == 0) {
        calculate_full_average();
    } else {
        calculate_partial_average();
    }
    Finalize();
    return 0;
}

/**
 * Manages the computations to split the array into chunks, send them to different processes and collect the results
 * back
 */
void calculate_full_average() {
    // all processes except this one is a worker, so worker count is the total amount of processes excluding current
    // process
    int worker_count = MPI::COMM_WORLD.Get_size() - 1;
    //generate an array of random numbers
    const int DATA_SIZE = 1000;
    int numbers[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; i++) {
        numbers[i] = i;
    }
    auto chunk_size = DATA_SIZE / worker_count;
    for (auto i = 0; i < worker_count; i++) {
        // send one part of the array to one of the processes. The array chunk indexes are from start_index to
        // end_index. If the data cannot be split evenly, the remainder goes to the last process.
        int end_index = (i == worker_count - 1 ? DATA_SIZE - 1: (i+1) * chunk_size - 1);
        int start_index = i * chunk_size;
        int current_chunk_size = end_index - start_index;  // Add 1 here
        MPI::COMM_WORLD.Send(numbers + start_index, current_chunk_size, INT, i + 1, TAG_PARTIAL_ARRAY);
    }
    int partial_averages[worker_count];
    for (auto i = 0; i < worker_count; i++) {
        // collect the partials sums back and store them in a local array of partial sums
        MPI::COMM_WORLD.Recv(&partial_averages[i], 1, MPI::DOUBLE, ANY_SOURCE, TAG_PARTIAL_AVERAGE);
    }
    auto total_sum = accumulate(&partial_averages[0], &partial_averages[worker_count], 0.0,  [](double x, double y) { return x + y;});
    auto total_average = total_sum / worker_count;
    cout << total_average << endl;
}

void calculate_partial_average() {
    Status status;
    MPI::COMM_WORLD.Probe(MAIN_PROCESS, TAG_PARTIAL_ARRAY, status);
    const auto item_count = status.Get_count(INT);
    int items[item_count];
    MPI::COMM_WORLD.Recv(items, item_count, INT, status.Get_source(), status.Get_tag());
    auto total_sum = accumulate(items, items + item_count, 0, [](int x, int y) { return x + y;});
    auto average = total_sum / item_count;
    MPI::COMM_WORLD.Send(&average, 1, MPI::DOUBLE, 0, TAG_PARTIAL_AVERAGE);
}
