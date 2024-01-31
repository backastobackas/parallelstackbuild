#include <iostream>
#include <mpi.h>
#include <random>

using namespace std;
using namespace MPI;

void calculate_full_sum();
void send_values();

const int MAIN_PROCESS = 0;
const int TAG_PARTIAL_ARRAY = 2;
const int VAL_SEND = 1;
const int LAST_ELEMENT = 3;
int main() {
    Init();
    auto rank = MPI::COMM_WORLD.Get_rank();
    if (rank == 0) {
        calculate_full_sum();
    } else {
        send_values();
    }
    Finalize();
    return 0;
}

/**
 * Manages the computations to split the array into chunks, send them to different processes and collect the results
 * back
 */
void calculate_full_sum() {

    auto process_count = MPI::COMM_WORLD.Get_size();
    int sum = 0;
    int num = 0;
    for (int i = 1; i < process_count; i++) {
        
        for (int j = 0; j <= 5 * i; j++)
        {
            MPI::Status status;
            MPI::COMM_WORLD.Probe(i, VAL_SEND, status);
            int received_item;
            MPI::COMM_WORLD.Recv(&received_item, 1, MPI::INT, status.Get_source(), status.Get_tag());    
            sum += received_item;
            std::cout << "SUM: " << sum << " RECEIVED: " << received_item << std::endl;
        }
    }
    
    cout << sum << endl;
}

void send_values() {

    int rank = MPI::COMM_WORLD.Get_rank();
    for (int i = 0; i <= 5 * rank; i++)
    {
        int value_to_send = i;
        MPI::COMM_WORLD.Send(&value_to_send, 1, MPI::INT, 0, VAL_SEND);
    }
}
