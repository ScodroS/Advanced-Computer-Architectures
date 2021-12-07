#include <chrono>
#include <iostream>
#include <random>
#include <omp.h>
#include "Timer.hpp"

int main() {
    using namespace timer;
    int N = 67108864;
    // int N = (1 << 25);

    int* Array = new int[N];
    const int to_find1 = 18;        //search [ .. ][ .. ][ 18 ][ 64 ][ .. ]
    const int to_find2 = 64;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 10000);

    for (int i = 0; i < N; ++i)
        Array[i] = distribution(generator);

    // ------------ SEQUENTIAL SEARCH ------------------------------------------
    Timer<HOST> TM;
    TM.start();

    int index = -1;
    for (int i = 0; i < N - 1; ++i) {
        if (Array[i] == to_find1 && Array[i + 1] == to_find2) {
            index = i;
            break;            // !!! error in OPENMP
        }
    }

    TM.stop();
    float time_seq = TM.duration();
    TM.print("Sequential Search");
    std::cout << index << std::endl;

    // ------------ PARALLEL SEARCH --------------------------------------------
    
    // my solution
    
    TM.start();

    index = -1;
    bool flag = true;
    int i;

    #pragma omp parallel for shared (index, flag) private(i)
    for (i = 0; i < N - 1; ++i) {
        if (Array[i] == to_find1 && Array[i + 1] == to_find2 && flag) {
            index = i;
            // i=N;            
            flag = false;
        }
    }

    TM.stop();

    /*
    // Other working solution
    TM.start();

    index = -1;
    int i;
    
    #pragma omp parallel for shared (index) private(i)
    for (i = 0; i < N - 1; ++i) {
        if (Array[i] == to_find1 && Array[i + 1] == to_find2) {
            index = i;
            i=N;
        }
    }

    TM.stop();
    */

    /*
    // uploaded solution
    TM.start();

    index = -1;
    bool flag = true;
    #pragma omp parallel for firstprivate(flag) shared(index)
    for (int i = 0; i < N - 1; ++i) {
        if (flag && Array[i] == to_find1 && Array[i + 1] == to_find2) {
            index = i;                        // index: concurrent value
            flag = false;
        }
    }

    TM.stop();
    */

    float time_omp = TM.duration();
    TM.print("Parallel Search");
    std::cout << index << std::endl;

    std::cout << "Speedup: " << time_seq/time_omp << "x" << std::endl;
}
