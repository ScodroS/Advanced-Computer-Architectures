#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <omp.h>
#include "Timer.hpp"

// ---------- DO NOT MODIFY ----------------------------------------------------
int partition(int array[], const int start, const int end) {
    int p = start;
    int pivot_element = array[end];
    for(int i = start; i < end; i++) {
        if(array[i] < pivot_element) {
            std::swap(array[i], array[p]);
            ++p;
        }
    }
    std::swap(array[p], array[end]);
    return p;
}
//------------------------------------------------------------------------------

void quick_sort(int array[], const int start, const int end) {
    if(start < end) {
        int pivot = partition(array, start, end);
        quick_sort(array, start, pivot - 1);
        quick_sort(array, pivot + 1, end);
    }
}

void quick_sortOMP(int array[], const int start, const int end) {
    if(start < end) {
        int pivot = partition(array, start, end);
        #pragma omp task
				{ quick_sort(array, start, pivot - 1); }
        #pragma omp task
        { quick_sort(array, pivot + 1, end); }
    }
}

template<typename T>
void print_array(T* array, int size, const char* str) {
    std::cout << str << "\n";
    //for (int i = 0; i < size; ++i)
    for (int i = 0; i < 100; ++i)
        std::cout << array[i] << ' ';
    std::cout << "\n" << std::endl;
}


int main() {

		using namespace timer;
    const int N = 1<<20;
    Timer<HOST> TM;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    int* input = new int[N];
    int* inputOMP = new int[N];
    for (int i = 0; i < N; ++i) {
        input[i] = distribution(generator);
        inputOMP[i] = input[i];
    }

    print_array(inputOMP, N, "\nInput:");

		TM.start();
    quick_sortOMP(inputOMP, 0, N - 1);
		TM.stop();
		float time_OMP = TM.duration();

    print_array(inputOMP, N, "Sorted:");

    print_array(input, N, "\nInput:");

		TM.start();
    quick_sort(input, 0, N - 1);
   	TM.stop();
		float time_seq = TM.duration();

    print_array(input, N, "Sorted:");
    
    std::cout << "\ntime OMP: " << time_OMP << "\n" << std::endl;
    std::cout << "\ntime sequential: " << time_seq << "\n" << std::endl;
    std::cout << "Speedup: " << time_seq/time_OMP << "x" << std::endl;
}
