#include <iostream>
#include <omp.h>
#include "Timer.hpp"

long long int fibonacci(long long int value, int level) {
    if (value <= 1)
        return 1;

    long long int fib_left, fib_right;
    fib_left  = fibonacci(value - 1, level + 1);
    fib_right = fibonacci(value - 2, level + 1);
		
    return fib_left + fib_right;
}


long long int fibonacciOMP(long long int value, int level) {
		long long int fib_left, fib_right;
		if (value <= 1)
			return 1;
		#pragma omp parallel sections 
		{
			#pragma omp section
			{ fib_left  = fibonacci(value - 1, level + 1); }
			#pragma omp section
			{ fib_right = fibonacci(value - 2, level + 1); }
		}
	 	return fib_left + fib_right;
}

int main() {
		using namespace timer;
    
    //  ------------------------- TEST FIBONACCI ----------------------
    omp_set_dynamic(0);
    int value = 44;
    
    Timer<HOST> TM;
    
    TM.start();
    long long int result_OMP = fibonacci(value, 1);
		TM.stop();
		float time_OMP = TM.duration();
		
		std::cout << "\nresult OMP: " << result_OMP << "\n" << std::endl;
    std::cout << "\ntime OMP: " << time_OMP << "\n" << std::endl;

		TM.start();
    long long int result_seq = fibonacci(value, 1);
		TM.stop();
		float time_seq = TM.duration();
		
    std::cout << "\nresult sequential: " << result_seq << "\n" << std::endl;
    std::cout << "\ntime sequential: " << time_seq << "\n" << std::endl;
    
    std::cout << "Speedup: " << time_seq/time_OMP << "x" << std::endl;
    
    return 0;
}
