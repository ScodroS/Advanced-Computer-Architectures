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


long long int fibonacciOMPsections(long long int value, int level) {
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

/*
long long int fibonacciOMPtasks(long long int value, int level) {
		long long int fib_left, fib_right;
		if (value <= 1)
			return 1;
    #pragma omp parallel
    {
      #pragma omp single
      { fib_left  = fibonacci(value - 1, level + 1); }
      #pragma omp single
      { fib_right = fibonacci(value - 2, level + 1); }
    }
	 	return fib_left + fib_right;
}
*/

long long int fibonacciOMPtasks(long long int value, int level) {
		long long int fib_left, fib_right;
		if (value <= 1)
			return 1;
    #pragma omp parallel
    {
      #pragma omp single
      {
        #pragma omp task
        { fib_left  = fibonacci(value - 1, level + 1); }
        #pragma omp task
        { fib_right = fibonacci(value - 2, level + 1); }
      }
    }
	 	return fib_left + fib_right;
}

int main() {
		using namespace timer;
    
    //  ------------------------- TEST FIBONACCI ----------------------
    omp_set_dynamic(0);
    int value = 44;
    float time_OMP, time_OMP2;
    long long int result_OMP, result_OMP2;
    
    Timer<HOST> TM;
    
    TM.start();
    result_OMP = fibonacciOMPsections(value, 1);
		TM.stop();
		time_OMP = TM.duration();
		
		std::cout << "\nresult OMP sections: " << result_OMP << "\n" << std::endl;
    std::cout << "\ntime OMP sections: " << time_OMP << "\n" << std::endl;

    TM.start();
    result_OMP2 = fibonacciOMPtasks(value, 1);
		TM.stop();
    time_OMP2 = TM.duration();
		
		std::cout << "\nresult OMP tasks: " << result_OMP2 << "\n" << std::endl;
    std::cout << "\ntime OMP tasks: " << time_OMP2 << "\n" << std::endl;

		TM.start();
    long long int result_seq = fibonacci(value, 1);
		TM.stop();
		float time_seq = TM.duration();
		
    std::cout << "\nresult sequential: " << result_seq << "\n" << std::endl;
    std::cout << "\ntime sequential: " << time_seq << "\n" << std::endl;
    
    std::cout << "Speedup sections: " << time_seq/time_OMP << "x" << std::endl;
    std::cout << "Speedup tasks: " << time_seq/time_OMP2 << "x" << std::endl;
    
    return 0;
}
