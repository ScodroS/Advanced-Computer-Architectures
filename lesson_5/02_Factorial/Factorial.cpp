#include <iostream>
#include <cstdio>   // std::printf
#include <omp.h>
#include "Timer.hpp"

int main() {
    using namespace timer;
    int N = 32;
    // int N = 268435456;
    double factorial = 1;
    
    /* Sequential implementation of factorial:*/
    
    Timer<HOST> TM;

    TM.start();
    for (int i = 1; i <= N; ++i)
        factorial *= i;
    TM.stop();

    float time_seq = TM.duration();
    TM.print("Sequential Factorial");
    std::cout << factorial << std::endl;
    
    //--------------------------------------------------------------------------
    /* Parallel implementation of Factorial: */

    double factorial_omp = 1; 
    int i;
    double tmp;

    /*
    TM.start();
    # pragma omp parallel private(i, tmp) 
    {
        tmp = 1;
        # pragma omp for
        for (i = 1; i <= N; ++i)
            tmp *= i;
        # pragma omp critical
        factorial_omp *= tmp;
    }
    TM.stop();
    */

    // My solution
    TM.start();
    # pragma omp parallel private(tmp)
    {
        tmp = 1;
        # pragma omp for private(i)
        for (i = 1; i <= N; ++i)
            tmp *= i;
        //std::printf("Thread number %d with tmp %f \n" , omp_get_thread_num(), tmp);
        # pragma omp atomic
        factorial_omp *= tmp;
    }
    TM.stop();
    
    /*
    // Uploaded solution
    auto results = new double[omp_get_max_threads()]();
    TM.start();
    
    #pragma omp parallel
    {
        int i;
        double ThFactorial = 1;
	    #pragma omp for firstprivate(N) private(i)
        for (i = 1; i <= N; ++i)
            ThFactorial *= i;
        results[omp_get_thread_num()] = ThFactorial;
    }
    
    for (int i = 0; i < omp_get_max_threads(); ++i)
        factorial_omp *= results[i];
    
    TM.stop();
    */

    float time_omp = TM.duration();
    TM.print("Parallel Factorial");
    std::cout << factorial_omp << std::endl;

    std::cout << "Speedup: " << time_seq/time_omp << "x" << std::endl;
}
