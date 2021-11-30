#include <iostream>
#include <omp.h>
#include "Timer.hpp"

int main() {
    using namespace timer;
    // int N = (1 << 30);
    int N = 268435456;
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
    TM.start();
    # pragma omp parallel for
    for (i = 1; i <= N; ++i)
        factorial_omp *= i;
    TM.stop();
    float time_omp = TM.duration();
    TM.print("Parallel Factorial");
    std::cout << factorial_omp << std::endl;

    std::cout << "Speedup: " << time_seq/time_omp << "x" << std::endl;
}
