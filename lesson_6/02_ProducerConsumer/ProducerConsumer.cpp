#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "Timer.hpp"

void test_producer_consumer(int Buffer[32]) {
	int i = 0;
	int count = 0;

	while (i < 35000) {					// number of test

		// PRODUCER
		if ((rand() % 50) == 0) {		// some random computations

			if (count < 31) {
				++count;
				std::cout << "Thread:\t" << omp_get_thread_num()
                          << "\tProduce on index: " << count << std::endl;
				Buffer[count] = omp_get_thread_num();
			}
		}

		// CONSUMER
		if ((std::rand() % 51) == 0) {		// some random computations

			if (count >= 1) {
				int var = Buffer[count];
				std::cout << "Thread:\t" << omp_get_thread_num()
                          << "\tConsume on index: " << count
                          << "\tvalue: " << var << std::endl;
				--count;
			}
		}
		i++;
	}
}

// Locks (mutex)
/*
void test_producer_consumerOMP(int Buffer[32]) {
	int i;
	int count = 0;
	omp_lock_t lock;
	omp_init_lock(&lock);

	#pragma omp parallel for ordered private(i) shared(Buffer, count)
	for(i=0; i<35000; i++) {					// number of test

		// PRODUCER
		
		if ((rand() % 50) == 0) {		// some random computations
			omp_set_lock(&lock);
			if (count < 31) {
				++count;
				std::cout << "Thread:\t" << omp_get_thread_num()
						<< "\tProduce on index: " << count << std::endl;
				Buffer[count] = omp_get_thread_num();
			}
			omp_unset_lock(&lock);
		}

		// CONSUMER
		if ((std::rand() % 51) == 0) {		// some random computations
			omp_set_lock(&lock);
			if (count >= 1) {
				
				int var = Buffer[count];
				std::cout << "Thread:\t" << omp_get_thread_num()
						<< "\tConsume on index: " << count
						<< "\tvalue: " << var << std::endl;
				--count;
			}
			omp_unset_lock(&lock);
		}
	}
}
*/

// critical regions

void test_producer_consumerOMP(int Buffer[32]) {
	int i = 0;
	int count = 0;

	#pragma omp parallel shared(i, Buffer, count)
		while (i < 35000) {					// number of test
			
			// PRODUCER
			if ((rand() % 50) == 0) {		// some random computations
				#pragma omp critical(buf)
				{
					if (count < 31) {
						++count;
						std::cout << "Thread:\t" << omp_get_thread_num()
								<< "\tProduce on index: " << count << std::endl;
						Buffer[count] = omp_get_thread_num();
					}
				}
			}

			// CONSUMER
			if ((std::rand() % 51) == 0) {		// some random computations
				#pragma omp critical(buf)
				{
					if (count >= 1) {
						
						int var = Buffer[count];
						std::cout << "Thread:\t" << omp_get_thread_num()
								<< "\tConsume on index: " << count
								<< "\tvalue: " << var << std::endl;
						--count;
					}
				}
			}
			#pragma omp critical(other)
			i++;
		}
}


int main() {

	using namespace timer;
	Timer<HOST> TM;
	int Buffer[32];
	std::srand(time(NULL));

	omp_set_num_threads(2);

	TM.start();
	test_producer_consumer(Buffer);
	TM.stop();
	float time_seq = TM.duration();

	TM.start();
	test_producer_consumerOMP(Buffer);
	TM.stop();
	float time_OMP = TM.duration();
	
	std::cout << "\ntime OMP: " << time_OMP << "\n" << std::endl;
	std::cout << "\ntime sequential: " << time_seq << "\n" << std::endl;
	std::cout << "Speedup: " << time_seq/time_OMP << "x" << std::endl;
	
}
