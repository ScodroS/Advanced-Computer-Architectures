#include <iostream>
#include <chrono>
#include <random>
#include <numeric>
#include <math.h>
#include "Timer.cuh"

using namespace timer;
using namespace timer_cuda;

const int BLOCK_SIZE = 512;

__device__ int counter = 0;

__global__ void PrefixScan(int* VectorIN, int N, int levels) {
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = 1;
	for ( int level = 0; level < levels; level++ ) {
		if ( level > 0 )
			offset *= 2;
		if ( global_id >= offset )
			VectorIN[global_id] = VectorIN[global_id - offset] + VectorIN[global_id];
		__syncthreads();
	}
}

void printArray(int* Array, int start, int end, const char str[] = "") {
	std::cout << str;
	for (int i = start; i < end; ++i)
		std::cout << std::setw(5) << Array[i] << ' ';
	std::cout << std::endl << std::endl;
}


#define DIV(a,b)	(((a) + (b) - 1) / (b))

int main() {
	const int blockDim = BLOCK_SIZE;
	// const int N = BLOCK_SIZE * 131072;
	const int N = BLOCK_SIZE * 8;
	const int levels = log2(N);
	std::cout << "Levels: " << levels << std::endl;


    // ------------------- INIT ------------------------------------------------

    // Random Engine Initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    timer::Timer<HOST> host_TM;
    timer_cuda::Timer<DEVICE> dev_TM;

	// ------------------ HOST INIT --------------------------------------------

	int* VectorIN = new int[N];
	for (int i = 0; i < N; ++i)
		VectorIN[i] = distribution(generator);

	// ------------------- CUDA INIT -------------------------------------------

	int* devVectorIN;
	__SAFE_CALL( cudaMalloc(&devVectorIN, N * sizeof(int)) );
  __SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );

	int* prefixScan = new int[N];
	float dev_time;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------

	dev_TM.start();
	PrefixScan<<<DIV(N, blockDim), blockDim>>>(devVectorIN, N, levels);
	dev_TM.stop();
	dev_time = dev_TM.duration();

	__SAFE_CALL(cudaMemcpy(prefixScan, devVectorIN, N * sizeof(int),
                           cudaMemcpyDeviceToHost) );

	// ------------------- CUDA ENDING -----------------------------------------

	std::cout << std::fixed << std::setprecision(1)
              << "KernelTime Naive  : " << dev_time << std::endl << std::endl;

	// ------------------- VERIFY ----------------------------------------------

  host_TM.start();

	int* host_result = new int[N];
	std::partial_sum(VectorIN, VectorIN + N, host_result);

  host_TM.stop();
    
  const char str[] = "";
  int start = BLOCK_SIZE * 1 -10;
  int end = BLOCK_SIZE * 1 + 10;
    
  printArray(prefixScan, start, end, str);
  printArray(host_result, start, end, str);

	if (!std::equal(host_result, host_result + blockDim - 1, prefixScan + 1)) {
		std::cerr << " Error! :  prefixScan" << std::endl << std::endl;
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}

    // ----------------------- SPEEDUP -----------------------------------------

    float speedup1 = host_TM.duration() / dev_time;
	std::cout << "Correct result" << std::endl
              << "(1) Speedup achieved: " << speedup1 << " x" << std::endl
              << std::endl << std::endl;

    std::cout << host_TM.duration() << ";" << dev_TM.duration() << ";" << host_TM.duration() / dev_TM.duration() << std::endl;
	
	delete[] host_result;
    delete[] prefixScan;
    
    __SAFE_CALL( cudaFree(devVectorIN) );
    
    cudaDeviceReset();
}
