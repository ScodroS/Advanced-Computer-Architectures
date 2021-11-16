#include <iostream>
#include <chrono>
#include <random>
#include <numeric>
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

// Macros
#define DIV(a, b)   (((a) + (b) - 1) / (b))

const int N  = 16777216; // 2^24
const bool TASK_PARALLELISM = 1;

#define BLOCK_SIZE 256
const int SegSize = BLOCK_SIZE * BLOCK_SIZE;


// standard version (global memory only)
__global__ void ReduceKernel0(int* VectorIN, int N) {
    
	int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 1; i < blockDim.x; i *= 2) {
		if (threadIdx.x % (i * 2) == 0)
			VectorIN[GlobalIndex] += VectorIN[GlobalIndex + i];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		VectorIN[blockIdx.x] = VectorIN[GlobalIndex];
}

// standard version with shared memory
__global__ void ReduceKernel1(int* VectorIN, int N) {
    
	__shared__ int SMem[1024];
	int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	SMem[threadIdx.x] = VectorIN[GlobalIndex];
	__syncthreads();
	for (int i = 1; i < blockDim.x; i *= 2) {
		if (threadIdx.x % (i * 2) == 0)
			SMem[threadIdx.x] += SMem[threadIdx.x + i];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		VectorIN[blockIdx.x] = SMem[0];
}

// second version (shared memory + multiple of warp size)
__global__ void ReduceKernel2(int* VectorIN, int N) {
    
    __shared__ int SMem[1024];
    int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    SMem[threadIdx.x] = VectorIN[GlobalIndex];
    // sync the threads once VectorIN has been copied into the shared memory in order to start the computation
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = threadIdx.x * i * 2;

        // instead of threadIdx.x % (i * 2) == 0, to avoid heavy branch divergence
        if (index < blockDim.x)
            SMem[index] += SMem[index + i];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        VectorIN[blockIdx.x] = SMem[0];

}

// <<DIV(N, SegSize), BLOCK_SIZE, 0, stream0>>>

__global__ void FinalSumKernel(int* VectorIN, int N) {

	int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
	VectorIN[GlobalIndex] = VectorIN[GlobalIndex * SegSize];

	__syncthreads();
	
	for (int i = 1; i < blockDim.x; i *= 2) {
		if (threadIdx.x % (i * 2) == 0)
			VectorIN[GlobalIndex] += VectorIN[GlobalIndex + i];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		VectorIN[blockIdx.x] = VectorIN[GlobalIndex];
}

// same as 2 for now...
__global__ void ReduceKernel3(int* VectorIN, int N) {
    
    __shared__ int SMem[1024];
    int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    SMem[threadIdx.x] = VectorIN[GlobalIndex];
    // sync the threads once VectorIN has been copied into the shared memory in order to start the computation
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = threadIdx.x * i * 2;

        // instead of threadIdx.x % (i * 2) == 0, to avoid heavy branch divergence
        if (index < blockDim.x)
            SMem[index] += SMem[index + i];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        VectorIN[blockIdx.x] = SMem[0];

}

int main() { {
    
    // ------------------- INIT ------------------------------------------------

    // Random Engine Initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    Timer<HOST> host_TM;
    Timer<DEVICE> dev_TM;

	// ------------------ HOST INIT --------------------------------------------

	int* VectorIN = new int[N];
	for (int i = 0; i < N; ++i)
		VectorIN[i] = distribution(generator);

	// ------------------- CUDA INIT -------------------------------------------

	int* devVectorIN;
	SAFE_CALL( cudaMalloc(&devVectorIN, N * sizeof(int)) );
	
	int sum = 0;
	float dev_time;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------

    std::cout<<"Starting computation on DEVICE "<<std::endl;
	if (!TASK_PARALLELISM) {
	
		SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN, N * sizeof(int),
       cudaMemcpyHostToDevice) );
	
		dev_TM.start();

		ReduceKernel2<<<DIV(N, BLOCK_SIZE), BLOCK_SIZE>>>
				(devVectorIN, N);
		ReduceKernel2<<<DIV(N, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE>>>
				 (devVectorIN, DIV(N, BLOCK_SIZE));
		ReduceKernel2<<<DIV(N, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE>>>
				 (devVectorIN, DIV(N, BLOCK_SIZE * BLOCK_SIZE));

		dev_TM.stop();
		
		SAFE_CALL( cudaMemcpy(&sum, devVectorIN, sizeof(int),
          cudaMemcpyDeviceToHost) );
		
		dev_time = dev_TM.duration();
		CHECK_CUDA_ERROR;
	}
	else {

		cudaStream_t stream0, stream1;
		cudaStreamCreate(&stream0);
		cudaStreamCreate(&stream1);

		/// N = 2^24
		/// BLOCK_SIZE = 2^8
		
		int* VectorINtmp = new int[N];
		
		// SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );
		
		dev_TM.start();

		for (int i = 0; i < N; i += SegSize * 2) {
			
			SAFE_CALL( cudaMemcpyAsync( devVectorIN + i, VectorIN + i, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream0 ));
			SAFE_CALL( cudaMemcpyAsync( devVectorIN + i + SegSize, VectorIN + i + SegSize, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream1 ));

			ReduceKernel0<<<DIV(SegSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream0>>> (devVectorIN + i, SegSize);
			ReduceKernel0<<<DIV(SegSize, BLOCK_SIZE), BLOCK_SIZE, 0, stream1>>> (devVectorIN + i + SegSize, SegSize);
			ReduceKernel0<<<DIV(SegSize, BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE, 0, stream0>>> (devVectorIN + i, DIV(SegSize, BLOCK_SIZE));
			ReduceKernel0<<<DIV(SegSize, BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE, 0, stream1>>> (devVectorIN + i + SegSize, DIV(SegSize, BLOCK_SIZE));
			
		}
		
		cudaStreamSynchronize(stream1);
		
		FinalSumKernel<<<DIV(N, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE, 0, stream0>>> (devVectorIN, DIV(N, BLOCK_SIZE * BLOCK_SIZE));

		dev_TM.stop();
		
		SAFE_CALL( cudaMemcpy(&sum, devVectorIN, sizeof(int), cudaMemcpyDeviceToHost) );
		
		dev_time = dev_TM.duration();
		CHECK_CUDA_ERROR;
	}

	// ------------------- HOST ------------------------------------------------
    host_TM.start();

	int host_sum = std::accumulate(VectorIN, VectorIN + N, 0);

    host_TM.stop();

    std::cout << std::setprecision(3)
              << "KernelTime Divergent: " << dev_time << std::endl
              << "HostTime            : " << host_TM.duration() << std::endl
              << std::endl;

    // ------------------------ VERIFY -----------------------------------------

    if (host_sum != sum) {
        std::cerr << std::endl
                  << "Error! Wrong result. Host value: " << host_sum
                  << " , Device value: " << sum
                  << std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }

    //-------------------------- SPEEDUP ---------------------------------------

    float speedup = host_TM.duration() / dev_time;

    std::cout << "Correct result" << std::endl
              << "Speedup achieved: " << std::setprecision(3)
              << speedup << " x" << std::endl << std::endl;

    std::cout << host_TM.duration() << ";" << dev_TM.duration() << ";" << host_TM.duration() / dev_TM.duration() << std::endl;

    delete[] VectorIN;
    SAFE_CALL( cudaFree(devVectorIN) ); }
    cudaDeviceReset();
}
