#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

const int RADIUS = 7;
const int THREADS_PER_BLOCK = 256;
const int BLOCK_SIZE = THREADS_PER_BLOCK;

__global__
void stencilKernel(const int* d_input, int N, int* d_output) {

    __shared__ int d_sm[THREADS_PER_BLOCK + 2 * RADIUS];
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // all moved to the right of RADIUS
    d_sm[threadIdx.x] = d_input[global_id];

    // some threads will have to copy 2 more values to the shared memory (0-RADIUS and THREADS_PER_BLOCK+RADIUS)
    if (threadIdx.x < RADIUS) {
        d_sm[threadIdx.x + THREADS_PER_BLOCK] = d_input[blockDim.x + global_id];
        d_sm[threadIdx.x + RADIUS + THREADS_PER_BLOCK] = d_input[global_id + RADIUS + blockDim.x];
    }
    // wait for the shared memory to be written by all threads before moving forward
    __syncthreads();

    // write the results in the output vector
    if (global_id < N - 2 * RADIUS) {
        int pvalue = 0;
        for (int k = 0; k < RADIUS * 2 + 1; k++)
            pvalue += d_sm[threadIdx.x + k];
        d_output[global_id + RADIUS] = pvalue;
    }
}

// https://www.olcf.ornl.gov/wp-content/uploads/2019/12/02-CUDA-Shared-Memory.pdf
__global__ void stencilKernel2(const int* in, int N, int* out) {
    
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;
    
    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }
    
    // Synchronize (ensure all the data is available)
    __syncthreads();
    
    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
        result += temp[lindex + offset];
    
    // Store the result
    out[gindex] = result;

}

const int N  = 100000000;

int main() {
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    int* h_input      = new int[N];
    int* h_output_tmp = new int[N]; // <-- used for device result
    int* h_output     = new int[N](); // initilization to zero

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N; i++)
        h_input[i] = distribution(generator);

    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    TM_host.start();

    for (int i = RADIUS; i < N - RADIUS; i++) {
        for (int j = i - RADIUS; j <= i + RADIUS; j++)
            h_output[i] += h_input[j];
    }

    TM_host.stop();
    TM_host.print("1DStencil host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    int *d_input, *d_output;
    SAFE_CALL( cudaMalloc( &d_input, N * sizeof(int) ) )
    SAFE_CALL( cudaMalloc( &d_output, N * sizeof(int) ) )

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVICE
    SAFE_CALL( cudaMemcpy( d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // did you miss something?
    dim3 DimGrid(N/THREADS_PER_BLOCK, 1, 1);
    if (N%THREADS_PER_BLOCK) DimGrid.x++;
    dim3 DimBlock(THREADS_PER_BLOCK, 1, 1);

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();

    stencilKernel<<<DimGrid, DimBlock>>>(d_input, N, d_output);
    //stencilKernel2<<<DimGrid, DimBlock>>>(d_input, N, d_output);

    TM_device.stop();
    CHECK_CUDA_ERROR

    TM_device.print("1DStencil device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL( cudaMemcpy( h_output_tmp, d_output, N * sizeof(int), cudaMemcpyDeviceToHost ) )
    /*
    for (int i = 0; i < 20; i++) {
        std::cout << h_output_tmp[i] << "\t";
    }
    std::cout << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << h_output[i] << "\t";
    }
    */
    // -------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < N; i++) {
        if (h_output[i] != h_output_tmp[i]) {
            std::cerr << "wrong result at: " << i
                      << "\nhost:   " << h_output[i]
                      << "\ndevice: " << h_output_tmp[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_tmp;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL( cudaFree( d_input ) )
    SAFE_CALL( cudaFree( d_output ) )

    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
