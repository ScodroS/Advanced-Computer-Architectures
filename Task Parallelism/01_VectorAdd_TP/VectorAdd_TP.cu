#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

__global__
void vectorAddKernel(const int* d_inputA,
                     const int* d_inputB,
                     int        N,
                     int*       output) {

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_id < N) output[global_id] = d_inputA[global_id] + d_inputB[global_id];

}

const int N = 32*1024*1024;
const int SegSize = 4096;

int main(){
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    int* h_inputA     = new int[N];
    int* h_inputB     = new int[N];
    int* h_output = new int[N];
    int* d_output_tmp = new int[N]; // <-- used for device result

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N; i++) {
        h_inputA[i] = distribution(generator);
        h_inputB[i] = distribution(generator);
    }

    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    std::cout<<"Starting computation on HOST.."<<std::endl;
    TM_host.start();

    for (int i = 0; i < N; i++)
        h_output[i] = h_inputA[i] + h_inputB[i];

    TM_host.stop();
    TM_host.print("vectorAdd host:   ");

    // -------------------------------------------------------------------------
    // DEVICE INIT
    // dim3 DimBlock(256, 1, 1);
    int DimBlock = 256;
              
    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    std::cout<<"Starting computation on DEVICE.."<<std::endl;

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    int* d_A0, * d_B0, * d_C0; // device memory for stream 0
    int* d_A1, * d_B1, * d_C1; // device memory for stream 1
    // float Tot = 0;

    SAFE_CALL( cudaMalloc( &d_A0, SegSize * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_B0, SegSize * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_C0, SegSize * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_A1, SegSize * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_B1, SegSize * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_C1, SegSize * sizeof(int) ));

    TM_device.start();

    for (int i = 0; i < N; i += SegSize * 2) {
        cudaMemcpyAsync(d_A0, h_inputA + i, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_B0, h_inputB + i, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_A1, h_inputA + i + SegSize, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_B1, h_inputB + i + SegSize, SegSize * sizeof(int), cudaMemcpyHostToDevice, stream1);
        // TM_device.start();
        vectorAddKernel <<< SegSize/DimBlock, DimBlock, 0, stream0 >>>(d_A0, d_B0, N, d_C0);
        vectorAddKernel <<< SegSize/DimBlock, DimBlock, 0, stream1 >>>(d_A1, d_B1, N, d_C1);
        // TM_device.stop();
        cudaMemcpyAsync(d_output_tmp + i, d_C0, SegSize * sizeof(int), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(d_output_tmp + i + SegSize, d_C1, SegSize * sizeof(int), cudaMemcpyDeviceToHost, stream1);
        // std::cout << "Check!\n";
        // Tot += TM_device.duration();
    }

    CHECK_CUDA_ERROR

    TM_device.stop();

    TM_device.print("vectorAdd device: ");
    std::cout << std::setprecision(1)
        // std::cout << Tot
        << "Speedup: " << TM_host.duration() / TM_device.duration()
        << "x\n\n";

    // -------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < N; i++) {
        if (h_output[i] != d_output_tmp[i]) {
            std::cerr << "wrong result at: " << i
                      << "\nhost:   " << h_output[i]
                      << "\ndevice: " << d_output_tmp[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    //	else printf("%i %i\n", h_output[i], d_output_tmp[i]);
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_inputA;
    delete[] h_inputB;
    delete[] h_output;
    delete[] d_output_tmp;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL( cudaFree( d_A0 ) );
    SAFE_CALL( cudaFree( d_B0 ) );
    SAFE_CALL( cudaFree( d_C0 ) );
    SAFE_CALL( cudaFree( d_A1 ) );
    SAFE_CALL( cudaFree( d_B1 ) );
    SAFE_CALL( cudaFree( d_C1 ) );

    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
