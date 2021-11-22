#include <chrono>
#include <math.h>
#include <iomanip>
#include <iostream>
#include "Timer.cuh"
#include "CheckError.cuh"
#include <opencv2/opencv.hpp>
// #include <imgcodecs.hpp> 
using namespace timer;

// N is the mask width / height (square mask)
const int N = 5;
#define WIDTH 1000
#define HEIGHT 500
#define CHANNELS 3

const int BLOCK_SIZE = 32;
const int TILE_WIDTH = BLOCK_SIZE;

template <class T>
void print5by5 (T* Image);
void createMask (float* Mask);

__global__
void GaussianBlurDevice(const unsigned char *image, 
												const float *mask, 
												unsigned char *image_out,
												int N) {

	int globalId_x = threadIdx.x + blockIdx.x * blockDim.x;
	int globalId_y = threadIdx.y + blockIdx.y * blockDim.y;
	// globalId_z = threadIdx.z + blockIdx.z * blockDim.z;

	if (globalId_x < WIDTH && globalId_y < HEIGHT) {
			// *CHANNELS = number of channels codified into a pixel (4 in this case, RGBA)
			//for(int channel = 0; channel < CHANNELS-1; channel++){
			for(int channel = 0; channel < CHANNELS; channel++){
				float pixel_value = 0;
				// N is the length of mask edge
				for(int u = 0; u < N; u++) {
					for(int v = 0; v < N; v++) {
						int new_x = min(WIDTH, max(0, globalId_x+u-N/2));
						int new_y = min(HEIGHT, max(0, globalId_y+v-N/2));
						pixel_value += mask[v*N+u]*image[(new_y*WIDTH+new_x)*CHANNELS+channel];
					}
				}
				image_out[(globalId_y*WIDTH+globalId_x)*CHANNELS+channel]=(unsigned char) (pixel_value < 255.0? pixel_value : 255.0);
			}
			//no transparency
			//image_out[(y*WIDTH+x)*CHANNELS+CHANNELS-1]=(unsigned char) 255;
	}
}

void GaussianBlurHost(const unsigned char *image, 
											const float *mask, 
											unsigned char *image_out) {

	for(int y = 0; y < HEIGHT; y++) {
		for(int x = 0; x < WIDTH; x++) {
			// *CHANNELS = number of channels codified into a pixel (4 in this case, RGBA)
			//for(int channel = 0; channel < CHANNELS-1; channel++){
			for(int channel = 0; channel < CHANNELS; channel++){
				float pixel_value = 0;
				// N is the length of mask edge
				for(int u = 0; u < N; u++) {
					for(int v = 0; v < N; v++) {
						int new_x = min(WIDTH, max(0, x+u-N/2));
						int new_y = min(HEIGHT, max(0, y+v-N/2));
						pixel_value += mask[v*N+u]*image[(new_y*WIDTH+new_x)*CHANNELS+channel];
					}
				}
				// temporary solution, need new filter mask...
				image_out[(y*WIDTH+x)*CHANNELS+channel]=(unsigned char) (pixel_value < 255.0? pixel_value : 255.0);
			}
			//no transparency
			//image_out[(y*WIDTH+x)*CHANNELS+CHANNELS-1]=(unsigned char) 255;
		}
	}
}

int main() {
	Timer<DEVICE> TM_device;
	Timer<HOST>   TM_host;

	// -------------------------------------------------------------------------
	// READ INPUT IMAGE
	cv::Mat I = cv::imread("../prova_resized.jpg");
	//cv::Mat I = cv::imread("../image.png");
	
	if (I.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		exit(0);
		// stop execution
	}

	cv::namedWindow("Display window");// Create a window for display.
	cv::imshow( "Display window", I ); 
	cv::waitKey(0);

	// -------------------------------------------------------------------------
	// HOST MEMORY ALLOCATION
	unsigned char *Image  = new unsigned char[WIDTH * HEIGHT * CHANNELS];
	unsigned char *Image_out = new unsigned char[WIDTH * HEIGHT * CHANNELS];
	unsigned char *hostImage_out = new unsigned char[WIDTH * HEIGHT * CHANNELS];
	
	Image = I.data;
	float Mask [] = { 1.0,   4.0,  67.0,  4.0,  1.0,
       						  4.0,  16.0,  26.0, 16.0,  4.0,
       						  7.0,  26.0,  41.0, 26.0,  7.0,
       						  4.0,  16.0,  26.0, 16.0,  4.0,
       						  1.0,   4.0,  67.0,  4.0,  1.0 };
  createMask(Mask);  
	
	std::cout << std::endl;
	for(int i=0; i<5; i++) {
		for(int j=0; j<5; j++) {
			std::cout << "[ ";
			std::cout << Mask[i*N+j];
			std::cout << "]";
		}		
		std::cout << std::endl;
	}
	std::cout << std::endl;
	
	// -------------------------------------------------------------------------
	// HOST EXECUTIION
	TM_host.start();

	GaussianBlurHost(Image, Mask, Image_out);

	TM_host.stop();
	TM_host.print("GaussianBlur host:   ");
	
	cv::Mat A(HEIGHT, WIDTH, CV_8UC3, Image_out);
	// cv::cvtColor(A, A, cv::COLOR_BGR2RGB);
	cv::imshow("Result of gaussian blur (host)", A);
	cv::waitKey(0);

	// -------------------------------------------------------------------------
	// DEVICE MEMORY ALLOCATION
	
	unsigned char *devImage, *devImage_out;
	float *devMask;
	
	SAFE_CALL( cudaMalloc( &devImage, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char) ) );
	SAFE_CALL( cudaMalloc( &devImage_out, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char) ) );
	SAFE_CALL( cudaMalloc( &devMask, N * N * sizeof(float) ) );
	
	// -------------------------------------------------------------------------
	// COPY DATA FROM HOST TO DEVICE
	
	SAFE_CALL( cudaMemcpy( devImage, Image, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice ) );
	SAFE_CALL( cudaMemcpy( devMask, Mask, N * N * sizeof(float), cudaMemcpyHostToDevice ) );

	// -------------------------------------------------------------------------
	// DEVICE EXECUTION
	
	dim3 block_size( BLOCK_SIZE, BLOCK_SIZE, 1 );
  dim3 num_blocks( ceil(float(WIDTH)/BLOCK_SIZE), ceil(float(HEIGHT)/BLOCK_SIZE), 1 );
	
	TM_device.start();

	GaussianBlurDevice<<<block_size, num_blocks>>>(devImage, devMask, devImage_out, N);

	TM_device.stop();
	CHECK_CUDA_ERROR
	TM_device.print("GaussianBlur device: ");

	std::cout << std::setprecision(1)
	      << "Speedup: " << TM_host.duration() / TM_device.duration()
	      << "x\n\n";

	// -------------------------------------------------------------------------
	// COPY DATA FROM DEVICE TO HOST

	SAFE_CALL( cudaMemcpy( hostImage_out, devImage_out, N * N * sizeof(unsigned char), cudaMemcpyDeviceToHost ) );

	// -------------------------------------------------------------------------
	// RESULT CHECK
	
	cv::Mat B(HEIGHT, WIDTH, CV_8UC3, hostImage_out);
	// cv::cvtColor(A, A, cv::COLOR_BGR2RGB);
	cv::imshow("Result of gaussian blur (device)", B);
	cv::waitKey(0);
	
	if (true /* Correct check here */) {
	std::cerr << "wrong result!" << std::endl;
	cudaDeviceReset();
	std::exit(EXIT_FAILURE);
	}
	std::cout << "<> Correct\n\n";

	// -------------------------------------------------------------------------
	// HOST MEMORY DEALLOCATION
  delete[] Image;
  delete[] Image_out;
  delete[] hostImage_out;


	// -------------------------------------------------------------------------
	// DEVICE MEMORY DEALLOCATION
	SAFE_CALL( cudaFree( devImage ) )
  SAFE_CALL( cudaFree( devImage_out ) )
  SAFE_CALL( cudaFree( devMask ) )

	// -------------------------------------------------------------------------
	//SAFE_CALL(cudaFree());
	cudaDeviceReset();
}

template <class T>
void print5by5 (T* Image) {
	std::cout << std::endl;
	for(int i=0; i<5; i++) {
		for(int j=0; j<5; j++) {
			std::cout << "[ ";
			for(int k=0; k<CHANNELS; k++) {
				std::cout << (short)Image[(i*WIDTH+j)*CHANNELS+k];
				k == CHANNELS-1? std::cout << " ": std::cout << " , "; 
			}
			std::cout << "]";
		}		
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// Just to calculate the mask
void createMask (float* Mask) {
	for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			Mask[i*N+j] = 1.0/273 * Mask[i*N+j];
		}
	}
}
