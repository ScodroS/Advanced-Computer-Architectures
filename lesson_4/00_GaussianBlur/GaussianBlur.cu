#include <chrono>
#include <math.h>
#include <iomanip>
#include <iostream>
#include "Timer.cuh"
#include "CheckError.cuh"
#include <opencv2/opencv.hpp>
#include <imgcodecs.hpp> 
using namespace timer;

// N is the mask width / height (square mask)
const int N = 5;
#define WIDTH 1000
#define HEIGHT 500
#define CHANNELS 4
#define MVARIANCE 64


template <class T>
void print5by5 (T* Image);
void createMask (float* Mask);

__global__
void GaussianBlurDevice() {

}

void GaussianBlurHost(unsigned char* image, float* mask, unsigned char* image_out) {

	// int test = 0;

	for(int y = 0; y < HEIGHT; y++) {
		for(int x = 0; x < WIDTH; x++) {
			// *CHANNELS = number of channels codified into a pixel (4 in this case, RGBA)
			for(int channel = 0; channel < CHANNELS-1; channel++){
				float pixel_value = 0;
				// N is the length of mask edge
				for(int u = 0; u < N; u++) {
					for(int v = 0; v < N; v++) {
						int new_x = min(WIDTH, max(0, x+u-N/2));
						int new_y = min(HEIGHT, max(0, y+v-N/2));
						pixel_value += mask[v*N+u]*image[(new_y*WIDTH+new_x)*CHANNELS+channel];
						/*if (test < 10)
							std::cout << pixel_value << " ";
						test++ ;*/
					}
				}
				image_out[(y*WIDTH+x)*CHANNELS+channel]=(unsigned char) pixel_value;
			}
			//no transparency
			image_out[(y*WIDTH+x)*CHANNELS+CHANNELS-1]=(unsigned char) 255;
		}
	}
	// print5by5(image_out);
}

int main() {
	Timer<DEVICE> TM_device;
	Timer<HOST>   TM_host;

	// -------------------------------------------------------------------------
	// READ INPUT IMAGE
	cv::Mat I = cv::imread("../image_resized.png");
	
	print5by5(I.data);
	
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
	unsigned char* Image  = new unsigned char[WIDTH * HEIGHT * CHANNELS];
	// float* Mask = new float[N * N];
	unsigned char* Image_out = new unsigned char[WIDTH * HEIGHT * CHANNELS];
	
	Image = I.data;
	float Mask [] = {1.0278445 ,  4.10018648,  6.49510362,  4.10018648,  1.0278445 ,
       						 4.10018648, 16.35610171, 25.90969361, 16.35610171,  4.10018648,
       						 6.49510362, 25.90969361, 41.0435344 , 25.90969361,  6.49510362,
       						 4.10018648, 16.35610171, 25.90969361, 16.35610171,  4.10018648,
       						 1.0278445 ,  4.10018648,  6.49510362,  4.10018648,  1.0278445 };
	
	print5by5(Image);
	
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
	
	cv::Mat A(HEIGHT, WIDTH, CV_8UC4, Image_out);
	cv::imshow("Result of gaussian blur", A);
	cv::waitKey(0);
	
	print5by5(Image_out);

	// -------------------------------------------------------------------------
	// DEVICE MEMORY ALLOCATION
	/*
	SAFE_CALL( cudaMalloc( &devImage, WIDTH * HEIGHT * CHANNELS * sizeof(int) ) );
	SAFE_CALL( cudaMalloc( &devMask, N * N * sizeof(int) ) );


	SAFE_CALL( cudaMemcpy( devImage, Image, WIDTH * HEIGHT * CHANNELS * sizeof(int), cudaMemcpyHostToDevice ) );
	SAFE_CALL( cudaMemcpy( devIMask, Mask, N * N * sizeof(int), cudaMemcpyHostToDevice ) );
	*/
	// -------------------------------------------------------------------------
	// COPY DATA FROM HOST TO DEVICE
	// SAFE_CALL();

	// -------------------------------------------------------------------------
	// DEVICE EXECUTION
	TM_device.start();

	/// GaussianBlurDevice<<<  >>>();

	TM_device.stop();
	CHECK_CUDA_ERROR
	TM_device.print("GaussianBlur device: ");

	std::cout << std::setprecision(1)
	      << "Speedup: " << TM_host.duration() / TM_device.duration()
	      << "x\n\n";

	// -------------------------------------------------------------------------
	// COPY DATA FROM DEVICE TO HOST


	// -------------------------------------------------------------------------
	// RESULT CHECK
	if (true /* Correct check here */) {
	std::cerr << "wrong result!" << std::endl;
	cudaDeviceReset();
	std::exit(EXIT_FAILURE);
	}
	std::cout << "<> Correct\n\n";

	// -------------------------------------------------------------------------
	// HOST MEMORY DEALLOCATION


	// -------------------------------------------------------------------------
	// DEVICE MEMORY DEALLOCATION


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

void createMask (float* Mask) {
	for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			Mask[i*N+j] = 1/(2 * M_PI * MVARIANCE)*exp(-(pow(i, 2)+pow(j, 2))/(2 * MVARIANCE));
		}
	}
}
