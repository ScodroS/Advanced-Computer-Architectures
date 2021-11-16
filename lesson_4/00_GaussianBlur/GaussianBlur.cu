#include <chrono>
#include <iomanip>
#include <iostream>
#include "Timer.cuh"
#include "CheckError.cuh"
#include <opencv2/opencv.hpp>
using namespace timer;

// N is the mask width / height (square mask)
const int N = 5;
#define WIDTH 4000
#define HEIGHT 2000

void GaussianBlurHost(int* image, int* mask, int* image_out) {
	for(int y = 0; y < HEIGHT; y++) {
		for(int x = 0; x < WIDTH; x++) {
			// *channel= number of channels codified into a pixel (4 in this case, RGBA)
			for(int channel = 0; channel < 4; channel++){
				float pixel_value = 0;
				for(int u = 0; u < N; u++) {
					for(int v = 0; v < N; v++) {
						int new_x = min(WIDTH, max(0, x+u-N/2));
						int new_y = min(HEIGHT, max(0, y+v-N/2));
						pixel_value += mask[v*N+u]*image[(new_y*WIDTH+new_x)*4+channel];
					}
				}
				image_out[(y*WIDTH+x)*4+channel]=(unsigned char) pixel_value;
			}
			//no transparency
			image_out[(y*WIDTH+x)*4+3]=(unsigned char) 255;
		}
	}
}

int main() {
	Timer<DEVICE> TM_device;
	Timer<HOST>   TM_host;

	// -------------------------------------------------------------------------
	// READ INPUT IMAGE
	cv::Mat I = cv::imread("../image.png", 0);
	if (I.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		exit(0);
		// don't let the execution continue, else imshow() will crash.
	}

	cv::namedWindow("Display window");// Create a window for display.
	cv::imshow( "Display window", I ); 
	cv::waitKey(0);

	// -------------------------------------------------------------------------
	// HOST MEMORY ALLOCATION
	int* image  = new int[WIDTH * HEIGHT];
	int* mask = new int[N * N];
	int* image_out = new int[WIDTH * HEIGHT];

	// -------------------------------------------------------------------------
	// HOST EXECUTIION
	TM_host.start();

	GaussianBlurHost(image, mask, image_out);

	TM_host.stop();
	TM_host.print("GaussianBlur host:   ");

	// -------------------------------------------------------------------------
	// DEVICE MEMORY ALLOCATION


	// -------------------------------------------------------------------------
	// COPY DATA FROM HOST TO DEVIE
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
	cudaDeviceReset();
}
