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

const int BLOCK_SIZE_1D = 1024;
const int BLOCK_SIZE = 32;
const int TILE_WIDTH = BLOCK_SIZE;

// print NxN sub-matrix (with first element in [0][0]) of 'Image' of width 'width' and channels 'channels'
template <class T>
void printNbyN (T* Image, int N, int width, int channels);
// create a Mask NxN given a certain weight sigma 
void createMask (float* Mask, int N, float sigma);
void createMask1D (float* Mask, int N, float sigma);

// 
__global__
void GaussianBlurDevice1Dhorizontal(const unsigned char *image, float *image_out, const float *mask, int N) {

  int globalId_x = (threadIdx.x + blockDim.x * blockIdx.x) % WIDTH;
  int globalId_y = (threadIdx.x + blockDim.x * blockIdx.x) / WIDTH;

  if ((threadIdx.x + blockDim.x * blockIdx.x) < (WIDTH * HEIGHT)) {
      for(int channel = 0; channel < CHANNELS; channel++) {
          float pixel_value = 0;
          for(int u = 0; u < N; u++) {
              int new_x = min(WIDTH, max(0, globalId_x+u-N/2));
              pixel_value += mask[u]*image[(globalId_y*WIDTH+new_x)*CHANNELS+channel];
          }
          image_out[(globalId_y*WIDTH+globalId_x)*CHANNELS+channel]= pixel_value ;
      }
  }
}

__global__
void GaussianBlurDevice1Dvertical(const float *image, unsigned char *image_out, const float *mask, int N) {

  int globalId_x = (threadIdx.x + blockDim.x * blockIdx.x) % WIDTH;
  int globalId_y = (threadIdx.x + blockDim.x * blockIdx.x) / WIDTH;

  if ((threadIdx.x + blockDim.x * blockIdx.x) < (WIDTH * HEIGHT)) {
      for(int channel = 0; channel < CHANNELS; channel++) {
          float pixel_value = 0;
          for(int u = 0; u < N; u++) {
						int new_y = min(HEIGHT, max(0, globalId_y+u-N/2));
						pixel_value += mask[u]*image[(new_y*WIDTH+globalId_x)*CHANNELS+channel];
          }
          image_out[(globalId_y*WIDTH+globalId_x)*CHANNELS+channel]=(unsigned char) pixel_value ;
      }
  }
}

__global__
void GaussianBlurDevice(const unsigned char *image, const float *mask, unsigned char *image_out, int N) {

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
				// previous solution due to fixed discrete filter mask
				// image_out[(globalId_y*WIDTH+globalId_x)*CHANNELS+channel]=(unsigned char) (pixel_value < 255.0? pixel_value : 255.0);
				image_out[(globalId_y*WIDTH+globalId_x)*CHANNELS+channel]=(unsigned char) pixel_value ;
			}
			//no transparency
			//image_out[(y*WIDTH+x)*CHANNELS+CHANNELS-1]=(unsigned char) 255;
	}
}

void GaussianBlurHost(const unsigned char *image, const float *mask, unsigned char *image_out) {

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
				// previous solution due to fixed discrete filter mask
				// image_out[(y*WIDTH+x)*CHANNELS+channel]=(unsigned char) (pixel_value < 255.0? pixel_value : 255.0);
				image_out[(y*WIDTH+x)*CHANNELS+channel]=(unsigned char) pixel_value ;
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

	cv::imshow( "Initial image", I ); 
	cv::waitKey(0);

	// -------------------------------------------------------------------------
	// HOST MEMORY ALLOCATION
	unsigned char *Image  = new unsigned char[WIDTH * HEIGHT * CHANNELS];
	unsigned char *Image_out = new unsigned char[WIDTH * HEIGHT * CHANNELS];
	unsigned char *hostImage_out = new unsigned char[WIDTH * HEIGHT * CHANNELS];
	float sigma;
	float *Mask = new float[N*N]; 
	float *Mask1D = new float[N];
	
	Image = I.data;
	sigma = 3.0; 						  
  createMask(Mask, N, sigma);  
  createMask1D(Mask1D, N, sigma);
  
  printNbyN(Mask, N, N, 1);
	
	// -------------------------------------------------------------------------
	// HOST EXECUTIION
	TM_host.start();

	GaussianBlurHost(Image, Mask, Image_out);

	TM_host.stop();
	TM_host.print("GaussianBlur host:   ");
	
	printNbyN(Image_out, 5, WIDTH, CHANNELS);
	
	cv::Mat A(HEIGHT, WIDTH, CV_8UC3, Image_out);
	// cv::cvtColor(A, A, cv::COLOR_BGR2RGB);
	cv::imshow("Result of gaussian blur (host)", A);
	cv::waitKey(0);
	
	// -------------------------------------------------------------------------
	// DEVICE MEMORY ALLOCATION
	
	unsigned char *devImage, *devImage_out;
	float *devImage_inter, *devMask, *devMask1D;
	
	SAFE_CALL( cudaMalloc( &devImage, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char) ) );
	SAFE_CALL( cudaMalloc( &devImage_inter, WIDTH * HEIGHT * CHANNELS * sizeof(float) ) );
	SAFE_CALL( cudaMalloc( &devImage_out, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char) ) );
	SAFE_CALL( cudaMalloc( &devMask, N * N * sizeof(float) ) );
	SAFE_CALL( cudaMalloc( &devMask1D, N * sizeof(float) ) );
	
	// -------------------------------------------------------------------------
	// COPY DATA FROM HOST TO DEVICE
	
	SAFE_CALL( cudaMemcpy( devImage, Image, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice ) );
	SAFE_CALL( cudaMemcpy( devMask, Mask, N * N * sizeof(float), cudaMemcpyHostToDevice ) );
	SAFE_CALL( cudaMemcpy( devMask1D, Mask1D, N * sizeof(float), cudaMemcpyHostToDevice ) );

	// -------------------------------------------------------------------------
	// DEVICE EXECUTION
	
	dim3 block_size( BLOCK_SIZE, BLOCK_SIZE, 1 );
  dim3 num_blocks( ceil(float(WIDTH)/BLOCK_SIZE), ceil(float(HEIGHT)/BLOCK_SIZE), 1 );
  dim3 block_size_1D( BLOCK_SIZE_1D, 1, 1);
  dim3 num_blocks_1D( WIDTH*HEIGHT%BLOCK_SIZE_1D > 0? WIDTH*HEIGHT/BLOCK_SIZE_1D+1 : WIDTH*HEIGHT/BLOCK_SIZE_1D, 1, 1 );
	
	TM_device.start();

	// GaussianBlurDevice<<<block_size, num_blocks>>>(devImage, devMask, devImage_out, N);
	
	GaussianBlurDevice1Dhorizontal <<< block_size_1D, num_blocks_1D >>> 
	(devImage, devImage_inter, devMask1D, N);
	GaussianBlurDevice1Dvertical <<< block_size_1D, num_blocks_1D >>> 
	(devImage_inter, devImage_out, devMask1D, N);
	
	TM_device.stop();
	
	CHECK_CUDA_ERROR
	TM_device.print("GaussianBlur device: ");

	std::cout << std::setprecision(1)
	      << "Speedup: " << TM_host.duration() / TM_device.duration()
	      << "x\n\n";

	// -------------------------------------------------------------------------
	// COPY DATA FROM DEVICE TO HOST

	SAFE_CALL( cudaMemcpy( hostImage_out, devImage_out, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost ) );
	printNbyN(hostImage_out, 5, WIDTH, CHANNELS);

	// -------------------------------------------------------------------------
	// RESULT CHECK
	
	cv::Mat B(HEIGHT, WIDTH, CV_8UC3, hostImage_out);
	// cv::cvtColor(A, A, cv::COLOR_BGR2RGB);
	cv::imshow("Result of gaussian blur (device)", B);
	cv::waitKey(0);
	
	// GPU and CPU different precision for floating point operations ???
	for( int i = 0 ; i < HEIGHT ; i++ ) {
		for ( int j = 0 ; j < WIDTH ; j++) {
			// if (hostImage_out[i*WIDTH+j] != Image_out[i*WIDTH+j]) {
			if (hostImage_out[i*WIDTH+j] < Image_out[i*WIDTH+j]-1 || hostImage_out[i*WIDTH+j] > Image_out[i*WIDTH+j]+1 ) {
					std::cerr << "wrong result at ["<< i <<"][" << j << "]!" << std::endl;
					std::cerr << "Image_out: " << (short)Image_out[i*WIDTH+j] << std::endl;
					std::cerr << "devImage_out: " << (short)hostImage_out[i*WIDTH+j] << std::endl;
					cudaDeviceReset();
					std::exit(EXIT_FAILURE);
			}
		}
	}
	std::cout << "<> Correct\n\n";

	// -------------------------------------------------------------------------
	// HOST MEMORY DEALLOCATION
  // "delete[] Image", "delete[] A", "delete[] B" not needed as they are in the stack, not heap;
  delete[] Image_out;
  delete[] hostImage_out;

	// -------------------------------------------------------------------------
	// DEVICE MEMORY DEALLOCATION
	SAFE_CALL( cudaFree( devImage ) )
  SAFE_CALL( cudaFree( devImage_out ) )
  SAFE_CALL( cudaFree( devMask ) )
  SAFE_CALL( cudaFree( devMask1D ) )
  SAFE_CALL( cudaFree( devImage_inter ) )

	// -------------------------------------------------------------------------
	//SAFE_CALL(cudaFree());
	cudaDeviceReset();
}

template <class T>
void printNbyN (T* Image, int N, int width, int channels) {
	
	std::cout << std::endl;
	for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			std::cout << "[ ";
			for(int k=0; k<channels; k++) {
				std::cout << (short)Image[(i*width+j)*channels+k];
				k == channels-1? std::cout << " ": std::cout << " , "; 
			}
			std::cout << "]";
		}		
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void createMask (float *Mask, int N, float sigma) {
	
	float sum = 0.0;
	
	for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			Mask[i*N+j] = 1 / ( 2 * M_PI * pow( sigma, 2 ) ) * exp( -( pow( abs( N / 2 - i ), 2 ) + pow( abs( N / 2 - j ), 2 ) ) / ( 2 * pow( sigma, 2 ) ) );
			sum += Mask[i*N+j];
		}
	}
	for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			// normalize kernel to avoid darkening of image
			Mask[i*N+j] = Mask[i*N+j] / sum;
		}
	}
}

void createMask1D (float *Mask, int N, float sigma) {

	float sum = 0.0;
	
	for(int j=0; j<N; j++) {
		Mask[j] = 1 / ( sqrt(2 * M_PI) * sigma ) * exp( -( pow( abs( N / 2 - j ), 2 ) ) / ( 2 * pow( sigma, 2 ) ) );
		sum += Mask[j];
	}
	
	std::cout << "1D Mask: [   ";
	
	for(int j=0; j<N; j++) {
		// normalize kernel to avoid darkening of image
		Mask[j] = Mask[j] / sum;
		std::cout << Mask[j] << "   ";
	}
	
	std::cout << "]" << std::endl;
}
