//required
#include "opencv2/opencv.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <windows.h>
#include <chrono>
#include <filesystem>

//#define DEBUG //for printing intermediate outputs to console
//#define SHOW_IMGS //for showing the final images
#define TIMER // for timing purposes

#define IM_WIDTH 512
#define IM_HEIGHT 512
#define IM_SIZE IM_WIDTH*IM_HEIGHT
#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define KERNEL_COUNT 2
#define SCALE 1
#define SHIFT 0
#define EPSILON 1e-7
#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/denominator) //Calculation for grid size
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace cv;
using namespace std;

//GLOBALS
const float channel_array[KERNEL_WIDTH * KERNEL_HEIGHT * KERNEL_COUNT] = { 1, 2, 1, 0, 0, 0, -1, -2, -1, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11 }; //Kernel values as array
const int block_size = 1024; //Used for some CUDA kernels

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) //For error checking
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void normalize_scale_shift_relu_Kernel(const float* __restrict__ in, float* __restrict__ out, const float batch_mean, const float const_val, const float shift)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x; //1D indexes

	float temp = 0;
	temp = (((in[threadId] - batch_mean) * const_val) + shift); //Final Batchnorm operation for all pixels

	//RELU starts
	if (temp < 0.f)
	{
		out[threadId] = 0;
	}
	else
	{
		out[threadId] = temp;
	}
}

__global__ void sumPixels_variance_Kernel(const float* __restrict__ a, float* __restrict__ out, const float batch_mean)
{
	int idx = threadIdx.x;
	float sum = 0;

	for (int i = idx; i < ((float)IM_SIZE); i += block_size)
	{
		sum += pow((a[i] - batch_mean), 2);  //Compute in parallel
	}
	__shared__ float r[block_size]; //Store to shared memory

	r[idx] = sum;

	__syncthreads();

	for (int size = block_size / 2; size > 0; size /= 2)
	{
		if (idx < size)
			r[idx] += r[idx + size]; //Compute the final value with parallel reduction

		__syncthreads();
	}
	if (idx == 0)
		*out = r[0] / ((float)IM_SIZE); //Variance
}

__global__ void sumPixels_mean_Kernel(const float* __restrict__ a, float* __restrict__ out)
{
	int idx = threadIdx.x;
	float sum = 0;

	for (int i = idx; i < ((float)IM_SIZE); i += block_size)
	{
		sum += a[i]; //Compute in parallel
	}
	__shared__ float r[block_size];
	r[idx] = sum;

	__syncthreads();

	for (int size = block_size / 2; size > 0; size /= 2)
	{
		if (idx < size)
			r[idx] += r[idx + size];  //Compute the final value with parallel reduction

		__syncthreads();
	}
	if (idx == 0)
		*out = r[0] / ((float)IM_SIZE); //Mean value
}

__global__ void kernel_conv_grey(const float* __restrict__ src, float* __restrict__ dst, const int width, const int height, const float* __restrict__ kernel, const int kernel_width, const int kernel_height)
{	//One kernel thread per pixel
	int i, j, k, l;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	/* convolution */
	float val = 0;

	for (i = x - (kernel_height % 2), k = 0; i <= x + (kernel_height % 2); i++, k++) //Boundary checks
	{
		for (j = y - (kernel_width % 2), l = 0; j <= y + (kernel_width % 2); j++, l++) //Boundary checks
		{
			if (i < 0 || j < 0 || i >= height || j >= width)
			{
				val = val + 0; //If out of bounds regard it as zero
			}
			else
			{
				val += src[width * i + j] * kernel[k * kernel_width + l]; //Multiply with the corresponding kernel value with the corresponding pixel
			}
		}
	}
	dst[width * x + y] = val;
}

class ImagesClass
{
public:
	ImagesClass(int b, int w, int h);
	virtual ~ImagesClass();
	int batch;
	int width;
	int height;
	vector<Mat_<float>> image_vec;
	vector<Mat_<float>> conv_imgs;
	vector<Mat_<float>> batchnorm_relu_imgs;

private:
};

ImagesClass::ImagesClass(int b, int w, int h) : batch{ b }, width{ w }, height{ h } {}

ImagesClass::~ImagesClass() {}

class KernelsClass
{
public:
	KernelsClass(int k, int l, int c);
	virtual ~KernelsClass();
	int kernel_width;
	int kernel_height;
	int channel_size;
	vector<vector<double>> kernel_vec2D;

private:
};

KernelsClass::KernelsClass(int k, int l, int c) : kernel_width{ k }, kernel_height{ l }, channel_size{ c } {}

KernelsClass::~KernelsClass() {}

void read_save_imgs(ImagesClass& Imgs, int batch_size, vector<String>& filenames)
{
	for (size_t i = 0; i < batch_size; i++)
	{
		Mat im = imread(filenames[i]);
		//Resize im to specific value
		resize(im, im, Size(IM_WIDTH, IM_HEIGHT), 0, 0, 1);

		//Convert the pixels to grayscale
		cvtColor(im, im, COLOR_BGR2GRAY);

		//Convert the pixels to float values for further processing
		im.convertTo(im, CV_32FC1);

		//Save the images in vector
		Imgs.image_vec.push_back(im);
	}
}

void convolveWithCuda(int batch_size, cudaStream_t* streams, float* dev_batch_img_float, float* dev_batch_conv_imgs, float* dev_channel_array)
{
	const int blockSize = 16;
	int gridX = FRACTION_CEILING(IM_HEIGHT, blockSize);
	int gridY = FRACTION_CEILING(IM_WIDTH, blockSize);
	dim3 block(blockSize, blockSize);
	dim3 grid(gridX, gridY);

	//Compute convolution concurrently in streams
#pragma unroll
	for (int i = 0; i < batch_size; i++)
	{
#pragma unroll
		for (int y = 0; y < KERNEL_COUNT; y++)
		{
			//Call kernel
			kernel_conv_grey << <grid, block, 0, streams[i * KERNEL_COUNT + y] >> > ((dev_batch_img_float + IM_SIZE * i), (dev_batch_conv_imgs + i * KERNEL_COUNT * IM_SIZE + y * IM_SIZE), (int)IM_WIDTH, (int)IM_HEIGHT, (dev_channel_array + y * KERNEL_WIDTH * KERNEL_HEIGHT), KERNEL_WIDTH, KERNEL_HEIGHT);
		}
	}
	cudaDeviceSynchronize();
}

void calcmeanWithCuda(int batch_size, cudaStream_t* streams, float* dev_batch_conv_imgs, float* dev_mean_vals)
{
	//Mean value of all the pixels
#pragma unroll
	for (int i = 0; i < batch_size; i++)
	{
#pragma unroll
		for (int y = 0; y < KERNEL_COUNT; y++)
		{
			//Call kernel
			sumPixels_mean_Kernel << <1, block_size, 0, streams[i * KERNEL_COUNT + y] >> > ((dev_batch_conv_imgs + i * KERNEL_COUNT * IM_SIZE + y * IM_SIZE), (dev_mean_vals + i * KERNEL_COUNT + y));
		}
	}
	cudaDeviceSynchronize();
}

void calcvarianceWithCuda(int batch_size, cudaStream_t* streams, float* dev_batch_conv_imgs, float* dev_variance_vals, float* mean_vals)
{
	//Variance value of all the pixels
#pragma unroll
	for (int i = 0; i < batch_size; i++)
	{
#pragma unroll
		for (int y = 0; y < KERNEL_COUNT; y++)
		{
			//Call kernel
			sumPixels_variance_Kernel << <1, block_size, 0, streams[i * KERNEL_COUNT + y] >> > ((dev_batch_conv_imgs + i * KERNEL_COUNT * IM_SIZE + y * IM_SIZE), (dev_variance_vals + i * KERNEL_COUNT + y), mean_vals[i * KERNEL_COUNT + y]);
		}
	}
	cudaDeviceSynchronize();
}

void bnormReluWithCuda(int batch_size, cudaStream_t* streams, float* dev_batch_conv_imgs, float* dev_batch_bnrelu_imgs, float* mean_vals, float* scale_normalizer)
{
	const int blockSize = 16;
	int gridX = FRACTION_CEILING(IM_HEIGHT, blockSize);
	int gridY = FRACTION_CEILING(IM_WIDTH, blockSize);
	dim3 block(blockSize, blockSize);
	dim3 grid(gridX, gridY);
	//Finalize batch norm & relu
#pragma unroll
	for (int i = 0; i < batch_size; i++)
	{
#pragma unroll
		for (int y = 0; y < KERNEL_COUNT; y++)
		{
			//Call kernel
			normalize_scale_shift_relu_Kernel << <grid, block, 0, streams[i * KERNEL_COUNT + y] >> > ((dev_batch_conv_imgs + i * KERNEL_COUNT * IM_SIZE + y * IM_SIZE), (dev_batch_bnrelu_imgs + i * KERNEL_COUNT * IM_SIZE + y * IM_SIZE), mean_vals[i * KERNEL_COUNT + y], scale_normalizer[i * KERNEL_COUNT + y], (float)SHIFT);
		}
	}
	cudaDeviceSynchronize();
}


int main(int argc, char** argv)
{
	cudaError_t cudaStatus;
	vector<String> filenames;

	// Get all jpg in the folder
	String image_path;
	if (argc <= 1){
		image_path = "C:/Users/aayte/Downloads/inputImageDir/*.jpg"; //change this to your image folder path
		std::cerr << "Default file path is used given as:\n" << image_path << std::endl;

	}
	else {
		image_path = argv[1];
	}

	cv::glob(image_path, filenames);

	int batch_size = filenames.size();
	ImagesClass Imgs(batch_size, IM_WIDTH, IM_HEIGHT);

	read_save_imgs(Imgs, batch_size, filenames); //Read and save the images as first step

	//Initialize host arrays
	float* batch_img_float = (float*)malloc(IM_SIZE * batch_size * sizeof(float)); //Original images
	float* batch_conv_imgs = (float*)malloc((float)IM_SIZE * batch_size * KERNEL_COUNT * sizeof(float)); //Convolved images
	float* batch_bnrelu_imgs = (float*)malloc((float)IM_SIZE * batch_size * KERNEL_COUNT * sizeof(float)); //Final images
	float* mean_vals = (float*)malloc(batch_size * KERNEL_COUNT * sizeof(float)); //Mean values as array
	float* variance_vals = (float*)malloc(batch_size * KERNEL_COUNT * sizeof(float)); //Variance values as array
	float* scale_normalizer = (float*)malloc(batch_size * KERNEL_COUNT * sizeof(float)); //This is SCALE*(1/sqrt(batch_variance+epsilon)). Combined for faster calculation

	for (int i = 0; i < batch_size; i++)
	{
		float* dataPointer = reinterpret_cast<float*>(Imgs.image_vec[i].data);
		std::memcpy((batch_img_float + i * IM_SIZE), dataPointer, IM_SIZE * sizeof(float)); //Vector to array for ease of use
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	gpuErrchk(cudaSetDevice(0));

	//CUDA Memory Allocations
	float* dev_batch_img_float = nullptr;  //Original images
	gpuErrchk(cudaMalloc((void**)&dev_batch_img_float, IM_SIZE * batch_size * sizeof(float)));

	float* dev_batch_conv_imgs = nullptr; //Convolved images
	gpuErrchk(cudaMalloc((void**)&dev_batch_conv_imgs, IM_SIZE * batch_size * KERNEL_COUNT * sizeof(float)));

	float* dev_batch_bnrelu_imgs = nullptr; //Final images
	gpuErrchk(cudaMalloc((void**)&dev_batch_bnrelu_imgs, IM_SIZE * batch_size * KERNEL_COUNT * sizeof(float)));

	float* dev_channel_array = nullptr;  //Kernel values as array
	gpuErrchk(cudaMalloc((void**)&dev_channel_array, KERNEL_WIDTH * KERNEL_HEIGHT * KERNEL_COUNT * sizeof(float)));

	float* dev_mean_vals = nullptr; //Mean values as array
	gpuErrchk(cudaMalloc((void**)&dev_mean_vals, batch_size * KERNEL_COUNT * sizeof(float)));

	float* dev_variance_vals = nullptr; //Variance values as array
	gpuErrchk(cudaMalloc((void**)&dev_variance_vals, batch_size * KERNEL_COUNT * sizeof(float)));

	//CUDA Mem Copies
	gpuErrchk(cudaMemcpy(dev_batch_img_float, batch_img_float, IM_SIZE * batch_size * sizeof(float), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(dev_channel_array, channel_array, KERNEL_WIDTH * KERNEL_HEIGHT * KERNEL_COUNT * sizeof(float), cudaMemcpyHostToDevice));

	int nstreams = batch_size * KERNEL_COUNT; //Calculate concurrent stream count
	cudaStream_t* streams = (cudaStream_t*)malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0; i < nstreams; i++)
	{
		gpuErrchk(cudaStreamCreate(&(streams[i]))); //Generate streams
	}

#ifdef TIMER
	auto start_time_0 = std::chrono::high_resolution_clock::now();
#endif // PRINT

	convolveWithCuda(batch_size, streams, dev_batch_img_float, dev_batch_conv_imgs, dev_channel_array); //Convolution kernel
#ifdef TIMER
	auto end_time_0 = std::chrono::high_resolution_clock::now();
	auto time0 = end_time_0 - start_time_0;
	std::cout <<"\n\nConvolution kernel Time is: "<< time0 / std::chrono::milliseconds(1) << "ms\n\n";
	auto start_time_1 = std::chrono::high_resolution_clock::now();
#endif // PRINT

	calcmeanWithCuda(batch_size, streams, dev_batch_conv_imgs, dev_mean_vals); //Mean calculation kernel
#ifdef TIMER
	auto end_time_1 = std::chrono::high_resolution_clock::now();
	auto time1 = end_time_1 - start_time_1;
	//std::cout << "\n\nMean calculation kernel total time is: " << time1 / std::chrono::milliseconds(1) << "ms\n\n";
#endif // PRINT


	gpuErrchk(cudaMemcpy(mean_vals, dev_mean_vals, batch_size * KERNEL_COUNT * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef DEBUG
	for (int i = 0; i < batch_size * KERNEL_COUNT; i++)
	{
		cout << "Mean val for image: " << i << "  Val: " << mean_vals[i] << endl;
	}
#endif // PRINT

#ifdef TIMER
	auto start_time_2 = std::chrono::high_resolution_clock::now();
#endif // PRINT
	calcvarianceWithCuda(batch_size, streams, dev_batch_conv_imgs, dev_variance_vals, mean_vals); //Variance calculation kernel
#ifdef TIMER
	auto end_time_2 = std::chrono::high_resolution_clock::now();
	auto time2 = end_time_2 - start_time_2;
	std::cout << "\n\nMean calculation kernel and Variance calculation  kernel total time is: " << (time1+time2) / std::chrono::milliseconds(1) << "ms\n\n";
#endif // PRINT


	gpuErrchk(cudaMemcpy(variance_vals, dev_variance_vals, batch_size * KERNEL_COUNT * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef DEBUG
	for (int i = 0; i < batch_size * KERNEL_COUNT; i++)
	{
		cout << "Variance for image: " << i << "  Val: " << variance_vals[i] << endl;
	}
#endif // PRINT

	//This is SCALE*(1/sqrt(batch_variance+epsilon)). Combined for faster calculation
	for (int i = 0; i < batch_size * KERNEL_COUNT; i++)
	{
		scale_normalizer[i] = ((float)SCALE) / ((float)(sqrt(variance_vals[i] + (float)EPSILON)));
#ifdef DEBUG
		cout << "Scale normalizer for image: " << i << "  Val: " << scale_normalizer[i] << endl;
#endif // PRINT
	}


#ifdef TIMER
	auto start_time_3 = std::chrono::high_resolution_clock::now();
#endif // PRINT
	bnormReluWithCuda(batch_size, streams, dev_batch_conv_imgs, dev_batch_bnrelu_imgs, mean_vals, scale_normalizer); //Call Batchnorm & ReLu kernel
#ifdef TIMER
	auto end_time_3 = std::chrono::high_resolution_clock::now();
	auto time3 = end_time_3 - start_time_3;
	auto time4 = end_time_3 - start_time_0;
	std::cout << "\n\nBatchnorm & ReLu kernel total time is: " << time3 / std::chrono::milliseconds(1) << "ms\n\n";
	std::cout << "\n\nOVERALL STACK TIME IS: " << time4 / std::chrono::milliseconds(1) << "ms\n\n";
#endif // PRINT
	

	gpuErrchk(cudaMemcpy(batch_bnrelu_imgs, dev_batch_bnrelu_imgs, IM_SIZE * batch_size * KERNEL_COUNT * sizeof(float), cudaMemcpyDeviceToHost));


#ifdef SHOW_IMGS
	for (int i = 0; i < batch_size * KERNEL_COUNT; i++)
	{
		for (int z = 0; z < IM_WIDTH; z++)
		{
			for (int t = 0; t < IM_HEIGHT; t++)
			{
				Imgs.image_vec[0].at<float>(z, t) = batch_bnrelu_imgs[(z * IM_HEIGHT + t) + IM_SIZE * i]; //Show output images
			}
		}
		imshow("", Imgs.image_vec[0]);
		waitKey();
	}
#endif // SHOW_IMGS


Error:
	free(batch_img_float);
	free(batch_conv_imgs);
	free(batch_bnrelu_imgs);
	free(mean_vals);
	free(variance_vals);
	free(scale_normalizer);
	cudaFree(dev_batch_img_float);
	cudaFree(dev_batch_conv_imgs);
	cudaFree(dev_batch_bnrelu_imgs);
	cudaFree(dev_channel_array);
	cudaFree(dev_mean_vals);
	cudaFree(dev_variance_vals);
	cudaFree(streams);



#ifdef TIMER
	auto end_time_4 = std::chrono::high_resolution_clock::now();
	auto time5 = end_time_4 - start_time_0;
	std::cout << "\n\nOVERALL STACK TIME (EXCLUDING DEVICE RESET) IS: " << time5 / std::chrono::milliseconds(1) << "ms\n\n";
#endif // PRINT

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	gpuErrchk(cudaDeviceReset());

	return 0;
}
