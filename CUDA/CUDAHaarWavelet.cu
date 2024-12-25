#include <iostream>
#include <opencv2/core.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAHaarWavelet.cuh"
#include "CudaWrapper.cuh"

namespace CUDAHaarWavelet {

	__global__ void gpu_dwt(
		const float* __restrict__ input,
		float* __restrict__ output,
		const int currLevel,
		const int imgRows,
		const int imgCols
	) {
		__shared__ float shared_mem[BLOCK_SIZE * BLOCK_SIZE * 4]; // Shared memory for the block (to solve global access too many times)

		int r = blockIdx.x * blockDim.x + threadIdx.x;
		int c = blockIdx.y * blockDim.y + threadIdx.y;

		// Check if the thread is within the image
		if (r >= (imgRows >> currLevel) || c >= (imgCols >> currLevel)) {
			return;
		}

		int coeffsRow = imgRows >> currLevel;
		int coeffsCol = imgCols >> currLevel;

		int operateRow = r * 2;
		int operateCol = c * 2;

		// Load data into shared memory
		int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;
		int operateIndex = sharedIndex * 4;

		shared_mem[operateIndex] = input[operateRow * imgCols + operateCol];
		shared_mem[operateIndex + 1] = input[operateRow * imgCols + operateCol + 1];
		shared_mem[operateIndex + 2] = input[(operateRow + 1) * imgCols + operateCol];
		shared_mem[operateIndex + 3] = input[(operateRow + 1) * imgCols + operateCol + 1];

		// Compute the DWT using shared memory
		// Imagine each coefficient is load into different portion of shared memory (each in contiguous memory)
		float topLeft = shared_mem[operateIndex];
		float topRight = shared_mem[operateIndex + 1];
		float bottomLeft = shared_mem[operateIndex + 2];
		float bottomRight = shared_mem[operateIndex + 3];

		output[r * imgCols + c] = (topLeft + topRight + bottomLeft + bottomRight) * 0.5f; // LL
		output[r * imgCols + c + coeffsCol] = (topLeft - topRight + bottomLeft - bottomRight) * 0.5f; // HL
		output[(r + coeffsRow) * imgCols + c] = (topLeft + topRight - bottomLeft - bottomRight) * 0.5f; // LH
		output[(r + coeffsRow) * imgCols + c + coeffsCol] = (topLeft - topRight - bottomLeft + bottomRight) * 0.5f; // HH
	}

	__global__ void gpu_idwt(
		const float* __restrict__ input,
		float* __restrict__ output,
		const int currLevel,
		const int imgRows,
		const int imgCols
	) {
		__shared__ float shared_mem[BLOCK_SIZE * BLOCK_SIZE * 4]; // Shared memory for the block (to solve global access too many times)

		int r = blockIdx.x * blockDim.x + threadIdx.x;
		int c = blockIdx.y * blockDim.y + threadIdx.y;

		// Check if the thread is within the image
		if (r >= (imgRows >> currLevel) || c >= (imgCols >> currLevel)) {
			return;
		}

		int coeffsRow = imgRows >> currLevel;
		int coeffsCol = imgCols >> currLevel;

		// Load data into shared memory

		// Rearrange the shared memory to store the coefficients in the order of LL, HL, LH, HH
		// this to ensure the shared memory is contiguous
		int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;
		int operateIndex = sharedIndex * 4;
		shared_mem[operateIndex] = input[r * imgCols + c];
		shared_mem[operateIndex + 1] = input[r * imgCols + c + coeffsCol];
		shared_mem[operateIndex + 2] = input[(r + coeffsRow) * imgCols + c];
		shared_mem[operateIndex + 3] = input[(r + coeffsRow) * imgCols + c + coeffsCol];

		// Compute the inverse DWT using shared memory
		float LL = shared_mem[operateIndex];	 // LL
		float HL = shared_mem[operateIndex + 1]; // HL
		float LH = shared_mem[operateIndex + 2]; // LH
		float HH = shared_mem[operateIndex + 3]; // HH

		int outRow = r * 2;
		int outCol = c * 2;

		output[outRow * imgCols + outCol] = (LL + HL + LH + HH) * 0.5f;        // Top Left
		output[outRow * imgCols + outCol + 1] = (LL - HL + LH - HH) * 0.5f;    // Top Right
		output[(outRow + 1) * imgCols + outCol] = (LL + HL - LH - HH) * 0.5f;    // Bottom Left
		output[(outRow + 1) * imgCols + outCol + 1] = (LL - HL - LH + HH) * 0.5f; // Bottom Right
	}


	void dwt(const cv::Mat& input, cv::Mat& output, int nIteration) {
		assert(nIteration > 0);
		assert(input.channels() == 1);
		assert(input.rows % 2 == 0 && input.cols % 2 == 0);

		std::cout << "Performing DWT with " << nIteration << " iterations" << std::endl;

		const int memory_size = input.total() * sizeof(float);

		output.create(input.size(), CV_32F);
		float* float_input_img = (float*)input.data;
		float* float_output_wavelet = (float*)output.data;

		float* d_input_img;
		float* d_output_wavelet;
		CudaWrapper::malloc((void**)&d_input_img, memory_size);
		CudaWrapper::malloc((void**)&d_output_wavelet, memory_size);

		CudaWrapper::memcpy(d_input_img, float_input_img, memory_size, cudaMemcpyHostToDevice);

		int sharedMemSize = 4 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid;

		for (int i = 1; i <= nIteration; i++) {
			std::cout << "Level: " << i << std::endl;

			dimGrid.x = (((input.cols >> i) - 1) / BLOCK_SIZE) + 1;
			dimGrid.y = (((input.rows >> i) - 1) / BLOCK_SIZE) + 1;

			std::cout << "Grid: " << dimGrid.x << " x " << dimGrid.y << std::endl;
			std::cout << "Block: " << dimBlock.x << " x " << dimBlock.y << std::endl;

			gpu_dwt << <dimGrid, dimBlock>> > (
				d_input_img,
				d_output_wavelet,
				i,
				input.rows,
				input.cols
				);

			std::cout << "Copy data from device to device" << std::endl;

			if (i < nIteration)
				CudaWrapper::memcpy(d_input_img, d_output_wavelet, memory_size, cudaMemcpyDeviceToDevice);
			//cudaMemcpy(d_input_img, d_output_wavelet, memory_size, cudaMemcpyDeviceToDevice);
		}

		CudaWrapper::memcpy(float_output_wavelet, d_output_wavelet, memory_size, cudaMemcpyDeviceToHost);

		CudaWrapper::free(d_input_img);
		CudaWrapper::free(d_output_wavelet);
	}


	void idwt(const cv::Mat& input, cv::Mat& output, int nIteration) {
		assert(nIteration > 0);
		assert(input.channels() == 1);
		assert(input.rows % 2 == 0 && input.cols % 2 == 0);

		std::cout << "Performing IDWT with " << nIteration << " iterations" << std::endl;

		const int memory_size = input.total() * sizeof(float);

		output = input.clone();
		float* float_input_wavelet = (float*)input.data;
		float* float_output_img = (float*)output.data;

		float* d_input_wavelet;
		float* d_output_img;
		CudaWrapper::malloc((void**)&d_input_wavelet, memory_size);
		CudaWrapper::malloc((void**)&d_output_img, memory_size);

		CudaWrapper::memcpy(d_input_wavelet, float_input_wavelet, memory_size, cudaMemcpyHostToDevice);
		CudaWrapper::memcpy(d_output_img, float_output_img, memory_size, cudaMemcpyHostToDevice);

		int sharedMemSize = 4 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid;

		for (int i = nIteration; i >= 1; i--) {
			std::cout << "Level: " << i << std::endl;

			dimGrid.x = (((input.cols >> i) - 1) / BLOCK_SIZE) + 1;
			dimGrid.y = (((input.rows >> i) - 1) / BLOCK_SIZE) + 1;

			std::cout << "Grid: " << dimGrid.x << " x " << dimGrid.y << std::endl;
			std::cout << "Block: " << dimBlock.x << " x " << dimBlock.y << std::endl;

			gpu_idwt << <dimGrid, dimBlock>> > (
				d_input_wavelet,
				d_output_img,
				i,
				input.rows,
				input.cols
				);

			if (i > 1)
				CudaWrapper::memcpy(d_input_wavelet, d_output_img, memory_size, cudaMemcpyDeviceToDevice);
		}

		CudaWrapper::memcpy(float_output_img, d_output_img, memory_size, cudaMemcpyDeviceToHost);

		CudaWrapper::free(d_input_wavelet);
		CudaWrapper::free(d_output_img);
	}
}


