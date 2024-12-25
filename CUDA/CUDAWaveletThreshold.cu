#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAHaarWavelet.cuh"
#include "CUDAWaveletThreshold.cuh"
#include "CudaWrapper.cuh"

//
// Wavelet-Based Thresholding (VisuShrink, NeighShrink, ModiNeighShrink)
//
namespace CUDAWaveletThreshold {

	/***************************************
	* HELPER functions.
	****************************************/

	__device__ __forceinline__ float custom_fabs(float x) {
		if (x < 0) {
			return -x;
		}
		return x;
	}

	__global__ void computeAbsoluteValues(
		const float* __restrict__ input,
		float* __restrict__ output,
		int size
	) {
		extern __shared__ float shared_mem[];

		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < size) {
			shared_mem[threadIdx.x] = custom_fabs(input[idx]);
			output[idx] = shared_mem[threadIdx.x];
		}
	}

	__device__ __forceinline__ float thresholdFunction(float x, float threshold, int mode) {
		switch (mode) {
		case 0: // Hard Shrinkage
			return std::abs(x) > threshold ? x : 0.0;
		case 1: // Soft Shrinkage
			return std::abs(x) > threshold ? std::copysign(std::abs(x) - threshold, x) : 0.0;
		case 2: // Garrot Shrinkage
			return std::abs(x) > threshold ? x - ((threshold * threshold) / x) : 0.0;
		default:
			return x;
		}
	}

	double calculateSigma(
		cv::Mat& coeffs
	) {
		cv::Mat temp = coeffs.clone();

		std::cout << "Calculating Sigma..." << std::endl;

		int img_size = coeffs.rows * coeffs.cols;

		std::cout << "Image Size: " << img_size << std::endl;

		// Allocate device memory
		int memory_size = sizeof(float) * img_size;

		float* input_data = (float*)temp.data;

		float* abs_values;
		CudaWrapper::hostAllocate((void**)&abs_values, memory_size, cudaHostAllocDefault);

		float* gpu_input;
		CudaWrapper::malloc((void**)&gpu_input, memory_size);
		float* gpu_abs_values;
		CudaWrapper::malloc((void**)&gpu_abs_values, memory_size);

		dim3 block_size(BLOCK_SIZE * BLOCK_SIZE);
		int sharedMemSize = block_size.x * sizeof(float);
		dim3 grid_size((img_size + block_size.x - 1) / block_size.x);

		CudaWrapper::memcpy(gpu_input, input_data, memory_size, cudaMemcpyHostToDevice);

		computeAbsoluteValues << <grid_size, block_size, sharedMemSize >> > (
			gpu_input,
			gpu_abs_values,
			img_size
			);

		CudaWrapper::memcpy(abs_values, gpu_abs_values, memory_size, cudaMemcpyDeviceToHost);

		std::nth_element(abs_values, abs_values + img_size / 2, abs_values + img_size); // partial sort to get the median

		double median = abs_values[img_size / 2]; // Get the median value

		double sigma = median / 0.6745; // Estimate noise standard deviation

		std::cout << "\nSigma: " << sigma << "\n" << std::endl;

		CudaWrapper::hostFree(abs_values);

		CudaWrapper::free(gpu_input);
		CudaWrapper::free(gpu_abs_values);

		return sigma;
	}

	double calculateUniversalThreshold(
		cv::Mat& highFreqBand
	) {

		double sigma = calculateSigma(highFreqBand); // Estimate noise standard deviation

		double threshold = sigma * sqrt(2 * std::log(highFreqBand.rows * highFreqBand.cols));

		return threshold;
	}


	/***************************************
	* VisuShrink thresholding function.
	****************************************/
	/**
	* VisuShrink thresholding function.
	*/
	__global__ void applyVisuShrinkKernel(
		float* __restrict__ coeffs,
		float threshold,
		int rows, int cols,
		int mode
	) {
		int r = blockIdx.y * blockDim.y + threadIdx.y;
		int c = blockIdx.x * blockDim.x + threadIdx.x;

		if (r >= rows || c >= cols) {
			return;
		}

		float value = coeffs[r * cols + c];
		coeffs[r * cols + c] = thresholdFunction(value, threshold, mode);
	}

	void applyVisuShrink(
		cv::Mat& HL,
		cv::Mat& LH,
		cv::Mat& HH,
		double threshold,
		CUDAThresholdMode mode
	) {
		int memorySize = sizeof(float) * HL.rows * HL.cols;

		float* float_HL = (float*)HL.data;
		float* float_LH = (float*)LH.data;
		float* float_HH = (float*)HH.data;

		float* gpu_HL, * gpu_LH, * gpu_HH;
		CudaWrapper::malloc((void**)&gpu_HL, memorySize);
		CudaWrapper::malloc((void**)&gpu_LH, memorySize);
		CudaWrapper::malloc((void**)&gpu_HH, memorySize);

		cudaStream_t stream1, stream2, stream3;
		CudaWrapper::streamCreate(&stream1);
		CudaWrapper::streamCreate(&stream2);
		CudaWrapper::streamCreate(&stream3);

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((HL.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (HL.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

		CudaWrapper::memcpy(gpu_HL, float_HL, memorySize, cudaMemcpyHostToDevice, stream1);
		applyVisuShrinkKernel << <grid, block, 0, stream1 >> > (
			gpu_HL, (float)threshold,
			HL.rows, HL.cols, (int)mode
			);

		CudaWrapper::memcpy(gpu_LH, float_LH, memorySize, cudaMemcpyHostToDevice, stream2);
		applyVisuShrinkKernel << <grid, block, 0, stream2 >> > (
			gpu_LH, (float)threshold,
			LH.rows, LH.cols, (int)mode
			);

		CudaWrapper::memcpy(gpu_HH, float_HH, memorySize, cudaMemcpyHostToDevice, stream3);
		applyVisuShrinkKernel << <grid, block, 0, stream3 >> > (
			gpu_HH, (float)threshold,
			HH.rows, HH.cols, (int)mode
			);

		CudaWrapper::memcpy(float_HL, gpu_HL, memorySize, cudaMemcpyDeviceToHost, stream1);
		CudaWrapper::memcpy(float_LH, gpu_LH, memorySize, cudaMemcpyDeviceToHost, stream2);
		CudaWrapper::memcpy(float_HH, gpu_HH, memorySize, cudaMemcpyDeviceToHost, stream3);

		CudaWrapper::streamSynchronize(stream1);
		CudaWrapper::streamSynchronize(stream2);
		CudaWrapper::streamSynchronize(stream3);

		CudaWrapper::free(gpu_HL);
		CudaWrapper::free(gpu_LH);
		CudaWrapper::free(gpu_HH);

		CudaWrapper::streamDestroy(stream1);
		CudaWrapper::streamDestroy(stream2);
		CudaWrapper::streamDestroy(stream3);
	}

	void visuShrink(
		const cv::Mat& input,
		cv::Mat& output,
		int level,
		CUDAThresholdMode mode
	) {

		if (input.empty() || level < 1) {
			throw std::invalid_argument("Invalid input parameters for VisuShrink.");
		}

		assert(input.type() == CV_32F || input.type() == CV_64F);

		/*
			1. apply wavelet transform to the input image
		*/
		std::cout << "Performing DWT" << std::endl;
		CUDAHaarWavelet::dwt(input, output, level); // dwt


		// Initialize variables
		int rows = output.rows;
		int cols = output.cols;

		/*
			2. Calc Universal Threshold
		*/
		std::cout << "Calculating Universal Threshold" << std::endl;
		// Based on many paper, the thresholding is applied to the high-frequency sub-bands on first level
		cv::Mat highFreqBand = output(
			cv::Rect(
				cols >> 1,
				rows >> 1,
				cols >> 1,
				rows >> 1
			)
		);
		double threshold = calculateUniversalThreshold(highFreqBand);

		/*
			3. Apply VisuShrink thresholding
		*/

		std::cout << "Applying VisuShrink" << std::endl;

		// Initialize the output image as dwtOutput
		for (int i = 1; i <= level; i++) {

			std::cout << "Performing VisuShrink level: " << i << std::endl;

			//LH
			cv::Rect lhRoi(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			);
			cv::Mat lhOutput = output(lhRoi).clone();

			//HL
			cv::Rect hlRoi(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			);
			cv::Mat hlOutput = output(hlRoi).clone();

			//HH
			cv::Rect hhRoi(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			);
			cv::Mat hhOutput = output(hhRoi).clone();

			applyVisuShrink(
				lhOutput,
				hlOutput,
				hhOutput,
				threshold, mode
			);

			lhOutput.copyTo(output(lhRoi));
			hlOutput.copyTo(output(hlRoi));
			hhOutput.copyTo(output(hhRoi));
			
		}
		std::cout << "VisuShrink Done" << std::endl;

		/*
			4. apply inverse wavelet transform to the output image
		*/
		std::cout << "Performing IDWT" << std::endl;
		CUDAHaarWavelet::idwt(output, output, level); // idwt
	}


	/***************************************
	* NeighShrink thresholding function.
	****************************************/
	__global__ void applyNeighShrinkKernel(
		const float* __restrict__ coeffs,
		float* __restrict__ output,
		int rows, int cols,
		float threshold, int halfWindow
	) {
		int r = blockIdx.y * blockDim.y + threadIdx.y;
		int c = blockIdx.x * blockDim.x + threadIdx.x;

		if (r >= rows || c >= cols) {
			return;
		}

		float squareSum = 0.0;

		for (int wr = -halfWindow; wr <= halfWindow; wr++) {
			for (int wc = -halfWindow; wc <= halfWindow; wc++) {
				int rr = r + wr;
				int cc = c + wc;

				if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) {
					float value = coeffs[rr * cols + cc];
					squareSum += value * value;
				}
			}
		}

		float value = coeffs[r * cols + c];
		float shrinkage = 1.0f - ((threshold * threshold) / squareSum);

		if (shrinkage < 0.0f) {
			shrinkage = 0.0f;
		}

		output[r * cols + c] = value * shrinkage;
	}

	void applyNeighShrink(
		cv::Mat& HL,
		cv::Mat& LH,
		cv::Mat& HH,
		double threshold,
		int windowSize
	) {

		int rows = HL.rows;
		int cols = HL.cols;
		int halfWindow = windowSize / 2;
		int memory_size = sizeof(float) * rows * cols;

		float* float_HL = (float*)HL.data;
		float* float_LH = (float*)LH.data;
		float* float_HH = (float*)HH.data;

		float* gpu_HL, * gpu_LH, * gpu_HH;
		CudaWrapper::malloc((void**)&gpu_HL, memory_size);
		CudaWrapper::malloc((void**)&gpu_LH, memory_size);
		CudaWrapper::malloc((void**)&gpu_HH, memory_size);

		float* output_HL_data, * output_LH_data, * output_HH_data;
		CudaWrapper::malloc((void**)&output_HL_data, memory_size);
		CudaWrapper::malloc((void**)&output_LH_data, memory_size);
		CudaWrapper::malloc((void**)&output_HH_data, memory_size);

		dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid_size((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

		std::cout << "grid: " << grid_size.x << "x" << grid_size.y << std::endl;
		std::cout << "block: " << block_size.x << "x" << block_size.y << std::endl;

		cudaStream_t stream1, stream2, stream3;
		CudaWrapper::streamCreate(&stream1);
		CudaWrapper::streamCreate(&stream2);
		CudaWrapper::streamCreate(&stream3);

		CudaWrapper::memcpy(gpu_HL, float_HL, memory_size, cudaMemcpyHostToDevice, stream1);
		applyNeighShrinkKernel << <grid_size, block_size, 0, stream1 >> > (
			gpu_HL, output_HL_data,
			rows, cols,
			(float)threshold,
			halfWindow
			);

		CudaWrapper::memcpy(gpu_LH, float_LH, memory_size, cudaMemcpyHostToDevice, stream2);
		applyNeighShrinkKernel << <grid_size, block_size, 0, stream2 >> > (
			gpu_LH, output_LH_data,
			rows, cols,
			(float)threshold,
			halfWindow
			);

		CudaWrapper::memcpy(gpu_HH, float_HH, memory_size, cudaMemcpyHostToDevice, stream3);
		applyNeighShrinkKernel << <grid_size, block_size, 0, stream3 >> > (
			gpu_HH, output_HH_data,
			rows, cols,
			(float)threshold,
			halfWindow
			);

		CudaWrapper::memcpy(float_HL, output_HL_data, memory_size, cudaMemcpyDeviceToHost, stream1);
		CudaWrapper::memcpy(float_LH, output_LH_data, memory_size, cudaMemcpyDeviceToHost, stream2);
		CudaWrapper::memcpy(float_HH, output_HH_data, memory_size, cudaMemcpyDeviceToHost, stream3);

		CudaWrapper::streamSynchronize(stream1);
		CudaWrapper::streamSynchronize(stream2);
		CudaWrapper::streamSynchronize(stream3);


		// Free memory
		CudaWrapper::free(output_HL_data);
		CudaWrapper::free(output_LH_data);
		CudaWrapper::free(output_HH_data);

		CudaWrapper::free(gpu_HL);
		CudaWrapper::free(gpu_LH);
		CudaWrapper::free(gpu_HH);

		CudaWrapper::streamDestroy(stream1);
		CudaWrapper::streamDestroy(stream2);
		CudaWrapper::streamDestroy(stream3);
	}

	/**
	* NeighShrink thresholding function.
	*/
	void neighShrink(
		const cv::Mat& input,
		cv::Mat& output,
		int level,
		int windowSize
	) {
		// Verify the input validity
		if (input.empty() || level < 1 || windowSize < 1) {
			throw std::invalid_argument("Invalid input parameters for NeighShrink.");
		}

		assert(input.type() == CV_32F || input.type() == CV_64F);

		/*
			1. Apply wavelet transform to the input image
		*/
		cv::Mat dwtOutput;
		CUDAHaarWavelet::dwt(input, dwtOutput, level); // dwt

		// Initialize variables
		int rows = dwtOutput.rows;
		int cols = dwtOutput.cols;
		int halfWindow = windowSize / 2;

		/*
			2. Calc Universal Threshold
		*/
		// Based on many paper, the thresholding is applied to the high-frequency sub-bands on first level
		cv::Mat highFreqBand = dwtOutput(
			cv::Rect(
				cols >> 1,
				rows >> 1,
				cols >> 1,
				rows >> 1
			)
		);

		/*  Universal Threshold Calculation */
		double threshold = calculateUniversalThreshold(highFreqBand); // Calculate VisuShrink threshold

		std::cout << "Threshold: " << threshold << std::endl;

		/*
			3. Apply NeighShrink thresholding
		*/
		// Apply NeighShrink thresholding
		// Loop through each level of the wavelet decomposition
		output = dwtOutput.clone();
		for (int i = 1; i <= level; ++i) {

			std::cout << "Performing NeighShrink level: " << i << std::endl;
			//LH
			cv::Rect lhRoi(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			);
			cv::Mat lhOutput = output(lhRoi).clone();

			//HL
			cv::Rect hlRoi(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			);
			cv::Mat hlOutput = output(hlRoi).clone();

			//HH
			cv::Rect hhRoi(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			);
			cv::Mat hhOutput = output(hhRoi).clone();

			applyNeighShrink(
				lhOutput,
				hlOutput,
				hhOutput,
				threshold, windowSize
			);

			lhOutput.copyTo(output(lhRoi));
			hlOutput.copyTo(output(hlRoi));
			hhOutput.copyTo(output(hhRoi));
		}

		/*
			4. Apply inverse wavelet transform to the output image
		*/

		CUDAHaarWavelet::idwt(output, output, level); // idwt
	}



	/***************************************
	* ModiNeighShrink thresholding function.
	****************************************/

	__global__ void applyModiNeighShrinkKernel(
		const float* __restrict__ coeffs,
		float* __restrict__ output,
		int rows, int cols,
		float threshold, int halfWindow
	) {
		int r = blockIdx.y * blockDim.y + threadIdx.y;
		int c = blockIdx.x * blockDim.x + threadIdx.x;

		if (r >= rows || c >= cols) {
			return;
		}

		float squareSum = 0.0;

		for (int wr = -halfWindow; wr <= halfWindow; wr++) {
			for (int wc = -halfWindow; wc <= halfWindow; wc++) {
				int rr = r + wr;
				int cc = c + wc;

				if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) {
					float value = coeffs[rr * cols + cc];
					squareSum += value * value;
				}
			}
		}

		float value = coeffs[r * cols + c];
		float shrinkage = 1.0f - ((3.0f / 4.0f) * (threshold * threshold) / squareSum);

		if (shrinkage < 0.0f) {
			shrinkage = 0.0f;
		}

		output[r * cols + c] = value * shrinkage;
	}

	void applyModiNeighShrink(
		cv::Mat& HL,
		cv::Mat& LH,
		cv::Mat& HH,
		double threshold,
		int windowSize
	) {

		int rows = HL.rows;
		int cols = HL.cols;
		int halfWindow = windowSize / 2;
		int memory_size = sizeof(float) * rows * cols;

		float* float_HL = (float*)HL.data;
		float* float_LH = (float*)LH.data;
		float* float_HH = (float*)HH.data;

		float* gpu_HL, * gpu_LH, * gpu_HH;
		CudaWrapper::malloc((void**)&gpu_HL, memory_size);
		CudaWrapper::malloc((void**)&gpu_LH, memory_size);
		CudaWrapper::malloc((void**)&gpu_HH, memory_size);

		float* output_HL_data, * output_LH_data, * output_HH_data;
		CudaWrapper::malloc((void**)&output_HL_data, memory_size);
		CudaWrapper::malloc((void**)&output_LH_data, memory_size);
		CudaWrapper::malloc((void**)&output_HH_data, memory_size);

		dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid_size((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

		std::cout << "grid: " << grid_size.x << "x" << grid_size.y << std::endl;
		std::cout << "block: " << block_size.x << "x" << block_size.y << std::endl;

		cudaStream_t stream1, stream2, stream3;
		CudaWrapper::streamCreate(&stream1);
		CudaWrapper::streamCreate(&stream2);
		CudaWrapper::streamCreate(&stream3);

		CudaWrapper::memcpy(gpu_HL, float_HL, memory_size, cudaMemcpyHostToDevice, stream1);
		applyModiNeighShrinkKernel << <grid_size, block_size, 0, stream1 >> > (
			gpu_HL, output_HL_data,
			rows, cols,
			(float)threshold,
			halfWindow
			);

		CudaWrapper::memcpy(gpu_LH, float_LH, memory_size, cudaMemcpyHostToDevice, stream2);
		applyModiNeighShrinkKernel << <grid_size, block_size, 0, stream2 >> > (
			gpu_LH, output_LH_data,
			rows, cols,
			(float)threshold,
			halfWindow
			);

		CudaWrapper::memcpy(gpu_HH, float_HH, memory_size, cudaMemcpyHostToDevice, stream3);
		applyModiNeighShrinkKernel << <grid_size, block_size, 0, stream3 >> > (
			gpu_HH, output_HH_data,
			rows, cols,
			(float)threshold,
			halfWindow
			);

		CudaWrapper::memcpy(float_HL, output_HL_data, memory_size, cudaMemcpyDeviceToHost, stream1);
		CudaWrapper::memcpy(float_LH, output_LH_data, memory_size, cudaMemcpyDeviceToHost, stream2);
		CudaWrapper::memcpy(float_HH, output_HH_data, memory_size, cudaMemcpyDeviceToHost, stream3);

		CudaWrapper::streamSynchronize(stream1);
		CudaWrapper::streamSynchronize(stream2);
		CudaWrapper::streamSynchronize(stream3);


		// Free memory
		CudaWrapper::free(output_HL_data);
		CudaWrapper::free(output_LH_data);
		CudaWrapper::free(output_HH_data);

		CudaWrapper::free(gpu_HL);
		CudaWrapper::free(gpu_LH);
		CudaWrapper::free(gpu_HH);

		CudaWrapper::streamDestroy(stream1);
		CudaWrapper::streamDestroy(stream2);
		CudaWrapper::streamDestroy(stream3);
	}

	/**
	* ModiNeighShrink thresholding function.
	*/
	void modineighShrink(
		const cv::Mat& input,
		cv::Mat& output,
		int level,
		int windowSize
	) {
		// Verify the input validity
		if (input.empty() || level < 1 || windowSize < 1) {
			throw std::invalid_argument("Invalid input parameters for ModiNeighShrink.");
		}

		assert(input.type() == CV_32F || input.type() == CV_64F);

		/*
			1. Apply wavelet transform to the input image
		*/
		cv::Mat dwtOutput;
		CUDAHaarWavelet::dwt(input, dwtOutput, level); // dwt

		// Initialize variables
		int rows = dwtOutput.rows;
		int cols = dwtOutput.cols;
		int halfWindow = windowSize / 2;

		/*
			2. Calc Universal Threshold
		*/
		// Based on many paper, the thresholding is applied to the high-frequency sub-bands on first level
		cv::Mat highFreqBand = dwtOutput(
			cv::Rect(
				cols >> 1,
				rows >> 1,
				cols >> 1,
				rows >> 1
			)
		);

		/*  Universal Threshold Calculation */
		double threshold = calculateUniversalThreshold(highFreqBand); // Calculate VisuShrink threshold

		std::cout << "Threshold: " << threshold << std::endl;

		/*
			3. Apply ModiNeighShrink thresholding
		*/
		// Apply ModiNeighShrink thresholding
		// Loop through each level of the wavelet decomposition
		for (int i = 1; i <= level; ++i) {

			std::cout << "Performing ModiNeighShrink level: " << i << std::endl;
			//LH
			cv::Rect lhRoi(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			);
			cv::Mat lhOutput = dwtOutput(lhRoi).clone();

			//HL
			cv::Rect hlRoi(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			);
			cv::Mat hlOutput = dwtOutput(hlRoi).clone();

			//HH
			cv::Rect hhRoi(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			);
			cv::Mat hhOutput = dwtOutput(hhRoi).clone();

			applyModiNeighShrink(
				lhOutput,
				hlOutput,
				hhOutput,
				threshold, windowSize
			);

			lhOutput.copyTo(dwtOutput(lhRoi));
			hlOutput.copyTo(dwtOutput(hlRoi));
			hhOutput.copyTo(dwtOutput(hhRoi));
		}

		/*
			4. Apply inverse wavelet transform to the output image
		*/

		CUDAHaarWavelet::idwt(dwtOutput, output, level); // idwt
	}


	/************************************
	* BayesShrink thresholding function.
	*************************************/

	__global__ void sumOfSquaresKernel(const float* __restrict__ input, float* __restrict__ output, int size) {
		__shared__ float shared_mem[BLOCK_SIZE * BLOCK_SIZE];

		int tid = threadIdx.x;
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		// Initialize shared memory
		shared_mem[tid] = (idx < size) ? input[idx] * input[idx] : 0.0f;

		__syncthreads();

		// Perform reduction in shared memory
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride) {
				shared_mem[tid] += shared_mem[tid + stride]; // Accumulate the sum
			}
			__syncthreads();
		}

		// Write the result of this block to the output
		if (tid == 0) {
			output[blockIdx.x] = shared_mem[0];
		}
	}

	__global__ void applyBayesShrinkKernel(
		float* __restrict__ coeffs, float threshold,
		int rows, int cols, int mode
	) {
		int r = blockIdx.y * blockDim.y + threadIdx.y;
		int c = blockIdx.x * blockDim.x + threadIdx.x;

		if (r >= rows || c >= cols) {
			return;
		}

		int idx = r * cols + c;

		coeffs[idx] = thresholdFunction(coeffs[idx], threshold, mode);
	}

	void applyBayesShrink(
		cv::Mat& HL,
		double threshold_HL,
		cv::Mat& LH,
		double threshold_LH,
		cv::Mat& HH,
		double threshold_HH,
		CUDAThresholdMode mode
	) {

		int memorySize = sizeof(float) * HL.rows * HL.cols;

		float* float_HL = (float*)HL.data;
		float* float_LH = (float*)LH.data;
		float* float_HH = (float*)HH.data;

		float* gpu_HL;
		CudaWrapper::malloc((void**)&gpu_HL, memorySize);
		float* gpu_LH;
		CudaWrapper::malloc((void**)&gpu_LH, memorySize);
		float* gpu_HH;
		CudaWrapper::malloc((void**)&gpu_HH, memorySize);

		cudaStream_t stream1, stream2, stream3;
		CudaWrapper::streamCreate(&stream1);
		CudaWrapper::streamCreate(&stream2);
		CudaWrapper::streamCreate(&stream3);

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((HL.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (HL.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

		CudaWrapper::memcpy(gpu_HL, float_HL, memorySize, cudaMemcpyHostToDevice, stream1);
		applyBayesShrinkKernel << <grid, block, 0, stream1 >> > (
			gpu_HL, (float)threshold_HL,
			HL.rows, HL.cols, (int)mode
			);

		CudaWrapper::memcpy(gpu_LH, float_LH, memorySize, cudaMemcpyHostToDevice, stream2);
		applyBayesShrinkKernel << <grid, block, 0, stream2 >> > (
			gpu_LH, (float)threshold_LH,
			LH.rows, LH.cols, (int)mode
			);

		CudaWrapper::memcpy(gpu_HH, float_HH, memorySize, cudaMemcpyHostToDevice, stream3);
		applyBayesShrinkKernel << <grid, block, 0, stream3 >> > (
			gpu_HH, (float)threshold_HH,
			HH.rows, HH.cols, (int)mode
			);

		CudaWrapper::memcpy(float_HL, gpu_HL, memorySize, cudaMemcpyDeviceToHost, stream1);
		CudaWrapper::memcpy(float_LH, gpu_LH, memorySize, cudaMemcpyDeviceToHost, stream2);
		CudaWrapper::memcpy(float_HH, gpu_HH, memorySize, cudaMemcpyDeviceToHost, stream3);

		CudaWrapper::streamSynchronize(stream1);
		CudaWrapper::streamSynchronize(stream2);
		CudaWrapper::streamSynchronize(stream3);

		CudaWrapper::free(gpu_HL);
		CudaWrapper::free(gpu_LH);
		CudaWrapper::free(gpu_HH);

		CudaWrapper::streamDestroy(stream1);
		CudaWrapper::streamDestroy(stream2);
		CudaWrapper::streamDestroy(stream3);
	}

	double calculateBayesThreshold(
		const cv::Mat& coeffs,
		double sigmaNoise
	) {
		assert(coeffs.type() == CV_32F);
		assert(coeffs.isContinuous());

		int img_size = coeffs.rows * coeffs.cols;
		int memory_size = sizeof(float) * img_size;

		dim3 block_size(BLOCK_SIZE * BLOCK_SIZE);
		int sharedMemSize = block_size.x * sizeof(float);
		dim3 grid_size((img_size + block_size.x - 1) / block_size.x);

		float* gpu_input;
		CudaWrapper::malloc((void**)&gpu_input, memory_size);
		float* gpu_totalVar;
		CudaWrapper::malloc((void**)&gpu_totalVar, grid_size.x * sizeof(float));

		float* input_data = (float*)coeffs.data;
		float* partialSum;
		CudaWrapper::hostAllocate((void**)&partialSum, grid_size.x * sizeof(float), cudaHostAllocDefault);

		CudaWrapper::memcpy(gpu_input, input_data, memory_size, cudaMemcpyHostToDevice);

		sumOfSquaresKernel << <grid_size, block_size, sharedMemSize >> > (
			gpu_input,
			gpu_totalVar,
			img_size
			);

		CudaWrapper::memcpy(partialSum, gpu_totalVar, grid_size.x * sizeof(float), cudaMemcpyDeviceToHost);

		float totalVar = 0;
		for (int i = 0; i < grid_size.x; i++) {
			totalVar += partialSum[i];
		}
		totalVar /= img_size;
		//double partialSum = cv::mean(coeffs.mul(coeffs))[0]; // Total variance

		double sigmaSignal = std::sqrt(std::max(totalVar - sigmaNoise * sigmaNoise, 0.0)); // Signal standard deviation
		double threshold = sigmaNoise * sigmaNoise / sigmaSignal; // BayesShrink threshold

		CudaWrapper::hostFree(partialSum);

		CudaWrapper::free(gpu_input);
		CudaWrapper::free(gpu_totalVar);

		return threshold;
	}

	/**
	* BayesShrink thresholding function.
	*/
	void bayesShrink(
		const cv::Mat& input,
		cv::Mat& output,
		int level,
		CUDAThresholdMode mode
	) {
		if (input.empty() || level < 1) {
			throw std::invalid_argument("Invalid input parameters for BayesShrink.");
		}

		assert(input.type() == CV_32F || input.type() == CV_64F);

		CUDAHaarWavelet::dwt(input, output, level); // dwt

		// Initialize variables
		int rows = input.rows;
		int cols = input.cols;

		/*
			2. Estimate noise standard deviation
		*/
		// Based on the paper the noise is estimated from the HH1 band
		std::cout << "Extracting high frequency band" << std::endl;
		cv::Mat highFreqBand = output(
			cv::Rect(
				cols >> 1,
				rows >> 1,
				cols >> 1,
				rows >> 1
			)
		);

		/* Calc noise from HH1  */
		std::cout << "Calculating noise standard deviation" << std::endl;
		double sigmaNoise = calculateSigma(highFreqBand);

		/*
			3. Apply BayesShrink thresholding
		*/
		for (int i = 1; i <= level; ++i) {

			std::cout << "Performing BayesShrink level: " << i << std::endl;

			cv::Rect lh_roi = cv::Rect(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			);
			cv::Mat lh = output(lh_roi).clone();
			double lhThreshold = calculateBayesThreshold(lh, sigmaNoise);

			cv::Rect hl_roi = cv::Rect(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			);
			cv::Mat hl = output(hl_roi).clone();
			double hlThreshold = calculateBayesThreshold(hl, sigmaNoise);

			cv::Rect hh_roi = cv::Rect(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			);
			cv::Mat hh = output(hh_roi).clone();
			double hhThreshold = calculateBayesThreshold(hh, sigmaNoise);

			applyBayesShrink(hl, hlThreshold, lh, lhThreshold, hh, hhThreshold, mode);

			hl.copyTo(output(hl_roi));
			lh.copyTo(output(lh_roi));
			hh.copyTo(output(hh_roi));
		}

		/*
			4. Apply inverse wavelet transform to the output image
		*/
		CUDAHaarWavelet::idwt(output, output, level); // idwt
	}
}


