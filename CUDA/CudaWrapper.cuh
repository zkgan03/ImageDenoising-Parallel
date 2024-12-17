#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define BLOCK_SIZE 32

namespace CudaWrapper {

	// Inline function for CUDA error checking
	inline void checkCudaCall(cudaError_t result, const char* func, const char* file, int line) {
		if (result != cudaSuccess) {
			std::cerr << "\nCUDA Error: " << cudaGetErrorString(result) <<
				"\nFunction: " << func <<
				"\nFile: " << file <<
				"\nLine: " << line << std::endl;

			exit(EXIT_FAILURE);
		}
	}

	// Macro to wrap CUDA calls with error checking
#define CUDA_CALL(call) CudaWrapper::checkCudaCall((call), #call, __FILE__, __LINE__)

// Memory management
	void malloc(void** devPtr, size_t size);
	void free(void* devPtr);

	void hostAllocate(void** hostPtr, size_t size, unsigned int flags);
	void hostFree(void* hostPtr);

	// Stream management
	void streamCreate(cudaStream_t* stream);
	void streamDestroy(cudaStream_t stream);
	void streamSynchronize(cudaStream_t stream);

	// Memory transfer
	void memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream = nullptr);

} // namespace CudaWrapper
