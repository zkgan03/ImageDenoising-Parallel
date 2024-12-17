
#include "CudaWrapper.cuh"

namespace CudaWrapper {

    // Memory management
    void malloc(void** devPtr, size_t size) {
        CUDA_CALL(cudaMalloc(devPtr, size));
    }

    void free(void* devPtr) {
        CUDA_CALL(cudaFree(devPtr));
    }

    void hostAllocate(void** hostPtr, size_t size, unsigned int flags) {
        CUDA_CALL(cudaHostAlloc(hostPtr, size, flags));
    }

    void hostFree(void* hostPtr) {
        CUDA_CALL(cudaFreeHost(hostPtr));
    }

    // Stream management
    void streamCreate(cudaStream_t* stream) {
        CUDA_CALL(cudaStreamCreate(stream));
    }

    void streamDestroy(cudaStream_t stream) {
        CUDA_CALL(cudaStreamDestroy(stream));
    }

    void streamSynchronize(cudaStream_t stream) {
        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    // Memory transfer
    void memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
        if (stream) {
            CUDA_CALL(cudaMemcpyAsync(dst, src, size, kind, stream));
        }
        else {
            CUDA_CALL(cudaMemcpy(dst, src, size, kind));
        }
    }

} // namespace CudaWrapper
