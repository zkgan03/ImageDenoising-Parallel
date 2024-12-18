#pragma once

#ifdef DLL_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>

/**
* This is a DLL that combines the functionality of the Sequential, OpenMP, and CUDA implementations of Image Denoising using Wavelet Transform.
*/

extern "C" {

    // CV Data types
    DLL_API int CV_TYPE_8U();
	DLL_API int CV_TYPE_8S();
	DLL_API int CV_TYPE_16U();
	DLL_API int CV_TYPE_16S();
	DLL_API int CV_TYPE_32S();
	DLL_API int CV_TYPE_32F();
	DLL_API int CV_TYPE_64F();

    // CUDA functions
    DLL_API void cuda_dwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration);
    DLL_API void cuda_idwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration);
    DLL_API void cuda_bayesShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level);
    DLL_API void cuda_visushrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level);
    DLL_API void cuda_neighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize);
    DLL_API void cuda_modiNeighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize);

    // OpenMP functions
    DLL_API void openmp_dwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration);
    DLL_API void openmp_idwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration);
    DLL_API void openmp_bayesShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level);
    DLL_API void openmp_visushrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level);
    DLL_API void openmp_neighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize);
    DLL_API void openmp_modiNeighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize);

    // Sequential functions
    DLL_API void sequential_dwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration);
    DLL_API void sequential_idwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration);
    DLL_API void sequential_bayesShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level);
    DLL_API void sequential_visushrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level);
    DLL_API void sequential_neighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize);
    DLL_API void sequential_modiNeighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize);
}
