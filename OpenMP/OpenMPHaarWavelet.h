#pragma once

#ifdef BUILDING_DLL
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

#include <opencv2/core.hpp>



namespace OpenMPHaarWavelet
{
	/**
	 * @brief Perform Haar wavelet decomposition
	 *
	 * This function performs Haar wavelet decomposition on the input image.
	 *
	 * @param input Input image
	 * @param output Output image
	 * @param nIteration Number of decomposition iterations
	 */
	void dwt(const cv::Mat& input, cv::Mat& output, int nIteration);

	/**
	 * @brief Perform Haar wavelet reconstruction
	 *
	 * This function performs Haar wavelet reconstruction on the input image.
	 *
	 * @param input Input image
	 * @param output Output image
	 * @param nIteration Number of decomposition iterations
	 */
	void idwt(const cv::Mat& input, cv::Mat& output, int nIteration);
};

