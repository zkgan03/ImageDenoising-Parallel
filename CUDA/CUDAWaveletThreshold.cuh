#pragma once

#include <opencv2/core.hpp>

enum class CUDAThresholdMode {
	HARD,
	SOFT,
	GARROTE,
};

namespace CUDAWaveletThreshold {

	/**
	* @brief Perform VisuShrink thresholding on the input image.
	*
	* This function performs VisuShrink thresholding on the input image.
	*
	* @param input Input image
	* @param output Output image
	* @param level Number of decomposition levels in the DWT
	* @param mode Thresholding mode (HARD, SOFT, GARROTE)
	*/
	void visuShrink(
		const cv::Mat& input,
		cv::Mat& output,
		int level,
		CUDAThresholdMode mode = CUDAThresholdMode::SOFT
	);

	/**
	* @brief Perform NeighShrink thresholding on the input image.
	*
	* This function performs NeighShrink thresholding on the input image.
	* Based on Paper: "IMAGE DENOISING USING NEIGHBOURING WAVELET COEFFICIENTS" by G. Y. Chen, T. D. Bui and A. Krzyzak
	*
	* @param input Input image
	* @param output Output image
	* @param level Number of decomposition levels in the DWT
	* @param windowSize Size of the window area
	* @param mode Thresholding mode (HARD, SOFT, GARROTE)
	*/
	void neighShrink(
		const cv::Mat& input,
		cv::Mat& output,
		int level,
		int windowSize);

	/**
	* @brief Perform ModineighShrink thresholding on the input image.
	*
	* This function performs ModineighShrink thresholding on the input image.
	* It modified the shrink factor that implemented in the NeighShrink function.
	*
	* Based on Paper: "Image Denoising using Discrete Wavelet transform" by S.Kother Mohideen†  Dr. S. Arumuga Perumal††, Dr. M.Mohamed Sathik
	*/
	void modineighShrink(
		const cv::Mat& input,
		cv::Mat& output,
		int level,
		int windowSize);

	/**
	* @brief Perform BayesShrink thresholding on the input image.
	*
	* This function performs BayesShrink thresholding on the input image.
	*
	* Based on Paper : "Adaptive wavelet thresholding for image denoising and compression" by S.G. Chang; Bin Yu; M. Vetterli
	*/
	void bayesShrink(
		const cv::Mat& input,
		cv::Mat& output,
		int level,
		CUDAThresholdMode mode = CUDAThresholdMode::SOFT
	);

}
