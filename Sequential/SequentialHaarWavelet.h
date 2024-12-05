#pragma once

#include <opencv2/core.hpp>

class SequentialHaarWavelet
{
public:
	SequentialHaarWavelet() {}

	/**
	 * @brief Perform Haar wavelet decomposition
	 *
	 * This function performs Haar wavelet decomposition on the input image.
	 *
	 * @param input Input image
	 * @param output Output image
	 * @param nIteration Number of decomposition iterations
	 */
	static void dwt(const cv::Mat& input, cv::Mat& output, int nIteration);

	/**
	 * @brief Perform Haar wavelet reconstruction
	 *
	 * This function performs Haar wavelet reconstruction on the input image.
	 *
	 * @param input Input image
	 * @param output Output image
	 * @param nIteration Number of decomposition iterations
	 */
	static void idwt(const cv::Mat& input, cv::Mat& output, int nIteration);
};

