#pragma once

#include <opencv2/core.hpp>

namespace SequentialHaarWavelet
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

	void dwt_improved(const cv::Mat& input, cv::Mat& output, int nIteration);
	void idwt_improved(const cv::Mat& input, cv::Mat& output, int nIteration);

};

