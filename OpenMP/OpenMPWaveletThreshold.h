#pragma once

#include <opencv2/core.hpp>




namespace OpenMPWaveletThreshold {

	enum class OpenMPThresholdMode {
		HARD,
		SOFT,
		GARROTE,
	};


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
		OpenMPThresholdMode mode = OpenMPThresholdMode::SOFT
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
		OpenMPThresholdMode mode = OpenMPThresholdMode::SOFT
	);

	namespace {
		void applyNeighShrink(
			const cv::Mat& coeffs,
			cv::Mat& output,
			double threshold,
			int halfWindow
		);

		void applyModiNeighShrink(
			const cv::Mat& coeffs,
			cv::Mat& output,
			double threshold,
			int halfWindow
		);

		void applyBayesShrink(
			cv::Mat& coeffs,
			double sigmaNoise,
			OpenMPThresholdMode mode = OpenMPThresholdMode::SOFT
		);

		/**
		 * @brief Calculate the median absolute deviation (MAD) of the input data.
		 *
		 * @param coeffs coefficients, normally the high-frequency band
		 * @return double MAD value
		 *
		*/
		double calculateSigma(cv::Mat& coeffs);

		/**
		 * @brief Sign function
		 *
		 * @param x Input value
		 *
		 * @return float
		*/
		float sign(float x);

		/**
		 * @brief Soft shrinkage
		 *
		 * Coefficients with a magnitude less than the threshold are set to zero,
		 * while coefficients greater than the threshold are shrunk towards zero.
		 *
		 * @param d Input value
		 * @param threshold Threshold
		 * @return float
		*/
		float soft_shrink(float d, float threshold);

		/**
		 * @brief Hard shrinkage
		 *
		 * Coefficients with a magnitude less than the threshold are set to zero.
		 *
		 * @param x Input value
		 * @param threshold Threshold
		 * @return float
		*/
		float hard_shrink(float x, float threshold);

		/**
		 * @brief Garrot shrinkage
		 *
		 * Coefficients with a magnitude less than the threshold are set to zero,
		 * while coefficients greater than the threshold are shrunk towards zero.
		 * The amount of shrinkage is proportional to the magnitude of the coefficient.
		 *
		 * @param x Input value
		 * @param threshold Threshold
		 * @return float
		*/
		float garrot_shrink(float x, float threshold);


		/**
		 * @brief Calculate the threshold value for a given high-frequency band using the VisuShrink method.
		 *
		 * Calculate the threshold value for a given high-frequency band using the Median Absolute Deviation (MAD) method.
		 *
		 * @param highFreqBand High-frequency band (highest level HH sub-band)
		 * @return double Threshold value
		 *
		*/
		double calculateUniversalThreshold(cv::Mat& highFreqBand);
	}
}
