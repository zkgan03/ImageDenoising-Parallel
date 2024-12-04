#include <iostream>
#include <opencv2/core.hpp>
#include <omp.h>

#include "WaveletThreshold.h"
#include "HaarWavelet.h"

//
// Wavelet-Based Thresholding (VisuShrink, NeighShrink, ModiNeighShrink)
//

/**
* VisuShrink thresholding function.
*/
void WaveletThreshold::visuShrink(
	const cv::Mat& input,
	cv::Mat& output,
	int level,
	ThresholdMode mode = ThresholdMode::SOFT
) {
	if (input.empty() || level < 1) {
		throw std::invalid_argument("Invalid input parameters for VisuShrink.");
	}

	output = input.clone();

	int rows = input.rows;
	int cols = input.cols;

	// Based on many paper, the thresholding is applied to the high-frequency sub-bands on first level
	cv::Mat highFreqBand = input(
		cv::Rect(
			cols >> 1,
			rows >> 1,
			cols >> 1,
			rows >> 1
		)
	);
	double threshold = calculateUniversalThreshold(highFreqBand);

	// select the thresholding function
	float (*thresholdFunction)(float x, float threshold) = soft_shrink;
	switch (mode) {
	case ThresholdMode::HARD:
		thresholdFunction = hard_shrink;
		break;
	case ThresholdMode::SOFT:
		thresholdFunction = soft_shrink;
		break;
	case ThresholdMode::GARROTE:
		thresholdFunction = garrot_shrink;
		break;
	}

	for (int i = 1; i <= level; i++) {

		std::cout << "Performing VisuShrink level: " << i << std::endl;

		// Apply threshold to high-frequency sub-bands
#pragma omp parallel for collapse(2)
		for (int r = 0; r < rows >> i; r++) {
			for (int c = 0; c < cols >> i; c++) {
				// LH band
				double& lh = output.at<double>(r + (rows >> i), c);
				lh = thresholdFunction(input.at<double>(r + (rows >> i), c), threshold);

				// HL band
				double& hl = output.at<double>(r, c + (cols >> i));
				hl = thresholdFunction(input.at<double>(r, c + (cols >> i)), threshold);

				// HH band
				double& hh = output.at<double>(r + (rows >> i), c + (cols >> i));
				hh = thresholdFunction(input.at<double>(r + (rows >> i), c + (cols >> i)), threshold);
			}
		}
	}

}


/**
* NeighShrink thresholding function.
*/
void WaveletThreshold::neighShrink(
	const cv::Mat& input,
	cv::Mat& output,
	int level,
	int windowSize
) {
	// Verify the input validity
	if (input.empty() || level < 1 || windowSize < 1) {
		throw std::invalid_argument("Invalid input parameters for NeighShrink.");
	}

	// Initialize variables
	int rows = input.rows;
	int cols = input.cols;
	int halfWindow = windowSize / 2;

	output = input.clone();

	// Based on many paper, the thresholding is applied to the high-frequency sub-bands on first level
	cv::Mat highFreqBand = input(
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

	// Apply NeighShrink thresholding
	// Loop through each level of the wavelet decomposition
	for (int i = 1; i <= level; ++i) {

		std::cout << "Performing NeighShrink level: " << i << std::endl;

		cv::Mat lh = output(
			cv::Rect(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat hl = output(
			cv::Rect(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat hh = output(
			cv::Rect(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		applyNeighShrink(lh, threshold, halfWindow);
		applyNeighShrink(hl, threshold, halfWindow);
		applyNeighShrink(hh, threshold, halfWindow);
	}
}

void WaveletThreshold::applyNeighShrink(
	cv::Mat& coeffs,
	double threshold,
	int halfWindow
) {

#pragma omp parallel for collapse(2)
	for (int r = 0; r < coeffs.rows; ++r) {
		for (int c = 0; c < coeffs.cols; ++c) {

			double squareSum = 0.0;

			// Loop through the window for each pixel
			for (int wr = -halfWindow; wr <= halfWindow; wr++) {
				for (int wc = -halfWindow; wc <= halfWindow; wc++) {

					// Check if the window is within the image boundaries
					if (r + wr >= 0 &&
						r + wr < coeffs.rows &&
						c + wc >= 0 &&
						c + wc < coeffs.cols) {

						double value = coeffs.at<double>(r + wr, c + wc);
						squareSum += value * value;
					}
				}
			}

			double& value = coeffs.at<double>(r, c);
			value *= std::max(1.0 - ((threshold * threshold) / squareSum), 0.0);
		}
	}
}


/**
* ModiNeighShrink thresholding function.
*/
void WaveletThreshold::modineighShrink(
	const cv::Mat& input,
	cv::Mat& output,
	int level,
	int windowSize
) {
	// Verify the input validity
	if (input.empty() || level < 1 || windowSize < 1) {
		throw std::invalid_argument("Invalid input parameters for ModiNeighShrink.");
	}

	// Initialize variables
	int rows = input.rows;
	int cols = input.cols;
	int halfWindow = windowSize / 2;

	output = input.clone();

	// Based on many paper, the thresholding is applied to the high-frequency sub-bands on first level
	cv::Mat highFreqBand = input(
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

	// Apply ModiNeighShrink thresholding
	// Loop through each level of the wavelet decomposition
	for (int i = 1; i <= level; ++i) {

		std::cout << "Performing ModiNeighShrink level: " << i << std::endl;

		cv::Mat lh = output(
			cv::Rect(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat hl = output(
			cv::Rect(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat hh = output(
			cv::Rect(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		applyModiNeighShrink(lh, threshold, halfWindow);
		applyModiNeighShrink(hl, threshold, halfWindow);
		applyModiNeighShrink(hh, threshold, halfWindow);
	}
}


void WaveletThreshold::applyModiNeighShrink(
	cv::Mat& coeffs,
	double threshold,
	int halfWindow
) {

#pragma omp parallel for collapse(2)
	for (int r = 0; r < coeffs.rows; ++r) {
		for (int c = 0; c < coeffs.cols; ++c) {

			double squareSum = 0.0;

			// Loop through the window for each pixel
			for (int wr = -halfWindow; wr <= halfWindow; wr++) {
				for (int wc = -halfWindow; wc <= halfWindow; wc++) {

					// Check if the window is within the image boundaries
					if (r + wr >= 0 &&
						r + wr < coeffs.rows &&
						c + wc >= 0 &&
						c + wc < coeffs.cols) {

						double value = coeffs.at<double>(r + wr, c + wc);
						squareSum += value * value;
					}
				}
			}

			double& value = coeffs.at<double>(r, c);
			value *= std::max(1.0 - ((3.0 / 4.0) * (threshold * threshold) / squareSum), 0.0);
		}
	}
}



/**
* BayesShrink thresholding function.
*/
void WaveletThreshold::bayesShrink(
	const cv::Mat& input,
	cv::Mat& output,
	int level,
	ThresholdMode mode
) {
	if (input.empty() || level < 1) {
		throw std::invalid_argument("Invalid input parameters for BayesShrink.");
	}

	output = input.clone();

	int rows = input.rows;
	int cols = input.cols;

	// Based on the paper the noise is estimated from the HH1 band
	cv::Mat highFreqBand = input(
		cv::Rect(
			cols >> 1,
			rows >> 1,
			cols >> 1,
			rows >> 1
		)
	);

	/* Calc noise from HH1  */
	double sigmaNoise = calculateSigma(highFreqBand);


	for (int i = 1; i <= level; ++i) {

		std::cout << "Performing BayesShrink level: " << i << std::endl;

		cv::Mat lh = output(
			cv::Rect(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat hl = output(
			cv::Rect(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat hh = output(
			cv::Rect(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		applyBayesShrink(lh, sigmaNoise, mode);
		applyBayesShrink(hl, sigmaNoise, mode);
		applyBayesShrink(hh, sigmaNoise, mode);
	}
}

void WaveletThreshold::applyBayesShrink(
	cv::Mat& coeffs,
	double sigmaNoise,
	ThresholdMode mode
) {
	// Select the thresholding function
	float (*thresholdFunction)(float x, float threshold) = soft_shrink;;
	switch (mode) {
	case ThresholdMode::HARD:
		thresholdFunction = hard_shrink;
		break;
	case ThresholdMode::SOFT:
		thresholdFunction = soft_shrink;
		break;
	case ThresholdMode::GARROTE:
		thresholdFunction = garrot_shrink;
		break;
	}

	double totalVar = cv::mean(coeffs.mul(coeffs))[0]; // Total variance
	double sigmaSignal = std::sqrt(std::max(totalVar - sigmaNoise * sigmaNoise, 0.0)); // Signal standard deviation
	double threshold = sigmaNoise * sigmaNoise / sigmaSignal; // BayesShrink threshold

#pragma omp parallel for collapse(2)
	for (int r = 0; r < coeffs.rows; ++r) {
		for (int c = 0; c < coeffs.cols; ++c) {

			double value = coeffs.at<double>(r, c);

			coeffs.at<double>(r, c) = thresholdFunction(value, threshold);
		}
	}
}


/**
* Other helper functions
*/
double WaveletThreshold::calculateUniversalThreshold(
	cv::Mat& highFreqBand
) {

	double sigma = calculateSigma(highFreqBand); // Estimate noise standard deviation

	double threshold = sigma * sqrt(2 * std::log(highFreqBand.rows * highFreqBand.cols));

	return threshold;
}


double WaveletThreshold::calculateSigma(
	cv::Mat& coeffs
) {
	// Flatten the high-frequency coefficients
	std::vector<double> coefficients;
	for (int i = 0; i < coeffs.rows; ++i) {
		for (int j = 0; j < coeffs.cols; ++j) {
			coefficients.push_back(std::abs(coeffs.at<double>(i, j)));
		}
	}

	// Calculate Median Absolute Deviation (MAD)
	std::nth_element(coefficients.begin(), coefficients.begin() + coefficients.size() / 2, coefficients.end());

	double median = coefficients[coefficients.size() / 2];

	double sigma = median / 0.6745; // Estimate noise standard deviation

	return sigma;
}

float WaveletThreshold::sign(float x)
{
	float res = 0;
	if (x == 0)
	{
		res = 0;
	}
	else if (x > 0)
	{
		res = 1;
	}
	else if (x < 0)
	{
		res = -1;
	}
	return res;
}

float WaveletThreshold::soft_shrink(float d, float threshold)
{
	return std::abs(d) > threshold ? std::copysign(std::abs(d) - threshold, d) : 0.0;
}

float WaveletThreshold::hard_shrink(float x, float threshold)
{
	return std::abs(x) > threshold ? x : 0.0;
}

float WaveletThreshold::garrot_shrink(float x, float threshold)
{
	return std::abs(x) > threshold ? x - ((threshold * threshold) / x) : 0.0;
}