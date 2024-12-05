#include <iostream>
#include <opencv2/opencv.hpp>

#include "CUDAWaveletThreshold.h"
#include "CUDAHaarWavelet.h"

//
// Wavelet-Based Thresholding (VisuShrink, NeighShrink, ModiNeighShrink)
//

/**
* VisuShrink thresholding function.
*/
void CUDAWaveletThreshold::visuShrink(
	const cv::Mat& input,
	cv::Mat& output,
	int level,
	CUDAThresholdMode mode
) {

	if (input.empty() || level < 1) {
		throw std::invalid_argument("Invalid input parameters for VisuShrink.");
	}


	/*
		1. apply wavelet transform to the input image
	*/
	std::cout << "Performing DWT" << std::endl;
	cv::Mat dwtOutput;
	CUDAHaarWavelet::dwt(input, dwtOutput, level); // dwt

	//display the dwt output
	//cv::Mat dwtOutputDisplay = dwtOutput.clone();
	//cv::normalize(dwtOutputDisplay, dwtOutputDisplay, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	//cv::imshow("DWT Output", dwtOutputDisplay);
	//cv::waitKey(0);

	// Initialize variables
	int rows = dwtOutput.rows;
	int cols = dwtOutput.cols;

	/*
		2. Calc Universal Threshold
	*/
	std::cout << "Calculating Universal Threshold" << std::endl;
	// Based on many paper, the thresholding is applied to the high-frequency sub-bands on first level
	cv::Mat highFreqBand = dwtOutput(
		cv::Rect(
			cols >> 1,
			rows >> 1,
			cols >> 1,
			rows >> 1
		)
	);
	double threshold = calculateUniversalThreshold(highFreqBand);

	/*
		3. select the thresholding function
	*/
	std::cout << "Selecting Threshold " << std::endl;
	float (*thresholdFunction)(float x, float threshold) = soft_shrink;
	switch (mode) {
	case CUDAThresholdMode::HARD:
		thresholdFunction = hard_shrink;
		break;
	case CUDAThresholdMode::SOFT:
		thresholdFunction = soft_shrink;
		break;
	case CUDAThresholdMode::GARROTE:
		thresholdFunction = garrot_shrink;
		break;
	}

	/*
		4. Apply VisuShrink thresholding
	*/

	std::cout << "Applying VisuShrink" << std::endl;

	// Initialize the output image as dwtOutput
	output = dwtOutput.clone();

	for (int i = 1; i <= level; i++) {

		std::cout << "Performing VisuShrink level: " << i << std::endl;

		// Apply threshold to high-frequency sub-bands
		for (int r = 0; r < rows >> i; r++) {
			for (int c = 0; c < cols >> i; c++) {
				// LH band
				double& lh = output.at<double>(r + (rows >> i), c);
				lh = thresholdFunction(dwtOutput.at<double>(r + (rows >> i), c), threshold);

				// HL band
				double& hl = output.at<double>(r, c + (cols >> i));
				hl = thresholdFunction(dwtOutput.at<double>(r, c + (cols >> i)), threshold);

				// HH band
				double& hh = output.at<double>(r + (rows >> i), c + (cols >> i));
				hh = thresholdFunction(dwtOutput.at<double>(r + (rows >> i), c + (cols >> i)), threshold);
			}
		}
	}
	std::cout << "VisuShrink Done" << std::endl;
	//cv::Mat outputDisplay = output.clone();
	//cv::normalize(outputDisplay, outputDisplay, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	//cv::imshow("VisuShrink Output", outputDisplay);
	//cv::waitKey(0);

	/*
		5. apply inverse wavelet transform to the output image
	*/
	std::cout << "Performing IDWT" << std::endl;
	CUDAHaarWavelet::idwt(output, output, level); // idwt

	//cv::Mat idwtOutputDisplay = output.clone();
	//cv::normalize(idwtOutputDisplay, idwtOutputDisplay, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	//cv::imshow("IDWT Output", idwtOutputDisplay);
	//cv::waitKey(0);
}


/**
* NeighShrink thresholding function.
*/
void CUDAWaveletThreshold::neighShrink(
	const cv::Mat& input,
	cv::Mat& output,
	int level,
	int windowSize
) {
	// Verify the input validity
	if (input.empty() || level < 1 || windowSize < 1) {
		throw std::invalid_argument("Invalid input parameters for NeighShrink.");
	}

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
		cv::Mat lhCoeffs = dwtOutput(
			cv::Rect(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat lhOutput = output(
			cv::Rect(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		//HL
		cv::Mat hlCoeffs = dwtOutput(
			cv::Rect(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat hlOutput = output(
			cv::Rect(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			)
		);

		//HH
		cv::Mat hhCoeffs = dwtOutput(
			cv::Rect(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat hhOutput = output(
			cv::Rect(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		applyNeighShrink(lhCoeffs, lhOutput, threshold, halfWindow);
		applyNeighShrink(hlCoeffs, hlOutput, threshold, halfWindow);
		applyNeighShrink(hhCoeffs, hhOutput, threshold, halfWindow);
	}

	/*
		4. Apply inverse wavelet transform to the output image
	*/
	CUDAHaarWavelet::idwt(output, output, level); // idwt
}

void CUDAWaveletThreshold::applyNeighShrink(
	const cv::Mat& coeffs,
	cv::Mat& output,
	double threshold,
	int halfWindow
) {

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

			double& value = output.at<double>(r, c);
			value *= std::max(1.0 - ((threshold * threshold) / squareSum), 0.0);
		}
	}
}


/**
* ModiNeighShrink thresholding function.
*/
void CUDAWaveletThreshold::modineighShrink(
	const cv::Mat& input,
	cv::Mat& output,
	int level,
	int windowSize
) {
	// Verify the input validity
	if (input.empty() || level < 1 || windowSize < 1) {
		throw std::invalid_argument("Invalid input parameters for NeighShrink.");
	}

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
		cv::Mat lhCoeffs = dwtOutput(
			cv::Rect(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat lhOutput = output(
			cv::Rect(
				0,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		//HL
		cv::Mat hlCoeffs = dwtOutput(
			cv::Rect(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat hlOutput = output(
			cv::Rect(
				cols >> i,
				0,
				cols >> i,
				rows >> i
			)
		);

		//HH
		cv::Mat hhCoeffs = dwtOutput(
			cv::Rect(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		cv::Mat hhOutput = output(
			cv::Rect(
				cols >> i,
				rows >> i,
				cols >> i,
				rows >> i
			)
		);

		applyModiNeighShrink(lhCoeffs, lhOutput, threshold, halfWindow);
		applyModiNeighShrink(hlCoeffs, hlOutput, threshold, halfWindow);
		applyModiNeighShrink(hhCoeffs, hhOutput, threshold, halfWindow);
	}

	/*
		4. Apply inverse wavelet transform to the output image
	*/
	CUDAHaarWavelet::idwt(output, output, level); // idwt
}


void CUDAWaveletThreshold::applyModiNeighShrink(
	const cv::Mat& coeffs,
	cv::Mat& output,
	double threshold,
	int halfWindow
) {

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

			double& value = output.at<double>(r, c);
			value *= std::max(1.0 - ((3.0 / 4.0) * (threshold * threshold) / squareSum), 0.0);
		}
	}
}



/**
* BayesShrink thresholding function.
*/
void CUDAWaveletThreshold::bayesShrink(
	const cv::Mat& input,
	cv::Mat& output,
	int level,
	CUDAThresholdMode mode
) {
	if (input.empty() || level < 1) {
		throw std::invalid_argument("Invalid input parameters for BayesShrink.");
	}

	/*
		1. Apply wavelet transform to the input image
	*/
	CUDAHaarWavelet::dwt(input, output, level); // dwt

	// Initialize variables
	int rows = input.rows;
	int cols = input.cols;


	/*
		2. Estimate noise standard deviation
	*/
	// Based on the paper the noise is estimated from the HH1 band
	cv::Mat highFreqBand = output(
		cv::Rect(
			cols >> 1,
			rows >> 1,
			cols >> 1,
			rows >> 1
		)
	);

	/* Calc noise from HH1  */
	double sigmaNoise = calculateSigma(highFreqBand);


	/*
		3. Apply BayesShrink thresholding
	*/
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

	/*
		4. Apply inverse wavelet transform to the output image
	*/
	CUDAHaarWavelet::idwt(output, output, level); // idwt
}

void CUDAWaveletThreshold::applyBayesShrink(
	cv::Mat& coeffs,
	double sigmaNoise,
	CUDAThresholdMode mode
) {

	// Select the thresholding function
	float (*thresholdFunction)(float x, float threshold) = soft_shrink;;
	switch (mode) {
	case CUDAThresholdMode::HARD:
		thresholdFunction = hard_shrink;
		break;
	case CUDAThresholdMode::SOFT:
		thresholdFunction = soft_shrink;
		break;
	case CUDAThresholdMode::GARROTE:
		thresholdFunction = garrot_shrink;
		break;
	}

	double totalVar = cv::mean(coeffs.mul(coeffs))[0]; // Total variance
	double sigmaSignal = std::sqrt(std::max(totalVar - sigmaNoise * sigmaNoise, 0.0)); // Signal standard deviation
	double threshold = sigmaNoise * sigmaNoise / sigmaSignal; // BayesShrink threshold

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
double CUDAWaveletThreshold::calculateUniversalThreshold(
	cv::Mat& highFreqBand
) {

	double sigma = calculateSigma(highFreqBand); // Estimate noise standard deviation

	double threshold = sigma * sqrt(2 * std::log(highFreqBand.rows * highFreqBand.cols));

	return threshold;
}


double CUDAWaveletThreshold::calculateSigma(
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

float CUDAWaveletThreshold::sign(float x)
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

float CUDAWaveletThreshold::soft_shrink(float d, float threshold)
{
	return std::abs(d) > threshold ? std::copysign(std::abs(d) - threshold, d) : 0.0;
}

float CUDAWaveletThreshold::hard_shrink(float x, float threshold)
{
	return std::abs(x) > threshold ? x : 0.0;
}

float CUDAWaveletThreshold::garrot_shrink(float x, float threshold)
{
	return std::abs(x) > threshold ? x - ((threshold * threshold) / x) : 0.0;
}