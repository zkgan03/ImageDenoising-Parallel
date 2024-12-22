#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>


#include "OpenMPWaveletThreshold.h"
#include "OpenMPHaarWavelet.h"

//
// Wavelet-Based Thresholding (VisuShrink, NeighShrink, ModiNeighShrink)
//

namespace OpenMPWaveletThreshold {

	float soft_shrink(float d, float threshold)
	{
		return std::abs(d) > threshold ? std::copysign(std::abs(d) - threshold, d) : 0.0;
	}

	float hard_shrink(float x, float threshold)
	{
		return std::abs(x) > threshold ? x : 0.0;
	}

	float garrot_shrink(float x, float threshold)
	{
		return std::abs(x) > threshold ? x - ((threshold * threshold) / x) : 0.0;
	}



	double calculateSigma(cv::Mat& coeffs) {
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

	float sign(float x)
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

	/**
	* Other helper functions
	*/
	double calculateUniversalThreshold(
		cv::Mat& highFreqBand
	) {

		double sigma = calculateSigma(highFreqBand); // Estimate noise standard deviation

		double threshold = sigma * sqrt(2 * std::log(highFreqBand.rows * highFreqBand.cols));

		return threshold;
	}


	void applyNeighShrink(const cv::Mat& coeffs, cv::Mat& output, double threshold, int halfWindow) {
		const int rows = coeffs.rows;
		const int cols = coeffs.cols;
		const double thresholdSq = threshold * threshold;

		// Create a temporary buffer to avoid race conditions
		cv::Mat tempOutput = output.clone();

		// Use parallel processing with optimized chunk size
		const int chunkSize = std::max(1, rows / (omp_get_max_threads() * 4));

#pragma omp parallel
		{
			// Thread-local storage for window calculations
			std::vector<double> windowValues((2 * halfWindow + 1) * (2 * halfWindow + 1));

#pragma omp for schedule(dynamic, chunkSize) nowait
			for (int r = 0; r < rows; ++r) {
				for (int c = 0; c < cols; ++c) {
					double squareSum = 0.0;
					int validCount = 0;

					// Calculate window sum using cache-friendly access pattern
					for (int wr = std::max(0, r - halfWindow);
						wr <= std::min(rows - 1, r + halfWindow); ++wr) {
						for (int wc = std::max(0, c - halfWindow);
							wc <= std::min(cols - 1, c + halfWindow); ++wc) {
							double value = coeffs.at<double>(wr, wc);
							squareSum += value * value;
							validCount++;
						}
					}

					// Apply threshold to temporary buffer
					tempOutput.at<double>(r, c) *= std::max(1.0 - (thresholdSq / squareSum), 0.0);
				}
			}
		}

		// Copy back results
		tempOutput.copyTo(output);
	}


	void applyModiNeighShrink(const cv::Mat& coeffs, cv::Mat& output, double threshold, int halfWindow) {
		const int rows = coeffs.rows;
		const int cols = coeffs.cols;
		const double thresholdSq = threshold * threshold;
		const double factor = 3.0 / 4.0;

		// Create padded matrices to eliminate boundary checks
		cv::Mat padded, tempOutput;
		cv::copyMakeBorder(coeffs, padded, halfWindow, halfWindow,
			halfWindow, halfWindow, cv::BORDER_CONSTANT, 0);
		tempOutput = output.clone();

		const int chunkSize = std::max(1, rows / (omp_get_max_threads() * 4));

#pragma omp parallel
		{
			// Thread-local storage
			std::vector<double> windowCache((2 * halfWindow + 1) * (2 * halfWindow + 1));

#pragma omp for schedule(dynamic, chunkSize) nowait
			for (int r = 0; r < rows; ++r) {
				for (int c = 0; c < cols; ++c) {
					double squareSum = 0.0;

					// Use pre-calculated indices for faster access
					const int startR = r;
					const int startC = c;
					const int endR = r + 2 * halfWindow + 1;
					const int endC = c + 2 * halfWindow + 1;

					// Efficient window processing using padded matrix
					for (int wr = startR; wr < endR; ++wr) {
						const double* windowRow = padded.ptr<double>(wr);
						for (int wc = startC; wc < endC; ++wc) {
							double value = windowRow[wc];
							squareSum += value * value;
						}
					}

					// Calculate and apply shrinkage to temporary buffer
					tempOutput.at<double>(r, c) *=
						std::max(1.0 - (factor * thresholdSq / squareSum), 0.0);
				}
			}
		}

		// Copy results back
		tempOutput.copyTo(output);
	}

	void applyBayesShrink(cv::Mat& coeffs, double sigmaNoise, OpenMPThresholdMode mode) {
		// Select threshold function (outside parallel region)
		float (*thresholdFunction)(float, float) = nullptr;
		switch (mode) {
		case OpenMPThresholdMode::HARD: thresholdFunction = hard_shrink; break;
		case OpenMPThresholdMode::SOFT: thresholdFunction = soft_shrink; break;
		case OpenMPThresholdMode::GARROTE: thresholdFunction = garrot_shrink; break;
		}

		// Calculate statistics outside parallel region
		const double sigmaNoiseSq = sigmaNoise * sigmaNoise;
		cv::Mat coeffsSquared;
		cv::multiply(coeffs, coeffs, coeffsSquared);
		double totalVar = cv::mean(coeffsSquared)[0];
		double sigmaSignal = std::sqrt(std::max(totalVar - sigmaNoiseSq, 0.0));
		const double threshold = sigmaNoiseSq / sigmaSignal;

		const int rows = coeffs.rows;
		const int cols = coeffs.cols;
		const int chunkSize = std::max(1, rows / (omp_get_max_threads() * 4));

		// Create temporary buffer
		cv::Mat tempCoeffs = coeffs.clone();

#pragma omp parallel for schedule(dynamic, chunkSize) collapse(2)
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				double value = coeffs.at<double>(r, c);
				tempCoeffs.at<double>(r, c) = thresholdFunction(value, threshold);
			}
		}

		// Copy results back
		tempCoeffs.copyTo(coeffs);
	}

	void visuShrink(const cv::Mat& input, cv::Mat& output, int level, OpenMPThresholdMode mode) {
		if (input.empty() || level < 1) {
			throw std::invalid_argument("Invalid input parameters for VisuShrink.");
		}

		// Perform DWT first
		cv::Mat dwtOutput;
		OpenMPHaarWavelet::dwt(input, dwtOutput, level);

		const int rows = dwtOutput.rows;
		const int cols = dwtOutput.cols;

		// Calculate threshold once outside parallel region
		cv::Mat highFreqBand = dwtOutput(cv::Rect(cols >> 1, rows >> 1, cols >> 1, rows >> 1));
		const double threshold = calculateUniversalThreshold(highFreqBand);

		// Select threshold function outside parallel region
		float (*thresholdFunction)(float, float) = nullptr;
		switch (mode) {
		case OpenMPThresholdMode::HARD: thresholdFunction = hard_shrink; break;
		case OpenMPThresholdMode::SOFT: thresholdFunction = soft_shrink; break;
		case OpenMPThresholdMode::GARROTE: thresholdFunction = garrot_shrink; break;
		}

		// Create temporary buffer to avoid race conditions
		cv::Mat tempOutput = dwtOutput.clone();

		// Process all levels
#pragma omp parallel
		{
			// Calculate optimal chunk size based on number of threads
			const int num_threads = omp_get_num_threads();
			const int chunk_size = std::max(1, (rows / (num_threads * 4)));

			for (int i = 1; i <= level; i++) {
				const int subRows = rows >> i;
				const int subCols = cols >> i;

				// Process LH band
#pragma omp for schedule(dynamic, chunk_size) nowait
				for (int r = 0; r < subRows; r++) {
					for (int c = 0; c < subCols; c++) {
						double value = dwtOutput.at<double>(r + subRows, c);
						tempOutput.at<double>(r + subRows, c) =
							thresholdFunction(value, threshold);
					}
				}

				// Process HL band
#pragma omp for schedule(dynamic, chunk_size) nowait
				for (int r = 0; r < subRows; r++) {
					for (int c = 0; c < subCols; c++) {
						double value = dwtOutput.at<double>(r, c + subCols);
						tempOutput.at<double>(r, c + subCols) =
							thresholdFunction(value, threshold);
					}
				}

				// Process HH band
#pragma omp for schedule(dynamic, chunk_size) nowait
				for (int r = 0; r < subRows; r++) {
					for (int c = 0; c < subCols; c++) {
						double value = dwtOutput.at<double>(r + subRows, c + subCols);
						tempOutput.at<double>(r + subRows, c + subCols) =
							thresholdFunction(value, threshold);
					}
				}

				// Synchronize threads before processing next level
#pragma omp barrier
			}
		}

		// Perform IDWT on the thresholded coefficients
		OpenMPHaarWavelet::idwt(tempOutput, output, level);
	}


	/**
	* NeighShrink thresholding function.
	*/
	void neighShrink(
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
		OpenMPHaarWavelet::dwt(input, dwtOutput, level); // dwt

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
		OpenMPHaarWavelet::idwt(output, output, level); // idwt
	}




	/**
	* ModiNeighShrink thresholding function.
	*/
	void modineighShrink(
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
		OpenMPHaarWavelet::dwt(input, dwtOutput, level); // dwt

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
			/*
				3. Apply ModiNeighShrink thresholding
			*/
			// Apply ModiNeighShrink thresholding
			// Loop through each level of the wavelet decomposition
			output = dwtOutput.clone();
			for (int i = 1; i <= level; ++i) {

				std::cout << "Performing ModiNeighShrink level: " << i << std::endl;

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
			OpenMPHaarWavelet::idwt(output, output, level); // idwt
		}


	}

	/**
	* BayesShrink thresholding function.
	*/
	void bayesShrink(
		const cv::Mat& input,
		cv::Mat& output,
		int level,
		OpenMPThresholdMode mode
	) {
		if (input.empty() || level < 1) {
			throw std::invalid_argument("Invalid input parameters for BayesShrink.");
		}

		/*
			1. Apply wavelet transform to the input image
		*/
		OpenMPHaarWavelet::dwt(input, output, level); // dwt

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
		OpenMPHaarWavelet::idwt(output, output, level); // idwt
	}



}


