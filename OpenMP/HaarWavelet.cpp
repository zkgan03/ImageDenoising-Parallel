#include <iostream>
#include <opencv2/core.hpp>
#include <omp.h>

#include "HaarWavelet.h"

void HaarWavelet::dwt(const cv::Mat& input, cv::Mat& output, int nIteration) {

	std::cout << "Performing DWT with " << nIteration << " iterations" << std::endl;

	cv::Mat temp = input.clone();
	temp.convertTo(temp, CV_64F);

	output = input.clone();

	output.convertTo(output, CV_64F);

	int rows = output.rows;
	int cols = output.cols;

	std::cout << "Image size: " << rows << " x " << cols << std::endl;

	/* Number of Level to perform DWT */
	for (int i = 1; i <= nIteration; i++) {

		std::cout << "Performing DWT level: " << i << std::endl;
		std::cout << "rows num :" << (rows >> i) << " cols num :" << (cols >> i) << std::endl;

		/* row / 2^i (Since the image will divided by 2 for each level) */
#pragma omp parallel for collapse(2)
		for (int r = 0; r < rows >> i; r++) {
			for (int c = 0; c < cols >> i; c++) {
				//std::cout << "current row x col: " << r << " x " << j << std::endl;
				int operateRow = r * 2;
				int opearateCol = c * 2;

				double topLeft = temp.at<double>(operateRow, opearateCol);
				double topRight = temp.at<double>(operateRow, opearateCol + 1);
				double bottomLeft = temp.at<double>(operateRow + 1, opearateCol);
				double bottomRight = temp.at<double>(operateRow + 1, opearateCol + 1);

				output.at<double>(r, c) = (topLeft + topRight + bottomLeft + bottomRight) * 0.25; // average (LL)
				output.at<double>(r, c + (cols >> i)) = (topLeft - topRight + bottomLeft - bottomRight) * 0.25; // vertical (HL)
				output.at<double>(r + (rows >> i), c) = (topLeft + topRight - bottomLeft - bottomRight) * 0.25; // horizontal (LH)
				output.at<double>(r + (rows >> i), c + (cols >> i)) = (topLeft - topRight - bottomLeft + bottomRight) * 0.25; // diagonal (HH)
			}
		}

		output.copyTo(temp);
	}
}

void HaarWavelet::idwt(const cv::Mat& input, cv::Mat& output, int nIteration) {
	std::cout << "Performing IDWT with " << nIteration << " iterations" << std::endl;

	cv::Mat temp = input.clone();
	temp.convertTo(temp, CV_64F);
	output = input.clone();
	output.convertTo(output, CV_64F);

	int rows = output.rows;
	int cols = output.cols;

	for (int i = nIteration; i >= 1; --i) {
		std::cout << "Performing IDWT level: " << i << std::endl;

		int currentRows = rows >> i;
		int currentCols = cols >> i;

#pragma omp parallel for collapse(2)
		for (int r = 0; r < currentRows; ++r) {
			for (int c = 0; c < currentCols; ++c) {
				// Retrieve DWT coefficients
				double ll = temp.at<double>(r, c);                               // Low-Low (LL)
				double hl = temp.at<double>(r, c + currentCols);                // High-Low (HL)
				double lh = temp.at<double>(r + currentRows, c);                // Low-High (LH)
				double hh = temp.at<double>(r + currentRows, c + currentCols);  // High-High (HH)

				// Compute the inverse
				output.at<double>(r * 2, c * 2) = (ll + hl + lh + hh) * 0.5;             // Top-left
				output.at<double>(r * 2, c * 2 + 1) = (ll - hl + lh - hh) * 0.5;         // Top-right
				output.at<double>(r * 2 + 1, c * 2) = (ll + hl - lh - hh) * 0.5;         // Bottom-left
				output.at<double>(r * 2 + 1, c * 2 + 1) = (ll - hl - lh + hh) * 0.5;     // Bottom-right
			}
		}

		output.copyTo(temp);
	}

}
