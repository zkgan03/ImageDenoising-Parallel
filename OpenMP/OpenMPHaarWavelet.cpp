#include <iostream>
#include <opencv2/core.hpp>
#include <omp.h>
#include "OpenMPHaarWavelet.h"

namespace OpenMPHaarWavelet
{
	void dwt(const cv::Mat& input, cv::Mat& output, int nIteration) {
		std::cout << "Performing DWT with " << nIteration << " iterations" << std::endl;

		cv::Mat temp = input.clone();
		temp.convertTo(temp, CV_64F);

		output = input.clone();
		output.convertTo(output, CV_64F);

		int rows = output.rows;
		int cols = output.cols;

#pragma omp parallel
        {
            for (int i = 1; i <= nIteration; i++) {
#pragma omp single
                {
                    std::cout << "Performing DWT level: " << i << std::endl;
                    temp = output.clone();
                }

                int currentRows = rows >> i;
                int currentCols = cols >> i;

                // Use schedule(static) for better cache utilization
#pragma omp for collapse(2) schedule(static)
                for (int r = 0; r < currentRows; r++) {
                    for (int c = 0; c < currentCols; c++) {
                        // Cache frequently accessed values
                        const int operateRow = r * 2;
                        const int operateCol = c * 2;

                        // Prefetch data
                        const double topLeft = temp.at<double>(operateRow, operateCol);
                        const double topRight = temp.at<double>(operateRow, operateCol + 1);
                        const double bottomLeft = temp.at<double>(operateRow + 1, operateCol);
                        const double bottomRight = temp.at<double>(operateRow + 1, operateCol + 1);

                        // Precalculate common expressions
                        const double sum = topLeft + topRight + bottomLeft + bottomRight;
                        const double diffH = topLeft - topRight;
                        const double diffV = bottomLeft - bottomRight;

                        output.at<double>(r, c) = sum * 0.5;  // LL
                        output.at<double>(r, c + currentCols) = (diffH + diffV) * 0.5;  // HL
                        output.at<double>(r + currentRows, c) = (sum - 2 * (bottomLeft + bottomRight)) * 0.5;  // LH
                        output.at<double>(r + currentRows, c + currentCols) = (diffH - diffV) * 0.5;  // HH
                    }
                }
            }
        }
    }

    void idwt(const cv::Mat& input, cv::Mat& output, int nIteration) {
        std::cout << "Performing IDWT with " << nIteration << " iterations" << std::endl;

        // Pre-allocate matrices
        cv::Mat temp;
        input.convertTo(temp, CV_64F);
        output = temp.clone();

        int rows = output.rows;
        int cols = output.cols;

        // Single parallel region for all iterations
#pragma omp parallel
        {
            for (int i = nIteration; i >= 1; --i) {
#pragma omp single
                {
                    std::cout << "Performing IDWT level: " << i << std::endl;
                    temp = output.clone();
                }

                int currentRows = rows >> i;
                int currentCols = cols >> i;

                // Use schedule(static) for better cache utilization
#pragma omp for collapse(2) schedule(static)
                for (int r = 0; r < currentRows; r++) {
                    for (int c = 0; c < currentCols; c++) {
                        // Cache frequently accessed values
                        const double ll = temp.at<double>(r, c);
                        const double hl = temp.at<double>(r, c + currentCols);
                        const double lh = temp.at<double>(r + currentRows, c);
                        const double hh = temp.at<double>(r + currentRows, c + currentCols);

                        // Precalculate common expressions
                        const double sum = ll + hl;
                        const double diff = ll - hl;
                        const double lhDiff = lh - hh;

                        const int r2 = r * 2;
                        const int c2 = c * 2;

                        output.at<double>(r2, c2) = (sum + lh + hh) * 0.5;
                        output.at<double>(r2, c2 + 1) = (diff + lhDiff) * 0.5;
                        output.at<double>(r2 + 1, c2) = (sum - lh - hh) * 0.5;
                        output.at<double>(r2 + 1, c2 + 1) = (diff - lhDiff) * 0.5;
                    }
                }
            }
        }
    }
}
