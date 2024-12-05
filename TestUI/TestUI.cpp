// TestUI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#include "OpenMP.h"
#include "CUDA.h"
#include "Sequential.h"

void add_gaussian_noise(cv::Mat& image, double mean = 0.0, double stddev = 10.0) {
	cv::Mat noise(image.size(), image.type());
	cv::randn(noise, mean, stddev);
	image += noise;
}


int main()
{
	// get image path input
	std::string imagePath;
	std::cout << "Enter image path: ";
	std::cin >> imagePath;

	// read image
	cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
	if (image.empty())
	{
		std::cout << "Could not read the image: " << imagePath << std::endl;
		return 1;
	}

	// add noise to image
	//add_gaussian_noise(image, 10,1);
	//add_gaussian_noise(image, 12, 5);
	add_gaussian_noise(image, 32, 50);

	// OpenMP
	OpenMP openMP;
	OpenMPWaveletThreshold waveletThreshold = openMP.waveletThreshold;
	cv::Mat outputOpenMP;

	waveletThreshold.bayesShrink(image, outputOpenMP, 3);
	cv::normalize(outputOpenMP, outputOpenMP, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::imshow("Wavelet Threshold OpenMP", outputOpenMP);



	// CUDA
	CUDA cuda;
	CUDAWaveletThreshold cudaWaveletThreshold = cuda.waveletThreshold;
	cv::Mat outputCUDA;

	cudaWaveletThreshold.bayesShrink(image, outputCUDA, 3);
	cv::normalize(outputCUDA, outputCUDA, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::imshow("Wavelet Threshold CUDA", outputCUDA);


	// Sequential
	Sequential sequential;
	SequentialWaveletThreshold sequentialWaveletThreshold = sequential.waveletThreshold;
	cv::Mat outputSeq;

	sequentialWaveletThreshold.bayesShrink(image, outputSeq, 3);
	cv::normalize(outputSeq, outputSeq, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::imshow("Wavelet Threshold Sequential", outputSeq);

	cv::waitKey(0);
	


	return 0;
}