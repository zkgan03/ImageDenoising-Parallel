#include <iostream>
#include <opencv2/opencv.hpp>

#include "OpenMP.h"


void OpenMP::TestFunctionOpenMP(cv::Mat& input)
{
	//Display image
	cv::imshow("OpenMP window", input);

	std::cout << "Hello from OpenMP!" << std::endl;
}
