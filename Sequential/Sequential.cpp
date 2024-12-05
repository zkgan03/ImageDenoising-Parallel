// Sequential.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Sequential.h"


void Sequential::TestFunctionSeq(cv::Mat& input)
{
	//Display image
	cv::imshow("Sequential window", input);

	std::cout << "Hello from Sequential!" << std::endl;
}
