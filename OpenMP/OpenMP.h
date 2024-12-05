#pragma once

#include "OpenMPWaveletThreshold.h"

class OpenMP
{
public:
	OpenMPWaveletThreshold waveletThreshold;
	void TestFunctionOpenMP(cv::Mat& input);

};
