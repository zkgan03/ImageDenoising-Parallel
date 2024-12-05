#pragma once

#include "CUDAWaveletThreshold.h"

class CUDA
{
public:
	CUDAWaveletThreshold waveletThreshold;
	void TestFunctionCUDA(cv::Mat& input);
};