#pragma once

#include "SequentialWaveletThreshold.h"

class Sequential
{
public:
	SequentialWaveletThreshold waveletThreshold;
	void TestFunctionSeq(cv::Mat& input);
};
