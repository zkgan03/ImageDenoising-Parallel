#include <iostream>
#include <opencv2/opencv.hpp>

namespace _CUDA_
{
	void TestFunctionCUDA(cv::Mat& input)
	{
		//Display image
		cv::imshow("CUDA window", input);

		std::cout << "Hello from CUDA!" << std::endl;
	}
}

