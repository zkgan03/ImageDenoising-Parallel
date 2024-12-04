// TestUI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "OpenMP.h"
#include "CUDA.h"
#include "Sequential.h"

int main()
{
	std::cout << "Hello from TestUI!" << std::endl;
    TestFunctionOpenMP();
	TestFunctionCUDA();
	TestFunctionSeq();

	return 0;
}