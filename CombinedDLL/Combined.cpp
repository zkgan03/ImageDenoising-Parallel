#include "pch.h" 
#include <omp.h>

#include "Combined.h"

void init_mats(
	const unsigned char* input_data, int input_rows, int input_cols, int input_data_type, int n_channels,
	unsigned char* output_data,
	cv::Mat& input, cv::Mat& output,
	std::vector<cv::Mat>& input_channels, std::vector<cv::Mat>& output_channels
) {
	// Initialize input and output matrices
	input = cv::Mat(input_rows, input_cols, CV_MAKETYPE(input_data_type, n_channels), (void*)input_data);
	if (n_channels > 1) {
		std::cout << "Converting input to 32FC3" << std::endl;
		input.convertTo(input, CV_32FC3); // Convert to float for processing
	}
	else {
		std::cout << "Converting input to 32F" << std::endl;
		input.convertTo(input, CV_32F); // Convert to float for processing
	}

	std::cout << "input channels: " << input.channels() << std::endl;

	input_channels = std::vector<cv::Mat>(n_channels);
	cv::split(input, input_channels);

	output_channels = std::vector<cv::Mat>(n_channels);
}

// CV Data types
int CV_TYPE_8U() { return CV_8U; }
int CV_TYPE_8S() { return CV_8S; }
int CV_TYPE_16U() { return CV_16U; }
int CV_TYPE_16S() { return CV_16S; }
int CV_TYPE_32S() { return CV_32S; }
int CV_TYPE_32F() { return CV_32F; }
int CV_TYPE_64F() { return CV_64F; }

void openmp_set_num_threads(int num_threads) {
	omp_set_num_threads(num_threads);
}

int openmp_get_num_threads() {
	return omp_get_max_threads();
}

// CUDA functions
void cuda_dwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration) {

	// Call the actual CUDA DWT function from the library
	cv::Mat input(input_rows, input_cols, input_data_type, (void*)input_data);
	cv::Mat output(input_rows, input_cols, input_data_type, (void*)output_data);

	CUDAHaarWavelet::dwt(input, output, nIteration);

	std::memcpy(output_data, output.data, input_rows * input_cols * input.elemSize());
}

void cuda_idwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration) {
	// Call the actual CUDA IDWT function from the library

	cv::Mat input(input_rows, input_cols, input_data_type, (void*)input_data);
	cv::Mat output(input_rows, input_cols, input_data_type, (void*)output_data);

	CUDAHaarWavelet::idwt(input, output, nIteration);

	std::memcpy(output_data, output.data, input_rows * input_cols * input.elemSize());

}

void cuda_bayesShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	std::cout << "Input channels: " << input_channels.size() << std::endl;
	std::cout << "Output channels: " << output_channels.size() << std::endl;

	for (int i = 0; i < input_channels.size(); i++) {
		CUDAWaveletThreshold::bayesShrink(input_channels[i], output_channels[i], level);
	}

	std::cout << "Combining output channels" << std::endl;
	cv::merge(output_channels, output);

	std::cout << "Converting output to 8U" << std::endl;

	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, output.total() * output.elemSize());
}

void cuda_visuShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	for (int i = 0; i < n_channels; i++) {
		CUDAWaveletThreshold::visuShrink(input_channels[i], output_channels[i], level);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}

void cuda_neighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	for (int i = 0; i < n_channels; i++) {
		CUDAWaveletThreshold::neighShrink(input_channels[i], output_channels[i], level, windowSize);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}

void cuda_modiNeighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	for (int i = 0; i < n_channels; i++) {
		CUDAWaveletThreshold::modineighShrink(input_channels[i], output_channels[i], level, windowSize);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}


//
// OpenMP functions
void openmp_dwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration) {
	// Call the actual OpenMP DWT function from the library
	cv::Mat input(input_rows, input_cols, input_data_type, (void*)input_data);
	cv::Mat output(input_rows, input_cols, input_data_type, (void*)output_data);

	OpenMPHaarWavelet::dwt(input, output, nIteration);

	std::memcpy(output_data, output.data, input_rows * input_cols * sizeof(unsigned char));
}

void openmp_idwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration) {
	// Call the actual OpenMP IDWT function from the library
	cv::Mat input(input_rows, input_cols, input_data_type, (void*)input_data);
	cv::Mat output(input_rows, input_cols, input_data_type, (void*)output_data);

	OpenMPHaarWavelet::idwt(input, output, nIteration);

	std::memcpy(output_data, output.data, input_rows * input_cols * sizeof(unsigned char));
}

void openmp_bayesShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	for (int i = 0; i < n_channels; i++) {
		OpenMPWaveletThreshold::bayesShrink(input_channels[i], output_channels[i], level);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}

void openmp_visuShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	for (int i = 0; i < n_channels; i++) {
		OpenMPWaveletThreshold::visuShrink(input_channels[i], output_channels[i], level);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}

void openmp_neighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	for (int i = 0; i < n_channels; i++) {
		OpenMPWaveletThreshold::neighShrink(input_channels[i], output_channels[i], level, windowSize);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}

void openmp_modiNeighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);


	for (int i = 0; i < n_channels; i++) {
		OpenMPWaveletThreshold::modineighShrink(input_channels[i], output_channels[i], level, windowSize);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}


//
// Sequential functions


void sequential_dwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration) {
	// Call the actual Sequential DWT function from the library
	cv::Mat input(input_rows, input_cols, input_data_type, (void*)input_data);
	cv::Mat output(input_rows, input_cols, input_data_type, (void*)output_data);

	SequentialHaarWavelet::dwt(input, output, nIteration);

	std::memcpy(output_data, output.data, input_rows * input_cols * sizeof(unsigned char));
}

void sequential_idwt(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, unsigned char* output_data, int nIteration) {
	// Call the actual Sequential IDWT function from the library
	cv::Mat input(input_rows, input_cols, input_data_type, (void*)input_data);
	cv::Mat output(input_rows, input_cols, input_data_type, (void*)output_data);

	SequentialHaarWavelet::idwt(input, output, nIteration);

	std::memcpy(output_data, output.data, input_rows * input_cols * sizeof(unsigned char));
}

void sequential_bayesShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	for (int i = 0; i < n_channels; i++) {
		SequentialWaveletThreshold::bayesShrink(input_channels[i], output_channels[i], level);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}

void sequential_visuShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	for (int i = 0; i < n_channels; i++) {
		SequentialWaveletThreshold::visuShrink(input_channels[i], output_channels[i], level);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}


void sequential_neighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	for (int i = 0; i < n_channels; i++) {
		SequentialWaveletThreshold::neighShrink(input_channels[i], output_channels[i], level, windowSize);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}


void sequential_modiNeighShrink(const unsigned char* input_data, int input_data_type, int input_rows, int input_cols, int n_channels, unsigned char* output_data, int level, int windowSize) {
	cv::Mat input, output;
	std::vector<cv::Mat> input_channels, output_channels;

	init_mats(input_data, input_rows, input_cols, input_data_type, n_channels, output_data, input, output, input_channels, output_channels);

	for (int i = 0; i < n_channels; i++) {
		SequentialWaveletThreshold::modineighShrink(input_channels[i], output_channels[i], level, windowSize);
	}

	cv::merge(output_channels, output);
	if (n_channels == 1)
		output.convertTo(output, CV_8U);
	else
		output.convertTo(output, CV_8UC3);

	std::memcpy(output_data, output.data, input_rows * input_cols * n_channels * sizeof(unsigned char));
}

