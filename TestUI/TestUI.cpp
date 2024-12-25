#include <windows.h>
#include <commdlg.h>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "OpenMP.h"
#include "_CUDA_.h"
#include "Sequential.h"

//#include "Combined.h"
//#include <omp.h>

#define IDC_BUTTON_OPEN 101 // id for the open button
#define IDC_COMBO_METHOD 102 // id for the method combo box
#define IDC_BUTTON_DENOISE 103 // id for the denoise button
#define IDC_COMBO_SHRINKAGE 104 // id for the shrinkage combo box
#define IDC_COMBO_IMAGE_TYPE 105 // id for the image type combo box
#define IDC_BUTTON_ADD_NOISE 106 // id for the add noise button
#define IDC_BUTTON_SAVE 107 // id for the save button

#define IDC_EDIT_LEVELS 200 // id for the levels input field
#define IDC_EDIT_WINDOW_SIZE 201 // id for the window size input field
#define IDC_EDIT_MEAN 202 // id for the mean input field
#define IDC_EDIT_STDDEV 203 // id for the standard deviation input field

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void add_gaussian_noise(cv::Mat& image, double mean = 0.0, double stddev = 10.0);
void openFileDialog(HWND hwnd, std::wstring& filePath);
void displayImage(HWND hwnd, cv::Mat& image, int x, int y, int width, int height);

cv::Mat originalImage, noisyImage, denoisedImage;
std::wstring imagePath;
int selectedMethod = 0;
int selectedShrinkage = 0;
int selectedImageType = 0; // 0 for BGR, 1 for Gray scale
double executionTime = 0.0;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	const wchar_t CLASS_NAME[] = L"Sample Window Class";

	WNDCLASS wc = {};
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = CLASS_NAME;

	RegisterClass(&wc);

	HWND hwnd = CreateWindowEx(
		0,
		CLASS_NAME,
		L"Image Denoising",
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, 1600, 800,
		NULL,
		NULL,
		hInstance,
		NULL
	);

	if (hwnd == NULL) {
		return 0;
	}

	ShowWindow(hwnd, nCmdShow);

	MSG msg = {};
	while (GetMessage(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return 0;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {

	switch (uMsg) {
	case WM_CREATE: {
		// Create controls

		// Create combo box for image type
		CreateWindow(L"COMBOBOX", NULL, CBS_DROPDOWN | CBS_HASSTRINGS | WS_CHILD | WS_OVERLAPPED | WS_VISIBLE,
			10, 10, 150, 100, hwnd, (HMENU)IDC_COMBO_IMAGE_TYPE, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

		// Create Open Image button
		CreateWindow(L"BUTTON", L"Open Image", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
			170, 10, 100, 30, hwnd, (HMENU)IDC_BUTTON_OPEN, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

		// Create combo box for methods
		CreateWindow(L"COMBOBOX", NULL, CBS_DROPDOWN | CBS_HASSTRINGS | WS_CHILD | WS_OVERLAPPED | WS_VISIBLE,
			10, 50, 150, 100, hwnd, (HMENU)IDC_COMBO_METHOD, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

		// Create combo box for shrinkage methods
		CreateWindow(L"COMBOBOX", NULL, CBS_DROPDOWN | CBS_HASSTRINGS | WS_CHILD | WS_OVERLAPPED | WS_VISIBLE,
			170, 50, 150, 100, hwnd, (HMENU)IDC_COMBO_SHRINKAGE, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

		// Create input field for number of levels
		CreateWindow(L"STATIC", L"Levels:", WS_VISIBLE | WS_CHILD,
			330, 50, 100, 20, hwnd, NULL, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);
		CreateWindow(L"EDIT", L"3", WS_VISIBLE | WS_CHILD | WS_BORDER | ES_NUMBER,
			390, 50, 50, 20, hwnd, (HMENU)IDC_EDIT_LEVELS, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

		// Create input field for window size
		CreateWindow(L"STATIC", L"Window Size:", WS_VISIBLE | WS_CHILD,
			450, 50, 100, 20, hwnd, NULL, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);
		CreateWindow(L"EDIT", L"3", WS_VISIBLE | WS_CHILD | WS_BORDER | ES_NUMBER,
			540, 50, 50, 20, hwnd, (HMENU)IDC_EDIT_WINDOW_SIZE, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

		// Create input field for mean
		CreateWindow(L"STATIC", L"Mean:", WS_VISIBLE | WS_CHILD,
			330, 10, 50, 20, hwnd, NULL, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);
		CreateWindow(L"EDIT", L"0", WS_VISIBLE | WS_CHILD | WS_BORDER | ES_NUMBER,
			380, 10, 50, 20, hwnd, (HMENU)IDC_EDIT_MEAN, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

		// Create input field for standard deviation
		CreateWindow(L"STATIC", L"Std Dev:", WS_VISIBLE | WS_CHILD,
			440, 10, 60, 20, hwnd, NULL, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);
		CreateWindow(L"EDIT", L"50", WS_VISIBLE | WS_CHILD | WS_BORDER | ES_NUMBER,
			500, 10, 50, 20, hwnd, (HMENU)IDC_EDIT_STDDEV, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

		// Create Add Noise button
		CreateWindow(L"BUTTON", L"Add Noise", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
			600, 10, 100, 30, hwnd, (HMENU)IDC_BUTTON_ADD_NOISE, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

		// Create Denoise button
		CreateWindow(L"BUTTON", L"Denoise", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
			600, 50, 100, 30, hwnd, (HMENU)IDC_BUTTON_DENOISE, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

		// Create Save Image button (initially hidden)
		CreateWindow(L"BUTTON", L"Save Image", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
			710, 50, 100, 30, hwnd, (HMENU)IDC_BUTTON_SAVE, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);


		// Add items to the combo box for methods
		HWND hComboMethod = GetDlgItem(hwnd, IDC_COMBO_METHOD);
		SendMessage(hComboMethod, CB_ADDSTRING, 0, (LPARAM)L"OpenMP");
		SendMessage(hComboMethod, CB_ADDSTRING, 0, (LPARAM)L"CUDA");
		SendMessage(hComboMethod, CB_ADDSTRING, 0, (LPARAM)L"Sequential");
		SendMessage(hComboMethod, CB_SETCURSEL, 0, 0);

		// Add items to the combo box for shrinkage methods
		HWND hComboShrinkage = GetDlgItem(hwnd, IDC_COMBO_SHRINKAGE);
		SendMessage(hComboShrinkage, CB_ADDSTRING, 0, (LPARAM)L"BayesShrink");
		SendMessage(hComboShrinkage, CB_ADDSTRING, 0, (LPARAM)L"VisuShrink");
		SendMessage(hComboShrinkage, CB_ADDSTRING, 0, (LPARAM)L"NeighShrink");
		SendMessage(hComboShrinkage, CB_ADDSTRING, 0, (LPARAM)L"ModiNeighShrink");
		SendMessage(hComboShrinkage, CB_SETCURSEL, 0, 0);

		// Add items to the combo box for image type
		HWND hComboImageType = GetDlgItem(hwnd, IDC_COMBO_IMAGE_TYPE);
		SendMessage(hComboImageType, CB_ADDSTRING, 0, (LPARAM)L"BGR");
		SendMessage(hComboImageType, CB_ADDSTRING, 0, (LPARAM)L"Gray scale");
		SendMessage(hComboImageType, CB_SETCURSEL, 0, 0);

		break;
	}
	case WM_COMMAND: {
		// Handle button click events

		// Open Image button
		if (LOWORD(wParam) == IDC_BUTTON_OPEN) {
			openFileDialog(hwnd, imagePath);

			if (imagePath.empty()) {
				MessageBox(hwnd, L"Failed to open image", L"Error", MB_OK);
				return 0;
			}

			denoisedImage.release();

			HWND hComboImageType = GetDlgItem(hwnd, IDC_COMBO_IMAGE_TYPE);
			selectedImageType = static_cast<int>(SendMessage(hComboImageType, CB_GETCURSEL, 0, 0));

			if (selectedImageType == 1) {
				originalImage = cv::imread(
					cv::String(imagePath.begin(), imagePath.end()),
					cv::IMREAD_GRAYSCALE
				);
			}
			else {
				originalImage = cv::imread(
					cv::String(imagePath.begin(), imagePath.end()),
					cv::IMREAD_COLOR
				);
			}

			if (originalImage.empty()) {
				MessageBox(hwnd, L"Failed to open image", L"Error", MB_OK);
				return 0;
			}

			// ensure the image dimensions are even
			if (originalImage.rows % 2 != 0) {
				originalImage = originalImage(cv::Rect(0, 0, originalImage.cols, originalImage.rows - 1));
			}

			if (originalImage.cols % 2 != 0) {
				originalImage = originalImage(cv::Rect(0, 0, originalImage.cols - 1, originalImage.rows));
			}

			noisyImage = originalImage.clone();
			InvalidateRect(hwnd, NULL, TRUE);
		}

		// Add Noise button
		else if (LOWORD(wParam) == IDC_BUTTON_ADD_NOISE) {
			if (!originalImage.empty()) {
				// Get mean and standard deviation from input fields
				wchar_t meanStr[10], stddevStr[10];
				GetWindowText(GetDlgItem(hwnd, IDC_EDIT_MEAN), meanStr, 10);
				GetWindowText(GetDlgItem(hwnd, IDC_EDIT_STDDEV), stddevStr, 10);

				double mean = _wtof(meanStr);
				double stddev = _wtof(stddevStr);

				noisyImage = originalImage.clone();
				add_gaussian_noise(noisyImage, mean, stddev);
				InvalidateRect(hwnd, NULL, TRUE);
			}
			else {
				MessageBox(hwnd, L"No image loaded", L"Error", MB_OK);
			}
		}

		// Save Image button
		else if (LOWORD(wParam) == IDC_BUTTON_SAVE) {

			if (denoisedImage.empty()) {
				MessageBox(hwnd, L"No denoised image to save", L"Error", MB_OK);
				return 0;
			}

			OPENFILENAME ofn;
			wchar_t szFile[260] = L"default_filename.jpg"; // Set default file name and extension

			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hwnd;
			ofn.lpstrFile = szFile;
			ofn.nMaxFile = sizeof(szFile);
			ofn.lpstrFilter = L"JPEG\0*.JPG\0PNG\0*.PNG\0All\0*.*\0";
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

			if (GetSaveFileName(&ofn) == TRUE) {
				cv::imwrite(cv::String(ofn.lpstrFile, ofn.lpstrFile + wcslen(ofn.lpstrFile)), denoisedImage);
			}

		}

		// Denoise button
		else if (LOWORD(wParam) == IDC_BUTTON_DENOISE) {
			if (noisyImage.empty()) {
				MessageBox(hwnd, L"No image loaded", L"Error", MB_OK);
				return 0;
			}

			HWND hComboMethod = GetDlgItem(hwnd, IDC_COMBO_METHOD);
			selectedMethod = static_cast<int>(SendMessage(hComboMethod, CB_GETCURSEL, 0, 0));

			HWND hComboShrinkage = GetDlgItem(hwnd, IDC_COMBO_SHRINKAGE);
			selectedShrinkage = static_cast<int>(SendMessage(hComboShrinkage, CB_GETCURSEL, 0, 0));

			HWND hEditLevels = GetDlgItem(hwnd, IDC_EDIT_LEVELS);

			wchar_t buffer[10];
			GetWindowText(hEditLevels, buffer, 10);
			int levels = _wtoi(buffer); // Convert input to integer
			if (levels > 5) {
				// DWT Levels should be less than or equal to 5
				// show popup and return
				MessageBox(hwnd, L"Levels should be less than or equal to 5", L"Error", MB_OK);
				return 0;
			}

			HWND hEditWindowSize = GetDlgItem(hwnd, IDC_EDIT_WINDOW_SIZE);
			GetWindowText(hEditWindowSize, buffer, 10);
			int windowSize = _wtoi(buffer); // Convert input to intege

			if ((windowSize % 2 == 0 || windowSize < 3) && (selectedShrinkage == 2 || selectedShrinkage == 3)) {
				// window size should be odd and greater than or equal to 3
				// only show when chosen shrinkage is NeighShrink or ModiNeighShrink
				MessageBox(hwnd, L"Window size should be an odd number greater than or equal to 3", L"Error", MB_OK);
				return 0;
			}

			std::vector<cv::Mat> noisyImageChannels;

			noisyImage.convertTo(noisyImage, CV_32F);
			cv::split(noisyImage, noisyImageChannels);

			std::vector<cv::Mat> denoisedImageChannels = std::vector<cv::Mat>(noisyImageChannels.size());

			//for (int i = 0; i < noisyImageChannels.size(); i++) {
			//	denoisedImageChannels[i] = cv::Mat(noisyImageChannels[i].size(), CV_8U);
			//}

			//if (!noisyImage.empty()) {

			//	double startTime = omp_get_wtime();

			//	switch (selectedMethod) {
			//	case 0: {
			//		switch (selectedShrinkage) {
			//		case 0:
			//			for (int i = 0; i < noisyImageChannels.size(); i++) {
			//				//OpenMPWaveletThreshold::bayesShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
			//				openmp_bayesShrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels);
			//			}
			//			break;
			//		case 1:
			//			for (int i = 0; i < noisyImageChannels.size(); i++) {
			//				//OpenMPWaveletThreshold::visuShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
			//				openmp_visushrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels);
			//			}
			//			break;
			//		case 2:
			//			for (int i = 0; i < noisyImageChannels.size(); i++) {
			//				//OpenMPWaveletThreshold::neighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
			//				openmp_neighShrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels, windowSize);
			//			}
			//			break;
			//		case 3:
			//			for (int i = 0; i < noisyImage.channels(); i++) {
			//				//OpenMPWaveletThreshold::modineighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
			//				openmp_modiNeighShrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels, windowSize);
			//			}
			//			break;
			//		}
			//		break;
			//	}
			//	case 1: {
			//		switch (selectedShrinkage) {
			//		case 0:
			//			for (int i = 0; i < noisyImageChannels.size(); i++)
			//				//CUDAWaveletThreshold::bayesShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
			//				cuda_bayesShrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels);
			//			break;
			//		case 1:
			//			for (int i = 0; i < noisyImageChannels.size(); i++)
			//				//CUDAWaveletThreshold::visuShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
			//				cuda_visushrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels);
			//			break;
			//		case 2:
			//			for (int i = 0; i < noisyImageChannels.size(); i++)
			//				//CUDAWaveletThreshold::neighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
			//				cuda_neighShrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels, windowSize);
			//			break;
			//		case 3:
			//			for (int i = 0; i < noisyImageChannels.size(); i++)
			//				//CUDAWaveletThreshold::modineighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
			//				cuda_modiNeighShrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels, windowSize);
			//			break;
			//		}
			//		break;
			//	}
			//	case 2: {
			//		switch (selectedShrinkage) {
			//		case 0:
			//			for (int i = 0; i < noisyImageChannels.size(); i++)
			//				//SequentialWaveletThreshold::bayesShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
			//				sequential_bayesShrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels);
			//			break;
			//		case 1:
			//			for (int i = 0; i < noisyImageChannels.size(); i++)
			//				//SequentialWaveletThreshold::visuShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
			//				sequential_visushrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels);
			//			break;
			//		case 2:
			//			for (int i = 0; i < noisyImageChannels.size(); i++)
			//				//SequentialWaveletThreshold::neighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
			//				sequential_neighShrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels, windowSize);
			//			break;
			//		case 3:
			//			for (int i = 0; i < noisyImageChannels.size(); i++)
			//				//SequentialWaveletThreshold::modineighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
			//				sequential_modiNeighShrink(noisyImageChannels[i].data, CV_TYPE_8U(), noisyImageChannels[i].rows,
			//					noisyImageChannels[i].cols, 1, denoisedImageChannels[i].data, levels, windowSize);
			//			break;
			//		}
			//		break;
			//	}
			//	}
			//	double endTime = omp_get_wtime();
			//	executionTime = endTime - startTime;

			//	cv::merge(denoisedImageChannels, denoisedImage);
			//	cv::normalize(denoisedImage, denoisedImage, 0, 255, cv::NORM_MINMAX);
			//	denoisedImage.convertTo(denoisedImage, CV_8UC3);
			//	noisyImage.convertTo(noisyImage, CV_8UC3);

			//	InvalidateRect(hwnd, NULL, TRUE);
			//}


			if (!noisyImage.empty()) {

				double startTime = omp_get_wtime();

				switch (selectedMethod) {
				case 0: {
					switch (selectedShrinkage) {
					case 0:
						for (int i = 0; i < noisyImageChannels.size(); i++) {
							OpenMPWaveletThreshold::bayesShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
						}
						break;
					case 1:
						for (int i = 0; i < noisyImageChannels.size(); i++) {
							OpenMPWaveletThreshold::visuShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
						}
						break;
					case 2:
						for (int i = 0; i < noisyImageChannels.size(); i++) {
							OpenMPWaveletThreshold::neighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
						}
						break;
					case 3:
						for (int i = 0; i < noisyImage.channels(); i++) {
							OpenMPWaveletThreshold::modineighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
						}
						break;
					}
					break;
				}
				case 1: {
					switch (selectedShrinkage) {
					case 0:
						for (int i = 0; i < noisyImageChannels.size(); i++)
							CUDAWaveletThreshold::bayesShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
						break;
					case 1:
						for (int i = 0; i < noisyImageChannels.size(); i++)
							CUDAWaveletThreshold::visuShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
						break;
					case 2:
						for (int i = 0; i < noisyImageChannels.size(); i++)
							CUDAWaveletThreshold::neighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
						break;
					case 3:
						for (int i = 0; i < noisyImageChannels.size(); i++)
							CUDAWaveletThreshold::modineighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
						break;
					}
					break;
				}
				case 2: {
					switch (selectedShrinkage) {
					case 0:
						for (int i = 0; i < noisyImageChannels.size(); i++)
							SequentialWaveletThreshold::bayesShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
						break;
					case 1:
						for (int i = 0; i < noisyImageChannels.size(); i++)
							SequentialWaveletThreshold::visuShrink(noisyImageChannels[i], denoisedImageChannels[i], levels);
						break;
					case 2:
						for (int i = 0; i < noisyImageChannels.size(); i++)
							SequentialWaveletThreshold::neighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
						break;
					case 3:
						for (int i = 0; i < noisyImageChannels.size(); i++)
							SequentialWaveletThreshold::modineighShrink(noisyImageChannels[i], denoisedImageChannels[i], levels, windowSize);
						break;
					}
					break;
				}
				}
				double endTime = omp_get_wtime();
				executionTime = endTime - startTime;

				cv::merge(denoisedImageChannels, denoisedImage);
				cv::normalize(denoisedImage, denoisedImage, 0, 255, cv::NORM_MINMAX);
				denoisedImage.convertTo(denoisedImage, CV_8UC3);
				noisyImage.convertTo(noisyImage, CV_8UC3);

				InvalidateRect(hwnd, NULL, TRUE);
			}
		}
		break;
	}
	case WM_PAINT: {
		// Display images

		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hwnd, &ps);

		// Display images
		if (!originalImage.empty()) {
			displayImage(hwnd, originalImage, 10, 120, 500, 500);
		}
		if (!noisyImage.empty()) {
			displayImage(hwnd, noisyImage, 520, 120, 500, 500);
		}
		if (!denoisedImage.empty()) {
			displayImage(hwnd, denoisedImage, 1030, 120, 500, 500);
		}
		else {
			cv::Mat temp = cv::Mat::ones(1, 1, CV_8U) * 255;
			displayImage(hwnd, temp, 1030, 120, 500, 500);
		}

		// Display execution time
		wchar_t timeBuffer[50];
		swprintf(timeBuffer, 50, L"Execution Time: %.5f seconds", executionTime);
		TextOut(hdc, 10, 630, timeBuffer, wcslen(timeBuffer));

		EndPaint(hwnd, &ps);

		break;
	}
	case WM_DESTROY: {
		// Clean up resources
		PostQuitMessage(0);
		break;
	}
	default:
		// Handle any messages the switch statement didn't handle
		return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
	return 0;
}

void add_gaussian_noise(cv::Mat& image, double mean, double stddev) {
	cv::Mat noise(image.size(), image.type());
	cv::randn(noise, mean, stddev);
	image += noise;
}

void openFileDialog(HWND hwnd, std::wstring& filePath) {
	WCHAR filename[MAX_PATH] = L"";
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFilter = L"Image Files\0*.BMP;*.JPG;*.JPEG;*.PNG;*.TIF;*.TIFF\0All Files\0*.*\0";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
	ofn.lpstrDefExt = L"";

	if (GetOpenFileName(&ofn)) {
		filePath = filename;
	}
}

void displayImage(HWND hwnd, cv::Mat& image, int x, int y, int displayWidth, int displayHeight) {
	if (image.empty()) return;

	// Get the device context for the window
	HDC hdc = GetDC(hwnd);
	HDC hdcMem = CreateCompatibleDC(hdc);
	HBITMAP hBitmap = CreateCompatibleBitmap(hdc, displayWidth, displayHeight);
	SelectObject(hdcMem, hBitmap);

	// Clear the background to black
	RECT rect = { 0, 0, displayWidth, displayHeight };
	HBRUSH blackBrush = CreateSolidBrush(RGB(0, 0, 0));
	FillRect(hdcMem, &rect, blackBrush);
	DeleteObject(blackBrush);

	// Calculate the aspect ratio and the new dimensions
	double imageAspect = static_cast<double>(image.cols) / image.rows;
	double displayAspect = static_cast<double>(displayWidth) / displayHeight;

	int newWidth, newHeight;
	if (imageAspect > displayAspect) {
		// Image is wider than the display area
		newWidth = displayWidth;
		newHeight = static_cast<int>(displayWidth / imageAspect);
	}
	else {
		// Image is taller than the display area
		newHeight = displayHeight;
		newWidth = static_cast<int>(displayHeight * imageAspect);
	}

	// Calculate offsets for centering
	int offsetX = (displayWidth - newWidth) / 2;
	int offsetY = (displayHeight - newHeight) / 2;

	// Resize the image while maintaining the aspect ratio
	cv::Mat resizedImage;
	cv::resize(image, resizedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);

	// Convert the resized image to BGR format for rendering
	if (resizedImage.channels() == 1) {
		cv::cvtColor(resizedImage, resizedImage, cv::COLOR_GRAY2BGR);
	}

	// Ensure DWORD alignment for bitmap rows
	int rowWidth = resizedImage.cols * 3; // Each pixel is 3 bytes (BGR)
	int paddedRowWidth = (rowWidth + 3) & ~3; // Align to the next multiple of 4
	std::vector<uint8_t> alignedData(paddedRowWidth * resizedImage.rows, 0);

	for (int i = 0; i < resizedImage.rows; ++i) {
		std::memcpy(
			alignedData.data() + i * paddedRowWidth, // Destination
			resizedImage.data + i * rowWidth,        // Source
			rowWidth                                // Copy only the actual row data
		);
	}

	// Create a BITMAPINFO structure
	BITMAPINFO bmi = {};
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = resizedImage.cols;
	bmi.bmiHeader.biHeight = -resizedImage.rows; // Negative for a top-down DIB
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 24;
	bmi.bmiHeader.biCompression = BI_RGB;

	// Draw the resized image centered in the display area
	StretchDIBits(
		hdcMem,
		offsetX, offsetY, newWidth, newHeight, // Destination rectangle
		0, 0, resizedImage.cols, resizedImage.rows, // Source rectangle
		alignedData.data(),
		&bmi,
		DIB_RGB_COLORS,
		SRCCOPY
	);

	// Copy the memory DC to the window DC
	BitBlt(hdc, x, y, displayWidth, displayHeight, hdcMem, 0, 0, SRCCOPY);

	// Clean up resources
	DeleteObject(hBitmap);
	DeleteDC(hdcMem);
	ReleaseDC(hwnd, hdc);
}

int main() {
	return WinMain(GetModuleHandle(NULL), NULL, GetCommandLineA(), SW_SHOWNORMAL);
}
