# DWT Image Denoising with Shrinkage Algorithms and Parallel Computing Technique
![image](https://github.com/user-attachments/assets/c773b70f-9ba7-4ca1-961e-e6b9157a7336)

## Parallel Platform
  - OpenMP
  - CUDA 12.6


## Image Denoising Algorithms
  - VisuShrink
  - BayesShrink
  - NeighShrink
  - ModiNeighShrink


## How To Run
### Cpp Project
> Important : Ensure **CUDA 12.6** is installed for simplicity, or update the vcxproj manually and load the project with dependency again
1. Open in Visual Studio 2022
2. Go to `View > Property Manager`, expand any project and open the `Common.props`
3. Update the `Include Directories` and `Library Directories`, to your pc's opencv path
    - This will automatically update and apply to all project, they share the same props file
 
    - <img src="https://github.com/user-attachments/assets/ed2c1273-b7c3-4bc4-8907-31b49a6bb12d" width="200" /> <img src="https://github.com/user-attachments/assets/ad553e8a-3410-46e4-b57c-5db3c126e318" width="400" />
    
4. Configure Startup Project as **`TestUI`**


### Performance Evaluation Notebook
1. Build and Compile the `CombinedDLL` project as dll files
2. Open `PerformanceEvaluation.ipynb` in `Jupyter Notebook` / `JupyterLab`
3. Update all the neccessary path inside the notebook
4. Run all and wait for result!


## Results for different algorithms

  - DWT Iteration = 3
  - Gausian Noise
    - mean = 0
    - standard deviation = 100

| **Algorithms**       | **Sample Image 1**                                                                                 |
|----------------------|----------------------------------------------------------------------------------------------------|
| **Original Image**   | ![fruits-512x512](https://github.com/user-attachments/assets/26c8a140-c021-49dc-8d63-21c11d9b5b38) |
| **Noisy Image**      | ![noisy_img](https://github.com/user-attachments/assets/d65c36a5-c851-4686-aad8-0ab4ce4cf190)      |
| **VisuShrink**       | ![visu](https://github.com/user-attachments/assets/2caeab35-2f92-43d8-8ac4-4032b17910cf)           | 
| **BayesShrink**      | ![bayes](https://github.com/user-attachments/assets/140db94f-723d-44e8-af8a-0e10ce10d057)          |
| **NeighShrink** <br> *window size = 3x3*     | ![neigh](https://github.com/user-attachments/assets/557a7ec8-0018-4faa-930b-68d74ed51c00)          |
| **ModiNeighShrink** <br> *window size = 3x3* | ![modineigh](https://github.com/user-attachments/assets/25a740ef-5787-4d0d-be13-7c94d0ee8643)      |

