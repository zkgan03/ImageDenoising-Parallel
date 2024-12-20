# Image Denoising with Parallel Computing Technique
![WhatsApp Image 2024-12-18 at 22 01 08_1da8a7d7](https://github.com/user-attachments/assets/a3ddd6ed-7923-455f-a52f-203dfa7567f6)


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
2. Open in `Jupyter Notebook` / `JupyterLab`
3. Update all the neccessary path inside the notebook
4. Run all and wait for result!
