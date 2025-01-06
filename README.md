# Stability & Accuracy of Radiomic Features

## 1) Download and prepare data for Feature Extraction
### Download
In a first step, the two datasets these experiments are based on must be downloaded.  
Both datasets are publicly available on The Cancer Imaging Archive's website.

Download CT-Phantom4Radiomics data from here: [CTP4R](https://www.cancerimagingarchive.net/collection/ct-phantom4radiomics/)  
Download Colorectal-Liver-Metastases data from here: [CRLM](https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/)

### Directories
The scan and segmentation data must be stored in the designated directories in this repository so the code can access them.  
A script to move the data is provided [here](https://github.com/Joingl/radiomic-features/blob/main/data/move_scan_and_mask.py).  
Specify the paths to the stored datasets in line 5 and 6 of the script and run it.

### Conversion to NumPy Arrays
Features will be extracted from NumPy arrays.  
Convert the selected DICOM images to NumPy arrays using the script provided [here](https://github.com/Joingl/radiomic-features/blob/main/data/generate_NumPy_images.py).

## 2) Feature Extraction
In step 2 all feature types are extracted from the NumPy arrays. Feature extraction must be performed on both the datasets.  
To extract features from the CTP4R data run the script provided [here](https://github.com/Joingl/radiomic-features/blob/main/feature%20extraction/ctp4r/extract_features_ctp4r.py)   
To extract features from the CRLM data run the script provided [here](https://github.com/Joingl/radiomic-features/blob/main/feature%20extraction/crlm/extract_features_crlm.py)  

**For Gabor feature extraction you need to have a working version of the pytorch gabor library installed.**

## 3) Experiment 1
In experiment 1 stability and discriminative power are measured and features are ranked considering both characteristics.  
The metrics and the ranking of the features are computed by executing the scripts provided in the respective directories [here](https://github.com/Joingl/radiomic-features/tree/main/experiments/experiment1).  
Additionally, code for generating plots to summarize the results can be found in the corresponding directories.  

## 4) Experiment 2
In experiment 2 various machine learning models are trained, validated and tested on the CTP4R and the CRLM datasets.  
First, a dataset holding all the extracted features from the 4 features types must be generated by running the script provided [here](https://github.com/Joingl/radiomic-features/blob/main/experiments/experiment2/prepare_data.py).  
Results are then computed in the script provided [here](https://github.com/Joingl/radiomic-features/blob/main/experiments/experiment2/compute_results.py).  
Finally, results can be visualized using the script [here](https://github.com/Joingl/radiomic-features/blob/main/experiments/experiment2/plot_performance.py).

## Notes
### Results
The repository provides CSV files with the extracted features and the pre-computed results. Recomputing might cause minor differences in results.

### Graphics
Graphics used to visualize the results can be found [here](https://github.com/Joingl/radiomic-features/tree/main/graphics).  

### Sources
Schaer, R., Bach, M., Obmann, M., Flouris, K., Konukoglu, E., Stieltjes, B., Müller, H., Aberle, C., Jimenez del Toro, O. A., & Depeursinge, A. (2023). Task-Based Anthropomorphic CT Phantom for Radiomics Stability and Discriminatory Power Analyses (CT-Phantom4Radiomics) [Data set]. [https://doi.org/10.7937/a1v1-rc66](https://doi.org/10.7937/a1v1-rc66)  

Simpson, A. L., Peoples, J., Creasy, J. M., Fichtinger, G., Gangai, N., Lasso, A., Keshava Murthy, K. N., Shia, J., D’Angelica, M. I., & Do, R. K. G. (2023). Preoperative CT and Survival Data for Patients Undergoing Resection of Colorectal Liver Metastases (Colorectal-Liver-Metastases) (Version 2) [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/QXK2-QG03](https://doi.org/10.7937/QXK2-QG03)  

Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.

Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, & Bingbing Ni. (2024). [MedMNIST+] 18x Standardized Datasets for 2D and 3D Biomedical Image Classification with Multiple Size Options: 28 (MNIST-Like), 64, 128, and 224 (3.0) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.10519652](https://doi.org/10.5281/zenodo.10519652)
