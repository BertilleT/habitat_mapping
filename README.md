# Project: Natural Habitats Mapping in the South of France with Deep Learning

MVA Master thesis peoject made at Inria Evergreen, Montpellier (FRANCE). 

This documentation is split into three main parts:

1. **Presentation of the Project**: recap the project's goal, data, methods and the challenges faced

2. **Directory Structure**: provides an overview of the project's folder organization.

3. **Folders Contents**: describes the contents and purpose of each folder.


## 1.  Presentation of the Project
- **Goal**: to classify natural habitats using Convolutional Neural Networks (CNNs) applied to Very High Resolution(VHR) aerial images. 

- **Data**: provided by the ecological consultancy firm ECO-MED. Please contact them directly to access it. 
    - **Images**: Very High Resolution (15 cm per pixel) RGB-NIR raster images.
    - **Labels**: Vector polygons in a shapefile, annotated by ecological experts according to the European Nature Information System(EUNIS) classification. 
    Maps of the geographical split of the data can be found at: https://bertillet.github.io/habitat_mapping/

- **Method**: 2 approaches were evaluated:
    1. Pixel-level classification: UNet model with an EfficientNet B7 encoder.
    2. Patch-level classification: ResNet18 model in a stepwise design: habitat classification in patches without boundaries, and then boundary detection task is included in the model. 

    Also, a post-processing step inspired by Conditional Random Fields (CRFs) is applied to smooth the regions predicted by the Patch-level classification model. 

- **Challenges**: dataset inconsistencies and missing data, geographical data shift, missing information about plant spcies names


## 2.  Directory Structure
```
├── csv
├── data
├── html
├── imgs
├── json
├── notebooks
├── results
├── src
|   ├── data_pre_processing
|   ├── get_information_data
|   ├── little_tools
|   └── processing
index.html

```
The data folder should be structured as follow: 
```
├── data
│   ├── patch64
│   │   ├── img
│   │   │   ├── zone1_0_0
│   │   │   │   ├── img_zone1_0_0_patch_8_48.tif
│   │   │   │   ├── img_zone1_0_0_patch_8_49.tif
│   │   │   │   └── ...
│   │   │   ├── zone2_0_0
│   │   │   └── ...
│   │   ├── msk
│   │   │   ├── zone1_0_0
│   │   │   │   ├── msk_zone1_0_0_patch_8_48.tif
│   │   │   │   ├── msk_zone1_0_0_patch_8_49.tif
│   │   │   │   └── ...
│   │   │   ├── zone2_0_0
│   │   │   └── ...
│   ├── patch128
│   │   ├── img
│   │   └── msk
│   ├── patch256
│   │   ├── img
│   │   └── msk
```

## 3.  Folders Content

- **csv**: CSV files used for data storage and manipulation.
- **html**: HTML files for the project's web interface, dedicated to plot the maps of geographical distribution of samples across train, validation, and test sets.
- **imgs**: images generated to view the data, and the results
- **json**: JSON dictionaries mapping legend items to integer classes in masks, along with associated colors for the plots. 
- **notebooks**: Jupyter notebooks including statistics made from the data. 
- **results**: output results from the project. The first part of the filename indicates the model used (e.g., `resnet18` or `unet`), the second part specifies the size of the patch used (e.g., `128`, `256`), and the third part represents the level of granularity of the EUNIS classification studied (e.g., `l1`, `l2`).
    - resnet18_128_l1
    - resnet18_256_l1
    ...
- **src**: Contains the source code for the project.
    - data_pre_processing: Scripts to pre_process the dataset from ECOMED. Key script is pipeline.py
    - get_information_data: Scripts to obtain various information about the data, such as class balance, list of heterogeneous patches, list of zones from annotated in 2023 dataset etc
    - processing: 
        - utils: Contains the scripts which includes dataloader, training/validation/testing, plotting functions used in main.py. 
        - main.py: Key script to load the dataset, train, validate test the model and plot the results. 
        - post_processing_MRF.py: Script to implement the post-processing step inspired from MRF
        - settings.py: Configuration file with all the parameters and hyperparameters to be chosen before running main.py
- **data**: pre-processed dataset
- **index.html**: The main HTML file for the project's web interface. 
- **report.pdf**: The master thesis report. 
