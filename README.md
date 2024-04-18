# Project: Natural Habitats Mapping with Deep Learning

This project aims to map natural habitats using deep learning techniques based on the classification produced by the European Nature Information System (EUNIS).

## Folder Structure
- **csv/**
  - `corrupted_tif_images.csv`: List of corrupted TIF images.
  - `corrupted_zip_files.csv`: List of corrupted ZIP files.
  - `intersection_shp_tif.csv`: Pivot table of shapefile polygon IDs and TIF image names when intersection > 0.75.

- **data/**
  - Initially contained the ECOMED dataset, which was cleaned, split into four parts, and zipped for upload to a remote workstation.
  - **data_1/**
    - **HABNAT**: Contains shapefile polygons with their EUNIS classes. 
    - **Sources.shp**: 
    - **zone1 to zone40**: Directories containing drone images (TIF format) captured with a resolution of 0.15. 
  - **data_2**
    - **zone41 to zone80** 
  - **data_3**
    - **zone81 to zone120** 
  - **data_4**
    - **zone121 to zone171** 

- **notebooks/**
  - `data_statistics.ipynb`: Selects intersecting polygons and images, generates and stores a pivot table, and analyzes data with statistics and plots.

- **src/**
  - `check_corrupted_tif.py`: Checks corrupted TIF images and stores their names.
  - `unzip_and_check_corrupted_zip.py`: Unzips zone folders data and stores the list of corrupted ZIP files.

## Note
The data folder is currently empty as the dataset is proprietary to the private ecological consulting firm Ecomed and not accessible in open access. For more information about Ecomed, please visit their website: [Ecomed](https://ecomed.fr/)