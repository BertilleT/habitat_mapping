#list all tif images path in the folder given in path
import os
from pathlib import Path
import rasterio

path = '../../original_data/data_'
#use rglob
tif_files = []
for i in range(1, 5):
    tif_files_intermed = list(Path(path+str(i)).rglob('*.tif'))
    tif_files.extend(tif_files_intermed)


# for each tif file, store zoneid, and centroid of the image geo location. The centroid should b computed by looking a 
#store in a dictionary

zone_dict = {}

# print metadata of first tif image

with rasterio.open(tif_files[0]) as src:
    