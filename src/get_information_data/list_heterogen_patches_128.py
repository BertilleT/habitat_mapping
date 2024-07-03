from pathlib import Path
import pandas as pd
import numpy as np
import random
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

# SETTINGS
msk_folder = Path(f'../../data/patch128/msk/')
msk_paths = list(msk_folder.rglob('*.tif'))
print(f'Number of unique masks: {len(set(msk_paths))}')

'''
def check_tif_integrity(file_paths):
    corrupt_files = []
    valid_files = []

    for file_path in file_paths:
        try:
            with rasterio.open(file_path) as src:
                # Attempt to read the file to check its integrity
                src.read()
            valid_files.append(file_path)
            print(f"{file_path} is valid.")
        except rasterio.errors.RasterioIOError as e:
            corrupt_files.append(file_path)
            print(f"{file_path} is corrupt. Error: {e}")

    return valid_files, corrupt_files

# Check the integrity of all TIFF files
valid_files, corrupt_files = check_tif_integrity(msk_paths)

# Output the results
print(f"Number of valid files: {len(valid_files)}")
print(f"Number of corrupt files: {len(corrupt_files)}")
if corrupt_files:
    print("List of corrupt files:")
    for corrupt_file in corrupt_files:
        print(corrupt_file)

'''


# check if masks copsoed of one single class, if so, then remove them
# open with rasterio
i = 0
heterogen_masks = []
for msk_path in msk_paths:
    print(f'Processing mask {i+1}/{len(msk_paths)}')
    with rasterio.open(msk_path) as src:
        msk = src.read(1)
    print(np.unique(msk))
    if len(np.unique(msk)) == 1:
        print(f'{msk_path} has only one class, we do not want to include it')
    else:
        heterogen_masks.append(msk_path)
    i += 1
print(f'Number of unique masks after removing single class masks: {len(set(heterogen_masks))}')

# save to csv the list of masks without masks single class msk_paths
df = pd.DataFrame(heterogen_masks)
df.to_csv(f'../../csv/heterogen_masks_128.csv', index=False)