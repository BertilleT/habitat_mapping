# load images and masks of size 256*256
# msks have 3 channels, First for l1, second for l2 and third for l3
# if l2 is 18, 19, 24 or 29, do not kep the image and mask
# if l3 is 67 or 70 neither
# the filtered dataset should be placed in img_filtered and msk_filtered

import os
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

#path_to_imgs = Path('../../data/patch256/img')
path_to_msks = Path('../../data/patch256/msk')

path_to_img_filtered = Path('../../data/patch256/img_filtered')
path_to_msk_filtered = Path('../../data/patch256/msk_filtered')

if not os.path.exists(path_to_img_filtered):
    os.makedirs(path_to_img_filtered)

if not os.path.exists(path_to_msk_filtered):
    os.makedirs(path_to_msk_filtered)


def filter_sub_classes_heterogenes(path_to_msks, path_to_msk_filtered):
    # Get the list of all files in directory tree at given path
    list_of_files = list(path_to_msks.rglob('*.tif'))
    for file in list_of_files:
        with rasterio.open(file) as src:
            msk = src.read()
            msk = msk[0]
            if 18 in msk or 19 in msk or 24 in msk or 29 in msk or 67 in msk or 70 in msk:
                pass
            else:
                print(f'Copying {file.name}')
                os.system(f'cp {file} {path_to_msk_filtered}') #os.system is used to run bash commands, it is equivalent to ! in jupyter notebook
                # Copy the corresponding image
                img_path = path_to_img_filtered / file.name
                img_path = img_path.with_suffix('.tif')
                os.system(f'cp {file.with_suffix(".tif")} {img_path}')

    print('Done')