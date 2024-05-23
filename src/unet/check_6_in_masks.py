# load masks from data/full_img_msk/msk/level123 and check unique values in it

import os
import numpy as np
import rasterio

# path to masks
path = '../data/full_img_msk/msk/level123'

# get all masks
masks = os.listdir(path)

# check unique values in masks
for mask in masks:
    with rasterio.open(os.path.join(path, mask)) as src:
        mask = src.read(2)
        print(f'{mask.shape} {np.unique(mask)}')

# output
# (512, 512) [0 6]
