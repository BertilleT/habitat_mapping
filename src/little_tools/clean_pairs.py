## load images and paths from the dataset
## clean the dataset by removing images with no corresponding caption

# for each img compute p2 and p98, print error when P2 = P98 or if nan in image and count them

import rasterio
from pathlib import Path
import numpy as np
import pandas as pd
'''

img_folder = Path(f'../../data/patch256/img/')
imgs = list(img_folder.rglob('*.tif'))
i = 0
for img in imgs:
    print(i, 'over', len(imgs))
    with rasterio.open(img) as src:
        img_array = src.read()
        p2 = np.percentile(img_array, 2)
        p98 = np.percentile(img_array, 98)
        if np.isnan(p2) or np.isnan(p98):
            print(f'Error in {img}: p2 = {p2}, p98 = {p98}, nan in image')
        elif p2 == p98:
            print(f'Error in {img}: p2 = {p2}, p98 = {p98}, p2 = p98')
    i += 1'''


img_folder = Path(f'../../data/patch64/img/')
imgs = list(img_folder.rglob('*.tif'))
i = 0
img_to_drop = []
for img in imgs:
    print(i, 'over', len(imgs))
    with rasterio.open(img) as src:
        img_array = src.read()
        p2 = np.percentile(img_array, 2)
        p98 = np.percentile(img_array, 98)
        if np.isnan(p2) or np.isnan(p98):
            print(f'Error in {img}: p2 = {p2}, p98 = {p98}, nan in image')
            img_to_drop.append(img)
        elif p2 == p98:
            print(f'Error in {img}: p2 = {p2}, p98 = {p98}, p2 = p98')
            # add the image to the list of images to drop
            img_to_drop.append(img)
        
    i += 1

print(f'Number of images to drop: {len(img_to_drop)}')
print(img_to_drop)

# print len of imgs and msks
img_folder = Path(f'../../data/patch64/img/')
msk_folder = Path(f'../../data/patch64/msk/')
imgs = list(img_folder.rglob('*.tif'))
msks = list(msk_folder.rglob('*.tif'))
print(f'Number of images: {len(imgs)}')
print(f'Number of masks: {len(msks)}')

# drop the images
print('Dropping images and masks')
for img in img_to_drop:
    img.unlink()
    # drop the corresponding mask
    msk = Path(str(img).replace('img', 'msk'))
    msk.unlink()

# print len of imgs and msks
img_folder = Path(f'../../data/patch64/img/')
msk_folder = Path(f'../../data/patch64/msk/')
imgs = list(img_folder.rglob('*.tif'))
msks = list(msk_folder.rglob('*.tif'))
print(f'Number of images: {len(imgs)}')
print(f'Number of masks: {len(msks)}')