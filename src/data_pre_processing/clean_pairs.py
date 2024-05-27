## load images and paths from the dataset
## clean the dataset by removing images with no corresponding caption

# for each img compute p2 and p98, print error when P2 = P98 or if nan in image and count them

import rasterio
from pathlib import Path
import numpy as np
import pandas as pd


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
    i += 1