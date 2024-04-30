import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from utils import *
from pathlib import Path
from rasterio.plot import show
import time

# PATHS and DIRECTORIES
data_dir = Path('../data/')
msk_dir = data_dir / 'full_img_msk' / 'msk'
level1_dir = msk_dir / 'level1'
level2_dir = msk_dir / 'level2'
level3_dir = msk_dir / 'level3'

# load all tif files in level1
print(len(list(level1_dir.rglob('*.tif'))))
counter = 0
for tif_path in list(level1_dir.rglob('*.tif')):
    counter += 1
    print(counter, 'out of', len(list(level1_dir.rglob('*.tif'))))
    # remove l1 at the begining of tif file name and replace it on l2 and then by l3
    l1 = tif_path.stem
    l2 = tif_path.stem.replace('l1', 'l2')
    l3 = tif_path.stem.replace('l1', 'l3')

    # from 3 masks in l1, l2 and l3, make only one mask in l123 with 3 channels
    mask_l1 = rasterio.open(tif_path)
    mask_l2 = rasterio.open(level2_dir / (l2 + '.tif'))
    mask_l3 = rasterio.open(level3_dir / (l3 + '.tif'))
    mask_l123 = np.stack([mask_l1.read(1), mask_l2.read(1), mask_l3.read(1)], axis=0)
    #when value is 999 in l3 change it to 250
    mask_l123[mask_l123 == 999] = 250

    new_name = l1.replace('l1', 'l123')
    mask_l123_path = msk_dir / 'level123' / (new_name + '.tif')

    mask_l123_meta = mask_l1.meta.copy()
    mask_l123_meta['count'] = 3
    mask_l123_meta['dtype'] = 'uint8'
    
    with rasterio.open(mask_l123_path, 'w', **mask_l123_meta) as dst:
        dst.write(mask_l123)

