#plot one img and pask from 128

import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

img_path = Path('../../data/patch128/img/zone1_0_0/img_zone1_0_0_patch_2_12_bottom_left.tif')
msk_path = Path('../../data/patch128/msk/zone1_0_0/msk_zone1_0_0_patch_2_12_bottom_left.tif')

# path 

with rasterio.open(img_path) as src:
    img = src.read()
with rasterio.open(msk_path) as src:
    msk = src.read()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img[0], cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(msk[0], cmap='gray')
ax[1].set_title('Mask')
#save
plt.savefig('img_msk_plot.png')

# get information about range of values, type of values, etc
print(f"Image: min: {img.min()}, max: {img.max()}, dtype: {img.dtype}")
print(f"Mask: min: {msk.min()}, max: {msk.max()}, dtype: {msk.dtype}")

img_path_2 = Path('../../data/patch256/img/zone1_0_0/img_zone1_0_0_patch_2_12.tif')
msk_path_2 = Path('../../data/patch256/msk/zone1_0_0/msk_zone1_0_0_patch_2_12.tif')

# path 

with rasterio.open(img_path_2) as src:
    img_2 = src.read()
with rasterio.open(msk_path_2) as src:
    msk_2 = src.read()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_2[0], cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(msk_2[0], cmap='gray')
ax[1].set_title('Mask')
#save
plt.savefig('img_msk_plot_256.png')

# get information about range of values, type of values, etc
print(f"Image: min: {img_2.min()}, max: {img_2.max()}, dtype: {img_2.dtype}")
print(f"Mask: {np.unique(msk_2[0])}, dtype: {msk_2.dtype}")