from pathlib import Path
from utils import split_img_msk_into_patches
import rasterio

patch_size = 256
tif_path = Path('../data/full_img_msk/img/img_zone16_0_0.tif')

name = tif_path.stem
mask_path = Path('../data/full_img_msk/msk/level123') / f'l123_msk_{name[4:]}.tif'

split_img_msk_into_patches(tif_path, mask_path, patch_size)

'''# laod img_zone16_0_0.tif an dprint shape
img = rasterio.open('../data/full_img_msk/img/img_zone16_0_0.tif')
print(img.shape)  # (3, 512, 512)

msk = rasterio.open('../data/full_img_msk/msk/level123/l123_msk_zone16_0_0.tif')
print(msk.shape)  # (1, 512, 512)'''