'''# for all pairs of img and msk, split them in 4 and save them into data/patch128 folder

import os
import rasterio
import numpy as np
import tifffile as tiff
from pathlib import Path


img_folder = Path(f'../../data/patch256/img/')
msk_folder = Path(f'../../data/patch256/msk/')

img_folder_128 = Path(f'../../data/patch128/img/')
msk_folder_128 = Path(f'../../data/patch128/msk/')
#create folders
os.makedirs(img_folder_128, exist_ok=True)
os.makedirs(msk_folder_128, exist_ok=True)

msk_paths = list(msk_folder.rglob('*.tif'))
img_paths = [img_folder / msk_path.parts[-2] / msk_path.name.replace('msk', 'img') for msk_path in msk_paths]

print(len(msk_paths), len(img_paths))

def split_img_msk(img_path, msk_path):
    with rasterio.open(img_path) as src:
        img = src.read()
    with rasterio.open(msk_path) as src:
        msk = src.read()
    #slice img and msk into 4 patches 128
    top_left = img[:128, :128]
    top_right = img[:128, 128:]
    bottom_left = img[128:, :128]
    bottom_right = img[128:, 128:]

    top_left_msk = msk[:128, :128]
    top_right_msk = msk[:128, 128:]
    bottom_left_msk = msk[128:, :128]
    bottom_right_msk = msk[128:, 128:]

    #save all f them in img_folder_128 and msk_folder_128
    img_parent = img_path.parts[-2]
    msk_parent = msk_path.parts[-2]
    img_name = img_path.name
    msk_name = msk_path.name
    img_name = img_name[:-4]
    msk_name = msk_name[:-4]
    #create dict img_folder_128 / img_parent
    os.makedirs(img_folder_128 / img_parent, exist_ok=True)
    os.makedirs(msk_folder_128 / msk_parent, exist_ok=True)
    tiff.imwrite(img_folder_128 / img_parent / f'{img_name}_top_left.tif', top_left)
    tiff.imwrite(img_folder_128 / img_parent / f'{img_name}_top_right.tif', top_right)
    tiff.imwrite(img_folder_128 / img_parent / f'{img_name}_bottom_left.tif', bottom_left)
    tiff.imwrite(img_folder_128 / img_parent / f'{img_name}_bottom_right.tif', bottom_right)


    tiff.imwrite(msk_folder_128 / msk_parent / f'{msk_name}_top_left.tif', top_left_msk)
    tiff.imwrite(msk_folder_128 / msk_parent / f'{msk_name}_top_right.tif', top_right_msk)
    tiff.imwrite(msk_folder_128 / msk_parent / f'{msk_name}_bottom_left.tif', bottom_left_msk)
    tiff.imwrite(msk_folder_128 / msk_parent / f'{msk_name}_bottom_right.tif', bottom_right_msk)

for img_path, msk_path in zip(img_paths, msk_paths):
    split_img_msk(img_path, msk_path)
    print(f'Split {img_path} and {msk_path} into 128x128 patches.')'''


from pathlib import Path
import os
import rasterio
import tifffile as tiff

img_folder = Path(f'../../data/patch256/img/')
msk_folder = Path(f'../../data/patch256/msk/')

img_folder_128 = Path(f'../../data/patch128/img/')
msk_folder_128 = Path(f'../../data/patch128/msk/')


os.makedirs(img_folder_128, exist_ok=True)
os.makedirs(msk_folder_128, exist_ok=True)

msk_paths = list(msk_folder.rglob('*.tif'))
img_paths = [img_folder / msk_path.parts[-2] / msk_path.name.replace('msk', 'img') for msk_path in msk_paths]

print(len(msk_paths), len(img_paths))
print(f'Number of unique images: {len(set(img_paths))}')
print(f'Number of unique masks: {len(set(msk_paths))}')

def split_img_msk(img_path, msk_path):
    with rasterio.open(img_path) as src:
        img = src.read()
        img_height, img_width = img.shape[1], img.shape[2]  # Channels, Height, Width
    with rasterio.open(msk_path) as src:
        msk = src.read()
        msk_height, msk_width = msk.shape[1], msk.shape[2]  # Channels, Height, Width

    print(f"Image shape: {img.shape}, Mask shape: {msk.shape}")

    # Ensure the images are at least 256x256
    if img_height < 256 or img_width < 256 or msk_height < 256 or msk_width < 256:
        print(f"Skipping {img_path} and {msk_path} due to insufficient dimensions.")
        return

    # Slice img and msk into 4 patches 128
    top_left = img[:, :128, :128]
    top_right = img[:, :128, 128:256]
    bottom_left = img[:, 128:256, :128]
    bottom_right = img[:, 128:256, 128:256]

    top_left_msk = msk[:, :128, :128]
    top_right_msk = msk[:, :128, 128:256]
    bottom_left_msk = msk[:, 128:256, :128]
    bottom_right_msk = msk[:, 128:256, 128:256]

    img_parent = img_path.parts[-2]
    msk_parent = msk_path.parts[-2]
    img_name = img_path.stem
    msk_name = msk_path.stem

    os.makedirs(img_folder_128 / img_parent, exist_ok=True)
    os.makedirs(msk_folder_128 / msk_parent, exist_ok=True)

    tiff.imwrite(img_folder_128 / img_parent / f'{img_name}_top_left.tif', top_left)
    tiff.imwrite(img_folder_128 / img_parent / f'{img_name}_top_right.tif', top_right)
    tiff.imwrite(img_folder_128 / img_parent / f'{img_name}_bottom_left.tif', bottom_left)
    tiff.imwrite(img_folder_128 / img_parent / f'{img_name}_bottom_right.tif', bottom_right)

    tiff.imwrite(msk_folder_128 / msk_parent / f'{msk_name}_top_left.tif', top_left_msk)
    tiff.imwrite(msk_folder_128 / msk_parent / f'{msk_name}_top_right.tif', top_right_msk)
    tiff.imwrite(msk_folder_128 / msk_parent / f'{msk_name}_bottom_left.tif', bottom_left_msk)
    tiff.imwrite(msk_folder_128 / msk_parent / f'{msk_name}_bottom_right.tif', bottom_right_msk)

    # print(f'Successfully split {img_path} and {msk_path} into 128x128 patches.')

i = 0
for img_path, msk_path in zip(img_paths, msk_paths):
    split_img_msk(img_path, msk_path)
    i += 1
    print(f'Processed {i}/{len(img_paths)} images.')
