from pathlib import Path
import os
import rasterio
import tifffile as tiff

img_folder = Path(f'../../data/patch256/img/')
msk_folder = Path(f'../../data/patch256/msk/')

img_folder_64 = Path(f'../../data/patch64/img/')
msk_folder_64 = Path(f'../../data/patch64/msk/')


os.makedirs(img_folder_64, exist_ok=True)
os.makedirs(msk_folder_64, exist_ok=True)

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

    # Ensure the images are at least 128x128
    if img_height < 256 or img_width < 256 or msk_height < 256 or msk_width < 256:
        print(f"Skipping {img_path} and {msk_path} due to insufficient dimensions.")
        return

    # Slice img and msk into 16 patches of 64x64
    patches_img = []
    patches_msk = []
    for i in range(0, img_height, 64):
        for j in range(0, img_width, 64):
            patch_img = img[:, i:i+64, j:j+64]
            patch_msk = msk[:, i:i+64, j:j+64]
            if patch_img.shape[1] == 64 and patch_img.shape[2] == 64:  # Check for patch size
                patches_img.append(patch_img)
                patches_msk.append(patch_msk)

    img_parent = img_path.parts[-2]
    msk_parent = msk_path.parts[-2]
    img_name = img_path.stem
    msk_name = msk_path.stem

    os.makedirs(img_folder_64 / img_parent, exist_ok=True)
    os.makedirs(msk_folder_64 / msk_parent, exist_ok=True)

    patch_names = ['top_left', 'top_middle_left', 'top_middle_right', 'top_right',
                   'middle_top_left', 'middle_top_middle_left', 'middle_top_middle_right', 'middle_top_right',
                   'middle_bottom_left', 'middle_bottom_middle_left', 'middle_bottom_middle_right', 'middle_bottom_right',
                   'bottom_left', 'bottom_middle_left', 'bottom_middle_right', 'bottom_right']

    for idx, (patch_img, patch_msk) in enumerate(zip(patches_img, patches_msk)):
        tiff.imwrite(img_folder_64 / img_parent / f'{img_name}_{patch_names[idx]}.tif', patch_img)
        tiff.imwrite(msk_folder_64 / msk_parent / f'{msk_name}_{patch_names[idx]}.tif', patch_msk)

i = 0
for img_path, msk_path in zip(img_paths, msk_paths):
    split_img_msk(img_path, msk_path)
    i += 1
    print(f'Processed {i}/{len(img_paths)} images.')
