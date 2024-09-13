import os
import rasterio


folder_path = '../../data/full_img_msk/msk/level1/'
# get all tif files in the folder
tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

# check values in each tif file
for tif in tif_files:
    # open the tif file
    mask = rasterio.open(folder_path + tif)
    mask_array = mask.read(1)
    print(f'{tif}: {set(mask_array.flatten())}')
    # if there is only one value in the tif which is 0, drop the mask
    if len(set(mask_array.flatten())) == 1 and 0 in set(mask_array.flatten()):
        os.remove(folder_path + tif)
        print(f'{tif} dropped')