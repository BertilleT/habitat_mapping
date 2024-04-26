import os


folder_path = '../../data/full_img_msk/msk/level1/'
# get all tif files in the folder
tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')] 
print(len(tif_files))

img_path = '../../data/full_img_msk/img/'
img_files = [f for f in os.listdir(img_path) if f.endswith('.tif')]
print(len(img_files))


for f in img_files:
    f2 = f.replace('img_', 'l1_msk_')
    if f2 not in tif_files:
        print(f2, 'does not exist')
        os.remove(img_path + f)

img_files = [f for f in os.listdir(img_path) if f.endswith('.tif')]
print(len(img_files))