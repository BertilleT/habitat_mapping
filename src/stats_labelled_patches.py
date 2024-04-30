import geopandas as gpd
from shapely.geometry import Point
import sys
import numpy as np
import rasterio
from rasterio import features
from utils import *
from pathlib import Path
from rasterio.plot import show
import csv

patch_size = 256
labelled_cover = 0.95
masks_path = Path(f'../data/patch{patch_size}/msk/l123')
save_csv = True
# I want to have nb patches kept in % for each threshold, also in absolute. 
# I want nb of patches with only 1 classes at level 1, in % and absolute
# I want nb of patches with 2 classes at level 1, in % and absolute
# I want nb of patches with 3 classes at level 1, in % and absolute
# I want nb of patches with 4 classes at level 1, in % and absolute
# I want nb of patches with 5 classes at level 1, in % and absolute
# I want nb of patches with 6 classes at level 1, in % and absolute
# I want nb of patches with 1 classes at level 2, in % and absolute
# I want nb of patches with 2 classes at level 2, in % and absolute
# ... 
# I want nb of patches with 26 classes at level 2, in % and absolute

def class_value_dict(mask_path, level):
    # Open the mask at channel 1 if level passed in argument is 1, 2 if level 2, 3 if level 3
    with rasterio.open(mask_path) as src:
        mask = src.read(level)
        unique, counts = np.unique(mask, return_counts=True)
        class_value = {k: v for k, v in zip(unique, counts)}
        nb_pixels = np.sum(counts)
    return class_value, nb_pixels

patches_kept_nb = 0
patches_kept_l1 = {}
patches_kept_l2 = {}
for key in range(1, 7):
    patches_kept_l1[key] = 0

for key in range(1, 27):
    patches_kept_l2[key] = 0

patches_kept_l1_un = patches_kept_l1.copy()
patches_kept_l2_un = patches_kept_l2.copy()

total_patches = len(list(masks_path.rglob('*.tif')))

counter = 0
for mask in masks_path.rglob('*.tif'):
    counter += 1
    if counter % 100 == 0:
        print(f'Processing {counter} over {total_patches} patches.')
    class_value_l1, nb_pixels = class_value_dict(mask, 1)
    # PATCHES KEPT WHEN MINIMUM LABELLED COVERAGE > THRESHOLD AT LEVEL 1
    # sum all values with key different from 0
    labelled_pixels = sum([v for k, v in class_value_l1.items() if k != 0])
    if labelled_pixels / nb_pixels >= labelled_cover:
        patches_kept_nb += 1
    # HETEROGENEITY OF LABELLED PATCHES AT LEVEL 1
        for key in class_value_l1:
            if key != 0:
                patches_kept_l1[len(class_value_l1)] += 1
    # HETEROGENEITY OF LABELLED PATCHES AT LEVEL 2
        class_value_l2, nb_pixels = class_value_dict(mask, 2)
        for key in class_value_l2:
            if key != 0:
                patches_kept_l2[len(class_value_l2)] += 1
    # Remove from class_value_l1 and from class_value_l2 the 0 values

        class_value_l1 = {k: v for k, v in class_value_l1.items() if v != 0}
        class_value_l2 = {k: v for k, v in class_value_l2.items() if v != 0}

        patches_kept_l1_un[len(class_value_l1)] += 1
        patches_kept_l2_un[len(class_value_l2)] += 1

# patches_kept_l1 and patches_kept_l1 in percentages
patches_kept_l1_un_per = {k: round((v * 100) / patches_kept_nb) for k, v in patches_kept_l1_un.items()}
patches_kept_l2_un_per = {k: round((v * 100) / patches_kept_nb) for k, v in patches_kept_l2_un.items()}

print('In total, there are ', total_patches, ' patches, there are all labelled because we did not save the unlabelled patches.')
print(f'--------------------- THRESHOLD {labelled_cover*100} % ---------------------')
print(f'{patches_kept_nb} patches kept, which is {patches_kept_nb * 100 / total_patches} % of the patches.')
print('--------------------- LEVEL1 ---------------------')
print('Heterogeneity of labelled patches at level 1:')
print(patches_kept_l1_un)
print('Heterogeneity of labelled patches at level 1 in %:')
print(patches_kept_l1_un_per)
print('--------------------- LEVEL2 ---------------------')
print('Heterogeneity of labelled patches at level 2 :')
print(patches_kept_l2_un)
print('Heterogeneity of labelled patches at level 2 in %:')
print(patches_kept_l2_un_per)


#save it to csv if save_csv is True
if save_csv == True:
    csv1_name = f'../csv/coverage_patch/patches_kept_l1_80p_{patch_size}_{labelled_cover}.csv'
    csv2_name = f'../csv/coverage_patch/patches_kept_l2_80p_{patch_size}_{labelled_cover}.csv'
    csv3_name = f'../csv/coverage_patch/patches_kept_nb_80p_{patch_size}_{labelled_cover}.csv'

    with open(csv1_name, 'w') as f:
        for key in patches_kept_l1_un.keys():
            f.write("%s,%s\n"%(key,patches_kept_l1_un[key]))

    with open(csv2_name, 'w') as f:
        for key in patches_kept_l2_un.keys():
            f.write("%s,%s\n"%(key,patches_kept_l2_un[key]))

    with open(csv3_name, 'w') as f:
        f.write("%s,%s\n"%('patches_kept_nb',patches_kept_nb))
        f.write("%s,%s\n"%('total_patches',total_patches))
        f.write("%s,%s\n"%('labelled_cover',labelled_cover))
