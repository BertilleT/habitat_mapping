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

masks_path = Path('../original_data/data_1/zone26/patch_64_msk_0_0')

def count_labelled_patches_threshold(per_path):
    if Path(per_path).exists():
        # print(f'{per_path} exists. Loading...')
        with open(per_path, mode='r') as infile:
            reader = csv.reader(infile)
            labels_patch_per = {rows[0]:rows[1] for rows in reader}
            # turn into int
            labels_patch_per = {k: float(v) for k, v in labels_patch_per.items()}
    else:   
        patches_masks = list(masks_path.rglob('*.tif'))
        labels_patch_per = {}
        processing_i = 0 
        for p in patches_masks:
            processing_i += 1
            with rasterio.open(p) as src:
                unique, counts = np.unique(src.read(1), return_counts=True)
                labbelled_pixels = 0
                for index, my_class in enumerate(unique):
                    if my_class != 0:
                        labbelled_pixels += counts[index]
                labels_patch_per[p] = labbelled_pixels / np.sum(counts)
                #print('Patch number ' + str(processing_i) + ' over ' + str(len(patches_masks)) + ' processed.')
        with open(per_path, 'w') as f:
            for key in labels_patch_per.keys():
                f.write("%s,%s\n"%(key,labels_patch_per[key]))
    return labels_patch_per

thresholds = [1, 0.75, 0.5, 0.25, 0.15, 0.1, 0.05]
for t in thresholds: 
    print('---------------------------------------------------------------------------------')
    print(f'For threshold {t}')
    per_path = f'../csv/per_labels_by_patch_{t}.csv'
    all_labels_patch_per = count_labelled_patches_threshold(per_path) 
    labelled_patches = {k: v for k, v in all_labels_patch_per.items() if v > 0}
    unlabelled_patches_per = round((len(all_labels_patch_per) - len(labelled_patches)) * 100 / len(all_labels_patch_per))
    threshold_labelled_patches = {k: v for k, v in all_labels_patch_per.items() if v >= t}
    per_labeled_patches_threshold = round(len(threshold_labelled_patches) * 100 / len(labelled_patches))
    per_l_t_2 = round(len(threshold_labelled_patches) * 100 / len(all_labels_patch_per))
    print(f'{unlabelled_patches_per} % of the patches are unlabelled.')
    print(f'Among the labeled patches, {per_labeled_patches_threshold} % have at least {t*100} % of labelled pixels.')
    print(f'It represents {per_l_t_2} % of all the patches.')

# HETEROGENEITY OF PATCHES 64_64
t = 1
print(f'For threshold {t}')
per_path = f'../csv/per_labels_by_patch_{t}.csv'
with open(per_path, mode='r') as infile:
    reader = csv.reader(infile)
    all_labels_patch_per = {rows[0]:rows[1] for rows in reader}
    all_labels_patch_per = {k: float(v) for k, v in all_labels_patch_per.items()}

nb_all_patches = len(all_labels_patch_per)
threshold_labelled_patches = {k: v for k, v in all_labels_patch_per.items() if v >= t}
nb_labelled_patchs_th = len(threshold_labelled_patches)

patches_masks = list(masks_path.rglob('*.tif'))
labels_patch_per = {}
processing_i = 0 
nb_labels_patchs = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0
}

for p in threshold_labelled_patches.keys():
    processing_i += 1
    with rasterio.open(p) as src:
        unique, counts = np.unique(src.read(1), return_counts=True)
        # remove all 0 from unique
        unique = unique[unique != 0]
        nb_labels_patchs[len(unique)] += 1

print(nb_labels_patchs)
# for each value from nb_labels_patchs, *100 and / nb_labelled_patchs
nb_labelled_patchs_per = {k: round((v * 100) / nb_labelled_patchs_th) for k, v in nb_labels_patchs.items()}
print(nb_labelled_patchs_per)