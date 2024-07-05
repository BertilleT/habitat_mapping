import numpy as np
from pathlib import Path
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import gc
import torchvision.models as models

from datetime import datetime

from utils.data_utils import *
from utils.plotting_utils import *
from utils.train_val_test_utils import *
from settings import *

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import albumentations as A
from albumentations.pytorch import ToTensorV2




new_colors_maps = {k: v for k, v in plotting_settings['my_colors_map'].items()}
new_colors_maps[6] = '#000000'  # Noir
new_colors_maps[7] = '#c7c7c7'  # Gris
new_colors_maps[8] = '#ffffff'  # Blanc

customs_color = list(new_colors_maps.values())
bounds = list(new_colors_maps.keys())
my_cmap = plt.cm.colors.ListedColormap(customs_color)
my_norm = plt.cm.colors.BoundaryNorm(bounds, my_cmap.N)


# Function to reassemble patches into a full image
def reassemble_patches(patches, i_indices, j_indices, patch_size):
    """
    Reassemble image patches into a full image.

    Parameters:
    patches (list of np.array): List of image patches (each patch is a 2D numpy array).
    i_indices (list of int): List of row indices for the patches.
    j_indices (list of int): List of column indices for the patches.
    patch_size (int): Size of each patch (default is 256).

    Returns:
    np.array: The reassembled image.
    """
    # Determine the size of the full image
    max_i = max(i_indices)
    min_i = min(i_indices)
    max_j = max(j_indices)
    min_j = min(j_indices)
    
    # get nb of patches from min max min max indices
    nb_patches = (max_i - min_i + 1) * (max_j - min_j + 1)
    # Calculate the full image dimensions
    full_image_height = (max_i - min_i + 1) * patch_size
    full_image_width = (max_j - min_j + 1) * patch_size
    full_image = np.ones((full_image_height, full_image_width), dtype=patches[0].dtype) * 8
    
    # Place each patch in the correct location in the full image
    for patch, i, j in zip(patches, i_indices, j_indices):
        row_start = (i - min_i) * patch_size
        row_end = row_start + patch_size
        col_start = (j - min_j) * patch_size
        col_end = col_start + patch_size 
        full_image[row_start:row_end, col_start:col_end] = patch[:patch_size, :patch_size]
        
    return full_image

def plot_reassembled_patches(zone, model, dataset, patch_size, alpha1, my_cmap, my_norm, path_to_save):
    # Predict all the masks from the dataset
    predictions = []
    i_indices = []
    j_indices = []
    k_indices = []
    l_indices = []
    img_classif_patches = []
    pixels_classif_patches = []
    predicted_patches = []

    for i in range(len(dataset)):
        img, msk, tif_path = dataset[i]
        # print shape
        # INDICES OF THE PATCH IN THE FULL IMAGE
        splitted = tif_path.stem.split('_')
        if patch_size == 256 or patch_size == 64:
            i = int(splitted[-2])
            j = int(splitted[-1])
        elif patch_size == 128:
            i = int(splitted[-4])
            j = int(splitted[-3])
            k = splitted[-2]
            l = splitted[-1]
        i_indices.append(i)
        j_indices.append(j)

        if patch_size == 256:
            original_patch = tiff.imread(tif_path)[:, :, :, 0] # dtype = uint8 (confirmed by print(patch.dtype))

        if patch_size == 64:
            original_patch = tiff.imread(tif_path)[0, :, :] # dtype = uint8 (confirmed by print(patch.dtype))
            original_patch = np.expand_dims(original_patch, axis=0)
            #original_patch = np.expand_dims(original_patch, axis=0)

        elif patch_size == 128:
            k_indices.append(k)
            l_indices.append(l)
            original_patch = tiff.imread(tif_path)[0,:,:]
            # dd a one dimension to the patch
            original_patch = np.expand_dims(original_patch, axis=0)

        group_under_represented_classes = {0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 2, 6: 5, 7: 3, 8: 4, 9: 5}
        # print unique values and their nb in original_patch
        unique, counts = np.unique(original_patch, return_counts=True)
        #print dtype
        group_under_represented_classes_uint8 = {np.uint8(k): np.uint8(v) for k, v in group_under_represented_classes.items()}
        patch = np.vectorize(group_under_represented_classes_uint8.get)(original_patch)

        # ORIGINAL PATCHES AT PIXEL LEVEL
        pixels_classif_patches.append(patch[0, :, :]) # we obtain a numpy array of (256, 256) of uint8 type

        # PATCHES IWTH ONE CLASS PER PATCH 
        if len(np.unique(patch)) > 1:
            # if multiple classes in the patch, then strip the patch
            striped = np.ones(patch.shape) * 6 # striped is a numpy array of (1, 256, 256)
            # Every 64 pixels in column, we set the value to 7 for 32 columns of pixels
            my_list = list(range(0, striped.shape[2], 64))
            for i in my_list:
                striped[0, :, i:i+32] = 7
            # turn to uint8
            striped = striped.astype(np.uint8) #  previously, striped was of type float64
            # Striped is an array of (1, 256, 256) with 6 and 7 values, uint8 type
            img_classif_patches.append(striped[0, :, :])            
        else:
            img_classif_patches.append(patch[0, :, :])

    print('Reassembling the patches...')
    reassembled_image_pixel = reassemble_patches(pixels_classif_patches, i_indices, j_indices, patch_size=64)
    reassembled_image = reassemble_patches(img_classif_patches, i_indices, j_indices, patch_size=64)

    # Create a figure with 1 row and 3 columns
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Display the images in the subplots
    axs[0].imshow(reassembled_image_pixel, cmap=my_cmap, norm=my_norm)
    axs[0].axis('off')
    axs[0].set_title('Pixel Level')

    axs[1].imshow(reassembled_image, cmap=my_cmap, norm=my_norm)
    axs[1].axis('off')
    axs[1].set_title('Image Level')

    plt.savefig(f'test_patch_64_{zone}_re_assembled', bbox_inches='tight', pad_inches=0)
    plt.close()


zones = ['zone1_0_0', 'zone100_0_0', 'zone133_0_0']
for zone in zones: 
    msk_paths = list(Path(f'../../data/patch64/msk/').rglob('*.tif'))
    msk_paths = [msk_path for msk_path in msk_paths if zone in msk_path.stem]

    dataset = EcomedDataset(msk_paths,Path('../../data/patch64/img/'), 1, 4, 'channel_by_channel', task = "image_classif", my_set = "test", labels = "multi", path_mask_name = True)
    plot_reassembled_patches(zone, None, dataset, 64, 0.6, my_cmap, my_norm, None)
