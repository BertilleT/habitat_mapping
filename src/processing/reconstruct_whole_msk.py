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

from utils import *
from settings import *

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import albumentations as A
from albumentations.pytorch import ToTensorV2

import tifffile as tiff


my_colors_map = {
    0: '#789262',  # Vert olive
    1: '#555555',  # Gris
    2: '#006400',  # Vert foncé
    3: '#00ff00',  # Vert vif
    4: '#ff4500',  # Rouge
    5: '#8a2be2',  # Violet
    6: '#000000',  # Noir
    7: '#c7c7c7',  # Gris
    8: '#ffffff'  # Blanc
}
customs_color = list(my_colors_map.values())
bounds = list(my_colors_map.keys())
my_cmap = plt.cm.colors.ListedColormap(customs_color)
my_norm = plt.cm.colors.BoundaryNorm(bounds, my_cmap.N)

# Function to reassemble patches into a full image
def reassemble_patches(patches, i_indices, j_indices, patch_size=256):
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
    print(patches[0].shape)
    full_image = np.ones((full_image_height, full_image_width), dtype=patches[0].dtype) * 8
    
    # Place each patch in the correct location in the full image
    for patch, i, j in zip(patches, i_indices, j_indices):
        row_start = (i - min_i) * patch_size
        row_end = row_start + patch_size
        col_start = (j - min_j) * patch_size
        col_end = col_start + patch_size 
        full_image[row_start:row_end, col_start:col_end] = patch[:patch_size, :patch_size]
        
    return full_image


#zone = 'zone1_0_0'
#zone = 'zone100_0_0'
zone = 'zone133_0_0'
patch_size = patch_level_param['patch_size']
msk_folder = data_loading_settings['msk_folder']
msk_paths = list(msk_folder.rglob('*.tif'))


# Keep only masks with zone in it
msk_paths = [msk_path for msk_path in msk_paths if zone in msk_path.stem]
dataset = EcomedDataset(msk_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], normalisation = data_loading_settings['normalisation'], task = model_settings['task'], my_set = "test", labels = model_settings['labels'], path_mask_name = True)
# Load the model
model = models.resnet18(weights=model_settings['pre_trained'])
num_channels = model_settings['in_channels']
num_filters = model.conv1.out_channels
kernel_size = model.conv1.kernel_size
stride = model.conv1.stride
padding = model.conv1.padding
conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
if model_settings['pre_trained']:
    mean_weights = model.conv1.weight.data.mean(dim=1, keepdim=True)
    conv1.weight.data = torch.cat([model.conv1.weight.data, mean_weights], dim=1)
    print('Pretrained weights loaded')
model.conv1 = conv1
if model_settings['labels'] == 'single':
    model.fc = nn.Linear(512, model_settings['classes'])
elif model_settings['labels'] == 'multi':
    model.fc = nn.Linear(512, model_settings['classes'] + 1)
model.load_state_dict(torch.load(model_settings['path_to_best_model'], map_location=torch.device('cpu')))
model.eval()

# Predict all the masks from the dataset
predictions = []
i_indices = []
j_indices = []
img_classif_patches = []
pixels_classif_patches = []
predicted_patches = []

for i in range(len(dataset)):
    img, msk, tif_path = dataset[i]
    # INDICES OF THE PATCH IN THE FULL IMAGE
    splitted = tif_path.stem.split('_')
    if patch_size == 256:
        i = int(splitted[-2])
        j = int(splitted[-1])
    elif patch_size == 128:
        i = int(splitted[-4])
        j = int(splitted[-3])
    i_indices.append(i)
    j_indices.append(j)

    if patch_size == 256:
        original_patch = tiff.imread(tif_path)[:, :, :, 0] # dtype = uint8 (confirmed by print(patch.dtype))
    elif patch_size == 128:
        original_patch = tiff.imread(tif_path)[0,:,:]
        # dd a one dimension to the patch
        original_patch = np.expand_dims(original_patch, axis=0)

    group_under_represented_classes = {0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 2, 6: 5, 7: 3, 8: 4, 9: 5}
    # print unique values and their nb in original_patch
    unique, counts = np.unique(original_patch, return_counts=True)
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

    # PREDICTION
    # img to tensor
    img = torch.unsqueeze(torch.tensor(img), 0)
    with torch.no_grad():
        pred = model(img)
        #remove first dimension
    pred = torch.sigmoid(pred)
    # heteroneity to 1 if last value of pred vector is > 0.5
    heterogeneity = 1 if pred[0, -1].item() > 0.5 else 0
    #if last value of pred vector is 1
    if heterogeneity == 1:
        # if multiple classes in the patch, then strip the patch
        pred_striped = np.ones(patch.shape) * 6 # striped is a numpy array of (1, 256, 256)
        # Every 64 pixels in column, we set the value to 7 for 32 columns of pixels
        pred_my_list = list(range(0, pred_striped.shape[2], 64))
        for i in pred_my_list:
            pred_striped[0, :, i:i+32] = 7
        # turn to uint8
        pred_striped = pred_striped.astype(np.uint8) #  previously, striped was of type float64
        # Striped is an array of (1, 256, 256) with 6 and 7 values, uint8 type
        predicted_patches.append(pred_striped[0, :, :])   
    else:
        # remove last value of pred vector
        pred = pred[:, :-1]
        # get the class with the highest probability
        pred = torch.argmax(pred, dim=1).item()
        # create a numpy array of (256, 256) full of the predicted class
        # turn pred which is an int to uint8
        pred = np.uint8(pred)
        array_pred = np.ones(patch.shape) * pred
        # to uint8
        array_pred = array_pred.astype(np.uint8)
        predicted_patches.append(array_pred[0, :, :])

# Reassemble the patches
print('Reassembling the patches...')
reassembled_image_pixel = reassemble_patches(pixels_classif_patches, i_indices, j_indices, patch_size=patch_size)
reassembled_image = reassemble_patches(img_classif_patches, i_indices, j_indices, patch_size=patch_size)
predicted_image = reassemble_patches(predicted_patches, i_indices, j_indices, patch_size=patch_size)

# Create a figure with 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Display the images in the subplots
axs[0].imshow(reassembled_image_pixel, cmap=my_cmap, norm=my_norm)
axs[0].axis('off')
axs[0].set_title('Pixel Level')

axs[1].imshow(reassembled_image, cmap=my_cmap, norm=my_norm)
axs[1].axis('off')
axs[1].set_title('Image Level')

axs[2].imshow(predicted_image, cmap=my_cmap, norm=my_norm)
axs[2].axis('off')
axs[2].set_title('Predicted')

# Save the combined figure
plt.savefig(f'../../imgs/reassembled_pred/{patch_size}/{zone}_combined.png', bbox_inches='tight', pad_inches=0)
plt.show()