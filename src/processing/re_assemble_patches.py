from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)

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
    full_image = np.ones((full_image_height, full_image_width), dtype=patches[0].dtype) * 8
    
    # Place each patch in the correct location in the full image
    for patch, i, j in zip(patches, i_indices, j_indices):
        row_start = (i - min_i) * patch_size
        row_end = row_start + patch_size
        col_start = (j - min_j) * patch_size
        col_end = col_start + patch_size 
        full_image[row_start:row_end, col_start:col_end] = patch[:patch_size, :patch_size]
        
    return full_image

zones = ['zone1_0_0', 'zone10_0_1', 'zone20_0_0', 'zone30_0_0', 'zone63_0_0', 'zone90_0_0', 'zone100_0_0', 'zone103_0_0', 'zone133_0_0', 'zone170_2_4']
patch_size = 128# 256 or 128
for zone in zones:
    print('-------------------------------------------------------------------')
    print(zone)
    print('-----------------')
    # Setup
    tif_paths = sorted(list(Path(f'../../data/patch{patch_size}/msk/{zone}').rglob('*.tif')))
    if len(tif_paths) == 0:
        print(f'No TIFF files found in {zone}.')
        continue

    # Read patches and store them along with their indices
    img_classif_patches = []
    pixels_classif_patches = []
    i_indices = []
    j_indices = []

    for tif_path in tif_paths:
        splitted = tif_path.stem.split('_')
        if patch_size == 256:
            i = int(splitted[-3])
            j = int(splitted[-2])
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
        pixels_classif_patches.append(patch[0, :, :]) # we obtain a numpy array of (256, 256) of uint8 type
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
        
    # Reassemble the full image
    reassembled_image_pixel = reassemble_patches(pixels_classif_patches, i_indices, j_indices, patch_size=patch_size)

    plt.imshow(reassembled_image_pixel, cmap = my_cmap, norm=my_norm)
    plt.axis('off')
    plt.savefig(f'../../imgs/reassembled/{patch_size}/{zone}_pixel_level.png', bbox_inches='tight', pad_inches=0)

    # Reassemble the full image
    reassembled_image = reassemble_patches(img_classif_patches, i_indices, j_indices, patch_size=patch_size)
    
    plt.imshow(reassembled_image, cmap = my_cmap, norm=my_norm)
    plt.axis('off')
    plt.savefig(f'../../imgs/reassembled/{patch_size}/{zone}_img_level.png', bbox_inches='tight', pad_inches=0)
    