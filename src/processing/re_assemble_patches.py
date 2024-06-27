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

 # Function to reassemble patches into a full image
def reassemble_patches(patches, i_indices, j_indices, patch_size=256):
    max_i = max(i_indices)
    min_i = min(i_indices)
    max_j = max(j_indices)
    min_j = min(j_indices)
    
    # Calculate the full image dimensions
    full_image_height = (max_i - min_i + 1) * patch_size
    full_image_width = (max_j - min_j + 1) * patch_size
    
    # Initialize the full image with 8 (assuming single-channel masks)
    #full_image = np.zeros((full_image_height, full_image_width), dtype=patches[0].dtype)
    full_image = np.ones((full_image_height, full_image_width), dtype=patches[0].dtype) * 8
    
    # Place each patch in the correct location in the full image
    for patch, i, j in zip(patches, i_indices, j_indices):
        row_start = (i - min_i) * patch_size
        row_end = row_start + patch_size
        col_start = (j - min_j) * patch_size
        col_end = col_start + patch_size 
        full_image[row_start:row_end, col_start:col_end] = patch[:patch_size, :patch_size]
    
    print(np.unique(full_image))
    return full_image

zones = ['zone1_0_0', 'zone10_0_1', 'zone20_0_0', 'zone30_0_0', 'zone63_0_0', 'zone78_10_8', 'zone78_16_10']

for zone in ['zone20_0_0']:#zones: 
    # Setup
    tif_paths = sorted(list(Path(f'../../data/patch256/msk/{zone}').rglob('*.tif')))

    # Make a DataFrame with the TIFF paths and extract row and column indices
    tif_paths_df = pd.DataFrame(tif_paths, columns=['tif_path'])
    tif_paths_df['row_index'] = tif_paths_df.tif_path.apply(lambda x: int(x.stem.split('_')[-2]))
    tif_paths_df['col_index'] = tif_paths_df.tif_path.apply(lambda x: int(x.stem.split('_')[-1]))

    min_row = tif_paths_df.row_index.min()
    max_row = tif_paths_df.row_index.max()
    min_col = tif_paths_df.col_index.min()
    max_col = tif_paths_df.col_index.max()

    # Read patches and store them along with their indices
    patches = []
    patches_pixels = []

    i_indices = []
    j_indices = []

    temp_patches = []
    pixel_temp_patches = []
    temp_i = []
    temp_j = []
    for tif_path in tif_paths:    
        patch = tiff.imread(tif_path)[:, :, :, 0]
        group_under_represented_classes = {0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 2, 6: 5, 7: 3, 8: 4, 9: 5}
        group_under_represented_classes_uint8 = {np.uint8(k): np.uint8(v) for k, v in group_under_represented_classes.items()}
        patch = np.vectorize(group_under_represented_classes_uint8.get)(patch)
        unique = np.unique(patch)
        patches_pixels.append(patch[0, :, :])
        # get dim of patch
        if len(unique) > 1:
            # if multiple classes in the patch, then rows of 6 and 7 to have lines
            striped = np.ones(patch.shape) * 6
            my_list = list(range(0, striped.shape[2], 64))
            for i in my_list:
                striped[0, :, i:i+32] = 7  
            #get size of patch
            patches.append(striped[0, :, :])
            # save patch png as test
            '''my_classes = np.unique(striped)
            customs_colors = [my_colors_map[i] for i in my_classes]
            my_cmap = plt.cm.colors.ListedColormap(customs_colors)
            plt.imshow(striped[0, :, :], cmap = my_cmap)
            plt.axis('off')
            plt.savefig(f'../../imgs/reassembled/test.png', bbox_inches='tight', pad_inches=0)
            '''
        else:
            
            #type of patch
            patches.append(patch[0, :, :])

        splitted = tif_path.stem.split('_')
        i = int(splitted[-2])
        j = int(splitted[-1]) 
        i_indices.append(i)
        j_indices.append(j)
        if j in range(8, 10):
            if i in range(7, 8):
                temp_i.append(i)
                temp_j.append(j)
                pixel_temp_patches.append(patch[0, :, :])

                if len(unique) > 1:
                    print('striped')
                    # print unique values in patch
                    print(np.unique(striped))
                    temp_patches.append(striped[0, :, :])
                else:
                    temp_patches.append(patch[0, :, :])

    # Reassemble the full image with temp
    print('Reassembling the full image, pixel level')
    reassembled_image_pixel = reassemble_patches(pixel_temp_patches, temp_i, temp_j)
    my_classes_pixels = np.unique(reassembled_image_pixel)
    print('Unique values in reassembled image:', my_classes_pixels)
    customs_colors_pixels = [my_colors_map[i] for i in my_classes_pixels]
    my_cmap_pixels = plt.cm.colors.ListedColormap(customs_colors_pixels)
    plt.imshow(reassembled_image_pixel, cmap = my_cmap_pixels)
    plt.axis('off')
    plt.savefig(f'../../imgs/reassembled/png/{zone}_pixel_level_temp_v2.png', bbox_inches='tight', pad_inches=0)

    # Reassemble the full image with temp at patch level
    print('Reassembling the full image, patch level')
    reassembled_image = reassemble_patches(temp_patches, temp_i, temp_j)
    my_classes = np.unique(reassembled_image)
    print('Unique values in reassembled image:', my_classes)
    customs_colors = [my_colors_map[i] for i in my_classes]
    my_cmap = plt.cm.colors.ListedColormap(customs_colors)
    plt.imshow(reassembled_image, cmap = my_cmap)
    plt.axis('off')
    plt.savefig(f'../../imgs/reassembled/png/{zone}_patch_level_temp_v2.png', bbox_inches='tight', pad_inches=0)
    

    '''    # Reassemble the full image
    print('Reassembling the full image, pixel level')
    reassembled_image_pixel = reassemble_patches(patches_pixels, i_indices, j_indices)
    my_classes_pixels = np.unique(reassembled_image_pixel)
    customs_colors_pixels = [my_colors_map[i] for i in my_classes_pixels]
    my_cmap_pixels = plt.cm.colors.ListedColormap(customs_colors_pixels)
    plt.imshow(reassembled_image_pixel, cmap = my_cmap_pixels)
    plt.axis('off')
    plt.savefig(f'../../imgs/reassembled/png/{zone}_pixel_level.png', bbox_inches='tight', pad_inches=0)

    print('Reassembling the full image, patch level')
    # Reassemble the full image
    reassembled_image = reassemble_patches(patches, i_indices, j_indices)
    output_path = f'../../imgs/reassembled/tif/{zone}_patch_level.tif'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)  
    tiff.imwrite(output_path, reassembled_image)
    my_classes = np.unique(reassembled_image)
    customs_colors = [my_colors_map[i] for i in my_classes]
    my_cmap = plt.cm.colors.ListedColormap(customs_colors)
    plt.imshow(reassembled_image, cmap = my_cmap)
    plt.axis('off')
    plt.savefig(f'../../imgs/reassembled/png/{zone}_patch_level.png', bbox_inches='tight', pad_inches=0)'''