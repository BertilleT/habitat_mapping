# i = row, j = col
# take zone1_0_0 for example
# tif images in this zone are like img_zone1_0_0_patch_2_12.tif. 2 is i and 12 is j

# plot the image by using the following code:

# create a grid with each item has 256 by 256 pixels
#sizee of the grids depends of max and min i, and max and min j

'''from pathlib import Path
import numpy as np
import pandas as pd

zone = 'zone1_0_0'
tif_paths = sorted(list(Path(f'../../data/patch256/msk/{zone}').rglob('*.tif')))

#make a df with the tif paths
#split the tif paths to get the row and col indexes
tif_paths_df = pd.DataFrame(tif_paths, columns=['tif_path'])
tif_paths_df['row_index'] = tif_paths_df.tif_path.apply(lambda x: int(x.stem.split('_')[-2]))
tif_paths_df['col_index'] = tif_paths_df.tif_path.apply(lambda x: int(x.stem.split('_')[-1]))
#remove limit on number of characters displayed
pd.set_option('display.max_colwidth', None)
print(tif_paths_df)

min_row = tif_paths_df.row_index.min()
max_row = tif_paths_df.row_index.max()
min_col = tif_paths_df.col_index.min()
max_col = tif_paths_df.col_index.max()

print(min_row, max_row, min_col, max_col)'''



from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

my_colors_map = {
    0: '#789262',  # Vert olive
    1: '#555555',  # Gris
    2: '#006400',  # Vert foncé
    3: '#00ff00',  # Vert vif
    4: '#ff4500',  # Rouge
    5: '#8a2be2',  # Violet, 6 noir, 7 gris
    6: '#000000',  # Noir
    7: '#808080',  # Gris
    8: '#ffffff'  # Blanc
}

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
    
    # Calculate the full image dimensions
    full_image_height = (max_i - min_i + 1) * patch_size
    full_image_width = (max_j - min_j + 1) * patch_size
    
    # Initialize the full image with zeros (assuming single-channel masks)
    #full_image = np.zeros((full_image_height, full_image_width), dtype=patches[0].dtype)
    #intialize it with 100 instead of 0
    full_image = np.ones((full_image_height, full_image_width), dtype=patches[0].dtype) * 8
    
    # Place each patch in the correct location in the full image
    for patch, i, j in zip(patches, i_indices, j_indices):
        row_start = (i - min_i) * patch_size
        row_end = row_start + patch_size
        col_start = (j - min_j) * patch_size
        col_end = col_start + patch_size
        full_image[row_start:row_end, col_start:col_end] = patch
    
    return full_image

# Setup
zone = 'zone1_0_0'
tif_paths = sorted(list(Path(f'../../data/patch256/msk/{zone}').rglob('*.tif')))

# Make a DataFrame with the TIFF paths and extract row and column indices
tif_paths_df = pd.DataFrame(tif_paths, columns=['tif_path'])
tif_paths_df['row_index'] = tif_paths_df.tif_path.apply(lambda x: int(x.stem.split('_')[-2]))
tif_paths_df['col_index'] = tif_paths_df.tif_path.apply(lambda x: int(x.stem.split('_')[-1]))

# Remove limit on number of characters displayed
pd.set_option('display.max_colwidth', None)
print(tif_paths_df)

min_row = tif_paths_df.row_index.min()
max_row = tif_paths_df.row_index.max()
min_col = tif_paths_df.col_index.min()
max_col = tif_paths_df.col_index.max()

print(min_row, max_row, min_col, max_col)

# Read patches and store them along with their indices
patches = []
i_indices = []
j_indices = []

for tif_path in tif_paths:
    #read first channel of the tif image
    
    patch = tiff.imread(tif_path)[:, :, :, 0]
    # get unique values in the patch
    unique = np.unique(patch)
    print(unique)
    if len(unique) > 1:
        # set ALL values to 6. I get a patch full of 6
        patch = np.ones(patch.shape) * 6
        my_list = list(range(0, patch.shape[2], 60))
        print('my_list', my_list)
        for i in my_list:
            print('i', i)
            print('i+10', i+30)
            if i+10 > 255:
                patch[:, i:] = 7
            patch[:, i:i+30] = 7
            #remove limit on print array
            np.set_printoptions(threshold=np.inf)
        
        '''#print('patch', patch)
        #plot the patch
        #[1, 256, 256] rmeove 1
        patch = patch[0]
        plt.imshow(patch, cmap='gray')
        plt.axis('off')
        #savefig
        plt.savefig(f'patch_test.png', bbox_inches='tight', pad_inches=0)
        #break
        break'''

    splitted = tif_path.stem.split('_')
    i = int(splitted[-2])
    j = int(splitted[-1])
    patches.append(patch)
    i_indices.append(i)
    j_indices.append(j)

# Reassemble the full image
reassembled_image = reassemble_patches(patches, i_indices, j_indices)

# Save the reassembled image
output_path = f'../../data/reassembled/{zone}_reassembled_v2.tif'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
tiff.imwrite(output_path, reassembled_image)
print(f'Reassembled image saved to {output_path}')

#plot the tif avd save as png
import matplotlib.pyplot as plt
#plot using colors from the dictionary
my_classes = np.unique(reassembled_image)
customs_colors = [my_colors_map[i] for i in my_classes]
my_cmap = plt.cm.colors.ListedColormap(customs_colors)

plt.imshow(reassembled_image, cmap=my_cmap)
plt.colorbar()
plt.axis('off')
plt.savefig(f'../../data/reassembled/{zone}_reassembled_v2.png', bbox_inches='tight', pad_inches=0)