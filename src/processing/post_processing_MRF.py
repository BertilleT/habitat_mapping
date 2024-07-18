## PSEUDO-CODE
## IMPROVE HOMOGENOUS PATCHES
## F1 ONLY ON HOMOGENOUS PATCHES

import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
from settings import data_loading_settings, model_settings, patch_level_param, training_settings, plotting_settings
from utils.data_utils import EcomedDataset
from utils.plotting_utils import plot_reassembled_patches
from pathlib import Path
import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# remove limit on df display size of the largeur of a column with head()
pd.set_option('display.max_colwidth', None)

new_colors_maps = {k: v for k, v in plotting_settings['my_colors_map'].items()}
new_colors_maps[6] = '#000000'  # Noir
new_colors_maps[7] = '#c7c7c7'  # Gris
new_colors_maps[8] = '#ffffff'  # Blanc

customs_color = list(new_colors_maps.values())
bounds = list(new_colors_maps.keys())
my_cmap = plt.cm.colors.ListedColormap(customs_color)
my_norm = plt.cm.colors.BoundaryNorm(bounds, my_cmap.N)
beta = 0.5

def energie(i, j):
    """
    Compute the energy of a patch at position i, j
    """
    energies = []
    neighbors = [(i + x, j + y) for x in (-1, 0, 1) for y in (-1, 0, 1) if not (x == 0 and y == 0)]
    # keep rows from homogenous_df with(i,j) from neighbors
    neighbor_patches = homogenous_df[homogenous_df.apply(lambda row: (row['i'], row['j']) in neighbors, axis=1)]
    # get the list of classes from neibors
    neighbor_classes = neighbor_patches['predicted_class'].tolist()
    #select probability_vector from homogenous_df with index i and j
    probability_vector = homogenous_df[(homogenous_df['i'] == i) & (homogenous_df['j'] == j)]['probability_vector'].values[0]
    #for my_class, p in probability_vector: 
    for my_class, p in enumerate(probability_vector): 
        nb_neigboors_diff = 0 
        for neighbor_class in neighbor_classes: 
            if my_class != neighbor_class:
                nb_neigboors_diff += 1
    
        E = -np.log(p) + beta * nb_neigboors_diff
        energies.append(E)
    # return as array
    return np.array(energies)


zone = 'zone133_0_0'

msk_paths = list(data_loading_settings['msk_folder'].rglob('*.tif'))
msk_paths = [msk_path for msk_path in msk_paths if zone in msk_path.stem]

dataset = EcomedDataset(msk_paths, Path(f'../../data/patch64/img/'), level=1, channels = 4, normalisation = "channel_by_channel", task = 'image_classif', my_set = "test", labels = "multi", path_mask_name = True)

model = models.resnet18(weights=False)
num_channels = 4
# Extract the first conv layer's parameters
num_filters = model.conv1.out_channels
kernel_size = model.conv1.kernel_size
stride = model.conv1.stride
padding = model.conv1.padding
conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
model.conv1 = conv1
model.fc = nn.Linear(512, model_settings['classes']) 

model.load_state_dict(torch.load(Path(f'../../results/resnet18_64_l1/random_shuffling/all/resnet18_multi_label_64_random_10epochs_bs256/models/unet_intermed_epoch10.pt')))
print('Model loaded')

for param in model.parameters():
    # to cpu
    param.to('cpu')

i_indices, j_indices, true_classes, true_heterogenity, predicted_classes, predicted_heterogenity, probability_vectors = plot_reassembled_patches(zone, model, dataset, patch_level_param['patch_size'], training_settings['alpha1'], None, None, plotting_settings['re_assemble_patches_path'][:-4], post_processing = True)
# FIRST STEP: 
# BUILD A DATAFRAME WITH COLUMN I, J, TRUE CLASS, TRUE HETEROGENITY, CLASS PREDICTED, HETEROGENITY PREDICTED
df = pd.DataFrame({'i': i_indices, 'j': j_indices, 'true_class': true_classes, 'true_heterogenity': true_heterogenity, 'predicted_class': predicted_classes, 'predicted_heterogenity': predicted_heterogenity, 'probability_vector': probability_vectors})
# probability vector must have only 6 items. Keep only the 6 first items
df['probability_vector'] = df['probability_vector'].apply(lambda x: [float(t.item()) for t in x[0][:6]])
# print item with i = 0 and j = 36
print('i0 and j36', df[(df['i'] == 0) & (df['j'] == 36)])
# FILTER TO KEEP ONLY PATCHES PREDICTED AS HOMOGENOUS WHEN PREDICTED_HETEROGENITY IS 0
homogenous_df = df[df['predicted_heterogenity'] == 0]
true_classes = homogenous_df['true_class'].tolist()
predicted_classes = homogenous_df['predicted_class'].tolist()
f1_by_class = f1_score(true_classes, predicted_classes, average=None)
print('f1_by_class: ', f1_by_class)


# SECOND STEP
# LOOP ON DATAFRAME,
# for i, j in homogenous_df: 
for i, j in zip(homogenous_df['i'], homogenous_df['j']):
    old_class = homogenous_df[(homogenous_df['i'] == i) & (homogenous_df['j'] == j)]['predicted_class'].values[0]
    energies = energie(i, j) # ernergie will be a vector of 5 classes
    # select the index of the vector with minimum energy
    new_class = np.argmin(energies) # return the index of the minimum value
    # update predicted class from homogenous_df with index i and j with new_class
    homogenous_df.loc[(homogenous_df['i'] == i) & (homogenous_df['j'] == j), 'predicted_class'] = new_class
    if old_class != new_class:
        print(f'Class {old_class} has been changed to {new_class}')

post_predicted_classes = homogenous_df['predicted_class'].tolist()
post_f1_by_class = f1_score(true_classes, post_predicted_classes, average=None)

print('post_f1_by_class', post_f1_by_class)

# check homogenous_df i = 0 and j = 36 ? 
# reassamble the image with the new predicted classes
plot_reassembled_patches(zone, model, dataset, patch_level_param['patch_size'], training_settings['alpha1'], my_cmap, my_norm, plotting_settings['re_assemble_patches_path'][:-4], False, True, homogenous_df)