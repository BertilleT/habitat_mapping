## PSEUDO-CODE
## IMPROVE HOMOGENOUS PATCHES
## F1 ONLY ON HOMOGENOUS PATCHES

import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
from settings import data_loading_settings, model_settings, patch_level_param, training_settings, plotting_settings

beta = 0.5

def energie(i, j): 
    energies = []
    neighbors = [(i + x, j + y) for x in (-1, 0, 1) for y in (-1, 0, 1) if not (x == 0 and y == 0)]
    # keep rows from homo_df with i,j from neighbors
    neighbor_patches = homo_df[homo_df.apply(lambda row: (row['i'], row['j']) in neighbors, axis=1)]
    # get the list of classes from neibors
    neighbor_classes = homo_df['predicted_class'].tolist()
    #select probability_vector from homo_df with index i and j
    probability_vector = homo_df[(homo_df['i'] == i) & (homo_df['j'] == j)]['probability_vector'].values[0]
    for my_class, p in probability_vector: 
        nb_neigboors_diff = 0 
        for neighbor_class in neighbor_classes: 
            if my_class != neighbor_class:
                nb_neigboors_diff += 1
        E = -np.log(p) + beta * nb_neigboors_diff
        energies.append(E)
    return energies


zone = ['zone133_0_0']

msk_paths = list(data_loading_settings['msk_folder'].rglob('*.tif'))
msk_paths = [msk_path for msk_path in msk_paths if zone in msk_path.stem]

dataset = EcomedDataset(msk_paths, Path(f'../../data/patch64/msk/'), level=1, 4, normalisation = "channel_by_channel", task = 'image_classif', my_set = "test", labels = "multi", path_mask_name = True)

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

model.load_state_dict(Path(f'../../results/resnet18_64_l1/random_shuffling/all/resnet18_multi_label_64_random_10epochs_bs256/models/unet_intermed_epoch10.pt'))
print('Model loaded')

for param in model.parameters():
    # to cpu
    param.to('cpu')


plot_reassembled_patches(zone, model, dataset, patch_level_param['patch_size'], training_settings['alpha1'], my_cmap, my_norm, plotting_settings['re_assemble_patches_path'][:-4])
# FIRST STEP: 
# BUILD A DATAFRAME WITH COLUMN I, J, TRUE CLASS, TRUE HETEROGENITY, CLASS PREDICTED, HETEROGENITY PREDICTED
df = pd.DataFrame(columns=['i', 'j', 'true_class', 'true_heterogenity', 'predicted_class', 'heterogenity', 'probability_vector']) # probability vector is with 6 items. 

# FILTER TO KEEP ONLY PATCHES PREDICTED AS HOMOGENOUS
homo_df = df[df['heterogenity'] == 0]
# probability vector must have only 6 items

true_classes = homo_df['true_class'].tolist()
predicted_classes = homo_df['predicted_class'].tolist()
f1_by_class = f1_score(true_classes, predicted_classes, average=None)
print('f1_by_class: ', f1_by_class)
# SECOND STEP
# LOOP ON DATAFRAME,
for i, j in homo_df: # 
    energies = energie(i, j) # ernergie will be a vector of 5 classes
    # select the index of the vector with minimum energy
    new_class = index_min(energies)
    # update predicted class from homo_df with index i and j with new_class
    homo_df.loc[(homo_df['i'] == i) & (homo_df['j'] == j), 'predicted_class'] = new_class

post_predicted_classes = homo_df['predicted_class'].tolist()
post_f1_by_class = f1_score(true_classes, post_predicted_classes, average=None)

print('post_f1_by_class', post_f1_by_class)