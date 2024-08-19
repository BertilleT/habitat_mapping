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
from utils.resnet_model import *
from settings import *
from utils.post_processing import *
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

my_colors_map = {
    0: '#789262',  # Vert olive
    1: '#555555',  # Gris
    2: '#006400',  # Vert foncé
    3: '#00ff00',  # Vert vif
    4: '#ff4500',  # Rouge
    5: '#8a2be2',  # Violet
}

customs_color = list(my_colors_map.values())
# N
N = len(customs_color)
bounds = list(my_colors_map.keys())
my_cmap = plt.cm.colors.ListedColormap(customs_color)
my_norm = plt.cm.colors.BoundaryNorm(bounds, my_cmap.N) # 

'''
print('----------------------- UNet -----------------------')
print(f'Patch size: {patch_level_param["patch_size"]}')
print(f'Classification level: {patch_level_param["level"]}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #
print(f'Using device: {device}')
torch.cuda.empty_cache()
gc.collect()


## DATA
print('Loading data...')
print('Data loading settings:')
print(f'Splitting data: {data_loading_settings["split"]}')
print(f'Stratified: {data_loading_settings["stratified"]}')
print(f'Year: {data_loading_settings["year"]}')
print(f'Patches: {data_loading_settings["patches"]}')
print(f'Batch size: {data_loading_settings["bs"]}')
print(f'Normalisation: {data_loading_settings["normalisation"]}')

if data_loading_settings['data_augmentation']: 
    print('Data augmentation')
    transform_rgb = A.Compose([
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ToTensorV2(),
    ])

    transform_all_channels = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(0.01, 0.05), p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomResizedCrop(height=patch_level_param['patch_size'], width=patch_level_param['patch_size'], scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
        ToTensorV2(),
    ])
    transform = [transform_rgb, transform_all_channels]
else: 
    print('No data augmentation')
    transform = [None, None]


train_paths, val_paths, test_paths = load_data_paths(**data_loading_settings)
#print(f'Train: {len(train_paths)} images, Val: {len(val_paths)} images, Test: {len(test_paths)} images')
train_ds = EcomedDataset(train_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], transform = transform, normalisation = data_loading_settings['normalisation'], task = model_settings['task'], my_set = "train", labels = model_settings['labels'])
train_dl = DataLoader(train_ds, batch_size=data_loading_settings['bs'], shuffle=True)
# check size of one img and masks, and value of masks at level 1
img, msk = next(iter(train_dl))
val_ds = EcomedDataset(val_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], normalisation = data_loading_settings['normalisation'], task = model_settings['task'], my_set = "val", labels = model_settings['labels'])
val_dl = DataLoader(val_ds, batch_size=data_loading_settings['bs'], shuffle=False)
test_ds = EcomedDataset(test_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], normalisation = data_loading_settings['normalisation'], task = model_settings['task'], my_set = "test", labels = model_settings['labels'])
test_dl = DataLoader(test_ds, batch_size=data_loading_settings['bs'], shuffle=False)



sys.stdout.flush()
## MODEL
print('Creating model...')
print('Model settings:')
print(f'Pretrained: {model_settings["pre_trained"]}')
print(f'Classes: {model_settings["classes"]}')

print('The model is: ', model_settings['model'])
model = models.resnet18(weights=False)
num_channels = model_settings['in_channels']
# Extract the first conv layer's parameters
num_filters = model.conv1.out_channels
kernel_size = model.conv1.kernel_size
stride = model.conv1.stride
padding = model.conv1.padding
# Replace the first conv layer with a new one
model.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
# Replace the classifier head
model.fc = nn.Linear(512, model_settings['classes'])
# load the model
model.load_state_dict(torch.load(model_settings['path_to_best_model']))
# remove the classifier head to extract the features
model = nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()
print('Model created')
'''
### ------------------------------------- EXTRACT FEATURES FOR TRAIN DATA, CLASS BY CLASS

'''all_features_train = [[] for _ in range(6)]
all_labels_train = [[] for _ in range(6)]

with torch.no_grad():
    for i, (img, msk) in enumerate(train_dl):
        print(f'Batch {i+1}/{len(train_dl)}', end='\r')
        img = img.to(device)
        msk = msk.to(device)
        output = model(img)

        msk_np = msk.cpu().numpy()
        output_np = output.cpu().numpy()
        
        for c in range(6):
            indices = np.where((msk_np[:, c] == 1) & (msk_np[:, -1] == 0))[0] # kep only homogene patches
            if indices.size > 0:
                all_labels_train[c].append(msk_np[indices])
                all_features_train[c].append(output_np[indices])

all_features_train = [np.concatenate(lst) if lst else np.array([]) for lst in all_features_train]
all_labels_train = [np.concatenate(lst) if lst else np.array([]) for lst in all_labels_train]
print('Finished extracting features for training data')'''

### ------------------------------------- EXTRACT FEATURES FOR TEST DATA
print('Starting to extract features for test data...')
'''all_features_test = [[] for _ in range(6)]
all_labels_test = [[] for _ in range(6)]

with torch.no_grad():
    for i, (img, msk) in enumerate(test_dl):
        print(f'Batch {i+1}/{len(test_dl)}', end='\r')
        img = img.to(device)
        msk = msk.to(device)
        output = model(img)

        msk_np = msk.cpu().numpy()
        output_np = output.cpu().numpy()

        for c in range(6):
            indices = np.where((msk_np[:, c] == 1) & (msk_np[:, -1] == 0))[0]
            if indices.size > 0:
                all_labels_test[c].append(msk_np[indices])
                all_features_test[c].append(output_np[indices])

all_features_test = [np.concatenate(lst) if lst else np.array([]) for lst in all_features_test]
all_labels_test = [np.concatenate(lst) if lst else np.array([]) for lst in all_labels_test]

print('Finished extracting features for test data')'''

### ------------------------------------- RUN t-SNE ON TRAIN AND TEST DATA
'''for c in range(6):
    features_train = all_features_train[c]
    labels_train = all_labels_train[c]
    features_test = all_features_test[c]
    labels_test = all_labels_test[c]
    
    # Combine train and test features
    features = np.concatenate([features_train, features_test])
    labels = np.concatenate([labels_train, labels_test])

    # Reshape the features to 2D
    features = np.reshape(features, (features.shape[0], -1))
    print(f'Features shape: {features.shape}')

    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    print('Running t-SNE on features...')
    # Run t-SNE
    features_embedded = tsne.fit_transform(features)

    scatter_test = plt.scatter(features_embedded[len(features_train):, 0], features_embedded[len(features_train):, 1], label='Test', color = 'green', s=5)
    scatter_train = plt.scatter(features_embedded[:len(features_train), 0], features_embedded[:len(features_train), 1], label='Train', color = 'red', s=5)
    plt.title(f't-SNE on training and test data for class {c} in zone shuffling by zone')
    plt.legend(loc='best')
    plt.savefig(f'stratified_shuffling_t_sne_class{c}_v1.png')
    plt.clf()

    scatter_train = plt.scatter(features_embedded[:len(features_train), 0], features_embedded[:len(features_train), 1], label='Train', color = 'red', s=5) 
    scatter_test = plt.scatter(features_embedded[len(features_train):, 0], features_embedded[len(features_train):, 1], label='Test', color = 'green', s=5)
    plt.title(f't-SNE on training and test data for class {c} in stratified shuffling by zone')
    plt.legend(loc='best')
    plt.savefig(f'stratified_shuffling_t_sne_class{c}_v2.png')
    plt.clf()

    print(len(features_train))
    np.save(f'strat_zone_features_embedded_c{c}.npy', features_embedded)
    # save features
    np.save(f'strat_zone_features_train_c{c}.npy', features_train)
    np.save(f'strat_zone_labels_train_c{c}.npy', labels_train)
    np.save(f'strat_zone_features_test_c{c}.npy', features_test)
    np.save(f'strat_zone_labels_test_c{c}.npy', labels_test)
    print('Saved features and labels')'''


### -------------------------------------EXTRACT FEATURES FOR TRAIN DATA, ALL CLASSES TOGETHER
'''
# run model ans save features for all classes from 0 to 5 when last value of vectir is 1
all_features_train = []
all_labels_train = []

with torch.no_grad():
    for i, (img, msk) in enumerate(train_dl):
        print(f'Batch {i+1}/{len(train_dl)}', end='\r')
        img = img.to(device)
        msk = msk.to(device)
        output = model(img)

        msk_np = msk.cpu().numpy()
        output_np = output.cpu().numpy()
        
        indices = np.where(msk_np[:, -1] == 0)[0]
        if indices.size > 0:
            all_labels_train.append(msk_np[indices])
            all_features_train.append(output_np[indices])

all_features_train = np.concatenate(all_features_train) if all_features_train else np.array([])
all_labels_train = np.concatenate(all_labels_train) if all_labels_train else np.array([])
print('Finished extracting features for training data')

# save them 
np.save(f'zone_features_train_all_30.npy', all_features_train)
np.save(f'zone_labels_train_all_30.npy', all_labels_train)

### EXTRACT FEATURES FOR TEST DATA, ALL CLASSES TOGETHER
all_features_test = []
all_labels_test = []

with torch.no_grad():
    for i, (img, msk) in enumerate(test_dl):
        print(f'Batch {i+1}/{len(test_dl)}', end='\r')
        img = img.to(device)
        msk = msk.to(device)
        output = model(img)

        msk_np = msk.cpu().numpy()
        output_np = output.cpu().numpy()
        
        indices = np.where(msk_np[:, -1] == 0)[0]
        if indices.size > 0:
            all_labels_test.append(msk_np[indices])
            all_features_test.append(output_np[indices])

all_features_test = np.concatenate(all_features_test) if all_features_test else np.array([])
all_labels_test = np.concatenate(all_labels_test) if all_labels_test else np.array([])
print('Finished extracting features for test data')

# save them
np.save(f'zone_features_test_all_30.npy', all_features_test)
np.save(f'zone_labels_test_all_30.npy', all_labels_test)
'''


#load all features train and labels train
all_features_train = np.load('zone_features_train_all_30.npy')
all_labels_train = np.load('zone_labels_train_all_30.npy')

all_features_test = np.load('zone_features_test_all_30.npy')
all_labels_test = np.load('zone_labels_test_all_30.npy')

# run tsne and plot. Each color with 
# concat train and test features
features = np.concatenate([all_features_train, all_features_test])
labels = np.concatenate([all_labels_train, all_labels_test])

#remove all last values of the labels
labels = labels[:, :-1]
print(f'Labels shape: {labels.shape}')
print(labels)
labels = np.argmax(labels, axis=1)
#load features from class 3 test
# features_test_c3 = np.load('strat_zone_features_test_c3.npy')
# labels_test_c3 = np.load('strat_zone_labels_test_c3.npy')

# Combine train and test features
#features = np.concatenate([features, features_test_c3])

# Reshape the features to 2D
features = np.reshape(features, (features.shape[0], -1))
print(f'Features shape: {features.shape}')

tsne = TSNE(n_components=2, random_state=42, n_iter=5000)
print('Running t-SNE on features...')
# Run t-SNE
features_embedded = tsne.fit_transform(features)

#save features
np.save(f'zone_features_embedded_all_35.npy', features_embedded)

#features_embedded has dimension (n_samples, 2), labels has dimension (n_samples, 1)
#plot class one by one
'''for c in range(6):
    indices_train = np.where(labels[:len(all_features_train)] == c)[0]
    indices_test = np.where(labels[len(all_features_train):] == c)[0]
    scatter = plt.scatter(features_embedded[indices_train, 0], features_embedded[indices_train, 1], label=f'Train class {c}', color = 'green', s=5)
    scatter = plt.scatter(features_embedded[indices_test, 0], features_embedded[indices_test, 1], label=f'Test class {c}', color = 'red', s=5)
    plt.title(f't-SNE for class {c} in zone shuffling')
    plt.legend(loc='best')
    plt.savefig(f'zone_t_sne_class{c}_35.png')
    plt.clf()'''


'''scatter_train = plt.scatter(features_embedded[:len(all_features_train), 0], features_embedded[:len(all_features_train), 1], c=labels, cmap=my_cmap, label='Train', s=5) # change symbol
scatter_test = plt.scatter(features_embedded[len(all_features_train):, 0], features_embedded[len(all_features_train):, 1], label='Test class 3', color = 'black', s=5)
plt.title(f't-SNE on training data for all classes in stratified shuffling by zone')
## Create a colorbar with the correct ticks and labels
cbar = plt.colorbar(scatter_train, ticks=np.arange(0, 6))
cbar.ax.set_yticklabels([f'Class {i}' for i in range(6)])
plt.legend(loc='best')
plt.savefig(f'strat_zone_t_sne_all_classes_train_c3test_8.png')
plt.clf()'''


scatter_train = plt.scatter(features_embedded[:len(all_features_train), 0], features_embedded[:len(all_features_train), 1], c=labels[:len(all_features_train)], cmap=my_cmap, label='Train', s=5, norm=my_norm) # change symbol
plt.title(f't-SNE on training data in zone shuffling')
## Create a colorbar with the correct ticks and labels
#cbar = plt.colorbar(scatter_train, ticks=np.arange(0, 6))
#cbar.ax.set_yticklabels([f'Class {i}' for i in range(6)])
plt.legend(loc='best')
plt.savefig(f'zone_t_sne_all_classes_train_35.png')
plt.clf()