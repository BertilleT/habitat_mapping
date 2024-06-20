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

from matplotlib.colors import ListedColormap

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
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussNoise(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),  # Ajout de la rotation avec une limite de 45 degrés et une probabilité de 50%
        A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),  # Ajout du recadrage aléatoire avec des paramètres définis
        ToTensorV2(),
    ])

else: 
    print('No data augmentation')
    transform = None


train_paths, val_paths, test_paths = load_data_paths(**data_loading_settings)
#print(f'Train: {len(train_paths)} images, Val: {len(val_paths)} images, Test: {len(test_paths)} images')

train_ds = EcomedDataset(train_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], transform = transform, normalisation = data_loading_settings['normalisation'], task = model_settings['task'])
train_dl = DataLoader(train_ds, batch_size=data_loading_settings['bs'], shuffle=True)
# check size of one img and masks, and value of masks at level 1
img, msk = next(iter(train_dl))
print(f'Image shape: {img.shape}, Mask shape: {msk.shape}')
print(f"Image: min: {img.min()}, max: {img.max()}, dtype: {img.dtype}")
print(f"Mask: {np.unique(msk[0])}, dtype: {msk.dtype}")
val_ds = EcomedDataset(val_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], normalisation = data_loading_settings['normalisation'], task = model_settings['task'])
val_dl = DataLoader(val_ds, batch_size=data_loading_settings['bs'], shuffle=False)
test_ds = EcomedDataset(test_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], normalisation = data_loading_settings['normalisation'], task = model_settings['task'])
test_dl = DataLoader(test_ds, batch_size=data_loading_settings['bs'], shuffle=False)


#run tsne on training data
print('Running t-SNE on training data... with rsnet18')
# Load the model
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
# remove the classifier head
model = nn.Sequential(*list(model.children())[:-1])

model.to(device)
model.eval()
features_train = []
labels_train = []
with torch.no_grad():
    for i, (img, msk) in enumerate(train_dl):
        print(f'Batch {i+1}/{len(train_dl)}', end='\r')
        img = img.to(device)
        msk = msk.to(device)
        output = model(img)
        features_train.append(output.cpu().numpy())
        labels_train.append(msk.cpu().numpy())
features_train = np.concatenate(features_train)
labels_train = np.concatenate(labels_train)
'''
# Get the features from the last layer of the model for validation data
features_val = []
labels_val = []
with torch.no_grad():
    for i, (img, msk) in enumerate(val_dl):
        print(f'Batch {i+1}/{len(val_dl)}', end='\r')
        img = img.to(device)
        msk = msk.to(device)
        output = model(img)
        features_val.append(output.cpu().numpy())
        labels_val.append(msk.cpu().numpy())
features_val = np.concatenate(features_val)
labels_val = np.concatenate(labels_val)

# Combine train and val features
features = np.concatenate([features_train, features_val])
labels = np.concatenate([labels_train, labels_val])

# Reshape the features to 2D
features = np.reshape(features, (features.shape[0], -1))
print(f'Features shape: {features.shape}')'''

my_colors_map = {
    0: '#789262',  # Vert olive
    1: '#555555',  # Gris
    2: '#006400',  # Vert foncé
    3: '#00ff00',  # Vert vif
    4: '#ff4500',  # Rouge
    5: '#8a2be2',  # Violet
}

# Run t-SNE
my_colors = [my_colors_map[i] for i in range(6)]
my_cmap = ListedColormap(my_colors)
'''
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
features_embedded = tsne.fit_transform(features)
print(f'Embedded features shape: {features_embedded.shape}')

# Plot training data with circles
scatter_train = plt.scatter(features_embedded[:len(features_train), 0], features_embedded[:len(features_train), 1], c=labels_train, cmap=my_cmap, label='Train', marker='o')

# Plot validation data with crosses
scatter_val = plt.scatter(features_embedded[len(features_train):, 0], features_embedded[len(features_train):, 1], c=labels_val, cmap=my_cmap, label='Validation', marker='x')

# Create a colorbar with the correct ticks and labels
cbar = plt.colorbar(scatter_train, ticks=np.arange(0, 6))
cbar.ax.set_yticklabels([f'Class {i}' for i in range(6)])

plt.title('t-SNE on training and validation data')
plt.legend(loc='best')

# Save the plot
plt.savefig('t_sne_train_val_random_shuffling.png')
# clean plot
plt.clf()'''

#on trainin and testing
print('Running t-SNE on training and testing data...')
# Get the features from the last layer of the model for test data
features_test = []
labels_test = []
with torch.no_grad():
    for i, (img, msk) in enumerate(test_dl):
        print(f'Batch {i+1}/{len(test_dl)}', end='\r')
        img = img.to(device)
        msk = msk.to(device)
        output = model(img)
        features_test.append(output.cpu().numpy())
        labels_test.append(msk.cpu().numpy())
features_test = np.concatenate(features_test)
labels_test = np.concatenate(labels_test)

# Combine train and test features
features = np.concatenate([features_train, features_test])
labels = np.concatenate([labels_train, labels_test])

# Reshape the features to 2D
features = np.reshape(features, (features.shape[0], -1))
print(f'Features shape: {features.shape}')

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
print('Running t-SNE on features...')
# Run t-SNE
features_embedded = tsne.fit_transform(features)
print(f'Embedded features shape: {features_embedded.shape}')

# Plot training data with circles
scatter_train = plt.scatter(features_embedded[:len(features_train), 0], features_embedded[:len(features_train), 1], c=labels_train, cmap=my_cmap, label='Train', marker='o')

# Plot test data with triangles
scatter_test = plt.scatter(features_embedded[len(features_train):, 0], features_embedded[len(features_train):, 1], c=labels_test, cmap=my_cmap, label='Test', marker='^')

# Create a colorbar with the correct ticks and labels
cbar = plt.colorbar(scatter_train, ticks=np.arange(0, 6))
cbar.ax.set_yticklabels([f'Class {i}' for i in range(6)])

plt.title('t-SNE on training and test data')
plt.legend(loc='best')

# Save the plot
plt.savefig('t_sne_test_stratified.png')
# clean plot
plt.clf()
'''
#save one with only the training
# Plot training data with circles
scatter_train = plt.scatter(features_embedded[:len(features_train), 0], features_embedded[:len(features_train), 1], c=labels_train, cmap=my_cmap, label='Train', marker='o')

plt.title('t-SNE on training data')
plt.legend(loc='best')

# Save the plot
plt.savefig('t_sne_train_random_shuffling.png')

#clean plot
plt.clf()'''
'''
from sklearn.manifold import TSNE
# Function to get images and labels
def get_images_and_labels(dataloader):
    images = []
    labels = []
    for img, msk in dataloader:
        images.append(img.numpy())
        labels.append(msk.numpy())
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    return images, labels

# Get images and labels for train and val sets
images_train, labels_train = get_images_and_labels(train_dl)
images_val, labels_val = get_images_and_labels(val_dl)
print(f'Train images shape: {images_train.shape}, Train labels shape: {labels_train.shape}')
# Flatten the images
images_train_flat = images_train.reshape((images_train.shape[0], -1))
images_val_flat = images_val.reshape((images_val.shape[0], -1))
print(f'Flattened train images shape: {images_train_flat.shape}')
# Combine train and val images
images_combined = np.concatenate([images_train_flat, images_val_flat])
labels_combined = np.concatenate([labels_train, labels_val])
print(f'Combined images shape: {images_combined.shape}')

# Run t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
print('Running t-SNE on images...')
images_embedded = tsne.fit_transform(images_combined)
print(f'Embedded images shape: {images_embedded.shape}')

# Custom colormap
my_colors_map = {
    0: '#789262',  # Vert olive
    1: '#555555',  # Gris
    2: '#006400',  # Vert foncé
    3: '#00ff00',  # Vert vif
    4: '#ff4500',  # Rouge
    5: '#8a2be2',  # Violet
}
my_colors = [my_colors_map[i] for i in range(6)]
my_cmap = ListedColormap(my_colors)

# Plot the t-SNE
plt.figure(figsize=(10, 10))

# Plot training data with circles
scatter_train = plt.scatter(images_embedded[:len(images_train_flat), 0], images_embedded[:len(images_train_flat), 1], c=labels_train, cmap=my_cmap, label='Train', marker='o')

# Plot validation data with crosses
scatter_val = plt.scatter(images_embedded[len(images_train_flat):, 0], images_embedded[len(images_train_flat):, 1], c=labels_val, cmap=my_cmap, label='Validation', marker='x')

# Create a colorbar with the correct ticks and labels
cbar = plt.colorbar(scatter_train, ticks=np.arange(0, 6))
cbar.ax.set_yticklabels([f'Class {i}' for i in range(6)])

plt.title('t-SNE on training and validation images')
plt.legend(loc='best')

# Save the plot
plt.savefig('t_sne_images.png')'''