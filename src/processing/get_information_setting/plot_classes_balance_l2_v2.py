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

import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
import matplotlib.patches as mpatches
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
# check train_ds length
print(len(train_ds))

train_dl = DataLoader(train_ds, batch_size=data_loading_settings['bs'], shuffle=True)
# check size of one img and masks, and value of masks at level 1
img, msk = next(iter(train_dl))
val_ds = EcomedDataset(val_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], normalisation = data_loading_settings['normalisation'], task = model_settings['task'], my_set = "val", labels = model_settings['labels'])
val_dl = DataLoader(val_ds, batch_size=data_loading_settings['bs'], shuffle=False)
test_ds = EcomedDataset(test_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], normalisation = data_loading_settings['normalisation'], task = model_settings['task'], my_set = "test", labels = model_settings['labels'])
test_dl = DataLoader(test_ds, batch_size=data_loading_settings['bs'], shuffle=False)


sys.stdout.flush()
print('Checking classes balance of train')
train_classes = check_classes_balance(train_dl, model_settings['classes'], model_settings['task'])
print(train_classes)
val_classes = check_classes_balance(val_dl, model_settings['classes'], model_settings['task'])
print(val_classes)
test_classes = check_classes_balance(test_dl, model_settings['classes'], model_settings['task'])
print(test_classes)

balance_classes = pd.DataFrame(train_classes.items(), columns=['class', 'train'])
balance_classes['val'] = balance_classes['class'].map(val_classes)
balance_classes['test'] = balance_classes['class'].map(test_classes)
balance_classes['total'] = balance_classes['train'] + balance_classes['val'] + balance_classes['test']
balance_classes = balance_classes.set_index('class')
print(balance_classes)

for col in balance_classes.columns:
    balance_classes[col] = round(balance_classes[col]*100 / balance_classes[col].sum(), 2)
#drop rows when values from total are 0
balance_classes = balance_classes[balance_classes['total'] != 0]
balance_classes_total = balance_classes['total']

print(plotting_settings)
print(plotting_settings['my_colors_map'])
print(balance_classes_total.index)

fig, ax = plt.subplots()
x = np.arange(len(balance_classes_total))
ax.bar(x, balance_classes_total, color=[plotting_settings['my_colors_map'][i] for i in balance_classes_total.index])
# add ercentage on top of bars round to 0 decimals
for i, v in enumerate(balance_classes_total):
    ax.text(i, v + 0.5, f'{v:.0f}%', ha='center', va='bottom')
ax.set_xticks(x)
ax.set_xticklabels(balance_classes_total.index)
plt.savefig(f'classes_balance_total_{data_loading_settings["stratified"]}_seed{data_loading_settings["random_seed"]}.png')

# create a legend with patch, '''

# Load the CSV file
df = pd.read_csv('../../csv/l2_dict.csv')
print(df.head())

# Create the legend
legend = []
for key, value in plotting_settings['my_colors_map'].items():
    if key != 16: 
        label = df[df['int_grouped'] == key]['name'].values[0]
    else:
        label = 'Autre'
    legend.append(mpatches.Patch(color=value, label=label))

# Create the plot
plt.figure(figsize=(14, 7))  # Adjust the figure size as needed
#increase space between rows
plt.subplots_adjust(hspace=0.5)
plt.axis('off')
plt.legend(handles=legend, frameon=False, fontsize=12)  # Adjust the font size as needed

# Save the plot
plt.savefig('l2_legend.png')