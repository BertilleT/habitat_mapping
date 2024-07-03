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

from datetime import datetime

from unet_utils import *
from unet_settings import *

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import albumentations as A
from albumentations.pytorch import ToTensorV2

from matplotlib.patches import Patch

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
print(f'Batch size: {data_loading_settings["bs"]}')

if data_loading_settings['data_augmentation']: 
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussNoise(p=0.5),
        ToTensorV2(), 
    ])
else: 
    print('No data augmentation')
    transform = None

train_paths, val_paths, test_paths = load_data_paths(**data_loading_settings)
train_paths = train_paths[:20]
val_paths = val_paths[:20]
test_paths = test_paths[:20]
train_ds = EcomedDataset(train_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], transform = transform)
train_dl = DataLoader(train_ds, batch_size=data_loading_settings['bs'], shuffle=True)
val_ds = EcomedDataset(val_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'])
val_dl = DataLoader(val_ds, batch_size=data_loading_settings['bs'], shuffle=False)
test_ds = EcomedDataset(test_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'])
test_dl = DataLoader(test_ds, batch_size=data_loading_settings['bs'], shuffle=False)

# print size of train, val and test and proportion it rperesents compared to the total size of the dataset
print(f'Train: {len(train_ds)} images, Val: {len(val_ds)} images, Test: {len(test_ds)} images')
print(f'Train: {len(train_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%, Val: {len(val_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%, Test: {len(test_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%')

sys.stdout.flush()
print('Checking classes balance of train')
train_classes = check_classes_balance(train_dl, model_settings['classes'])
print(train_classes)
val_classes = check_classes_balance(val_dl, model_settings['classes'])
print(val_classes)
test_classes = check_classes_balance(test_dl, model_settings['classes'])
print(test_classes)

balance_classes = pd.DataFrame(train_classes.items(), columns=['class', 'train'])
balance_classes['val'] = balance_classes['class'].map(val_classes)
balance_classes['test'] = balance_classes['class'].map(test_classes)
balance_classes['total'] = balance_classes['train'] + balance_classes['val'] + balance_classes['test']
balance_classes = balance_classes.set_index('class')
for col in balance_classes.columns:
    balance_classes[col] = round(balance_classes[col]*100 / balance_classes[col].sum(), 2)
#drop rows when values from total are 0
balance_classes = balance_classes[balance_classes['total'] != 0]
balance_classes_total = balance_classes['total']
# drop rows with 0 values
# get the classes unique in ascending order
balance_classes_total = balance_classes_total.sort_values(ascending=True)
print(balance_classes_total)

print(balance_classes)
# barplot of classes balance
fig, ax = plt.subplots()
ax.grid(axis='y')
x = np.arange(len(balance_classes))
width = 0.2
colors = {'train': 'red', 'val': 'blue', 'test': 'green', 'total': 'black'}
for i, col in enumerate(balance_classes.columns):
    ax.bar(x + i*width, balance_classes[col], width, label=col, color=colors[col])
# add grid

ax.set_xticks(x + width)
ax.set_xticklabels(balance_classes.index)
ax.legend()
plt.savefig(f'classes_balance_l2_{data_loading_settings["stratified"]}_seed{data_loading_settings["random_seed"]}.png')


fig, ax = plt.subplots()
x = np.arange(len(balance_classes_total))
ax.bar(x, balance_classes_total, color=[plotting_settings['my_colors_map'][i] for i in balance_classes_total.index])
# add ercentage on top of bars round to 0 decimals
for i, v in enumerate(balance_classes_total):
    ax.text(i, v + 0.5, f'{v:.0f}%', ha='center', va='bottom')
ax.set_xticks(x)
ax.set_xticklabels(balance_classes_total.index)
#nb of rows in balances_classes_total
'''classes = balance_classes_total.index
class_color = {c: plotting_settings['my_colors_map'][c] for c in classes}
unique_labels = set()
legend_elements = []
for l, color in class_color.items():
    label = plotting_settings['habitats_dict'][l]
    legend_elements.append(Patch(facecolor=color, label=label))
    unique_labels.add(label)

fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1.2), loc='upper center')
plt.tight_layout(rect=[0, 0, 1, 0.6])'''
#plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f'classes_balance_l2_total_{data_loading_settings["stratified"]}_seed{data_loading_settings["random_seed"]}.png')
'''for col in balance_classes.columns:
    balance_classes[col] = round(balance_classes[col]*100 / balance_classes[col].sum(), 2)
print(balance_classes)'''
# save df to csv
# balance_classes.to_csv(data_loading_settings['classes_balance'], index=False)