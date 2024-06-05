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
train_paths = train_paths
val_paths = val_paths
test_paths = test_paths
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
print(balance_classes)

# barplot of classes balance
#load csv l2_map_l1.csv
l2_map_l1_path = Path('../../csv/l2_map_l1.csv')
l2_map_l1 = pd.read_csv(l2_map_l1_path)
# add column l1_int to balance_classes based ont the mapping of l2_map_l1
balance_classes['l1_int'] = balance_classes.index.map(l2_map_l1.set_index('int')['l1_int'])
l2_map_dict = l2_map_l1.set_index('int')['name'].to_dict()

balance_classes_total = balance_classes[['total', 'l1_int']]
# drop rows with 0 values
# get the classes unique in ascending order
balance_classes_total = balance_classes_total.sort_values(ascending=True, by='l1_int')

fig, ax = plt.subplots()
ax.grid(axis='y')
width = 0.2
colors = {'train': 'red', 'val': 'blue', 'test': 'green', 'total': 'black'}
# get uique values from l1_int in ascending order
l1_ints = balance_classes['l1_int'].unique()
l1_ints = np.sort(l1_ints)
#plot prop of classes at level 2 for each class at level1

import textwrap

# Define a function to wrap text with a maximum width
def wrap_labels(labels, max_width):
    wrapped_labels = []
    for label in labels:
        wrapped_labels.append('\n'.join(textwrap.wrap(label, max_width)))
    return wrapped_labels

for l1_int in l1_ints:
    fig, ax = plt.subplots()
    #keep only rows where l1_int == l1_int
    sub_balance_classes_l1 = balance_classes[balance_classes['l1_int'] == l1_int]
    # drop l1_int column
    sub_balance_classes_l1 = sub_balance_classes_l1.drop(columns='l1_int')
    x = np.arange(len(sub_balance_classes_l1))

    for i, col in enumerate(sub_balance_classes_l1.columns):
        ax.bar(x + i*width, sub_balance_classes_l1[col], width, label=col, color=colors[col])
    # add grid

    ax.set_xticks(x + width)
    class_labels = [f"{index}: {l2_map_dict.get(index, 'Unknown')}" for index in sub_balance_classes_l1.index]
    # Example usage in your plotting code
    max_char_width = 30  # Set the maximum number of characters per line

    # Create class labels with line breaks
    wrapped_class_labels = wrap_labels(class_labels, max_char_width)
    ax.set_xticklabels(wrapped_class_labels, rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    ax.legend()
    # add title
    l1_int_name = plotting_settings['habitats_dict'].get(l1_int, 'Unknown')
    ax.set_title(f'Balance of level 2 classes derived from the level 1 class {l1_int} : \n {l1_int_name}')
    plt.savefig(f'classes_balance_l2_c{l1_int}_{data_loading_settings["stratified"]}_seed{data_loading_settings["random_seed"]}.png')
    plt.close(fig)

fig, ax = plt.subplots()
x = np.arange(len(balance_classes_total))
# create color_map based on my_colors_maps and l1_int values
color_map = balance_classes_total['l1_int'].map(plotting_settings['my_colors_map'])
# drop l1_int column
balance_classes_total = balance_classes_total.drop(columns='l1_int')
# to serie 
print(balance_classes_total)

ax.bar(x, balance_classes_total['total'], color=color_map)
# add ercentage on top of bars round to 0 decimals
for i, v in enumerate(balance_classes_total['total']):
    #rotation to ahvee text in vertical
    ax.text(i, v + 0.15, f'{v:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
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
# title
ax.set_title('Balance classes at level 2 in the whole dataset')
plt.savefig(f'classes_balance_l2_total_{data_loading_settings["stratified"]}_seed{data_loading_settings["random_seed"]}.png')
'''for col in balance_classes.columns:
    balance_classes[col] = round(balance_classes[col]*100 / balance_classes[col].sum(), 2)
print(balance_classes)'''
# save df to csv
# balance_classes.to_csv(data_loading_settings['classes_balance'], index=False)