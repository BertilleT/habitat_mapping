# check classes balance in train, valid and test. 

#load traing dataset as done in Unet and check the classes balance
#load the dataset
import numpy as np
from pathlib import Path
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import gc

from unet_utils import *
from unet_settings import *

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


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
print(f'Splitting data: {data_loading_settings["splitting"]}')
print(f'Stratified: True')
print(f'Batch size: {data_loading_settings["bs"]}')

train_paths, val_paths, test_paths = load_data_paths(**data_loading_settings)
train_ds = EcomedDataset(train_paths, data_loading_settings['img_folder'], level=patch_level_param['level'])
train_dl = DataLoader(train_ds, batch_size=data_loading_settings['bs'], shuffle=True)
val_ds = EcomedDataset(val_paths, data_loading_settings['img_folder'], level=patch_level_param['level'])
val_dl = DataLoader(val_ds, batch_size=data_loading_settings['bs'], shuffle=False)
test_ds = EcomedDataset(test_paths, data_loading_settings['img_folder'], level=patch_level_param['level'])
test_dl = DataLoader(test_ds, batch_size=data_loading_settings['bs'], shuffle=False)


#  check the classes balance in train, valid and test
# add indicator of time for this fct
import time

def check_classes_balance(dl):
    classes = {i: 0 for i in range(1, 7)}
    len_dl = len(dl)
    i = 0
    for img, msk in dl:
        i += 1
        print(f'Batch {i}/{len_dl}')
        for i in range(1, 7):
            classes[i] += torch.sum(msk == i).item()
    return classes

print('Checking classes balance of train')
train_classes = check_classes_balance(train_dl)
print('Train classes:', train_classes)

print('Checking classes balance of val')
val_classes = check_classes_balance(val_dl)
print('Val classes:', val_classes)

print('Checking classes balance of test')
test_classes = check_classes_balance(test_dl)
print('Test classes:', test_classes)