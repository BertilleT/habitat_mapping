import numpy as np
from pathlib import Path
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import gc

from unet_utils import *
from unet_settings import *

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


my_models_paths = Path('../unet256_randomshuffling/models').glob('*.pt')
my_models_paths = [str(model.name) for model in my_models_paths]
# to df 
my_models_df = pd.DataFrame(my_models_paths)
my_models_df['nb_epoch'] = my_models_df.apply(lambda x: int(x[0][19:-3]), axis=1)
# sort by nb_epoch
my_models_df = my_models_df.sort_values(by='nb_epoch')
# turn back to list
ordered_models_paths = my_models_df[0].tolist()
ordered_models_paths = [Path(f'../unet256_randomshuffling/models/{model}') for model in ordered_models_paths]
print(ordered_models_paths)
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
print(f'Stratified: {data_loading_settings["stratified"]}')
print(f'Batch size: {data_loading_settings["bs"]}')

train_paths, val_paths, test_paths = load_data_paths(**data_loading_settings)
train_ds = EcomedDataset(train_paths, data_loading_settings['img_folder'], level=patch_level_param['level'])
train_dl = DataLoader(train_ds, batch_size=data_loading_settings['bs'], shuffle=True)
val_ds = EcomedDataset(val_paths, data_loading_settings['img_folder'], level=patch_level_param['level'])
val_dl = DataLoader(val_ds, batch_size=data_loading_settings['bs'], shuffle=False)
test_ds = EcomedDataset(test_paths, data_loading_settings['img_folder'], level=patch_level_param['level'])
test_dl = DataLoader(test_ds, batch_size=data_loading_settings['bs'], shuffle=False)

## MODEL
print('Creating model...')
print('Model settings:')
print(f'Encoder name: {model_settings["encoder_name"]}')
print(f'Pretrained: {model_settings["encoder_weights"]}')
print(f'Classes: {model_settings["classes"]}')
model = smp.Unet(
    encoder_name=model_settings['encoder_name'],        
    encoder_weights=model_settings['encoder_weights'], 
    in_channels=model_settings['in_channels'], 
    classes=model_settings['classes']
)
# OPTIMIZER
print('Creating optimizer...')
print('Training settings:')
print(f'Learning rate: {training_settings["lr"]}')
print(f'Criterion: {training_settings["criterion"]}')
if training_settings['criterion'] == 'CrossEntropy':
    criterion = nn.CrossEntropyLoss()
elif training_settings['criterion'] == 'Dice':
    criterion = smp.losses.DiceLoss(mode='multiclass', eps=0.0000001)
else:
    raise ValueError('Criterion not implemented')
print(f'Optimizer: {training_settings["optimizer"]}')
if training_settings['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=training_settings['lr'])

training_losses = []
training_ious = []
validation_losses = []
validation_ious = []

for index, model_path in enumerate(ordered_models_paths):
    # get tr_mIoU and val_miou
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    with torch.no_grad():
        train_loss, train_mIoU = valid(model, train_dl, criterion, device)
        training_losses.append(train_loss)
        training_ious.append(train_mIoU)
        val_loss, val_mIoU = valid(model, val_dl, criterion, device)   
        validation_losses.append(val_loss)
        validation_ious.append(val_mIoU)
        print(f'Batch {index+1}/{len(ordered_models_paths)}')
        print(f'Training loss: {train_loss}')
        print(f'Validation loss: {val_loss}')
        print(f'Training mIoU: {train_mIoU}')
        print(f'Validation mIoU: {val_mIoU}')
    # Plot the training and validation losses

plt.plot(training_losses, label='train')
plt.plot(validation_losses, label='val')
plt.legend()
plt.savefig(plotting_settings['losses_path'])

plt.plot(training_ious, label='train')
plt.plot(validation_ious, label='val')
plt.legend()
plt.savefig(plotting_settings['ious_path'])

# same with 
# get miou on tr and val    
print(f'Training mIoU: {train_mIoU:.4f}') # after 10 epochs
print(f'Validation mIoU: {val_mIoU:.4f}') #after 10 epochs