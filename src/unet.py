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

from unet_utils import *
from unet_settings import *

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

print('----------------------- UNet -----------------------')
print(f'Patch size: {patch_level_param["patch_size"]}')
print(f'Classification level: {patch_level_param["level"]}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
torch.cuda.empty_cache()


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
# Sanity check of the dataloader
for img, msk in train_dl:
    print(img.shape, msk.shape)
    break



## MODEL
print('Creating model...')
print('Model settings:')
print(f'Encoder name: {model_settings["encoder_name"]}')
print(f'Pretrained: {model_settings["encoder_weights"]}')

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
else:
    raise ValueError('Criterion not implemented')
print(f'Optimizer: {training_settings["optimizer"]}')
if training_settings['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=training_settings['lr'])



## FUNCTION FOR TRAINING, VALIDATION AND TESTING
def train(model, train_dl, criterion, optimizer, device):
    print('Training')
    running_loss = 0.0
    tr_IoUs = []
    for i, (img, msk) in enumerate(train_dl):
        if i % 50 == 0:
            print( 'Batch:', i, ' over ', len(train_dl))
        img, msk = img.to(device), msk.to(device)
        optimizer.zero_grad()
        out = model(img)
        msk = msk.long()
        loss = criterion(out, msk)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        out = torch.argmax(out, dim=1)
        out = out.int()
        tr_IoUs.append(IoU(out, msk))
    tr_IoUs = [x.item() for x in tr_IoUs]
    mIoU = np.mean(tr_IoUs)
    return running_loss / len(train_dl), mIoU


def valid(model, val_dl, criterion, device):
    print('Validation')
    running_loss = 0.0
    val_IoUs = []
    for i, (img, msk) in enumerate(val_dl):
        if i % 50 == 0:
            print( 'Batch:', i, ' over ', len(val_dl))
        img, msk = img.to(device), msk.to(device)
        out = model(img)
        msk = msk.long()
        loss = criterion(out, msk)
        running_loss += loss.item()
        out = torch.argmax(out, dim=1)
        out = out.int()
        val_IoUs.append(IoU(out, msk))
    val_IoUs = [x.item() for x in val_IoUs]
    mIoU = np.mean(val_IoUs)
    return running_loss / len(val_dl), mIoU

def test(model, test_dl, criterion, device):
    print('Testing')
    running_loss = 0.0
    test_IoUs = []
    for i, (img, msk) in enumerate(test_dl):
        if i % 50 == 0:
            print( 'Batch:', i, ' over ', len(test_dl))
        img, msk = img.to(device), msk.to(device)
        out = model(img)
        msk = msk.long()
        loss = criterion(out, msk)
        running_loss += loss.item()
        out = torch.argmax(out, dim=1)
        out = out.int()
        test_IoUs.append(IoU(out, msk))
    test_IoUs = [x.item() for x in test_IoUs]
    mIoU = np.mean(test_IoUs)
    return running_loss / len(test_dl), mIoU



## TRAINING AND VALIDATION
if training_settings['training']:
    print('Training...')
    training_losses = []
    validation_losses = []
    training_iou = []
    validation_iou = []
    for epoch in range(training_settings['nb_epochs']):
        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}')
        model.to(device)
        model.train()
        train_loss, train_mIoU = train(model, train_dl, criterion, optimizer, device)
        training_losses.append(train_loss)
        training_iou.append(train_mIoU)

        model.eval()
        with torch.no_grad():
            val_loss, val_mIoU = valid(model, val_dl, criterion, device)   
            validation_losses.append(val_loss)
            validation_iou.append(val_mIoU)

        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}: train loss {train_loss:.4f}, val loss {val_loss:.4f}')
        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}: train mIoU {train_mIoU:.4f}, val mIoU {val_mIoU:.4f}')

        if epoch % 2 == 0:
            torch.save(model.state_dict(), model_settings['path_to_intermed_model'] + f'_epoch{epoch+1}.pt')

    # save last model state and optim
    torch.save(model.state_dict(), model_settings['path_to_model'])
    torch.save(optimizer.state_dict(), model_settings['path_to_optim'])

    # plot losses
    plt.plot(training_losses, label='train')
    plt.plot(validation_losses, label='val')
    plt.legend()
    plt.savefig(plotting_settings['losses_path'])

    # get miou on tr and val    
    print(f'Training mIoU: {train_mIoU:.4f}') # after 10 epochs
    print(f'Validation mIoU: {val_mIoU:.4f}') #after 10 epochs
else: 
    model.to(device)
    model.load_state_dict(torch.load(model_settings['path_to_model']))
    for param in model.parameters():
        param.to(device)



# TESTING
model.eval()
with torch.no_grad():
    test_loss, test_mIoU = test(model, test_dl, criterion, device)
print(f'Test mIoU: {test_mIoU:.4f}')



# PLOTTING TEST PREDICTIONS
if plotting_settings['plot_test']:
    # plot img, original mask and prediction
    model.eval()
    img, msk = next(iter(test_dl))
    img, msk = img.to(device), msk.to(device)
    out = model(img)
    out = torch.argmax(out, dim=1)
    out = out.int()
    img = img.cpu().numpy()
    msk = msk.cpu().numpy()
    out = out.cpu().numpy()
    plot_pred(img, msk, out, plotting_settings['pred_plot_path'], plotting_settings['my_colors_map'], plotting_settings['nb_plots'])