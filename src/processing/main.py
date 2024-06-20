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


train_ds_plot = EcomedDataset_to_plot(train_paths, data_loading_settings['img_folder'], channels = model_settings['in_channels'], task = model_settings['task'])
plot_patch_class_by_class(train_ds_plot, 20, plotting_settings['habitats_dict'], plotting_settings['l2_habitats_dict'], 'training set')
val_ds_plot = EcomedDataset_to_plot(val_paths, data_loading_settings['img_folder'], channels = model_settings['in_channels'], task = model_settings['task'])
plot_patch_class_by_class(val_ds_plot, 20, plotting_settings['habitats_dict'], plotting_settings['l2_habitats_dict'], 'validation set')
test_ds_plot = EcomedDataset_to_plot(test_paths, data_loading_settings['img_folder'], channels = model_settings['in_channels'], task = model_settings['task'])
plot_patch_class_by_class(test_ds_plot, 20, plotting_settings['habitats_dict'], plotting_settings['l2_habitats_dict'], 'test set')

# print size of train, val and test and proportion it rperesents compared to the total size of the dataset
print(f'Train: {len(train_ds)} images, Val: {len(val_ds)} images, Test: {len(test_ds)} images')
print(f'Train: {len(train_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%, Val: {len(val_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%, Test: {len(test_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%')

sys.stdout.flush()
## MODEL
print('Creating model...')
print('Model settings:')
print(f'Pretrained: {model_settings["encoder_weights"]}')
print(f'Classes: {model_settings["classes"]}')
if model_settings['model'] == 'Unet':
    print('The model is Unet')
    print(f'Encoder name: {model_settings["encoder_name"]}')
    model = smp.Unet(
        encoder_name=model_settings['encoder_name'],        
        encoder_weights=model_settings['encoder_weights'], 
        in_channels=model_settings['in_channels'], 
        classes=model_settings['classes'], 
    )

elif model_settings['model'] == 'Resnet18':
    print('The model is Resnet18')
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
    
if training_settings['restart_training'] is not None:
    model.load_state_dict(torch.load(model_settings['path_to_last_model']))
    print('Model from last epoch', training_settings['restart_training'], ' loaded')
model.to(device)

# METRIC
if model_settings['model'] == 'Unet':
    metric = 'mIoU'
elif model_settings['model'] == 'Resnet18':
    metric = 'mF1'

# OPTIMIZER
if training_settings['criterion'] == 'CrossEntropy':
    criterion = nn.CrossEntropyLoss()
elif training_settings['criterion'] == 'Dice':
    criterion = smp.losses.DiceLoss(mode='multiclass', eps=0.0000001)
else:
    raise ValueError('Criterion not implemented')
if training_settings['training']:
    print('Creating optimizer...')
    print('Training settings:')
    print(f'Learning rate: {training_settings["lr"]}')
    print(f'Criterion: {training_settings["criterion"]}')
    print(f'Optimizer: {training_settings["optimizer"]}')

if training_settings['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=training_settings['lr'])
if training_settings['optimizer'] == 'AdamW':
    print('Using AdamW optimizer')
    optimizer = optim.AdamW(model.parameters(), lr=training_settings['lr'])
if training_settings['optimizer'] == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=training_settings['lr'], momentum=0.9)


if training_settings['restart_training'] is not None:
    torch.cuda.empty_cache()
    optimizer.load_state_dict(torch.load(model_settings['path_to_last_optim']))

## METRIC

## TRAINING AND VALIDATION
if training_settings['training']:
    print('Training...')
    training_losses = []
    validation_losses = []
    training_metric = []
    validation_metric = []
    count = 0
    best_val_loss = np.inf 

    if training_settings['restart_training']:
        # load losses
        df = pd.read_csv(training_settings['losses_metric_path'])
        training_losses = df['training_losses'].tolist()
        validation_losses = df['validation_losses'].tolist()
        training_metric = df['training_metric'].tolist()
        validation_metric = df['validation_metric'].tolist()

        best_val_loss = min(validation_losses)
        print('training_losses: ', training_losses)
        print('validation_losses: ', validation_losses)
        print('training_metric: ', training_metric)
        print('validation_metric: ', validation_metric)
        print('best_val_loss', best_val_loss)
        print('Losses and metric loaded')

    for epoch in range(training_settings['nb_epochs']):
        if training_settings['restart_training'] and epoch < training_settings['restart_training']:
            print(f'Skipping epoch {epoch+1}/{training_settings["nb_epochs"]}')
            continue
        
        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}')
        # get time
        now = datetime.now()
        model.to(device)
        model.train()
        
        train_loss, tr_metric = train(model, train_dl, criterion, optimizer, device, model_settings['classes'], model_settings['model'])
        training_losses.append(train_loss)
        training_metric.append(tr_metric)
        model.eval()
        with torch.no_grad():
            print('Validation')
            val_loss, val_metric = valid_test(model, val_dl, criterion, device, model_settings['classes'], 'valid', model_settings['model'])
            validation_losses.append(val_loss)
            validation_metric.append(val_metric)

        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}: train loss {train_loss:.4f}, val loss {val_loss:.4f}')
        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}: train {metric} {tr_metric:.4f}, val {metric} {val_metric:.4f}')
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_settings['path_to_intermed_model'] + f'_epoch{epoch+1}.pt')
            torch.save(optimizer.state_dict(), model_settings['path_to_intermed_optim'] + f'_epoch{epoch+1}.pt')
        else:
            count += 1
        
        if training_settings['early_stopping'] and count == training_settings['patience']:
            print(f'Early stopping at epoch {epoch+1}')
            break

        df = pd.DataFrame({'training_losses': training_losses, 'validation_losses': validation_losses, f'training_{metric}': training_metric, f'validation_{metric}': validation_metric})
        df.to_csv(training_settings['losses_metric_path'])
        sys.stdout.flush()

        # one epoch time
        print('Time:', datetime.now()-now)
            
    # save last model state and optim
    torch.save(model.state_dict(), model_settings['path_to_last_model'])
    torch.save(optimizer.state_dict(), model_settings['path_to_last_optim'])
    # plot losses and metric using csv file and fct plot_losses_metrics
    plot_losses_metrics(training_settings['losses_metric_path'], plotting_settings['losses_path'], plotting_settings['metrics_path'], metric)

    # load epoch for which best_val
    model.load_state_dict(torch.load(model_settings['path_to_intermed_model'] + f'_epoch{np.argmin(validation_losses)+1}.pt'))
    for param in model.parameters():
        param.to(device)

else: 
    plot_losses_metrics(training_settings['losses_metric_path'], plotting_settings['losses_path'], plotting_settings['metrics_path'], metric)
    model.to(device)
    model.load_state_dict(torch.load(model_settings['path_to_best_model']))
    print('Model ', model_settings['path_to_best_model'], ' loaded')

    for param in model.parameters():
        param.to(device)

# TESTING
model.eval()
with torch.no_grad():
    print('Testing')
    test_loss, metrics = valid_test(model, test_dl, criterion, device, model_settings['classes'], 'test', model_settings['model'])
if model_settings['model'] == 'Unet':
    print(f'Test IoU by class: {metrics["IoU_by_class"]}')
    print(f'Test mIoU: {metrics["mIoU"]}')
    metrics['IoU_by_class']['mean'] = metrics['mIoU']
    metrics['IoU_by_class'] = {k: round(v, 2) for k, v in metrics['IoU_by_class'].items()}
    iou_df = pd.DataFrame(metrics['IoU_by_class'].items(), columns=['class', 'IoU'])
    iou_df.to_csv(plotting_settings['IoU_path'], index=False)

print(f'Test F1 by class: {metrics["F1_by_class"]}')
print(f'Test mF1: {metrics["mF1"]}')
# turn metrics['F1_by_class'] from array to dict
metrics['F1_by_class'] = {k: v for k, v in enumerate(metrics['F1_by_class'])}
metrics['F1_by_class']['mean'] = metrics['mF1']
metrics['F1_by_class'] = {k: round(v, 2) for k, v in metrics['F1_by_class'].items()}
f1_df = pd.DataFrame(metrics['F1_by_class'].items(), columns=['class', 'F1'])
f1_df.to_csv(plotting_settings['F1_path'], index=False)

#plot confusion matrix and save it
confusion_matrix = metrics['confusion_matrix']
confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
#sns.set(font_scale=0.8)
plt.figure(figsize=(10, 10))

ax = sns.heatmap(confusion_matrix_normalized, annot=True, fmt=".2f", cmap='Blues', cbar=False)#, xticklabels=, yticklabels=)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Normalized confusion matrix')
plt.savefig(plotting_settings['confusion_matrix_path'])

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
    plot_pred(img, msk, out, plotting_settings['pred_plot_path'], plotting_settings['my_colors_map'], plotting_settings['nb_plots'], plotting_settings['habitats_dict'], model_settings['task'])