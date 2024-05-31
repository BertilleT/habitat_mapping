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
#print(f'Train: {len(train_paths)} images, Val: {len(val_paths)} images, Test: {len(test_paths)} images')
train_ds = EcomedDataset(train_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'])
train_dl = DataLoader(train_ds, batch_size=data_loading_settings['bs'], shuffle=True)
val_ds = EcomedDataset(val_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'])
val_dl = DataLoader(val_ds, batch_size=data_loading_settings['bs'], shuffle=False)
test_ds = EcomedDataset(test_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'])
test_dl = DataLoader(test_ds, batch_size=data_loading_settings['bs'], shuffle=False)

# print size of train, val and test and proportion it rperesents compared to the total size of the dataset
print(f'Train: {len(train_ds)} images, Val: {len(val_ds)} images, Test: {len(test_ds)} images')
print(f'Train: {len(train_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%, Val: {len(val_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%, Test: {len(test_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%')

# print shape of images and masks
print('Shape of images and masks:')
img, msk = next(iter(train_dl))
print('Image shape:', img.shape)
# stotre train_imgs_id = [img_id1, img_id2, ...]. Zone_id can be found 
'''sys.stdout.flush()
print('Checking classes balance of train')
train_classes = check_classes_balance(train_dl)
print(train_classes)
val_classes = check_classes_balance(val_dl)
print(val_classes)
test_classes = check_classes_balance(test_dl)
print(test_classes)

# turn dict to pandas df
balance_classes = pd.DataFrame(train_classes.items(), columns=['class', 'train'])
balance_classes['val'] = balance_classes['class'].map(val_classes)
balance_classes['test'] = balance_classes['class'].map(test_classes)
balance_classes['total'] = balance_classes['train'] + balance_classes['val'] + balance_classes['test']
balance_classes = balance_classes.set_index('class')

for col in balance_classes.columns:
    balance_classes[col] = round(balance_classes[col]*100 / balance_classes[col].sum(), 2)
print(balance_classes)
# save df to csv
balance_classes.to_csv(data_loading_settings['classes_balance'], index=False)'''

sys.stdout.flush()
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
    classes=model_settings['classes'], 
)

if training_settings['restart_training'] is not None:
    model.load_state_dict(torch.load(model_settings['path_to_last_model']))
    print('Model from last epoch', training_settings['restart_training'], ' loaded')
model.to(device)

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

if training_settings['restart_training'] is not None:
    torch.cuda.empty_cache()
    optimizer.load_state_dict(torch.load(model_settings['path_to_last_optim']))
    #optimizer_to(optimizer,device)
    #device = next(optimizer.param_groups[0]['params']).device
    #print("Optimizer is running on:", device)
    #print('Optimizer from epoch', training_settings['restart_training'], ' loaded')

## METRIC

## TRAINING AND VALIDATION
if training_settings['training']:
    print('Training...')
    training_losses = []
    validation_losses = []
    training_miou = []
    validation_miou = []
    count = 0
    best_val_loss = np.inf 

    if training_settings['restart_training']:
        # load losses
        df = pd.read_csv(training_settings['losses_mious_path'])
        training_losses = df['training_losses'].tolist()
        validation_losses = df['validation_losses'].tolist()
        training_miou = df['training_miou'].tolist()
        validation_miou = df['validation_miou'].tolist()

        best_val_loss = min(validation_losses)
        print('training_losses: ', training_losses)
        print('validation_losses: ', validation_losses)
        print('training_miou: ', training_miou)
        print('validation_miou: ', validation_miou)
        print('best_val_loss', best_val_loss)
        print('Losses and iou loaded')

    for epoch in range(training_settings['nb_epochs']):
        if training_settings['restart_training'] and epoch < training_settings['restart_training']:
            print(f'Skipping epoch {epoch+1}/{training_settings["nb_epochs"]}')
            continue
        
        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}')
        # get time
        now = datetime.now()
        model.to(device)
        model.train()
        
        train_loss, tr_miou = train(model, train_dl, criterion, optimizer, device, model_settings['classes'])
        training_losses.append(train_loss)
        training_miou.append(tr_miou)
        model.eval()
        with torch.no_grad():
            print('Validation')
            val_loss, val_mIoU = valid_test(model, val_dl, criterion, device, model_settings['classes'], 'valid')
            validation_losses.append(val_loss)
            validation_miou.append(val_mIoU)

        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}: train loss {train_loss:.4f}, val loss {val_loss:.4f}')
        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}: train mIoU {tr_miou:.4f}, val mIoU {val_mIoU:.4f}')
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
            
        #every 10, save losses values in csv
        #if (epoch+1) % 5 == 0:
        df = pd.DataFrame({'training_losses': training_losses, 'validation_losses': validation_losses, 'training_miou': training_miou, 'validation_miou': validation_miou})
        df.to_csv(training_settings['losses_mious_path'])
        sys.stdout.flush()

        # time
        # one epoch time
        print('Time:', datetime.now()-now)
            
    # save last model state and optim
    torch.save(model.state_dict(), model_settings['path_to_last_model'])
    torch.save(optimizer.state_dict(), model_settings['path_to_last_optim'])
    # plot losses and iou using csv file and fct plot_losses_ious
    plot_losses_ious(training_settings['losses_mious_path'], plotting_settings['losses_path'], plotting_settings['mious_path'])

    # load epoch for which best_val
    model.load_state_dict(torch.load(model_settings['path_to_intermed_model'] + f'_epoch{np.argmin(validation_losses)+1}.pt'))
    for param in model.parameters():
        param.to(device)

else: 
    plot_losses_ious(training_settings['losses_mious_path'], plotting_settings['losses_path'], plotting_settings['mious_path'])
    model.to(device)
    model.load_state_dict(torch.load(model_settings['path_to_best_model']))
    print('Model ', model_settings['path_to_best_model'], ' loaded')

    for param in model.parameters():
        param.to(device)

# TESTING
model.eval()
with torch.no_grad():
    print('Testing')
    test_loss, metrics = valid_test(model, test_dl, criterion, device, model_settings['classes'], 'test')
print(f'Test IoU by class: {metrics["IoU_by_class"]}')
print(f'Test F1 by class: {metrics["F1_by_class"]}')
print(f'Test mIoU: {metrics["mIoU"]}')
print(f'Test mF1: {metrics["mF1"]}')
#save metrics to csv
# add one key value to metrics IouBy clas called mIoU with the value of mIoU
metrics['IoU_by_class']['mean'] = metrics['mIoU']
#same for F1
metrics['F1_by_class']['mean'] = metrics['mF1']
# from metrics['IoU_by_class'] and metrics['F1_by_class'] create a df and save it to csv
iou_df = pd.DataFrame(metrics['IoU_by_class'].items(), columns=['class', 'IoU'])
iou_df.to_csv(plotting_settings['IoU_path'], index=False)
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
    plot_pred(img, msk, out, plotting_settings['pred_plot_path'], plotting_settings['my_colors_map'], plotting_settings['nb_plots'])