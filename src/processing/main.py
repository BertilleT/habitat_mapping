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

#laod one img and msk
img, msk = next(iter(train_dl))
# print img high and low values
print(f'Image: min: {img.min()}, max: {img.max()}')
# print unique values of the mask
print(f'Mask unique values: {np.unique(msk[0])}')
# nb of dimensions of the mask
print(f'Mask nb of dimensions: {msk[0].shape}')
'''
train_ds_plot = EcomedDataset_to_plot(train_paths, data_loading_settings['img_folder'], channels = model_settings['in_channels'], transform = [transform_rgb, transform_all_channels], task = model_settings['task'])
plot_patch_class_by_class(train_ds_plot, 20, plotting_settings['habitats_dict'], plotting_settings['l2_habitats_dict'], 'training set')
val_ds_plot = EcomedDataset_to_plot(val_paths, data_loading_settings['img_folder'], channels = model_settings['in_channels'], task = model_settings['task'])
plot_patch_class_by_class(val_ds_plot, 20, plotting_settings['habitats_dict'], plotting_settings['l2_habitats_dict'], 'validation set')
test_ds_plot = EcomedDataset_to_plot(test_paths, data_loading_settings['img_folder'], channels = model_settings['in_channels'], task = model_settings['task'])
plot_patch_class_by_class(test_ds_plot, 20, plotting_settings['habitats_dict'], plotting_settings['l2_habitats_dict'], 'test set')
'''
# print size of train, val and test and proportion it rperesents compared to the total size of the dataset
print(f'Train: {len(train_ds)} images, Val: {len(val_ds)} images, Test: {len(test_ds)} images')
print(f'Train: {len(train_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%, Val: {len(val_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%, Test: {len(test_ds)/len(train_ds+val_ds+test_ds)*100:.2f}%')

sys.stdout.flush()
## MODEL
print('Creating model...')
print('Model settings:')
print(f'Pretrained: {model_settings["pre_trained"]}')
print(f'Classes: {model_settings["classes"]}')
if model_settings['model'] == 'UNet':
    print('The model is UNet')
    print(f'Encoder name: {model_settings["encoder_name"]}')
    model = smp.Unet(
        encoder_name=model_settings['encoder_name'],        
        encoder_weights=model_settings['encoder_weights'], 
        in_channels=model_settings['in_channels'], 
        classes=model_settings['classes'], 
    )

elif model_settings['model'] in ['Resnet18', 'Resnet34']:
    print('The model is: ', model_settings['model'])
    model = Ecomed_ResNet(model_settings)
      
if training_settings['restart_training'] is not None:
    model.load_state_dict(torch.load(model_settings['path_to_last_model']))
    print('Model from last epoch', training_settings['restart_training'], ' loaded')
model.to(device)

# METRIC
if model_settings['model'] == 'UNet':
    metric = 'mIoU'
elif model_settings['model'] in ['Resnet18', 'Resnet34']:
    metric = 'mF1'

# OPTIMIZER
if training_settings['criterion'] == 'CrossEntropy':
    criterion = nn.CrossEntropyLoss()
elif training_settings['criterion'] == 'Dice':
    criterion = smp.losses.DiceLoss(mode='multiclass', eps=0.0000001)
elif training_settings['criterion'] == 'BCEWithDigits':
    criterion = smp.losses.SoftBCEWithLogitsLoss()
    print('Using BCEWithDigits criterion')
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
    best_val_metric = 0

    if training_settings['restart_training']:
        # load losses
        df = pd.read_csv(training_settings['losses_metric_path'])
        training_losses = df['training_losses'].tolist()
        validation_losses = df['validation_losses'].tolist()
        training_metric = df['training_mF1'].tolist()
        validation_metric = df['validation_mF1'].tolist()

        best_val_loss = min(validation_losses)
        best_val_metric = max(validation_metric)
        print('training_losses: ', training_losses)
        print('validation_losses: ', validation_losses)
        print('training_metric: ', training_metric)
        print('validation_metric: ', validation_metric)
        print('best_val_loss', best_val_loss)
        print('best_val_metric', best_val_metric)
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
        
        train_loss, tr_metric = train(model, train_dl, criterion, optimizer, device, model_settings['classes'], model_settings['model'], model_settings['labels'], training_settings['alpha1'], training_settings['alpha2'], model_settings['nb_output_heads'])
        training_losses.append(train_loss)
        training_metric.append(tr_metric)
        model.eval()
        with torch.no_grad():
            print('Validation')
            val_loss, val_metric = valid_test(model, val_dl, criterion, device, model_settings['classes'], 'valid', model_settings['model'], model_settings['labels'], training_settings['alpha1'], training_settings['alpha2'])
            validation_losses.append(val_loss)
            validation_metric.append(val_metric)

        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}: train loss {train_loss:.4f}, val loss {val_loss:.4f}')
        print(f'Epoch {epoch+1}/{training_settings["nb_epochs"]}: train {metric} {tr_metric:.4f}, val {metric} {val_metric:.4f}')
        
        if val_loss <= best_val_loss or val_metric >= best_val_metric:
            count = 0
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
            elif val_metric >= best_val_metric:
                best_val_metric = val_metric
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
    #plot_losses_metrics(training_settings['losses_metric_path'], plotting_settings['losses_path'], plotting_settings['metrics_path'], metric)
    model.to(device)
    model.load_state_dict(torch.load(model_settings['path_to_best_model']))
    print('Model ', model_settings['path_to_best_model'], ' loaded')

    for param in model.parameters():
        param.to(device)

# TUNING ALPHA1 FOR MULTI LABEL CLASSIFICATION
if training_settings['tune_alpha1']:
    model.eval()
    print('Tuning alpha1')
    alpha1_values = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    print(alpha1_values)
    precisions = []
    recalls = []
    f1s = []
    for alpha1 in alpha1_values:
        print(f'Alpha1: {alpha1}')
        with torch.no_grad():
            print('Tuning alpha1 on validation set')
            precision, recall, f1 = tune_alpha1_valid(model, val_dl, device, alpha1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
    # close last fig
    plt.close()
    #plot precision against recall. Precision in y axis, recall in x axis. Add the index of each point

    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    for i, txt in enumerate(alpha1_values):
        plt.annotate(txt, (recalls[i], precisions[i]))
    # add a red line horizontal line at 1 from 0 yo 1 x axis
    plt.axhline(y=1, color='r', linestyle='--')
    # add a red line vertical line at 1 from 0 yo 1 y axis
    plt.axvline(x=1, color='r', linestyle='--')
        
    plt.title('Precision against Recall')
    plt.savefig('precision_recall.png')
    #close
    plt.close()

    print(np.argmax(f1s))
    best_alpha1 = alpha1_values[np.argmax(f1s)]
    #plot f1s against alpha1
    plt.plot(alpha1_values, f1s)
    plt.xlabel('Alpha1')
    plt.ylabel('F1')
    plt.title('F1 against Alpha1')
    # set in red the best alpha1
    plt.axvline(x=best_alpha1, color='r', linestyle='--')
    plt.savefig('f1_alpha1.png')
    print(f'Best alpha1: {best_alpha1}')


# TUNING ALPHA2 FOR MULTI LABEL CLASSIFICATION
if training_settings['tune_alpha2']:
    alpha1 = 0.6
    model.eval()
    alpha2_values = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    precisions = []
    recalls = []
    f1s = []
    for alpha2 in alpha2_values:
        print(f'Alpha2: {alpha2}')
        with torch.no_grad():
            print('Tuning alpha2 on validation set')
            precision, recall, f1 = tune_alpha2_valid(model, val_dl, device, alpha1, alpha2)
            mean_precision = np.mean(precision)
            mean_recall = np.mean(recall)
            mean_f1 = np.mean(f1)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
            print(f'MEAN Precision: {mean_precision}, Recall: {mean_recall}, F1: {mean_f1}')
            # mean precision, recall and f1
    print(np.argmax(f1s))

    # select only precision and recall of the class 0
    precisions_0 = [precision[0] for precision in precisions]
    recalls_0 = [recall[0] for recall in recalls]
    precision_1 = [precision[1] for precision in precisions]
    recall_1 = [recall[1] for recall in recalls]
    precisions_2 = [precision[2] for precision in precisions]
    recalls_2 = [recall[2] for recall in recalls]
    precisions_3 = [precision[3] for precision in precisions]
    recalls_3 = [recall[3] for recall in recalls]
    precisions_4 = [precision[4] for precision in precisions]
    recalls_4 = [recall[4] for recall in recalls]
    precisions_5 = [precision[5] for precision in precisions]
    recalls_5 = [recall[5] for recall in recalls]

    plt.plot(recalls_0, precisions_0, color='#789262')
    plt.plot(recall_1, precision_1, color='#555555')
    plt.plot(recalls_2, precisions_2, color='#006400')
    plt.plot(recalls_3, precisions_3, color='#00ff00')
    plt.plot(recalls_4, precisions_4, color='#ff4500')
    plt.plot(recalls_5, precisions_5, color='#8a2be2')
    plt.plot
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision against Recall by class')
    plt.savefig('precision_recall_by_class.png')
    #close
    plt.close()

    # drop the 5th for each precision and recall
    precisions = [precision[:5] for precision in precisions]
    recalls = [recall[:5] for recall in recalls]
    # mean precision and recall
    mean_precisions = [np.mean(precision) for precision in precisions]
    mean_recalls = [np.mean(recall) for recall in recalls]

    # plot mean precision against mean recall
    plt.plot(mean_recalls, mean_precisions)
    for i, txt in enumerate(alpha2_values):
        plt.annotate(txt, (mean_recalls[i], mean_precisions[i]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.axvline(x=1, color='r', linestyle='--')
    plt.title('Mean Precision against Mean Recall')
    plt.savefig('mean_precision_recall.png')    
    #close
    plt.close()

    #plot F1 against alpha2 by color
    f1s_0 = [f1[0] for f1 in f1s]
    f1s_1 = [f1[1] for f1 in f1s]
    f1s_2 = [f1[2] for f1 in f1s]
    f1s_3 = [f1[3] for f1 in f1s]
    f1s_4 = [f1[4] for f1 in f1s]
    f1s_5 = [f1[5] for f1 in f1s]

    plt.plot(alpha2_values, f1s_0, color='#789262')
    plt.plot(alpha2_values, f1s_1, color='#555555')
    plt.plot(alpha2_values, f1s_2, color='#006400')
    plt.plot(alpha2_values, f1s_3, color='#00ff00')
    plt.plot(alpha2_values, f1s_4, color='#ff4500')
    plt.plot(alpha2_values, f1s_5, color='#8a2be2')
    plt.xlabel('Alpha2')
    plt.ylabel('F1')
    plt.title('F1 against Alpha2 by class')
    plt.savefig('f1_alpha2_by_class.png')
    #close
    plt.close()

    # remove F1 of the 5th class
    f1s = [f1[:5] for f1 in f1s]
    #plot mean f1 against alpha2
    mean_f1s = [np.mean(f1) for f1 in f1s]
    plt.plot(alpha2_values, mean_f1s)
    plt.xlabel('Alpha2')
    plt.ylabel('F1')
    plt.title('Mean F1 against Alpha2')
    plt.axvline(x=alpha2_values[np.argmax(mean_f1s)], color='r', linestyle='--')
    plt.savefig('mean_f1_alpha2.png')
    #close
    plt.close()

# TESTING
if training_settings['testing']:
    model.eval()
    with torch.no_grad():
        print('Testing')
        test_loss, metrics = valid_test(model, test_dl, criterion, device, model_settings['classes'], 'test', model_settings['model'], model_settings['labels'], training_settings['alpha1'], training_settings['alpha2'])
    if model_settings['model'] == 'UNet':
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

    if labels == 'single':
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

    elif labels == 'multi':
        #plot confusion matrix and save it
        confusion_matrices = metrics['confusion_matrix']
        # normalize each conf matrix from confusion_matrices
        confusion_matrices_normalized = []
        for conf_matrix in confusion_matrices:
            conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            confusion_matrices_normalized.append(conf_matrix_normalized)
        #plot each conf matrixes 
        for i, conf_matrix_normalized in enumerate(confusion_matrices_normalized):
            plt.figure(figsize=(10, 10))
            ax = sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap='Blues', cbar=False)#, xticklabels=, yticklabels=)
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title(f'Normalized confusion matrix for class {i}')
            plt.savefig(plotting_settings['confusion_matrix_path'].replace('.png', f'_class_{i}.png'))
        
        confusion_matrix_homo = metrics['confusion_matrix_homo']
        #plot it
        

# PLOTTING TEST PREDICTIONS
if plotting_settings['plot_test']:
    alpha1 = training_settings['alpha1']
    alpha2 = training_settings['alpha2']
    # plot img, original mask and prediction
    model.eval()
    test_dl_iter = iter(test_dl)
    first_batch = next(test_dl_iter)
    img, msk = next(test_dl_iter)
    img, msk = img.to(device), msk.to(device)
    out = model(img)
    if model_settings['labels'] == 'single':
        out = torch.argmax(out, dim=1)
        out = out.int()
    elif model_settings['labels'] == 'multi':
        out = torch.sigmoid(out)
        # remove heterogenity from out
        out_classes_ecosystems = out[:, :-1]
        # turn to 1 when proba is > alpha1 for heterogenity
        heterogenity_predicted = torch.where(out[:, -1] >= alpha1, torch.tensor(1).to(device), torch.tensor(0).to(device))
        # turn to 1 when proba is > alpha2 for each class except heterogenity
        binary_pred = torch.where(out_classes_ecosystems >= alpha2, torch.tensor(1).to(device), torch.tensor(0).to(device))
        # concat heterogenity to binary_pred
        binary_pred = torch.cat((binary_pred, heterogenity_predicted.unsqueeze(1)), dim=1)
            
        for i, vector in enumerate(out):
            # if homogenity is predicted
            if vector[-1] < alpha1:
                # keep the class with the max proba
                max_class = torch.argmax(out_classes_ecosystems[i])
                binary_pred[i, :-1] = 0
                binary_pred[i, max_class] = 1  
        out = binary_pred 
    img = img.cpu().numpy()
    msk = msk.cpu().numpy()
    out = out.cpu().numpy()
    plot_pred(img, msk, out, plotting_settings['pred_plot_path'], plotting_settings['my_colors_map'], plotting_settings['nb_plots'], plotting_settings['habitats_dict'], model_settings['task'], model_settings['labels'])

if plotting_settings['plot_re_assemble']:
    model.eval()
    model.to('cpu')

    new_colors_maps = {k: v for k, v in plotting_settings['my_colors_map'].items()}
    if patch_level_param['level'] == 1:
        new_colors_maps[6] = '#000000'  # Noir
        new_colors_maps[7] = '#c7c7c7'  # Gris
        new_colors_maps[8] = '#ffffff'  # Blanc
    elif patch_level_param['level'] == 2:
        new_colors_maps[17] = '#000000'
        new_colors_maps[18] = '#c7c7c7'
        new_colors_maps[19] = '#ffffff'

    customs_color = list(new_colors_maps.values())
    bounds = list(new_colors_maps.keys())
    my_cmap = plt.cm.colors.ListedColormap(customs_color)
    my_norm = plt.cm.colors.BoundaryNorm(bounds, my_cmap.N)

    zones = ['zone1_0_0', 'zone100_0_0', 'zone133_0_0']
    for zone in zones: 
        msk_paths = list(data_loading_settings['msk_folder'].rglob('*.tif'))
        msk_paths = [msk_path for msk_path in msk_paths if zone in msk_path.stem]

        dataset = EcomedDataset(msk_paths, data_loading_settings['img_folder'], level=patch_level_param['level'], channels = model_settings['in_channels'], normalisation = data_loading_settings['normalisation'], task = model_settings['task'], my_set = "test", labels = model_settings['labels'], path_mask_name = True)
        plot_reassembled_patches(zone, model, dataset, patch_level_param['patch_size'], training_settings['alpha1'], my_cmap, my_norm, plotting_settings['re_assemble_patches_path'][:-4])
        if plotting_settings['post_processing']:
            i_indices, j_indices, true_classes, true_heterogenity, predicted_classes, predicted_heterogenity, probability_vectors = plot_reassembled_patches(zone, model, dataset, patch_level_param['patch_size'], training_settings['alpha1'], None, None, plotting_settings['re_assemble_patches_path'][:-4], post_processing = True) 
            # FIRST STEP: 
            # BUILD A DATAFRAME WITH COLUMN I, J, TRUE CLASS, TRUE HETEROGENITY, CLASS PREDICTED, HETEROGENITY PREDICTED
            df = pd.DataFrame({'i': i_indices, 'j': j_indices, 'true_class': true_classes, 'true_heterogenity': true_heterogenity, 'predicted_class': predicted_classes, 'predicted_heterogenity': predicted_heterogenity, 'probability_vector': probability_vectors})
            # probability vector must have only 6 items. Keep only the 6 first items
            df['probability_vector'] = df['probability_vector'].apply(lambda x: [float(t.item()) for t in x[0][:6]])
            # print item with i = 0 and j = 36
            # FILTER TO KEEP ONLY PATCHES PREDICTED AS HOMOGENOUS WHEN PREDICTED_HETEROGENITY IS 0
            homogenous_df = df[df['predicted_heterogenity'] == 0]
            true_classes = homogenous_df['true_class'].tolist()
            predicted_classes = homogenous_df['predicted_class'].tolist()
            f1_by_class = f1_score(true_classes, predicted_classes, average=None)
            print('f1_by_class: ', f1_by_class)

            # SECOND STEP
            # LOOP ON DATAFRAME,
            # for i, j in homogenous_df: 
            for i, j in zip(homogenous_df['i'], homogenous_df['j']):
                old_class = homogenous_df[(homogenous_df['i'] == i) & (homogenous_df['j'] == j)]['predicted_class'].values[0]
                energies = energie(i, j, homogenous_df, training_settings['beta'])
                # select the index of the vector with minimum energy
                new_class = np.argmin(energies) # return the index of the minimum value
                # update predicted class from homogenous_df with index i and j with new_class
                homogenous_df.loc[(homogenous_df['i'] == i) & (homogenous_df['j'] == j), 'predicted_class'] = new_class

            post_predicted_classes = homogenous_df['predicted_class'].tolist()
            post_f1_by_class = f1_score(true_classes, post_predicted_classes, average=None)

            print('post_f1_by_class', post_f1_by_class)

            # check homogenous_df i = 0 and j = 36 ? 
            # reassamble the image with the new predicted classes
            plot_reassembled_patches(zone, model, dataset, patch_level_param['patch_size'], training_settings['alpha1'], my_cmap, my_norm, plotting_settings['re_assemble_patches_path'][:-4], False, True, homogenous_df)