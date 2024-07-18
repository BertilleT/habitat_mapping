from pathlib import Path
import pandas as pd
import rasterio
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib.patches import Patch
import torchvision.transforms.functional as F
# F1 score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
# multilabel
from sklearn.metrics import multilabel_confusion_matrix
import tifffile as tiff

not_mediterranean_zones = ['zone65', 'zone66', 'zone67', 'zone68', 'zone69','zone78',  'zone167', 'zone169', 'zone170', 'zone171', 'zone172']


def IoU_F1_from_confmatrix(conf_matrix):
    ious = {c: 0 for c in range(conf_matrix.shape[0])}
    f1s = {c: 0 for c in range(conf_matrix.shape[0])}
    for c in range(conf_matrix.shape[0]):
        TP = conf_matrix[c, c]
        FP = np.sum(conf_matrix[:, c]) - TP
        FN = np.sum(conf_matrix[c, :]) - TP
        i = TP
        u = TP + FP + FN
        if u != 0:
            ious[c] = i / u
        else:
            ious[c] = float('nan')
        f1s[c] = TP/(TP + 0.5*(FP + FN))
        if (TP + 0.5 * (FP + FN)) != 0: 
            f1s[c] = TP / (TP + 0.5 * (FP + FN))
        else:
            f1s[c] = float('nan') 
    return ious, f1s


## FUNCTION FOR TRAINING, VALIDATION AND TESTING
def train(model, train_dl, criterion, optimizer, device, nb_classes, model_name, labels, alpha1, alpha2, nb_output_heads):
    print('Training')
    running_loss = 0.0
    patch_confusion_matrices = []
    all_preds = []
    all_labels = []

    for i, (img, msk) in enumerate(train_dl):
        if i % 50 == 0:
            print( 'Batch:', i, ' over ', len(train_dl))
        img, msk = img.to(device), msk.to(device)
        optimizer.zero_grad()
        out = model(img)
        if labels == 'single':
            msk = msk.long() 
        #get shape of out
        loss = criterion(out, msk)
        # if nb_output_heads == 2
        # give a weight of 2 to the heterogenity term
        if labels == 'multi' and nb_output_heads == 2:
            w1 = 0
            w2 = 1
            #take only the 6 first rows of out
            habitats_loss = criterion(out[:, :-1], msk[:, :-1])
            fronteer_loss = criterion(out[:, -1], msk[:, -1])
            loss = w1 * habitats_loss + w2 * fronteer_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if model_name == 'UNet':
            out = torch.argmax(out, dim=1)
            out = out.int() #int means int32 on cpu and int64 on gpu
            patch_confusion_matrices.append(confusion_matrix(msk.flatten().cpu().numpy(), out.flatten().cpu().numpy(), labels=range(nb_classes)))
        elif model_name in ['Resnet18', 'Resnet34']:
            if labels == 'single':
                _, preds = torch.max(out, 1)
                # to int
                preds = preds.int()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(msk.cpu().numpy())
            elif labels == 'multi':
                # turn to 1 when proba is > 0.5
                out = torch.sigmoid(out)
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

                all_preds.extend(binary_pred.cpu().numpy())
                all_labels.extend(msk.cpu().numpy())

    if model_name == 'UNet':
        sum_confusion_matrix = np.sum(patch_confusion_matrices, axis=0)
        IoU_by_class, _ = IoU_F1_from_confmatrix(sum_confusion_matrix)
        IoU_by_class = {k: v for k, v in IoU_by_class.items() if not np.isnan(v)}
        mIoU = np.mean(list(IoU_by_class.values()))
        return running_loss / len(train_dl), mIoU


    elif model_name in ['Resnet18', 'Resnet34']:
        F1_by_class = f1_score(all_labels, all_preds, average=None)
        mF1 = np.mean(F1_by_class)
        return running_loss / len(train_dl), mF1

def valid_test(model, dl, criterion, device, nb_classes, valid_or_test, model_name, labels, alpha1, alpha2):
    running_loss = 0.0
    patch_confusion_matrices = []
    patch_confusion_matrices_1c = []
    patch_confusion_matrices_multic = []
    all_preds = []
    all_labels = []
    for i, (img, msk) in enumerate(dl):
        batch_confusion_matrices_1c = []
        batch_confusion_matrices_multic = []
        if i % 50 == 0:
            print( 'Batch:', i, ' over ', len(dl))
        img, msk = img.to(device), msk.to(device)
        out = model(img)
        if labels == 'single':
            msk = msk.long()
        loss = criterion(out, msk)
        running_loss += loss.item()
        if model_name == 'UNet':
            out = torch.argmax(out, dim=1)
            out = out.int()
            patch_confusion_matrices.append(confusion_matrix(msk.flatten().cpu().numpy(), out.flatten().cpu().numpy(), labels=range(nb_classes)))
        elif model_name in ['Resnet18', 'Resnet34']:
            if labels == 'single':
                _, preds = torch.max(out, 1)
                # to int
                preds = preds.int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(msk.cpu().numpy())
            elif labels == 'multi':
                # turn to 1 when proba is > alpha1 for heterogenity
                #apply sigmoid
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
                        
                all_preds.extend(binary_pred.cpu().numpy())
                all_labels.extend(msk.cpu().numpy())

    if model_name == 'UNet':   
        sum_confusion_matrix = np.sum(patch_confusion_matrices, axis=0)
        IoU_by_class, F1_by_class = IoU_F1_from_confmatrix(sum_confusion_matrix)
        IoU_by_class = {k: v for k, v in IoU_by_class.items() if not np.isnan(v)}
        mIoU = np.mean(list(IoU_by_class.values()))
        mF1 = np.mean(list(F1_by_class.values()))
        metrics = {
            'confusion_matrix': sum_confusion_matrix,
            'IoU_by_class': IoU_by_class,
            'F1_by_class': F1_by_class,
            'mIoU': mIoU,
            'mF1': mF1
        }
    elif model_name in ['Resnet18', 'Resnet34']:
        if labels == 'single':
            cm = confusion_matrix(all_labels, all_preds, labels=range(nb_classes))
        elif labels == 'multi':
            # multi labels conf matrix
            cm = multilabel_confusion_matrix(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds) # normalize=True by default 
        F1_by_class = f1_score(all_labels, all_preds, average=None)
        mF1 = np.mean(F1_by_class)

        metrics = {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'F1_by_class': F1_by_class,
            'mF1': mF1
        }

    if valid_or_test == 'valid':
        if model_name == 'UNet':
            return running_loss / len(dl), mIoU
        else:
            return running_loss / len(dl), mF1
    else:
        return running_loss / len(dl), metrics

def tune_alpha1_valid(model, dl, device, alpha1):
    all_preds = []
    all_labels = []
    for i, (img, msk) in enumerate(dl):
        if i % 50 == 0:
            print( 'Batch:', i, ' over ', len(dl))
        img, msk = img.to(device), msk.to(device) # img is a tensor of shape (batch_size, 4, 256, 256)
        out = model(img)
        out = torch.sigmoid(out)
        # take only the last value of the vector out to get the heterogenity
        heterogenity_predicted = out[:, -1] 
        # turn to 1 when proba is > alpha1
        heterogenity_predicted = torch.where(heterogenity_predicted >= alpha1, torch.tensor(1).to(device), torch.tensor(0).to(device)) # 
        all_preds.extend(heterogenity_predicted.cpu().numpy())

        true_heterogenity = msk[:, -1]
        all_labels.extend(true_heterogenity.cpu().numpy())

    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds)
    F1 = f1_score(all_labels, all_preds, zero_division=1)
    return precision, recall, F1

def tune_alpha2_valid(model, dl, device, alpha1, alpha2):
    all_preds = []
    all_labels = []
    for i, (img, msk) in enumerate(dl):
        if i % 50 == 0:
            print( 'Batch:', i, ' over ', len(dl))
        img, msk = img.to(device), msk.to(device) # img is a tensor of shape (batch_size, 4, 256, 256)
        out = model(img)
        out = torch.sigmoid(out)
        out_classes_ecosystems = out[:, :-1]
        # turn to 1 when proba is > alpha2
        predicted_class = torch.argmax(out_classes_ecosystems, dim=1)
        binary_pred = torch.where(out_classes_ecosystems >= alpha2, torch.tensor(1).to(device), torch.tensor(0).to(device))
        for i, vector in enumerate(out):
            # if homogenity is predicted
            if vector[-1] < alpha1:
                # max proba from out_classes_ecosystems
                max_class = torch.argmax(out_classes_ecosystems[i])
                binary_pred[i, :] = 0
                binary_pred[i, max_class] = 1    
        all_preds.extend(binary_pred.cpu().numpy())   
        # remoe the last column from msk
        true_classes_ecosystems = msk[:, :-1]
        all_labels.extend(true_classes_ecosystems.cpu().numpy())    
    
    precision_by_class = precision_score(all_labels, all_preds, zero_division=1, average=None)
    recall_by_class = recall_score(all_labels, all_preds, average=None, zero_division=1)
    F1_by_class = f1_score(all_labels, all_preds, zero_division=1, average=None)
    return precision_by_class, recall_by_class, F1_by_class

def optimizer_to(optim, device):
    # get number of values
    i = 0
    for param in optim.state.values(): 
        i += 1
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)