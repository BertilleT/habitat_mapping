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

not_mediterranean_zones = ['zone65', 'zone66', 'zone67', 'zone68', 'zone69','zone78',  'zone167', 'zone169', 'zone170', 'zone171', 'zone172']

class EcomedDataset(Dataset):
    def __init__(self, msk_paths, img_dir, level=1, channels=4, transform = None, normalisation = "all_channels_together", task = "pixel_classif"):
        self.img_dir = img_dir
        self.level = level
        self.msks = msk_paths
        self.imgs = [self.img_dir / msk_path.parts[-2] / msk_path.name.replace('msk', 'img').replace('l123/', '') for msk_path in self.msks]
        self.channels = channels
        self.transform = transform
        self.normalisation = normalisation
        self.task = task

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        msk_path = self.msks[idx]
        with rasterio.open(msk_path) as src:
            msk = src.read(self.level)
            # print unique values in msk
            img_label = msk[0, 0]

            #get type of values in msk
            if self.level == 1:
                group_under_represented_classes = {0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 2, 6: 5, 7: 3, 8: 4, 9: 5}
                group_under_represented_classes_uint8 = {np.uint8(k): np.uint8(v) for k, v in group_under_represented_classes.items()}
                #if tasl is pixel_classif, then group under represented classes
                if self.task == "pixel_classif":
                    msk_mapped = np.vectorize(group_under_represented_classes_uint8.get)(msk)
                elif self.task == "image_classif":  
                    # map value from img_label using group_under_represented_classes_uint8
                    img_label_mapped = group_under_represented_classes_uint8[img_label]
            else:
                msk_mapped = msk
                img_label_mapped = img_label

        with rasterio.open(img_path) as src:
            img = src.read()
            if self.channels == 3:
                img = img[:3]
            # turn to values betw 0 and 1 and to float
            # linear normalisation with p2 and p98
            if self.normalisation == "all_channels_together":
                p2, p98 = np.percentile(img, (2, 98))
                img = np.clip(img, p2, p98)
                img = (img - p2) / (p98 - p2)
                normalized_img = img.astype(np.float32)
            elif self.normalisation == "channel_by_channel":
                #print('Normalisation channel by channel')
            # Normalize each channel separately
                normalized_img = np.zeros_like(img, dtype=np.float32)
                for c in range(self.channels):
                    channel = img[c, :, :]
                    p2, p98 = np.percentile(channel, (2, 98))
                    channel = np.clip(channel, p2, p98)
                    channel = (channel - p2) / (p98 - p2)
                    normalized_img[c, :, :] = channel.astype(np.float32)
        if self.transform:
            if self.task == "pixel_classif":
                augmented = self.transform(image=normalized_img[0:3].transpose(1, 2, 0), mask=msk_mapped)
                aug_img = augmented['image']
                msk_mapped = augmented['mask']
                # add fourth channel from img to img
                img = np.concatenate((aug_img, normalized_img[3:4]), axis=0)
            elif self.task == "image_classif":
                augmented = self.transform(image=normalized_img.transpose(1, 2, 0))
                img = augmented['image']
        if self.task == "pixel_classif":
            return normalized_img, msk_mapped
        elif self.task == "image_classif":
            return normalized_img, img_label_mapped


class EcomedDataset_to_plot(Dataset):
    def __init__(self, msk_paths, img_dir, channels=4, transform = None, task = "pixel_classif"):
        self.img_dir = img_dir
        self.msks = msk_paths
        self.imgs = [self.img_dir / msk_path.parts[-2] / msk_path.name.replace('msk', 'img').replace('l123/', '') for msk_path in self.msks]
        self.channels = channels
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        msk_path = self.msks[idx]
        with rasterio.open(msk_path) as src:
            msk = src.read(1)
            # print unique values in msk
            img_label_1 = msk[0, 0]

            #get type of values in msk
            group_under_represented_classes = {0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 2, 6: 5, 7: 3, 8: 4, 9: 5}
            group_under_represented_classes_uint8 = {np.uint8(k): np.uint8(v) for k, v in group_under_represented_classes.items()}
            #if tasl is pixel_classif, then group under represented classes
            if self.task == "pixel_classif":
                msk_mapped = np.vectorize(group_under_represented_classes_uint8.get)(msk)
            elif self.task == "image_classif":  
                # map value from img_label using group_under_represented_classes_uint8
                img_label_mapped = group_under_represented_classes_uint8[img_label_1]
        with rasterio.open(msk_path) as src:
            msk = src.read(2)
            # img_label_2 is the class the most present in msk_patch
            img_label_2 = np.argmax(np.bincount(msk.flatten()))

        with rasterio.open(img_path) as src:
            img = src.read()
            if self.channels == 3:
                img = img[:3]
            normalized_img = np.zeros_like(img, dtype=np.float32)
            for c in range(self.channels):
                channel = img[c, :, :]
                p2, p98 = np.percentile(channel, (2, 98))
                channel = np.clip(channel, p2, p98)
                channel = (channel - p2) / (p98 - p2)
                normalized_img[c, :, :] = channel.astype(np.float32)
        if self.transform:
            if self.task == "pixel_classif":
                augmented = self.transform(image=normalized_img[0:3].transpose(1, 2, 0), mask=msk_mapped)
                aug_img = augmented['image']
                msk_mapped = augmented['mask']
                # add fourth channel from img to img
                img = np.concatenate((aug_img, normalized_img[3:4]), axis=0)
            elif self.task == "image_classif":
                augmented = self.transform(image=normalized_img.transpose(1, 2, 0))
                img = augmented['image']

        if self.task == "pixel_classif":
            return normalized_img, msk_mapped # deal with label at level 2. 
        elif self.task == "image_classif":
            return normalized_img, img_label_mapped, img_label_2

def load_data_paths(img_folder, msk_folder, stratified, random_seed, split, **kwargs):
    print('The seed to shuffle the data is ', str(random_seed))
    print('The data are from ', kwargs['year'], ' year')
    #extract unique zones from alltif names
    if kwargs['patches'] == 'heterogeneous':
        #load the csv file with the list of heterogen masks '../../csv/heterogen_masks.csv'
        heterogen_masks = pd.read_csv(kwargs['heterogen_patches_path'])
        msk_paths = heterogen_masks['0'].values # get the values of the column 
        # turn to paths
        msk_paths = [Path(msk_path) for msk_path in msk_paths]
        print(len(msk_paths), 'heterogneous masks found')
    elif kwargs['patches'] == 'homogeneous':
        msk_paths = list(msk_folder.rglob('*.tif'))
        heterogen_masks = pd.read_csv(kwargs['heterogen_patches_path'])
        heterogen_masks_ = heterogen_masks['0'].values
        heterogen_masks_ = [Path(msk_path) for msk_path in heterogen_masks_]
        msk_paths = [msk_path for msk_path in msk_paths if msk_path not in heterogen_masks_]
        print(len(msk_paths), 'homogeneous masks found')

    elif kwargs['patches'] == 'all':
        msk_paths = list(msk_folder.rglob('*.tif'))
    if kwargs['year'] == '2023':
        zones_2023 = pd.read_csv(kwargs['2023_zones'])
        msk_paths = [msk_path for msk_path in msk_paths if str(msk_path).split('/')[-2].split('_')[0] in zones_2023['zone_AJ'].values]
        #msk_paths = [msk_path for msk_path in msk_paths if str(msk_path).split('/').parts[-2].split('_')[0] in zones_2023['zone_id'].values]
        print(len(msk_paths), ' kept masks from 2023 zones')
        print('The data are from 2023')
    if stratified == 'random':
        # Shuffle with random_seed fixed and split in 60 20 20
        np.random.seed(random_seed)
        np.random.shuffle(msk_paths)
        n = len(msk_paths)
        train_paths = msk_paths[:int(split[0]*n)]
        val_paths = msk_paths[int(split[0]*n):int((split[0]+split[1])*n)]
        test_paths = msk_paths[int((split[0]+split[1])*n):]
    elif stratified == 'zone' or stratified == 'image':
        msk_df = pd.DataFrame(msk_paths, columns=['mask_path'])
        msk_df['zone_id'] = msk_df['mask_path'].apply(lambda x: x.parts[-2])
        # zone_id is zone100_0_0, extract only zone100 using split
        if stratified == 'zone':
            msk_df['zone_id'] = msk_df['zone_id'].apply(lambda x: x.split('_')[0])
            # if zone in not_mediterranean_zones, then remove the path from the df
            if location == 'mediteranean':
                msk_df = msk_df[~msk_df['zone_id'].isin(not_mediterranean_zones)]
        zone_ids = msk_df['zone_id'].unique()
        np.random.seed(random_seed)
        np.random.shuffle(zone_ids)
        n = len(zone_ids)
        print(zone_ids)
        # print unique values of zone_ids
        print('Nbumber of unique zones:', n)
        train_zone_ids = zone_ids[:int(split[0]*n)]# there are not the same number of images by zone. To get 60 20 20 split, tune by hand 0.67. 
        val_zone_ids = zone_ids[int(split[0]*n):int((split[0]+split[1])*n)] # 0.67 0.9
        test_zone_ids = zone_ids[int((split[0]+split[1])*n):]

        train_zone_ids_str = ','.join(train_zone_ids)
        val_zone_ids_str = ','.join(val_zone_ids)
        test_zone_ids_str = ','.join(test_zone_ids)
        #save to csv file train_zone_ids = [...], val_zone_ids = [...], test_zone_ids = [...]
        my_dict = {'train_img_ids': train_zone_ids_str, 'val_img_ids': val_zone_ids_str, 'test_img_ids': test_zone_ids_str}
        # from dict to df. Each k dict is a riw in df. df with ione single oclumn
        df = pd.DataFrame(list(my_dict.items()), columns=['set', 'img_ids'])
        df.to_csv(kwargs['img_ids_by_set'])
        print('Train, val and test zones saved in csv file at:', kwargs['img_ids_by_set'])
        
        train_paths = []
        val_paths = []
        test_paths = []
        for zone_id in train_zone_ids:
            train_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
        for zone_id in val_zone_ids:
            val_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
        for zone_id in test_zone_ids:
            test_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
    elif stratified == 'acquisition':
        #for each subfolder in msk_folder, set 60% of the patches to train, 20% to val and 20% to test
        train_paths = []
        val_paths = []
        test_paths = []
        for subfolder in msk_folder.iterdir():
            msk_paths_subfolder = list(subfolder.rglob('*.tif'))
            msk_df = pd.DataFrame(msk_paths_subfolder, columns=['mask_path'])
            msk_df['index_row'] = msk_df['mask_path'].apply(lambda x: x.parts[-1].split('_')[-2]).astype(int)
            msk_df['index_col'] = msk_df['mask_path'].apply(lambda x: x.parts[-1].split('_')[-1].split('.')[0]).astype(int)

            msk_df = msk_df.sort_values(by=['index_row', 'index_col'])
            n = len(msk_paths_subfolder)
            #np.random.seed(random_seed)
            #np.random.shuffle(msk_paths_subfolder)
            train_paths += msk_paths_subfolder[:int(split[0]*n)]
            val_paths += msk_paths_subfolder[int(split[0]*n):int((split[0]+split[1])*n)]
            test_paths += msk_paths_subfolder[int((split[0]+split[1])*n):]
    return train_paths, val_paths, test_paths
  
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

def classes_balance(zone_list, path_pixels_by_zone):
    zones = list(set(zone_list))
    nb_pix_byz_df = pd.read_csv(path_pixels_by_zone)
    # keep only the columns corresponding to the zones and the int column
    nb_pix_byz_df = nb_pix_byz_df[zones + ['int']]
    # get the nb of pixels by class, sum across columns for one row
    nb_pix_byz_df['per'] = nb_pix_byz_df.sum(axis=1)
    # get the total number of pixels
    total = nb_pix_byz_df['per'].sum()
    # keep only col int and total
    nb_pix_byz_df = nb_pix_byz_df[['int', 'per']]
    # get the proportion of pixels by class
    nb_pix_byz_df['per'] = round(nb_pix_byz_df['per'] / total, 2)
    return nb_pix_byz_df

def check_classes_balance(dl, nb_class, task):
    classes = {i: 0 for i in range(nb_class)}
    len_dl = len(dl)
    c = 0
    if task == 'pixel_classif':
        for img, msk in dl:
            c += 1
            print(f'Batch {c}/{len_dl}')
            for i in range(nb_class):
                classes[i] += torch.sum(msk == i).item()
    elif task == 'image_classif':
        for img, label in dl:
            c += 1
            print(f'Batch {c}/{len_dl}')
            for l in label:
                classes[l.item()] += 1
    return classes


## FUNCTION FOR TRAINING, VALIDATION AND TESTING
def train(model, train_dl, criterion, optimizer, device, nb_classes, model_name):
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
        msk = msk.long()
        loss = criterion(out, msk)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if model_name == 'UNet':
            out = torch.argmax(out, dim=1)
            out = out.int() #int means int32 on cpu and int64 on gpu
            patch_confusion_matrices.append(confusion_matrix(msk.flatten().cpu().numpy(), out.flatten().cpu().numpy(), labels=range(nb_classes)))
        elif model_name == 'Resnet18':
            _, preds = torch.max(out, 1)
             # to int
            preds = preds.int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(msk.cpu().numpy())

    if model_name == 'UNet':
        sum_confusion_matrix = np.sum(patch_confusion_matrices, axis=0)
        IoU_by_class, _ = IoU_F1_from_confmatrix(sum_confusion_matrix)
        IoU_by_class = {k: v for k, v in IoU_by_class.items() if not np.isnan(v)}
        mIoU = np.mean(list(IoU_by_class.values()))
        return running_loss / len(train_dl), mIoU
    elif model_name == 'Resnet18':
        F1_by_class = f1_score(all_labels, all_preds, average=None)
        mF1 = np.mean(F1_by_class)
        return running_loss / len(train_dl), mF1

def valid_test(model, dl, criterion, device, nb_classes, valid_or_test, model_name):
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
        msk = msk.long()
        loss = criterion(out, msk)
        running_loss += loss.item()
        if model_name == 'UNet':
            out = torch.argmax(out, dim=1)
            out = out.int()
            patch_confusion_matrices.append(confusion_matrix(msk.flatten().cpu().numpy(), out.flatten().cpu().numpy(), labels=range(nb_classes)))
        elif model_name == 'Resnet18':
            _, preds = torch.max(out, 1)
            # to int
            preds = preds.int()
            all_preds.extend(preds.cpu().numpy())
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
    elif model_name == 'Resnet18':
        cm = confusion_matrix(all_labels, all_preds, labels=range(nb_classes))
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

def optimizer_to(optim, device):
    # get number of values
    i = 0
    for param in optim.state.values(): 
        i += 1
        print(i)
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

def plot_pred(img, msk, out, pred_plot_path, my_colors_map, nb_imgs, habitats_dict, task):
    if task == 'pixel_classif':
        classes_msk = np.unique(msk)
        legend_colors_msk = [my_colors_map[c] for c in classes_msk]
        custom_cmap_msk = mcolors.ListedColormap(legend_colors_msk)
        classes_out = np.unique(out)
        legend_colors_out = [my_colors_map[c] for c in classes_out]
        custom_cmap_out = mcolors.ListedColormap(legend_colors_out)

        fig, axs = plt.subplots(nb_imgs, 3, figsize=(15, 5*nb_imgs))
        # conatenate classes from classes_out and classes_msk
        classes = np.concatenate((classes_msk, classes_out))
        # get the unique classes
        classes = np.unique(classes)
        class_color = {c: my_colors_map[c] for c in classes}

        unique_labels = set()
        legend_elements = []
        for l, color in class_color.items():
            label = habitats_dict[l]
            legend_elements.append(Patch(facecolor=color, label=label))
            unique_labels.add(label)
        

        for i in range(nb_imgs):
            # get values from img[i]
            print('img[i].shape', img[i].shape)
            print(img[i])

            rgb_img = img[i][:3, :, :]
            #rgb_img[1] *= 0.8
            # augment the brightness of the image
            #rgb_img = rgb_img + 0.1
            # turn to values betw 0 and 1
            #rgb_img = np.clip(rgb_img, 0, 1)

            rgb_img = rgb_img.transpose(1, 2, 0)

            axs[i, 0].imshow(rgb_img)#/float(5000.0))
            axs[i, 0].set_title('Image')
            axs[i, 1].imshow(msk[i], cmap=custom_cmap_msk)
            axs[i, 1].set_title('Mask')
            axs[i, 2].imshow(out[i], cmap=custom_cmap_out)
            axs[i, 2].set_title('Prediction')

        fig.legend(handles=legend_elements, loc='upper center', fontsize=18)#, labelspacing = 1.15)
        plt.savefig(pred_plot_path)

    elif task == 'image_classif':
        # Créer une grille de sous-graphes avec 2 colonnes
        fig, axs = plt.subplots((nb_imgs + 1) // 2, 2, figsize=(20, 10 * ((nb_imgs + 1) // 2)))
        fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.9, bottom=0.02)
        
        my_class = np.unique(msk)
        
        for i in range(nb_imgs):
            row = i // 2
            col = i % 2
            rgb_img = img[i][:3, :, :]
            rgb_img = rgb_img.transpose(1, 2, 0)
            axs[row, col].imshow(rgb_img)
            axs[row, col].text(0, -20, f'True class: {msk[i]}: {habitats_dict[msk[i].item()]}\nPredicted class: {out[i]}: {habitats_dict[out[i].item()]}', fontsize=16)
            axs[row, col].axis('off')
        
        # Supprimer les axes inutilisés si nb_imgs est impair
        if nb_imgs % 2 != 0:
            fig.delaxes(axs[-1, -1])
        
        plt.savefig(pred_plot_path)
    print('Plot saved at:', pred_plot_path)

def plot_patch_msk(img, msk, pred_plot_path, my_colors_map, nb_imgs, habitats_dict, task):        
    classes_msk = np.unique(msk)
    legend_colors_msk = [my_colors_map[c] for c in classes_msk]
    custom_cmap_msk = mcolors.ListedColormap(legend_colors_msk)

    fig, axs = plt.subplots(nb_imgs, 2, figsize=(15, 5*nb_imgs))
    fig.subplots_adjust(top=0.9, bottom=0.02)
    classes = classes_msk
    # get the unique classes
    classes = np.unique(classes)
    class_color = {c: my_colors_map[c] for c in classes}

    unique_labels = set()
    legend_elements = []
    for l, color in class_color.items():
        label = habitats_dict[l]
        legend_elements.append(Patch(facecolor=color, label=label))
        unique_labels.add(label)
    

    for i in range(nb_imgs):
        # get values from img[i]
        print('img[i].shape', img[i].shape)
        print(img[i])

        rgb_img = img[i][:3, :, :]
        #rgb_img[1] *= 0.8
        # augment the brightness of the image
        #rgb_img = rgb_img + 0.1
        # turn to values betw 0 and 1
        #rgb_img = np.clip(rgb_img, 0, 1)
        print('rgb_img.shape', rgb_img.shape)
        rgb_img = rgb_img.permute(1, 2, 0)

        axs[i, 0].imshow(rgb_img)#/float(5000.0))
        axs[i, 0].set_title('Image')
        axs[i, 1].imshow(msk[i], cmap=custom_cmap_msk)
        axs[i, 1].set_title('Mask')
    fig.legend(handles=legend_elements, loc='upper center', fontsize=18)#, labelspacing = 1.15)
    plt.savefig(pred_plot_path)
    print('Plot saved at:', pred_plot_path)

def plot_patch_class_by_class(dataset, nb_imgs, habitats_dict, l2_habitats_dict, set_name):
    # 2 images by class 20 images. 
    # loop on my datast to look for 20 images with the same class at level1. In dataset I have img, class level 1, class level 2

    # 6 classes unique between 0 &nd 5 at level 1
    for c in range(6):        
        
        fig, axs = plt.subplots((nb_imgs + 1) // 2, 2, figsize=(20, 10 * ((nb_imgs + 1) // 2)))
        fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.95, bottom=0.02)
        i = 0
        for img, label, label_2 in dataset:
            if label == c and i < nb_imgs:
                row = i // 2
                col = i % 2
                rgb_img = img[:3, :, :]
                rgb_img = rgb_img.transpose(1, 2, 0)
                axs[row, col].imshow(rgb_img)
                axs[row, col].text(0, -20, f'Class level 2: {label_2}: {l2_habitats_dict[label_2]}', fontsize=22)
                axs[row, col].axis('off')
                i += 1

        plt.suptitle(f'{set_name} : {c} {habitats_dict[c]} class', fontsize=30)
        plt.savefig(f'{set_name}_{c}_class')
        print(f'Plot saved at: {set_name} : {c} class')
        plt.clf()
    
def plot_losses_metrics(losses_metric_path, losses_plot_path, metrics_plot_path, metric):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(losses_metric_path)

    # Extract the data
    epochs = df.iloc[:, 0]
    training_losses = df.iloc[:, 1]
    validation_losses = df.iloc[:, 2]
    training_metric = df.iloc[:, 3]
    validation_metric = df.iloc[:, 4]

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label='Training Losses')
    plt.plot(epochs, validation_losses, label='Validation Losses')
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    #savefig 
    plt.savefig(losses_plot_path)

    # Plot the IoU
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_metric, label=f'Training {metric}')
    plt.plot(epochs, validation_metric, label=f'Validation {metric}')
    plt.title(f'Training and Validation {metric} Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric}')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(metrics_plot_path)