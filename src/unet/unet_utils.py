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


class EcomedDataset(Dataset):
    def __init__(self, msk_paths, img_dir, level=1, channels=4):
        self.img_dir = img_dir
        self.level = level
        self.msks = msk_paths
        self.imgs = [self.img_dir / msk_path.parts[-2] / msk_path.name.replace('msk', 'img').replace('l123/', '') for msk_path in self.msks]
        self.channels = channels

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        msk_path = self.msks[idx]
        with rasterio.open(msk_path) as src:
            msk = src.read(self.level)
            #get type of values in msk
            group_under_represented_classes = {0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 2, 6: 5, 7: 3, 8: 4, 9: 5}
            group_under_represented_classes_uint8 = {np.uint8(k): np.uint8(v) for k, v in group_under_represented_classes.items()}
            msk_mapped = np.vectorize(group_under_represented_classes_uint8.get)(msk)

        with rasterio.open(img_path) as src:
            img = src.read()
            if self.channels == 3:
                img = img[:3]
            # turn to values betw 0 and 1 and to float
            # linear normalisation with p2 and p98
            p2, p98 = np.percentile(img, (2, 98))
            img = np.clip(img, p2, p98)
            img = (img - p2) / (p98 - p2)
            img = img.astype(np.float32)
        
        return img, msk_mapped
    
def load_data_paths(img_folder, msk_folder, stratified, random_seed=3, split=[0.6, 0.2, 0.2], **kwargs):
    msk_paths = list(msk_folder.rglob('*.tif'))
    
    if stratified == 'random':
        # Shuffle with random_seed fixed and split in 60 20 20
        np.random.seed(random_seed)
        np.random.shuffle(msk_paths)
        n = len(msk_paths)
        train_paths = msk_paths[:int(split[0]*n)]
        val_paths = msk_paths[int(split[0]*n):int((split[0]+split[1])*n)]
        test_paths = msk_paths[int((split[0]+split[1])*n):]
    elif stratified == 'zone':
        msk_df = pd.DataFrame(msk_paths, columns=['mask_path'])
        msk_df['zone_id'] = msk_df['mask_path'].apply(lambda x: x.parts[-2])
        # zone_id is zone100_0_0, extract only zone100 using split
        msk_df['zone_id'] = msk_df['zone_id'].apply(lambda x: x.split('_')[0])
        #print(msk_df['zone_id'].unique())
        zone_ids = msk_df['zone_id'].unique()
        np.random.seed(random_seed)
        np.random.shuffle(zone_ids)
        n = len(zone_ids)
        train_zone_ids = zone_ids[:int(0.55*n)] # there are not the same number of images by zone. To get 60 20 20 split, tune by hand 0.67. 
        val_zone_ids = zone_ids[int(0.55*n):int(0.79*n)] # 0.67 0.9
        test_zone_ids = zone_ids[int(0.79*n):]
        train_paths = []
        val_paths = []
        test_paths = []
        for zone_id in train_zone_ids:
            train_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
        for zone_id in val_zone_ids:
            val_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
        for zone_id in test_zone_ids:
            test_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
    elif stratified == 'image':
        msk_df = pd.DataFrame(msk_paths, columns=['mask_path'])
        msk_df['zone_id'] = msk_df['mask_path'].apply(lambda x: x.parts[-2])
        zone_ids = msk_df['zone_id'].unique()
        np.random.seed(random_seed)
        np.random.shuffle(zone_ids)
        n = len(zone_ids)
        train_zone_ids = zone_ids[:int(0.6*n)] #0.55
        val_zone_ids = zone_ids[int(0.6*n):int(0.8*n)] #0.79
        test_zone_ids = zone_ids[int(0.8*n):]
        train_zone_ids_str = ','.join(train_zone_ids)
        val_zone_ids_str = ','.join(val_zone_ids)
        test_zone_ids_str = ','.join(test_zone_ids)
        #save to csv file train_zone_ids = [...], val_zone_ids = [...], test_zone_ids = [...]
        my_dict = {'train_img_ids': train_zone_ids_str, 'val_img_ids': val_zone_ids_str, 'test_img_ids': test_zone_ids_str}
        # from dict to df. Each k dict is a riw in df. df with ione single oclumn
        df = pd.DataFrame(list(my_dict.items()), columns=['set', 'img_ids'])
        df.to_csv(kwargs['img_ids_by_set'])

        train_paths = []
        val_paths = []
        test_paths = []
        for zone_id in train_zone_ids:
            train_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
        for zone_id in val_zone_ids:
            val_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
        for zone_id in test_zone_ids:
            test_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
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

def check_classes_balance(dl):
    classes = {i: 0 for i in range(6)}
    len_dl = len(dl)
    c = 0
    for img, msk in dl:
        c += 1
        print(f'Batch {c}/{len_dl}')
        for i in range(6):
            classes[i] += torch.sum(msk == i).item()
    return classes

## FUNCTION FOR TRAINING, VALIDATION AND TESTING
def train(model, train_dl, criterion, optimizer, device, nb_classes):
    print('Training')
    running_loss = 0.0
    patch_confusion_matrices = []
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
        out = out.int() #int means int32 on cpu and int64 on gpu
        patch_confusion_matrices.append(confusion_matrix(msk.flatten().cpu().numpy(), out.flatten().cpu().numpy(), labels=range(nb_classes)))

    sum_confusion_matrix = np.sum(patch_confusion_matrices, axis=0)
    IoU_by_class, _ = IoU_F1_from_confmatrix(sum_confusion_matrix)
    IoU_by_class = {k: v for k, v in IoU_by_class.items() if not np.isnan(v)}
    mIoU = np.mean(list(IoU_by_class.values()))
    return running_loss / len(train_dl), mIoU

def valid_test(model, dl, criterion, device, nb_classes, valid_or_test):
    running_loss = 0.0
    patch_confusion_matrices = []
    patch_confusion_matrices_1c = []
    patch_confusion_matrices_multic = []
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
        out = torch.argmax(out, dim=1)
        # out to uint8
        out = out.int()
        #print("y_true", msk.flatten().cpu().numpy())
        #print("y_pred", out.flatten().cpu().numpy())
        patch_confusion_matrices.append(confusion_matrix(msk.flatten().cpu().numpy(), out.flatten().cpu().numpy(), labels=range(nb_classes)))
        '''# for each img and mask in the batch, check if the msk is composed of one single class
        for i in range(len(msk)):
            if len(np.unique(msk[i].cpu())) == 1:
                batch_confusion_matrices_1c.append(confusion_matrix(msk[i].flatten().cpu().numpy(), out[i].flatten().cpu().numpy(), labels=range(nb_classes)))
            else:
                batch_confusion_matrices_multic.append(confusion_matrix(msk[i].flatten().cpu().numpy(), out[i].flatten().cpu().numpy(), labels=range(nb_classes)))
        
        sum_batch_confusion_matrices_1c = np.sum(batch_confusion_matrices_1c, axis=0)
        #print('sum_batch_confusion_matrices_1c: ', sum_batch_confusion_matrices_1c)
        sum_batch_confusion_matrices_multic = np.sum(batch_confusion_matrices_multic, axis=0)
        #print('sum_batch_confusion_matrices_multic: ', sum_batch_confusion_matrices_multic)

        patch_confusion_matrices_1c.append(sum_batch_confusion_matrices_1c)
        patch_confusion_matrices_multic.append(sum_batch_confusion_matrices_multic)'''
    sum_confusion_matrix = np.sum(patch_confusion_matrices, axis=0)
    '''sum_confusion_matrix_1c = np.sum(patch_confusion_matrices_1c, axis=0)
    sum_confusion_matrix_multic = np.sum(patch_confusion_matrices_multic, axis=0)'''

    IoU_by_class, F1_by_class = IoU_F1_from_confmatrix(sum_confusion_matrix)
    '''IoU_by_class_1c, F1_by_class_1c = IoU_F1_from_confmatrix(sum_confusion_matrix_1c)
    IoU_by_class_multic, F1_by_class_multic = IoU_F1_from_confmatrix(sum_confusion_matrix_multic)'''
    # mean of IoU_by_class for all classes
    #remove nan from IoU_by_class and F1_by_class
    IoU_by_class = {k: v for k, v in IoU_by_class.items() if not np.isnan(v)}
    '''IoU_by_class_1c = {k: v for k, v in IoU_by_class_1c.items() if not np.isnan(v)}
    IoU_by_class_multic = {k: v for k, v in IoU_by_class_multic.items() if not np.isnan(v)}'''
    F1_by_class = {k: v for k, v in F1_by_class.items() if not np.isnan(v)}
    '''F1_by_class_1c = {k: v for k, v in F1_by_class_1c.items() if not np.isnan(v)}
    F1_by_class_multic = {k: v for k, v in F1_by_class_multic.items() if not np.isnan(v)}'''
    
    mIoU = np.mean(list(IoU_by_class.values()))
    ''' mIoU_1c = np.mean(list(IoU_by_class_1c.values()))
    mIoU_multic = np.mean(list(IoU_by_class_multic.values()))'''
    mF1 = np.mean(list(F1_by_class.values()))
    ''' mF1_1c = np.mean(list(F1_by_class_1c.values()))
    mF1_multic = np.mean(list(F1_by_class_multic.values()))'''

    metrics = {
        'confusion_matrix': sum_confusion_matrix,
        'IoU_by_class': IoU_by_class,
        'F1_by_class': F1_by_class,
        'mIoU': mIoU,
        'mF1': mF1
    }

    '''metrics_1c = {
        'confusion_matrix': sum_confusion_matrix_1c,
        'IoU_by_class': IoU_by_class_1c,
        'F1_by_class': F1_by_class_1c,
        'mIoU': mIoU_1c,
        'mF1': mF1_1c
    }

    metrics_multic = {
        'confusion_matrix': sum_confusion_matrix_multic,
        'IoU_by_class': IoU_by_class_multic,
        'F1_by_class': F1_by_class_multic,
        'mIoU': mIoU_multic,
        'mF1': mF1_multic
    }
    
    print('Metrics for single labelled patches')
    print(metrics_1c)
    print('Metrics for multi labelled patches')
    print(metrics_multic)'''
    if valid_or_test == 'valid':
        return running_loss / len(dl), mIoU
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

def plot_pred(img, msk, out, pred_plot_path, my_colors_map, nb_imgs):
    classes_msk = np.unique(msk)
    legend_colors_msk = [my_colors_map[c] for c in classes_msk]
    custom_cmap_msk = mcolors.ListedColormap(legend_colors_msk)
    classes_out = np.unique(out)
    legend_colors_out = [my_colors_map[c] for c in classes_out]
    custom_cmap_out = mcolors.ListedColormap(legend_colors_out)

    fig, axs = plt.subplots(nb_imgs, 3, figsize=(15, 5*nb_imgs))
    for i in range(nb_imgs):
        axs[i, 0].imshow(img[i, 0], cmap='gray') # change to color! 
        axs[i, 0].set_title('Image')
        axs[i, 1].imshow(msk[i], cmap=custom_cmap_msk)
        axs[i, 1].set_title('Mask')
        axs[i, 2].imshow(out[i], cmap=custom_cmap_out)
        axs[i, 2].set_title('Prediction')

    plt.savefig(pred_plot_path)

def plot_losses_ious(losses_ious_path, losses_plot_path, ious_plot_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(losses_ious_path)

    # Extract the data
    epochs = df.iloc[:, 0]
    training_losses = df.iloc[:, 1]
    validation_losses = df.iloc[:, 2]
    training_iou = df.iloc[:, 3]
    validation_iou = df.iloc[:, 4]

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
    plt.plot(epochs, training_iou, label='Training IoU')
    plt.plot(epochs, validation_iou, label='Validation IoU')
    plt.title('Training and Validation IoU Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(ious_plot_path)