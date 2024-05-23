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
    def __init__(self, msk_paths, img_dir, level=1):
        self.img_dir = img_dir
        self.level = level
        self.msks = msk_paths
        self.imgs = [self.img_dir / msk_path.parts[-2] / msk_path.name.replace('msk', 'img').replace('l123/', '') for msk_path in self.msks]
                
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        msk_path = self.msks[idx]
        with rasterio.open(msk_path) as src:
            msk = src.read(self.level)
            # to torch long
        with rasterio.open(img_path) as src:
            img = src.read()
            # turn to values betw 0 and 1 and to float
            # linear normalisation with p2 and p98
            p2, p98 = np.percentile(img, (2, 98))
            img = np.clip(img, p2, p98)
            img = (img - p2) / (p98 - p2)
            img = img.astype(np.float32)
        return img, msk
    
def load_data_paths(img_folder, msk_folder, msks_256_fully_labelled, stratified=False, random_seed=42, split=[0.6, 0.2, 0.2], **kwargs):
    msk_paths = list(msks_256_fully_labelled['mask_path'])
    #add ../ to all paths
    msk_paths_ = [Path('../') / Path(p) for p in msk_paths]
    msk_paths = []

    # KEEP ONLY EXISTING MASKS PATHS (FILTER THE ONES WITH INVALID DATES)
    for msk_path in msk_paths_:
        if msk_path.exists():
            msk_paths.append(msk_path)

    # BUILD THE IMG PATHS CORRESPONDING TO THE MASKS PATHS
    img_paths = [img_folder / msk_path.parts[-2] / msk_path.name.replace('msk', 'img').replace('l123/', '') for msk_path in msk_paths]

    # CHECK IF THE IMAGES HAVE ZERO VALUES
    imgs_zero_paths = []
    for img_path in img_paths:
        with rasterio.open(img_path) as src:
            img = src.read()
        if np.count_nonzero(img) == 0:
            imgs_zero_paths.append(img_path)

    # LOOK FOR MASKS MATCHING THE IMAGES WITH ZERO VALUES
    msks_paths_zero = [msk_folder / img_path.parts[-2] / img_path.name.replace('img', 'msk') for img_path in imgs_zero_paths]

    final_msk_paths = []

    for msk_path in msk_paths:
        if msk_path in msks_paths_zero:
            continue
        final_msk_paths.append(msk_path)
    
    if not stratified:
        # Shuffle with random_seed fixed and split in 60 20 20
        np.random.seed(random_seed)
        np.random.shuffle(final_msk_paths)
        n = len(final_msk_paths)
        train_paths = final_msk_paths[:int(split[0]*n)]
        val_paths = final_msk_paths[int(split[0]*n):int((split[0]+split[1])*n)]
        test_paths = final_msk_paths[int((split[0]+split[1])*n):]
    else:
        msk_df = pd.DataFrame(final_msk_paths, columns=['mask_path'])
        msk_df['zone_id'] = msk_df['mask_path'].apply(lambda x: x.parts[-2])
        # zone_id is zone100_0_0, extract only zone100 using split
        msk_df['zone_id'] = msk_df['zone_id'].apply(lambda x: x.split('_')[0])
        #print(msk_df['zone_id'].unique())
        zone_ids = msk_df['zone_id'].unique()
        np.random.seed(random_seed)
        np.random.shuffle(zone_ids)
        n = len(zone_ids)
        train_zone_ids = zone_ids[:int(0.67*n)] # there are not the same number of images by zone. To get 60 20 20 split, tune by hand 0.67. 
        val_zone_ids = zone_ids[int(0.67*n):int(0.9*n)]
        test_zone_ids = zone_ids[int(0.9*n):]
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
    ious = {c: 0 for c in range(1, conf_matrix.shape[0]+1)}
    f1s = {c: 0 for c in range(1, conf_matrix.shape[0]+1)}
    for c in range(1, conf_matrix.shape[0]+1):
        TP = conf_matrix[c-1, c-1]
        FP = np.sum(conf_matrix[:, c-1]) - TP
        FN = np.sum(conf_matrix[c-1, :]) - TP
        i = TP
        u = TP + FP + FN
        ious[c] = i/u
        f1s[c] = TP/(TP + 0.5*(FP + FN))
    return ious, f1s

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
        out = out.int()
        patch_confusion_matrices.append(confusion_matrix(msk.flatten().cpu().numpy(), out.flatten().cpu().numpy(), labels=range(1, nb_classes+1)))

    sum_confusion_matrix = np.sum(patch_confusion_matrices, axis=0)
    IoU_by_class, _ = IoU_F1_from_confmatrix(sum_confusion_matrix)
    IoU_by_class = {k: v for k, v in IoU_by_class.items() if not np.isnan(v)}
    mIoU = np.mean(list(IoU_by_class.values()))
    return running_loss / len(train_dl), mIoU

def valid_test(model, dl, criterion, device, nb_classes, valid_or_test):
    running_loss = 0.0
    patch_confusion_matrices = []
    for i, (img, msk) in enumerate(dl):
        if i % 50 == 0:
            print( 'Batch:', i, ' over ', len(dl))
        img, msk = img.to(device), msk.to(device)
        out = model(img)
        msk = msk.long()
        loss = criterion(out, msk)
        running_loss += loss.item()
        out = torch.argmax(out, dim=1)
        out = out.int()
        patch_confusion_matrices.append(confusion_matrix(msk.flatten().cpu().numpy(), out.flatten().cpu().numpy(), labels=range(1, nb_classes+1)))
        
    sum_confusion_matrix = np.sum(patch_confusion_matrices, axis=0)
    IoU_by_class, F1_by_class = IoU_F1_from_confmatrix(sum_confusion_matrix)
    # mean of IoU_by_class for all classes
    #remove nan from IoU_by_class and F1_by_class
    IoU_by_class = {k: v for k, v in IoU_by_class.items() if not np.isnan(v)}
    F1_by_class = {k: v for k, v in F1_by_class.items() if not np.isnan(v)}
    
    mIoU = np.mean(list(IoU_by_class.values()))
    mF1 = np.mean(list(F1_by_class.values()))

    metrics = {
        'confusion_matrix': sum_confusion_matrix,
        'IoU_by_class': IoU_by_class,
        'F1_by_class': F1_by_class,
        'mIoU': mIoU,
        'mF1': mF1
    }

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
        axs[i, 0].imshow(img[i, 0], cmap='gray')
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