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


def plot_pred(img, msk, out, pred_plot_path, my_colors_map, nb_imgs, habitats_dict, task, labels):
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

            rgb_img = img[i][:3, :, :]
            rgb_img = rgb_img.transpose(1, 2, 0)

            axs[i, 0].imshow(rgb_img)
            axs[i, 0].set_title('Image')
            axs[i, 1].imshow(msk[i], cmap=custom_cmap_msk)
            axs[i, 1].set_title('Mask')
            axs[i, 2].imshow(out[i], cmap=custom_cmap_out)
            axs[i, 2].set_title('Prediction')

        fig.legend(handles=legend_elements, loc='upper center', fontsize=18)
        plt.savefig(pred_plot_path)

    elif task == 'image_classif':
        if labels == 'single':
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

        elif labels == 'multi':
            # 3 col, multiple rows
            fig, axs = plt.subplots(nb_imgs, 3, figsize=(20, 5*nb_imgs))
            for i in range(nb_imgs):
                rgb_img = img[i][:3, :, :]
                rgb_img = rgb_img.transpose(1, 2, 0)
                axs[i, 0].imshow(rgb_img)#/float(5000.0))
                axs[i, 0].set_title('Image')
                true_heterogenity = msk[i][-1]
                pred_heterogenity = out[i][-1]
                if true_heterogenity == 1:
                    true_heterogenity = 'HETEROGENE'
                else:
                    true_heterogenity = 'HOMOGENE'
                if pred_heterogenity == 1:
                    pred_heterogenity = 'HETEROGENE'
                else:
                    pred_heterogenity = 'HOMOGENE'
                
                # garder que 6 premiers éléments
                msk_ = msk[i][:-1]
                out_ = out[i][:-1]
                # col 2 = textes classes vraies et heterogenity. Col 3 = textes classes prédites et heterogenity
                # pour chaque classe, afficher le nom de la classe
                #créer un texte avecun rteour à al ligne netre chaque classe
                true_classes_text = ''
                # keep the indexes of the values 1
                msk_ = np.where(msk_ == 1)[0]
                for j in msk_:
                    true_classes_text += f'_{habitats_dict[j]}_'
                #remove all \n in true_classes_text
                true_classes_text = true_classes_text.replace('\n', '')
                # add \n every 35 characters in true_classes_text
                true_classes_text = '\n'.join([true_classes_text[i:i+35] for i in range(0, len(true_classes_text), 35)])

                true_classes_text += f'\n {true_heterogenity}'
                #afficher le texte dans la col 2
                axs[i, 1].text(0, 0, true_classes_text, fontsize=16)
                axs[i, 1].axis('off')

                out_ = np.where(out_ == 1)[0]
                pred_classes_text = ''
                for j in out_:
                    pred_classes_text += f'_{habitats_dict[j]}_'
                
                #remove all \n in pred_classes_text
                pred_classes_text = pred_classes_text.replace('\n', '')
                pred_classes_text = '\n'.join([pred_classes_text[i:i+35] for i in range(0, len(pred_classes_text), 35)])
                pred_classes_text += f'\n {pred_heterogenity}'
                axs[i, 2].text(0, 0, pred_classes_text, fontsize=16)
                axs[i, 2].axis('off')
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
        rgb_img = img[i][:3, :, :]
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
                rgb_img = rgb_img.permute(1, 2, 0)

                axs[row, col].imshow(rgb_img)
                axs[row, col].text(0, -20, f'Class level 2: {label_2}: {l2_habitats_dict[label_2]}', fontsize=22)
                axs[row, col].axis('off')
                i += 1

        plt.suptitle(f'{set_name} : {c} {habitats_dict[c]} class', fontsize=30)
        plt.savefig(f'{set_name}_{c}_class')
        print(f'Plot saved at: {set_name} : {c} class')
        plt.clf()

# Function to reassemble patches into a full image
def reassemble_patches(patches, i_indices, j_indices, patch_size=256):
    """
    Reassemble image patches into a full image.

    Parameters:
    patches (list of np.array): List of image patches (each patch is a 2D numpy array).
    i_indices (list of int): List of row indices for the patches.
    j_indices (list of int): List of column indices for the patches.
    patch_size (int): Size of each patch (default is 256).

    Returns:
    np.array: The reassembled image.
    """
    # Determine the size of the full image
    max_i = max(i_indices)
    min_i = min(i_indices)
    max_j = max(j_indices)
    min_j = min(j_indices)
    
    # get nb of patches from min max min max indices
    nb_patches = (max_i - min_i + 1) * (max_j - min_j + 1)
    # Calculate the full image dimensions
    full_image_height = (max_i - min_i + 1) * patch_size
    full_image_width = (max_j - min_j + 1) * patch_size
    full_image = np.ones((full_image_height, full_image_width), dtype=patches[0].dtype) * 8
    
    # Place each patch in the correct location in the full image
    for patch, i, j in zip(patches, i_indices, j_indices):
        row_start = (i - min_i) * patch_size
        row_end = row_start + patch_size
        col_start = (j - min_j) * patch_size
        col_end = col_start + patch_size 
        full_image[row_start:row_end, col_start:col_end] = patch[:patch_size, :patch_size]
        
    return full_image

def plot_reassembled_patches(zone, model, patch_size, msk_folder, alpha1, my_cmap, my_norm, path_to_save, level, model_settings, data_loading_settings):
    msk_paths = list(msk_folder.rglob('*.tif'))
    msk_paths = [msk_path for msk_path in msk_paths if zone in msk_path.stem]
    dataset = EcomedDataset(msk_paths, data_loading_settings['img_folder'], level=level, channels = model_settings['in_channels'], normalisation = data_loading_settings['normalisation'], task = model_settings['task'], my_set = "test", labels = model_settings['labels'], path_mask_name = True)

    # Predict all the masks from the dataset
    predictions = []
    i_indices = []
    j_indices = []
    img_classif_patches = []
    pixels_classif_patches = []
    predicted_patches = []

    for i in range(len(dataset)):
        img, msk, tif_path = dataset[i]
        # INDICES OF THE PATCH IN THE FULL IMAGE
        splitted = tif_path.stem.split('_')
        if patch_size == 256:
            i = int(splitted[-2])
            j = int(splitted[-1])
        elif patch_size == 128:
            i = int(splitted[-4])
            j = int(splitted[-3])
        i_indices.append(i)
        j_indices.append(j)

        if patch_size == 256:
            original_patch = tiff.imread(tif_path)[:, :, :, 0] # dtype = uint8 (confirmed by print(patch.dtype))
        elif patch_size == 128:
            original_patch = tiff.imread(tif_path)[0,:,:]
            # dd a one dimension to the patch
            original_patch = np.expand_dims(original_patch, axis=0)

        group_under_represented_classes = {0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 2, 6: 5, 7: 3, 8: 4, 9: 5}
        # print unique values and their nb in original_patch
        unique, counts = np.unique(original_patch, return_counts=True)
        group_under_represented_classes_uint8 = {np.uint8(k): np.uint8(v) for k, v in group_under_represented_classes.items()}
        patch = np.vectorize(group_under_represented_classes_uint8.get)(original_patch)

        # ORIGINAL PATCHES AT PIXEL LEVEL
        pixels_classif_patches.append(patch[0, :, :]) # we obtain a numpy array of (256, 256) of uint8 type

        # PATCHES IWTH ONE CLASS PER PATCH 
        if len(np.unique(patch)) > 1:
            # if multiple classes in the patch, then strip the patch
            striped = np.ones(patch.shape) * 6 # striped is a numpy array of (1, 256, 256)
            # Every 64 pixels in column, we set the value to 7 for 32 columns of pixels
            my_list = list(range(0, striped.shape[2], 64))
            for i in my_list:
                striped[0, :, i:i+32] = 7
            # turn to uint8
            striped = striped.astype(np.uint8) #  previously, striped was of type float64
            # Striped is an array of (1, 256, 256) with 6 and 7 values, uint8 type
            img_classif_patches.append(striped[0, :, :])            
        else:
            img_classif_patches.append(patch[0, :, :])
        # PREDICTION
        # img to tensor
        img = torch.unsqueeze(torch.tensor(img), 0)
        with torch.no_grad():
            pred = model(img)
            #remove first dimension
        pred = torch.sigmoid(pred)
        # heteroneity to 1 if last value of pred vector is > alpha1
        heterogeneity = 1 if pred[0, -1].item() >= alpha1 else 0
        #if last value of pred vector is 1
        if heterogeneity == 1:
            # if multiple classes in the patch, then strip the patch
            pred_striped = np.ones(patch.shape) * 6 # striped is a numpy array of (1, 256, 256)
            # Every 64 pixels in column, we set the value to 7 for 32 columns of pixels
            pred_my_list = list(range(0, pred_striped.shape[2], 64))
            for i in pred_my_list:
                pred_striped[0, :, i:i+32] = 7
            # turn to uint8
            pred_striped = pred_striped.astype(np.uint8) #  previously, striped was of type float64
            # Striped is an array of (1, 256, 256) with 6 and 7 values, uint8 type
            predicted_patches.append(pred_striped[0, :, :])   
        else:
            # remove last value of pred vector
            pred = pred[:, :-1]
            # get the class with the highest probability
            pred = torch.argmax(pred, dim=1).item()
            # create a numpy array of (256, 256) full of the predicted class
            # turn pred which is an int to uint8
            pred = np.uint8(pred)
            array_pred = np.ones(patch.shape) * pred
            # to uint8
            array_pred = array_pred.astype(np.uint8)
            predicted_patches.append(array_pred[0, :, :])

    # Reassemble the patches
    print('Reassembling the patches...')
    reassembled_image_pixel = reassemble_patches(pixels_classif_patches, i_indices, j_indices, patch_size=patch_size)
    reassembled_image = reassemble_patches(img_classif_patches, i_indices, j_indices, patch_size=patch_size)
    predicted_image = reassemble_patches(predicted_patches, i_indices, j_indices, patch_size=patch_size)

    # Create a figure with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Display the images in the subplots
    axs[0].imshow(reassembled_image_pixel, cmap=my_cmap, norm=my_norm)
    axs[0].axis('off')
    axs[0].set_title('Pixel Level')

    axs[1].imshow(reassembled_image, cmap=my_cmap, norm=my_norm)
    axs[1].axis('off')
    axs[1].set_title('Image Level')

    axs[2].imshow(predicted_image, cmap=my_cmap, norm=my_norm)
    axs[2].axis('off')
    axs[2].set_title('Predicted')

    re_assemble_patches_path = path_to_save + f'_{zone}.png'
    plt.savefig(re_assemble_patches_path, bbox_inches='tight', pad_inches=0)
    plt.close()

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

    # close the plot
    plt.close()