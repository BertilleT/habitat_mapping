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

class EcomedDataset(Dataset):
    def __init__(self, msk_paths, img_dir, level=1, channels=4, transform = [None, None], normalisation = "all_channels_together", task = "pixel_classif", my_set = "train", labels = 'single', path_mask_name = None):
        self.img_dir = img_dir
        self.level = level
        self.msks = msk_paths
        self.imgs = [self.img_dir / msk_path.parts[-2] / msk_path.name.replace('msk', 'img').replace('l123/', '') for msk_path in self.msks]
        self.channels = channels
        self.transform_rgb = transform[0]
        self.transform_all_channels = transform[1]
        self.normalisation = normalisation
        self.task = task
        self.set = my_set
        self.labels = labels
        self.path_mask_name = path_mask_name

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        msk_path = self.msks[idx]
        with rasterio.open(msk_path) as src:
            '''msk_l2 = src.read(2)
            # if 18, 19, 24 ou 29 in msk_l2, then filter the image
            if 18 in msk_l2 or 19 in msk_l2 or 24 in msk_l2 or 29 in msk_l2:
                #print(msk_path)
            msk_l3 = src.read(3)
            # if 67 or 70 in msk_l3, then filter the image
            if 67 in msk_l3 or 70 in msk_l3:
                #print(msk_path)'''
            msk = src.read(self.level)
            # print unique values in msk
            img_label = msk[0, 0]

            #get type of values in msk
            if self.level == 1:
                group_under_represented_classes = {0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 2, 6: 5, 7: 3, 8: 4, 9: 5}
            elif self.level == 2:
                group_under_represented_classes = {0: 16,1: 16,2: 16,3: 16,4: 16,5: 16,6: 16,7: 16,8: 0,9: 1,10: 16,11: 2,12: 16,13: 3,14: 16,15: 4,16: 5,17: 6,18: 16,19: 7,20: 8,21: 9,22: 10,23: 16,24: 11,25: 16,26: 16,27: 16,28: 12,29: 16,30: 13,31: 14,32: 16,33: 15,34: 16,35: 16,255: 16}
            group_under_represented_classes_uint8 = {np.uint8(k): np.uint8(v) for k, v in group_under_represented_classes.items()}
            #if tasl is pixel_classif, then group under represented classes
            if self.task == "pixel_classif":
                msk_mapped = np.vectorize(group_under_represented_classes_uint8.get)(msk)
            elif self.task == "image_classif":  
                # map value from img_label using group_under_represented_classes_uint8
                if self.labels == 'single':
                    img_label_mapped = group_under_represented_classes_uint8[img_label] # PB HERE 
                elif self.labels == 'multi':
                    # store in img_labels all uniques values from masks
                    labels = np.unique(msk)
                    labels_mapped = np.vectorize(group_under_represented_classes_uint8.get)(labels)
                    # unique
                    unique_labels_mapped = np.unique(labels_mapped)
                    if self.level == 1:
                        binary_labels_mapped = np.zeros(7)
                        for i in range(6):
                            if i in unique_labels_mapped:
                                binary_labels_mapped[i] = 1
                        if len(unique_labels_mapped) > 1:
                            binary_labels_mapped[6] = 1
                    elif self.level == 2:
                        binary_labels_mapped = np.zeros(18)
                        for i in range(17):
                            if i in unique_labels_mapped:
                                binary_labels_mapped[i] = 1
                        if len(unique_labels_mapped) > 1:
                            binary_labels_mapped[17] = 1
                        # img_label_mapped and to float32 because of the loss function
                    img_label_mapped = binary_labels_mapped.astype(np.float32) 

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

        if self.transform_rgb and self.set == 'train':
            if self.task == "pixel_classif":
                augmented = self.transform_rgb(image=normalized_img[0:3].transpose(1, 2, 0))
                rgb_transformed_img = augmented['image']
                temp_img = np.concatenate((rgb_transformed_img, normalized_img[3:4]), axis=0)
                augmented = self.transform_all_channels(image=temp_img.transpose(1, 2, 0), mask=msk_mapped)
                img = augmented['image']
                msk_mapped = augmented['mask']
            elif self.task == "image_classif":
                augmented = self.transform_rgb(image=normalized_img[0:3].transpose(1, 2, 0))
                rgb_transformed_img = augmented['image']
                temp_img = np.concatenate((rgb_transformed_img, normalized_img[3:4]), axis=0)
                augmented = self.transform_all_channels(image=temp_img.transpose(1, 2, 0))
                img = augmented['image']

        else: 
            img = normalized_img
        if self.task == "pixel_classif":
            return img, msk_mapped
        elif self.task == "image_classif":
            if self.path_mask_name:
                return img, img_label_mapped, msk_path
            return img, img_label_mapped



class EcomedDataset_to_plot(Dataset):
    def __init__(self, msk_paths, img_dir, channels=4, transform = None, task = "pixel_classif", my_set = "train"):
        self.img_dir = img_dir
        self.msks = msk_paths
        self.imgs = [self.img_dir / msk_path.parts[-2] / msk_path.name.replace('msk', 'img').replace('l123/', '') for msk_path in self.msks]
        self.channels = channels
        self.transform_rgb = transform[0]
        self.transform_all_channels = transform[1]
        self.task = task
        self.set = my_set

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
        if self.transform_rgb and self.set == 'train':
            if self.task == "pixel_classif":
                augmented = self.transform_rgb(image=normalized_img[0:3].transpose(1, 2, 0))
                rgb_transformed_img = augmented['image']
                temp_img = np.concatenate((rgb_transformed_img, normalized_img[3:4]), axis=0)
                augmented = self.transform_all_channels(image=temp_img.transpose(1, 2, 0), mask=msk_mapped)
                img = augmented['image']
                msk_mapped = augmented['mask']
            elif self.task == "image_classif":
                augmented = self.transform_rgb(image=normalized_img[0:3].transpose(1, 2, 0))
                rgb_transformed_img = augmented['image']
                temp_img = np.concatenate((rgb_transformed_img, normalized_img[3:4]), axis=0)
                augmented = self.transform_all_channels(image=temp_img.transpose(1, 2, 0))
                img = augmented['image']
        else: 
            img = normalized_img
        if self.task == "pixel_classif":
            return img, msk_mapped # deal with label at level 2. 
        elif self.task == "image_classif":
            return img, img_label_mapped, img_label_2

def load_data_paths(img_folder, msk_folder, stratified, random_seed, split, **kwargs):
    print('The seed to shuffle the data is ', str(random_seed))
    print('The data are from ', kwargs['year'], ' year')    
    #load csv file from ../../csv/path_to_filter.csv, one single col with name file_path
    path_to_filter = pd.read_csv('../../csv/paths_to_filter.csv')
    # get the values of the column
    path_to_filter = path_to_filter['file_path'].values
    # turn to paths
    path_to_filter = [Path(path) for path in path_to_filter]
    
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
        # filter
        #msk_paths = [msk_path for msk_path in msk_paths if msk_path not in path_to_filter]
    if kwargs['year'] == '2023':
        zones_2023 = pd.read_csv(kwargs['2023_zones'])
        msk_paths = [msk_path for msk_path in msk_paths if str(msk_path).split('/')[-2].split('_')[0] in zones_2023['zone_AJ'].values]
        #msk_paths = [msk_path for msk_path in msk_paths if str(msk_path).split('/').parts[-2].split('_')[0] in zones_2023['zone_id'].values]
        print(len(msk_paths), ' kept masks from 2023 zones')
        print('The data are from 2023')

    zones_to_plot = []
    #for msk_path in msk_paths:
    #    if 'zone1_0_0' in str(msk_path) or 'zone100_0_0' in str(msk_path) or 'zone133_0_0' in str(msk_path):
    #        zones_to_plot.append(msk_path)
    # drop the paths from msk_paths
   # msk_paths = [msk_path for msk_path in msk_paths if msk_path not in zones_to_plot]
    if stratified == 'random':
        # Shuffle with random_seed fixed and split in 60 20 20
        np.random.seed(random_seed)
        np.random.shuffle(msk_paths)
        n = len(msk_paths)
        train_paths = msk_paths[:int(split[0]*n)]
        val_paths = msk_paths[int(split[0]*n):int((split[0]+split[1])*n)]
        test_paths = msk_paths[int((split[0]+split[1])*n):]
        if kwargs['location'] == 'mediteranean':   
            msk_df = pd.DataFrame(msk_paths, columns=['mask_path'])
            msk_df['zone_id'] = msk_df['mask_path'].apply(lambda x: x.parts[-2]) 
            msk_df['zone_id'] = msk_df['zone_id'].apply(lambda x: x.split('_')[0])
            msk_df = msk_df[~msk_df['zone_id'].isin(not_mediterranean_zones)]
            print('Only mediterranean zones kept')
            print('Number of masks:', len(msk_df))
    elif stratified == 'zone' or stratified == 'image':
        msk_df = pd.DataFrame(msk_paths, columns=['mask_path'])
        msk_df['zone_id'] = msk_df['mask_path'].apply(lambda x: x.parts[-2])
        # zone_id is zone100_0_0, extract only zone100 using split
        if stratified == 'zone':
            msk_df['zone_id'] = msk_df['zone_id'].apply(lambda x: x.split('_')[0])
            # if zone in not_mediterranean_zones, then remove the path from the df
        if kwargs['location'] == 'mediteranean':            
            msk_df = msk_df[~msk_df['zone_id'].isin(not_mediterranean_zones)]
            print('Only mediterranean zones kept')
            print('Number of masks:', len(msk_df))

        zone_ids = msk_df['zone_id'].unique()
        np.random.seed(random_seed)
        np.random.shuffle(zone_ids)
        n = len(zone_ids)
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
        #df.to_csv(kwargs['img_ids_by_set'])
        #print('Train, val and test zones saved in csv file at:', kwargs['img_ids_by_set'])
        
        '''train_zone_ids = ['zone72', 'zone69', 'zone50', 'zone120', 'zone112', 'zone75', 'zone7', 'zone56', 'zone57', 'zone74', 'zone84', 'zone148', 'zone171', 'zone157', 'zone143', 'zone106', 'zone37', 'zone33', 'zone136', 'zone24', 'zone88', 'zone113', 'zone129', 'zone144', 'zone155', 'zone156', 'zone93', 'zone14', 'zone30', 'zone127', 'zone159', 'zone98', 'zone172', 'zone10', 'zone54', 'zone147', 'zone77', 'zone145', 'zone4', 'zone123', 'zone161', 'zone38', 'zone134', 'zone101', 'zone11', 'zone15', 'zone48', 'zone5', 'zone154', 'zone121', 'zone22', 'zone17', 'zone126', 'zone6', 'zone169', 'zone95', 'zone85', 'zone19', 'zone26', 'zone65', 'zone68', 'zone39', 'zone160', 'zone139', 'zone115', 'zone80', 'zone104', 'zone165', 'zone76', 'zone45', 'zone34', 'zone21', 'zone167']
        val_zone_ids = [
            'zone27', 'zone162', 'zone142', 'zone114', 'zone71', 'zone25', 
            'zone63', 'zone133', 'zone66', 'zone20', 'zone117', 'zone41', 
            'zone2', 'zone78', 'zone47', 'zone3', 'zone102', 'zone137', 
            'zone16', 'zone59', 'zone44', 'zone164'
        ]
        test_zone_ids = [
            'zone90', 'zone170', 'zone132', 'zone28', 'zone12', 'zone96', 
            'zone116', 'zone97', 'zone73', 'zone158', 'zone53', 'zone51', 'zone67'
        ]'''
        train_paths = []
        val_paths = []
        test_paths = []
        for zone_id in train_zone_ids:
            print('hey')
            train_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
        for zone_id in val_zone_ids:
            print('hey2')
            val_paths += list(msk_df[msk_df['zone_id'] == zone_id]['mask_path'])
        for zone_id in test_zone_ids:
            print('hey3')
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
    #concat test_paths with zones_to_plot
    test_paths = test_paths + zones_to_plot
    return train_paths, val_paths, test_paths


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
