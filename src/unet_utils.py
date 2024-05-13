from pathlib import Path
import pandas as pd
import rasterio
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
    msk_paths_ = [Path(p) for p in msk_paths]
    msk_paths = []

    # KEEP ONLY EXISTING MASKS PATHS (THE ONES WITH INVALID DATES)
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
  
def IoU(pred, target):
    i = torch.sum(pred & target)
    u = torch.sum(pred | target)
    return i/u

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