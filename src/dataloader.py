import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import rasterio
from utils import *

# create a dataloader. I need an argument to tell which level of classification I am working with, and load channel of mask accordingly
class EcomedDataset(Dataset):
    def __init__(self, img_dir, msk_dir, level):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.level = level
        self.imgs = list(img_dir.rglob('*.tif'))
        self.msks = list(msk_dir.rglob('*.tif'))
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        msk_path = self.msks[idx]
        with rasterio.open(img_path) as src:
            img = src.read()
        with rasterio.open(msk_path) as src:
            msk = src.read(self.level)
        img = img.astype(np.int16)  # You can choose a different dtype if needed
        return img, msk
    
# Create a dataloader for level 1
img_dir = Path('/media/bertille/My Passport/natural_habitats_mapping/data/patch64/img/zone3_0_0')
msk_dir = Path('/media/bertille/My Passport/natural_habitats_mapping/data/patch64/msk/l123/zone3_0_0')
level = 1
dataset = EcomedDataset(img_dir, msk_dir, level)
dataloader = DataLoader(dataset, batch_size=11, shuffle=True)

#plot one image and mask
# Assuming dataloader is your PyTorch DataLoader instance
for i, (img_batch, msk_batch) in enumerate(dataloader):
    if i >= 10:
        break  # Stop after processing 10 batches
    
    # list unique values in masks
    print(np.unique(msk_batch.numpy()))
    
