# list all folders in ../../data/patch256/img
import os
import rasterio
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = Path('../../data/patch256/img/')

subfolders = [x for x in path.iterdir() if x.is_dir()]
empty_folders = [x for x in subfolders if not list(x.iterdir())] 
# remove emptyfolders from subfolders
subfolders = [x for x in subfolders if x not in empty_folders]

#make a df from the subfolders
df = pd.DataFrame(subfolders, columns=['path'])
df['img'] = df['path'].apply(lambda x: str(x).split('/')[-1])
# split the zone to get the first lement
df['zone'] = df['img'].apply(lambda x: x.split('_')[0])

# get nb of unique zones
print('Number of unique zones:', len(df['zone'].unique()))
print('Number of images:', len(df))

# create a df with the unque zones
df_zone = pd.DataFrame(df['zone'].unique(), columns=['zone'])
# add a column where we count the number of rows from the df with this zone
df_zone['nb_img'] = df_zone['zone'].apply(lambda x: len(df[df['zone']==x]))

print(df_zone)

#boxplot from the number of images per zone
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_zone, x='nb_img')
plt.title('Number of images per zone')
# savefig
# plt.savefig('nb_img_per_zone.png')

msk_path = Path('../../data/patch256/msk/')
# count the total nb of patches in the msk folder
nb_msk = len(list(msk_path.rglob('*.tif')))
print(msk_path)
print('Number of patches in the msk folder:', nb_msk)
# each msk is of size 256x256
# count the number total of pixels
total_pixels = nb_msk*256*256
print('Total number of pixels:', total_pixels)
print('Total number of patches:', nb_msk)
# compute the surface ara covered by the patches. One pixel is 0.15cm*0.15cm
total_surface = total_pixels*0.15*0.15
print('Total surface covered by the patches:', total_surface, 'm2')
# to km2
total_surface = total_surface/1000000
print('Total surface covered by the patches:', total_surface, 'km2')
