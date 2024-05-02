import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from utils import *
from pathlib import Path
from rasterio.plot import show
import time
import pandas as pd

# PATHS and DIRECTORIES
data_dir = Path('../data/')
full_img_dir = data_dir / 'full_img_msk' / 'img'
tif_paths = list(full_img_dir.rglob('*.tif'))
msk_dir = data_dir / 'full_img_msk' / 'msk'
msk_dir.mkdir(exist_ok=True)
level1_dir = msk_dir / 'level1'
level1_dir.mkdir(exist_ok=True)
level2_dir = msk_dir / 'level2'
level2_dir.mkdir(exist_ok=True)
level3_dir = msk_dir / 'level3'
level3_dir.mkdir(exist_ok=True)
csv_dir = Path('../csv/')
shapefile_path = data_dir / 'shp/HABNATs.shp'
pt_name = 'final_pivot_table.csv'
pivot_table_path = csv_dir / pt_name
path_to_un_classes = csv_dir / 'classes_grouped_3l.csv'

# Load the csv of unique classes
unique_classes = pd.read_csv(path_to_un_classes)

'''# DICTIONNARY CLASS-INTEGER FOR 3 LEVELS
# L1
#--------------------------------------------------------------

classes_l1 = unique_classes['l1'].dropna().to_dict() #the key is the class and the value is the integer
classes_l1 = {value: int(1) for key, value in classes_l1.items()}
# order in alphabetical order
classes_l1 = dict(sorted(classes_l1.items()))
int_classes_l1 = {}
counter = 1
# give as value an interger strating from 1 
for key in sorted(classes_l1.keys()):
    if len(key) != 1: 
        int_classes_l1[key] = 9
    else:
        int_classes_l1[key] = counter
        counter += 1

# save in csv/label_map_to_int_dict the dict
pd.DataFrame(int_classes_l1.items(), columns=['class', 'int']).to_csv(csv_dir / 'old_l1.csv', index=False)

# L2
#--------------------------------------------------------------

classes_l2 = unique_classes['l2'].dropna().to_dict() #the key is the class and the value is the integer
classes_l2 = {value: int(1) for key, value in classes_l2.items()}
classes_l2 = dict(sorted(classes_l2.items()))
int_classes_l2 = {}
counter = 1
for key in sorted(classes_l2.keys()):
    if len(key) != 2: 
        int_classes_l2[key] = 99
    else:
        int_classes_l2[key] = counter
        counter += 1

pd.DataFrame(int_classes_l2.items(), columns=['class', 'int']).to_csv(csv_dir / 'old_l2.csv', index=False)
# L3
#--------------------------------------------------------------

classes_l3 = unique_classes['l3'].dropna().to_dict() #the key is the class and the value is the integer
classes_l3 = {value: int(1) for key, value in classes_l3.items()}
classes_l3 = dict(sorted(classes_l3.items()))
int_classes_l3 = {}
counter = 1
for key in sorted(classes_l3.keys()):
    if len(key) != 3: 
        int_classes_l3[key] = 999
    else:
        int_classes_l3[key] = counter
        counter += 1

pd.DataFrame(int_classes_l3.items(), columns=['class', 'int']).to_csv(csv_dir / 'old_l3.csv', index=False)
#--------------------------------------------------------------'''

# Load the csv of unique classes
#load int_classes_l1
int_classes_l1 = pd.read_csv(csv_dir / 'l1_int.csv')
int_classes_l1 = int_classes_l1.set_index('class')['int'].to_dict()
#load int_classes_l2
int_classes_l2 = pd.read_csv(csv_dir / 'l2_int.csv')
int_classes_l2 = int_classes_l2.set_index('class')['int'].to_dict()
#load int_classes_l3
int_classes_l3 = pd.read_csv(csv_dir / 'l3_int.csv')
int_classes_l3 = int_classes_l3.set_index('class')['int'].to_dict()

#print(int_classes_l1)
#print(int_classes_l2)
#print(int_classes_l3)

# RASTERIZE
total_images = len(tif_paths)
count_rasterized_image = 0
shp = load_all_shapefile(shapefile_path)
intersect_df = pd.read_csv(pivot_table_path)
# timer
start = time.time()
# print len of files tif in data/full_img_msk/msk/level1
print(f'Number of files in level1: {len(list(level1_dir.rglob("*.tif")))}')
print(f'Number of files in level2: {len(list(level2_dir.rglob("*.tif")))}')
print(f'Number of files in level3: {len(list(level3_dir.rglob("*.tif")))}')
for full_img in tif_paths:
    count_rasterized_image += 1
    if 'zone16_0_0' in str(full_img):
        # Renaming img_blabla to msk_blabla
        split_name = full_img.stem.split('_')
        new_name = '_'.join(split_name[-3:])
        new_name = 'msk_' + new_name + '.tif'
        l1_name = 'l1_' + new_name
        l2_name = 'l2_' + new_name
        l3_name = 'l3_' + new_name
        print(full_img)
        polygons_one_tif, _ = filter_shapefile(shp, intersect_df, full_img)
        print(polygons_one_tif)

        polygons_one_tif['CDEUNIS_1_l1'] = polygons_one_tif['CDEUNIS_1'].apply(lambda x: x[0] if isinstance(x, str) else x)
        polygons_one_tif['CDEUNIS_1_l2'] = polygons_one_tif['CDEUNIS_1'].apply(lambda x: x[0:2] if isinstance(x, str) else x)
        polygons_one_tif['CDEUNIS_1_l3'] = polygons_one_tif['CDEUNIS_1'].apply(lambda x: x[0:4] if isinstance(x, str) else x)
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'] = polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'].apply(lambda x: x.replace('.', ''))

        polygons_one_tif['CDEUNIS_1_l1'] = polygons_one_tif['CDEUNIS_1_l1'].map(int_classes_l1)
        polygons_one_tif['CDEUNIS_1_l2'] = polygons_one_tif['CDEUNIS_1_l2'].map(int_classes_l2)
        polygons_one_tif['CDEUNIS_1_l3'] = polygons_one_tif['CDEUNIS_1_l3'].map(int_classes_l3)
        # values which have not been mapped to -1
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l1'].isnull(), 'CDEUNIS_1_l1'] = 0
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l2'].isnull(), 'CDEUNIS_1_l2'] = 0
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l3'].isnull(), 'CDEUNIS_1_l3'] = 0
        polygons_one_tif['CDEUNIS_1_l1'] = polygons_one_tif['CDEUNIS_1_l1'].astype(int)
        polygons_one_tif['CDEUNIS_1_l2'] = polygons_one_tif['CDEUNIS_1_l2'].astype(int)
        polygons_one_tif['CDEUNIS_1_l3'] = polygons_one_tif['CDEUNIS_1_l3'].astype(int)

        #polygons_one_tif['label_levels123'] = polygons_one_tif['CDEUNIS_1_l1'].astype(str) + '_' + polygons_one_tif['CDEUNIS_1_l2'].astype(str) + '_' + polygons_one_tif['CDEUNIS_1_l3'].astype(str)
        #polygons_one_tif = polygons_one_tif[['geometry', 'label_levels123']]
        #print(polygons_one_tif['label_levels123'].unique())

        try:
            # RASTERIZE AT LEVEL 1
            polygons_one_tif_l1 = polygons_one_tif[['geometry', 'CDEUNIS_1_l1']]
            mask_l1_path = level1_dir / l1_name
            if not mask_l1_path.exists():
                rasterize(polygons_one_tif_l1, full_img, mask_l1_path)

            # RASTERIZE AT LEVEL 2
            polygons_one_tif_l2 = polygons_one_tif[['geometry', 'CDEUNIS_1_l2', 'index']]
            mask_l2_path = level2_dir / l2_name
            if not mask_l2_path.exists():
                rasterize(polygons_one_tif_l2, full_img, mask_l2_path)

            # RASTERIZE AT LEVEL 3
            polygons_one_tif_l3 = polygons_one_tif[['geometry', 'CDEUNIS_1_l3', 'index']]
            mask_l3_path = level3_dir / l3_name
            if not mask_l3_path.exists():
                rasterize(polygons_one_tif_l3, full_img, mask_l3_path)

            print(f'{count_rasterized_image}/{total_images} images rasterized.')

        except ValueError as e:
            # print polygons_one_tif_l1[index]
            print("Problem of geometry with shp ")
            #print(polygons_one_tif_l1)
            print(polygons_one_tif_l2)
            print(polygons_one_tif_l3)
            print(f'Skipping rasterization due to error: {e}')

# timer
end = time.time()
print(f'Number of files in level1: {len(list(level1_dir.rglob("*.tif")))}')
print(f'Number of files in level2: {len(list(level2_dir.rglob("*.tif")))}')
print(f'Number of files in level3: {len(list(level3_dir.rglob("*.tif")))}')
print(f'{end - start} seconds to rasterize {total_images} images.')