import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from utils import *
from pathlib import Path
from rasterio.plot import show
import time

# PATHS and DIRECTORIES
data_dir = Path('../data/')
full_img_dir = data_dir / 'full_img_msk' / 'img'
tif_paths = list(full_img_dir.rglob('*.tif'))
msk_dir = data_dir / 'full_img_msk' / 'msk'
msk_dir.mkdir(exist_ok=True)
level123_dir = msk_dir / 'level123'
level123_dir.mkdir(exist_ok=True)
csv_dir = Path('../csv/')
shapefile_path = data_dir / 'shp/HABNATs.shp'
threshold = 0.99
pt_name = 'intersection_shp_tif_' + str(threshold) + '.csv'
pivot_table_path = csv_dir / pt_name
path_to_un_classes = csv_dir / 'classes_grouped_3l.csv'

# Load the csv of unique classes
unique_classes = pd.read_csv(path_to_un_classes)

# DICTIONNARY CLASS-INTEGER FOR 3 LEVELS
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

#--------------------------------------------------------------
total_images = len(tif_paths)
count_rasterized_image = 0
shp = load_all_shapefile(shapefile_path)
intersect_df = pd.read_csv(pivot_table_path)
# timer
start = time.time()

for full_img in tif_paths[:3]:
    count_rasterized_image += 1
    # Renaming img_blabla to msk_blabla
    split_name = full_img.stem.split('_')
    new_name = '_'.join(split_name[-3:])
    new_name = 'msk_' + new_name + '.tif'
    l1_name = 'l1_' + new_name
    l2_name = 'l2_' + new_name
    l3_name = 'l3_' + new_name

    polygons_one_tif, _ = filter_shapefile(shp, intersect_df, full_img)

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
    # turn values to int 
    polygons_one_tif['CDEUNIS_1_l1'] = polygons_one_tif['CDEUNIS_1_l1'].astype(int)
    polygons_one_tif['CDEUNIS_1_l2'] = polygons_one_tif['CDEUNIS_1_l2'].astype(int)
    polygons_one_tif['CDEUNIS_1_l3'] = polygons_one_tif['CDEUNIS_1_l3'].astype(int)
    # if CDEUNIS_1_l2 has less than 2 digits, add a 0 at the beginning
    polygons_one_tif['CDEUNIS_1_l2'] = polygons_one_tif['CDEUNIS_1_l2'].apply(lambda x: str(x).zfill(2))
    polygons_one_tif['CDEUNIS_1_l3'] = polygons_one_tif['CDEUNIS_1_l3'].apply(lambda x: str(x).zfill(3))
    print(polygons_one_tif['CDEUNIS_1_l1'].unique())
    print(polygons_one_tif['CDEUNIS_1_l2'].unique())
    print(polygons_one_tif['CDEUNIS_1_l3'].unique())
    try: 
        #concat the 3 columns values

        polygons_one_tif['label_levels123'] = polygons_one_tif.apply(lambda row: str(row['CDEUNIS_1_l1']) + '' + str(row['CDEUNIS_1_l2']) + '' + str(row['CDEUNIS_1_l3']), axis=1)
        # turn to one single integer
        print(polygons_one_tif['label_levels123'].unique())
        polygons_one_tif['label_levels123'] = polygons_one_tif['label_levels123'].astype('uint32')

        polygons_one_tif = polygons_one_tif[['geometry', 'label_levels123']]
        print(polygons_one_tif['label_levels123'].unique())
         # RASTERIZE AT LEVEL 123
        polygons_one_tif = polygons_one_tif[['geometry', 'label_levels123']]
        mask_l123_path = level123_dir / new_name
        if not mask_l123_path.exists():
            rasterize(polygons_one_tif, full_img, mask_l123_path)
        print(f'{count_rasterized_image}/{total_images} images rasterized.')
    except ValueError as e:
        # print polygons_one_tif_l1[index]
        print("Problem of geometry with shp of id ", polygons_one_tif.index)
        print(f'Skipping rasterization due to error: {e}')

# timer
end = time.time()
print(f'{end - start} seconds to rasterize {total_images} images.')