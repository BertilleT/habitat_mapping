import geopandas as gpd
from data_preparation_utils import *
from classification import *
# time
import time
import warnings
import torch
warnings.filterwarnings('ignore')

pre_processing_steps = {
    '0_create_pivot_table': False, 
    '1_create_labels_dict': False,
    '2_create_masks': False,
    '3_create_patches': False,
    '4_rm_nan_img_msk': False,
    '5_count_pixels_class_by_zone': True,  
}

original_data_dir = Path('../../original_data/')
shapefile_path = original_data_dir / 'data_1/HABNAT/HABNATs.shp'
pivot_table_path = f'../../csv/final_pivot_table.csv'
l1_dict_path = f'../../csv/l1_dict.csv'
l2_dict_path = f'../../csv/l2_dict.csv'
l3_dict_path = f'../../csv/l3_dict.csv'
data_dir = Path('../../data/')
msk_dir = data_dir / 'full_img_msk' / 'msk' 
full_img_dir = data_dir / 'full_img_msk' / 'img'
patch_size = 256
msk_folder = Path('../../data/patch256/msk/')
img_folder = Path('../../data/patch256/img/')
l1_nb_pixels_by_zone_path = Path('../../csv/l1_nb_pixels_by_zone.csv')
l2_nb_pixels_by_zone_path = Path('../../csv/l2_nb_pixels_by_zone.csv')
l3_nb_pixels_by_zone_path = Path('../../csv/l3_nb_pixels_by_zone.csv')

# ------------------------------STEP 0: CREATE PIVOT TABLE--------------------------------#

if pre_processing_steps['0_create_pivot_table']:
    # Load the tif files in original_data
    my_tif_paths = list(original_data_dir.rglob('*.tif'))

    # Read shapefile
    shapefile = gpd.read_file(shapefile_path)
    # Add unique index
    shapefile['index'] = range(len(shapefile))

    # Store the shp id and the tif path when intersect > 0 in a csv
    my_polygons = shapefile[['index', 'geometry']]
    intersect_df = generate_pivot_table_intersect(my_polygons, my_tif_paths, pivot_table_path)
    print('STEP 0, create_pivot_table, done.')
else: 
    print('STEP 0, create_pivot_table, skipped.')
    intersect_df = pd.read_csv(pivot_table_path)

# ------------------------------STEP 1: CREATE LABELS DICT--------------------------------#

if pre_processing_steps['1_create_labels_dict']: 
    # UNIQUE CLASSES AT 3 LEVELS FOR ALL MY POLYGONS
    all_polygons = load_all_shapefile(shapefile_path)
    all_polygons, _ = correct_filter_shapefile(all_polygons, intersect_df, None)

    all_polygons['CDEUNIS_1_l1'] = all_polygons['CDEUNIS_1'].apply(lambda x: x[0] if isinstance(x, str) else x)
    all_polygons['CDEUNIS_1_l2'] = all_polygons['CDEUNIS_1'].apply(lambda x: x[0:2] if isinstance(x, str) else x)
    all_polygons['CDEUNIS_1_l3'] = all_polygons['CDEUNIS_1'].apply(lambda x: x[0:4] if isinstance(x, str) else x)
    # if not in the well represented classes, replace by Z for Others
    selected_classes = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J']

    all_polygons.loc[~all_polygons['CDEUNIS_1_l1'].isin(selected_classes), 'CDEUNIS_1_l1'] = 'Z'

    # if not None, remove the . in the CDEUNIS_1_l3
    all_polygons.loc[all_polygons['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'] = all_polygons.loc[all_polygons['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'].apply(lambda x: x.replace('.', ''))
    # keep CDEUNIS_1_l2 value only if its starts by one value from CDEUNIS_1_l1
    all_polygons.loc[all_polygons['CDEUNIS_1_l2'].notnull(), 'CDEUNIS_1_l2'] = all_polygons.loc[all_polygons['CDEUNIS_1_l2'].notnull(), 'CDEUNIS_1_l2'].apply(lambda x: x if x[0] in all_polygons['CDEUNIS_1_l1'].unique() else None)
    # keep CDEUNIS_1_l3 value only if its starts by two values from CDEUNIS_1_l2
    all_polygons.loc[all_polygons['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'] = all_polygons.loc[all_polygons['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'].apply(lambda x: x if x[0:2] in all_polygons['CDEUNIS_1_l2'].unique() else None)
    # turn values None, A and J from CDEUNIS_1_l2 to U
    all_polygons.loc[all_polygons['CDEUNIS_1_l2'].isin(['A', 'J', None]), 'CDEUNIS_1_l2'] = 'U9'
    # turn values None, or with less than 3 characters from CDEUNIS_1_l3 to Z99
    all_polygons.loc[all_polygons['CDEUNIS_1_l3'].isnull(), 'CDEUNIS_1_l3'] = 'Z99'
    all_polygons.loc[all_polygons['CDEUNIS_1_l3'].apply(lambda x: len(x) < 3), 'CDEUNIS_1_l3'] = 'Z99'
    print(all_polygons['CDEUNIS_1_l1'].unique())
    print(all_polygons['CDEUNIS_1_l2'].unique())
    #print(all_polygons['CDEUNIS_1_l3'].unique())

    # create dict_l1 with name, code and int columns
    codes_l1 = all_polygons['CDEUNIS_1_l1'].unique()
    #order by alphabetical order
    codes_l1 = sorted(codes_l1)
    names_l1 = [my_classes_l1[code] for code in codes_l1]
    # remove "" from names when there are
    names_l1 = [name.replace('"', '') for name in names_l1]
    integers_l1 = range(len(codes_l1))
    dict_l1 = {
        'code': codes_l1,
        'name': names_l1,
        'int': integers_l1,
    }

    # create dict_l2 with name, code and int columns
    codes_l2 = all_polygons['CDEUNIS_1_l2'].unique()
    codes_l2 = sorted(codes_l2)
    names_l2 = [my_classes_l2[code] for code in codes_l2]
    integers_l2 = range(len(codes_l2))
    #change last values of integers_l2 by 254
    integers_l2 = [254 if i == len(codes_l2) - 1 else i for i in integers_l2]
    dict_l2 = {
        'code': codes_l2,
        'name': names_l2,
        'int': integers_l2,
    }

    # create dict_l3 with name, code and int columns
    codes_l3 = all_polygons['CDEUNIS_1_l3'].unique()
    codes_l3 = sorted(codes_l3)
    #names_l3 = [all_polygons[all_polygons['CDEUNIS_1_l3'] == code]['CDEUNIS'].unique()[0] for code in codes_l3]
    integers_l3 = range(len(codes_l3))
    integers_l3 = [254 if i == len(codes_l3) - 1 else i for i in integers_l3]
    dict_l3 = {
        'code': codes_l3,
        'int': integers_l3,
    }

    # Save the dictionaries
    dict_l1_df = pd.DataFrame(dict_l1)
    dict_l2_df = pd.DataFrame(dict_l2)
    dict_l3_df = pd.DataFrame(dict_l3)
    dict_l1_df.to_csv(l1_dict_path, index=False)
    dict_l2_df.to_csv(l2_dict_path, index=False)
    dict_l3_df.to_csv(l3_dict_path, index=False)
    int_classes_l1 = dict_l1_df.set_index('code')['int'].to_dict()
    int_classes_l2 = dict_l2_df.set_index('code')['int'].to_dict()
    int_classes_l3 = dict_l3_df.set_index('code')['int'].to_dict()
    print('STEP 1, create_labels_dict, done.')
else:
    print('STEP 1, create_labels_dict, skipped.')
    dict_l1_df = pd.read_csv(l1_dict_path)
    dict_l2_df = pd.read_csv(l2_dict_path)
    dict_l3_df = pd.read_csv(l3_dict_path)
    #keys are in the column code and values in the column int
    int_classes_l1 = dict_l1_df.set_index('code')['int'].to_dict()
    int_classes_l2 = dict_l2_df.set_index('code')['int'].to_dict()
    int_classes_l3 = dict_l3_df.set_index('code')['int'].to_dict()


# ------------------------------STEP 2: RASTERIZE POLYGONS--------------------------------#

if pre_processing_steps['2_create_masks']:
    # time
    start = time.time()

    all_polygons = load_all_shapefile(shapefile_path)
    # add '../' to tif_path_name in all_polygons
    intersect_df['tif_path_name'] = intersect_df['tif_path_name'].apply(lambda x: '../' + x)

    tif_paths = list(full_img_dir.rglob('*.tif'))
    #order by alphatical order
    tif_paths = sorted(tif_paths)
    count_rasterized_image = 0
    for full_img in tif_paths[-1:]:
        count_rasterized_image += 1
        # Renaming img_blabla to msk_blabla
        split_name = full_img.stem.split('_')
        new_name = '_'.join(split_name[-3:])
        new_name_ = 'msk_' + new_name + '.tif'
        polygons_one_tif, _ = correct_filter_shapefile(all_polygons, intersect_df, full_img)

        polygons_one_tif['CDEUNIS_1_l1'] = polygons_one_tif['CDEUNIS_1'].apply(lambda x: x[0] if isinstance(x, str) else x)
        polygons_one_tif['CDEUNIS_1_l2'] = polygons_one_tif['CDEUNIS_1'].apply(lambda x: x[0:2] if isinstance(x, str) else x)
        polygons_one_tif['CDEUNIS_1_l3'] = polygons_one_tif['CDEUNIS_1'].apply(lambda x: x[0:4] if isinstance(x, str) else x)
        selected_classes = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J']
        polygons_one_tif.loc[~polygons_one_tif['CDEUNIS_1_l1'].isin(selected_classes), 'CDEUNIS_1_l1'] = 'Z'
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'] = polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'].apply(lambda x: x.replace('.', ''))
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l2'].notnull(), 'CDEUNIS_1_l2'] = polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l2'].notnull(), 'CDEUNIS_1_l2'].apply(lambda x: x if x[0] in polygons_one_tif['CDEUNIS_1_l1'].unique() else None)
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'] = polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'].apply(lambda x: x if x[0:2] in polygons_one_tif['CDEUNIS_1_l2'].unique() else None)
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l2'].isin(['A', 'J', None]), 'CDEUNIS_1_l2'] = 'U9'
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l3'].isnull(), 'CDEUNIS_1_l3'] = 'Z99'
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l3'].apply(lambda x: len(x) < 3), 'CDEUNIS_1_l3'] = 'Z99'
        
        polygons_one_tif['CDEUNIS_1_l1'] = polygons_one_tif['CDEUNIS_1_l1'].map(int_classes_l1)
        polygons_one_tif['CDEUNIS_1_l2'] = polygons_one_tif['CDEUNIS_1_l2'].map(int_classes_l2)
        polygons_one_tif['CDEUNIS_1_l3'] = polygons_one_tif['CDEUNIS_1_l3'].map(int_classes_l3)
        # values which have not been mapped set to 254
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l1'].isnull(), 'CDEUNIS_1_l1'] = 254
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l2'].isnull(), 'CDEUNIS_1_l2'] = 254
        polygons_one_tif.loc[polygons_one_tif['CDEUNIS_1_l3'].isnull(), 'CDEUNIS_1_l3'] = 254
        polygons_one_tif['CDEUNIS_1_l1'] = polygons_one_tif['CDEUNIS_1_l1'].astype(int)
        polygons_one_tif['CDEUNIS_1_l2'] = polygons_one_tif['CDEUNIS_1_l2'].astype(int)
        polygons_one_tif['CDEUNIS_1_l3'] = polygons_one_tif['CDEUNIS_1_l3'].astype(int)
        polygons_one_tif = polygons_one_tif[['geometry', 'CDEUNIS_1_l1', 'CDEUNIS_1_l2', 'CDEUNIS_1_l3']]
        # add + 1 to eevery col
        polygons_one_tif['CDEUNIS_1_l1'] = polygons_one_tif['CDEUNIS_1_l1'] + 1
        polygons_one_tif['CDEUNIS_1_l2'] = polygons_one_tif['CDEUNIS_1_l2'] + 1
        polygons_one_tif['CDEUNIS_1_l3'] = polygons_one_tif['CDEUNIS_1_l3'] + 1
        try:
            rasterize_multichannel(polygons_one_tif, full_img, msk_dir / (new_name_))
            print(f'{count_rasterized_image}/{len(tif_paths)}: {new_name_} done.')
            
        except Exception as e:
            print(f'Error with {full_img}')
            print(e)
    
    print(f'Time elapsed: {time.time() - start}')

# ------------------------------STEP 3: CREATE PATCHES--------------------------------#

if pre_processing_steps['3_create_patches']:
    tif_paths = list(full_img_dir.rglob('*.tif'))
    #order in alphabetical order
    tif_paths = sorted(tif_paths)
    print(len(tif_paths))  # 375
    #list all directries name in Path('../data/patch64/msk/l123')
    down = 300
    up = len(tif_paths)
    for tif_path in tif_paths[down:]:
        name = tif_path.stem
        mask_path = Path('../../data/full_img_msk/msk') / f'msk_{name[4:]}.tif'
        split_img_msk_into_patches(tif_path, mask_path, patch_size)
        print('Processing image number ' + str(down) + ' over ' + str(up) + ' done.')
        down += 1

# ------------------------------STEP 4: REMOVE IMG,MSK when IMG FULL OF 0--------------------------------#
if pre_processing_steps['4_rm_nan_img_msk']:
    # remove the pairs of images when img is full of zero. 
    msk_paths = list(msk_folder.rglob('*.tif'))
    print(f'Number of masks: {len(msk_paths)}')
    # BUILD THE IMG PATHS CORRESPONDING TO THE MASKS PATHS
    img_paths = [img_folder / msk_path.parts[-2] / msk_path.name.replace('msk', 'img') for msk_path in msk_paths]
    print(f'Number of images: {len(img_paths)}')

    # CHECK IF THE IMAGES HAVE ZERO VALUES
    imgs_zero_paths = []
    for img_path in img_paths:
        with rasterio.open(img_path) as src:
            img = src.read()
        if np.count_nonzero(img) == 0:
            imgs_zero_paths.append(img_path)

    print(f'Number of images with zero values: {len(imgs_zero_paths)}')
    # LOOK FOR MASKS MATCHING THE IMAGES WITH ZERO VALUES
    msks_paths_zero = [msk_folder / img_path.parts[-2] / img_path.name.replace('img', 'msk') for img_path in imgs_zero_paths]

    # drop img and maks with zero values
    for img_path, msk_path in zip(imgs_zero_paths, msks_paths_zero):
        img_path.unlink()
        msk_path.unlink()
    print('Images with zero values removed')

    msk_paths = list(msk_folder.rglob('*.tif'))
    print(f'Number of masks: {len(msk_paths)}')
    # BUILD THE IMG PATHS CORRESPONDING TO THE MASKS PATHS
    img_paths = [img_folder / msk_path.parts[-2] / msk_path.name.replace('msk', 'img') for msk_path in msk_paths]
    print(f'Number of images: {len(img_paths)}')

# ------------------------------STEP 5: COUNT PIXELS BY CLASS AND BY ZONE--------------------------------#

if pre_processing_steps['5_count_pixels_class_by_zone']:
    zones = []
    for zone in msk_folder.iterdir():
        zones.append(zone.name.split('_')[0])
    zones = list(set(zones))
    #print(zones)

    #load available classes at level 1. Check int column in l1_dict.csv
    l1_list = pd.read_csv(l1_dict_path)
    l2_list = pd.read_csv(l2_dict_path)
    l3_list = pd.read_csv(l3_dict_path)
    #l1_dict to list from int column
    l1_list = l1_list['int'].tolist()
    l2_list = l2_list['int'].tolist()
    l3_list = l3_list['int'].tolist()
    #print(l1_list)

    per_l1_cl_by_zone = pd.DataFrame(0, index=range(len(l1_list)), columns=zones)
    per_l2_cl_by_zone = pd.DataFrame(0, index=range(len(l2_list)), columns=zones)
    per_l3_cl_by_zone = pd.DataFrame(0, index=range(len(l3_list)), columns=zones)
    #print(per_l1_cl_by_zone)
    nb_zones = len(zones)
    z = 0
    for zone in zones:
        print('--------------------------Processing ', zone, '--------------------------')
        z += 1
        print('Zone number ', z, ' out of ', nb_zones)
        #list all masks path from data/patch256/msk which containes the zone name   msk_zone102_0_0_etc.tif
        masks = list(msk_folder.rglob(f'*_{zone}_*'))
        masks = [mask for mask in masks if mask.suffix == '.tif']
        # create an empty dict to store the number of pixels for each class
        l1_nb_pixels_bycl = {i: 0 for i in l1_list}
        l2_nb_pixels_bycl = {i: 0 for i in l2_list}
        l3_nb_pixels_bycl = {i: 0 for i in l3_list}

        #iterate over masks
        c = 0
        for mask_path in masks:
            #print('Processing mask ', c, ' out of ', len(masks))
            # Open mask
            with rasterio.open(mask_path) as mask:
                # Read band 1
                mask_band1 = mask.read(1)
                # Iterate over classes in l1_list
                for i in l1_list:
                    # Add the number of pixels for each class
                    l1_nb_pixels_bycl[i] += torch.sum(torch.tensor(mask_band1) == i).item()
                mask_band2 = mask.read(2)
                for i in l2_list:
                    l2_nb_pixels_bycl[i] += torch.sum(torch.tensor(mask_band2) == i).item()
                mask_band3 = mask.read(3)
                for i in l3_list:
                    l3_nb_pixels_bycl[i] += torch.sum(torch.tensor(mask_band3) == i).item()
            c += 1

        for i in l1_list:
            per_l1_cl_by_zone.loc[i, zone] = l1_nb_pixels_bycl[i]
        
        for i in l2_list:
            per_l2_cl_by_zone.loc[i, zone] = l2_nb_pixels_bycl[i]

        for i in l3_list:
            per_l3_cl_by_zone.loc[i, zone] = l3_nb_pixels_bycl[i]

    per_l1_cl_by_zone.to_csv(l1_nb_pixels_by_zone_path)
    per_l2_cl_by_zone.to_csv(l2_nb_pixels_by_zone_path)
    per_l3_cl_by_zone.to_csv(l3_nb_pixels_by_zone_path)  