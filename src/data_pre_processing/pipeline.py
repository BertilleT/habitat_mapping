import geopandas as gpd
from data_preparation_utils import *
from classification import *
# time
import time
import warnings
warnings.filterwarnings('ignore')

pre_processing_steps = {
    '0_create_pivot_table': False, 
    '1_create_labels_dict': False,
    '2_create_masks': False,
    '3_create_patches': True,
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

# ------------------------------STEP 1: RASTERIZE POLYGONS--------------------------------#

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