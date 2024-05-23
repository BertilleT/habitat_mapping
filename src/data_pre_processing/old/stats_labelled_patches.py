import numpy as np
import rasterio
from utils import *
from pathlib import Path
import csv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

patch_size = 128
labelled_cover = 0.95
save_csv = False
print_stats = True

masks_path = Path(f'../data/patch{patch_size}/msk/l123')
csv1_name = f'../csv/coverage_patch/class_count_l1_{patch_size}.csv'
csv2_name = f'../csv/coverage_patch/class_count_l2_{patch_size}.csv'

def class_value_dict(mask_path, level):
    with rasterio.open(mask_path) as src:
        mask = src.read(level)
        unique, counts = np.unique(mask, return_counts=True)
        class_value = {k: v for k, v in zip(unique, counts)}
    return class_value

if save_csv:
    my_masks = list(masks_path.rglob('*.tif'))
    #shuffle
    np.random.shuffle(my_masks)
    # select 25% of the patches
    my_masks = my_masks[:int(len(my_masks)*0.20)]

    total_patches = len(my_masks)
    l1_df = pd.DataFrame(columns=["mask_path", 0, 1, 2, 3, 4, 5, 6])
    # Set "mask_path" as the index
    l1_df.set_index('mask_path', inplace=True)

    l2_df = pd.DataFrame(columns=["mask_path"] + list(range(1, 27)))
    l2_df.set_index('mask_path', inplace=True)

    counter = 0

    print(f'-----------------------------Patch size: {patch_size}')
    for mask in my_masks:
        counter += 1
        if counter % 100 == 0:
            print(f'Processing {counter} over {total_patches} patches.')
        
        # HETEROGENEITY OF LABELLED PATCHES AT LEVEL 1
        class_value_l1 = class_value_dict(mask, 1)
        # add class_value_l1 to l1_dict
        l1_df.loc[mask] = class_value_l1
        
        # HETEROGENEITY OF LABELLED PATCHES AT LEVEL 2
        class_value_l2 = class_value_dict(mask, 2)
        l2_df.loc[mask] = class_value_l2

    if save_csv:
        l1_df.reset_index().to_csv(csv1_name, index=False)
    if save_csv:
        l2_df.reset_index().to_csv(csv2_name, index=False)

    print('CSV files saved.')

if print_stats:
    #Load the csv 
    l1_dict = pd.read_csv(csv1_name)
    l2_dict = pd.read_csv(csv2_name)
    total_patches = l1_dict.shape[0]
    patches_kept_nb = 0
    l1_by_class = {k: 0 for k in range(1, 7)}
    l2_by_class = {k: 0 for k in range(1, 27)}
    count = 0

    # loop on csv1_name column mask_path
    for mask_path in l1_dict["mask_path"]:
        count += 1
        if count % 100 == 0:
            print(f'Processing {count} over {total_patches} patches.')
        l1_dict_temp = l1_dict[l1_dict["mask_path"] == mask_path]
        l2_dict_temp = l2_dict[l2_dict["mask_path"] == mask_path]

        labelled_pixels = l1_dict_temp[[str(i) for i in range(1, 7)]].sum(axis=1).iloc[0]
        nb_pixels = l1_dict_temp[[str(i) for i in range(0, 7)]].sum(axis=1).iloc[0]
        
        if labelled_pixels / nb_pixels >= labelled_cover:
            patches_kept_nb += 1
            # remove comumns with 0 and None
            temp_df_1 = l1_dict_temp[[str(i) for i in range(1, 7)]]
            temp_df_1 = temp_df_1.loc[:, (temp_df_1 != 0).any() & temp_df_1.notna().any()]
            # nb of columns remaining in temp_df_1
            len_df1 = len(temp_df_1.columns)
            l1_by_class[len_df1] += 1

            temp_df_2 = l2_dict_temp[[str(i) for i in range(1, 27)]]
            temp_df_2 = temp_df_2.loc[:, (temp_df_2 != 0).any() & temp_df_2.notna().any()]
            len_df2 = len(temp_df_2.columns)
            if len_df2 == 0: 
                print('It seems that the mask below has no pixel labelled at level 2.')
                print(mask_path)
                l1_dict = l1_dict[l1_dict["mask_path"] != mask_path]
                l2_dict = l2_dict[l2_dict["mask_path"] != mask_path]
            else: 
                l2_by_class[len_df2] += 1
        else:
            #remove it from the df 
            l1_dict = l1_dict[l1_dict["mask_path"] != mask_path]
            l2_dict = l2_dict[l2_dict["mask_path"] != mask_path]

    # patches_kept_l1 and patches_kept_l1 in percentages
    heterogeneity_l1_per = {k: round((v * 100) / patches_kept_nb) for k, v in l1_by_class.items()}
    heterogeneity_l2_per = {k: round((v * 100) / patches_kept_nb) for k, v in l2_by_class.items()}
    print(heterogeneity_l1_per)

    print('In total, there are ', total_patches, ' patches, there are all labelled because we did not save the unlabelled patches.')
    print(f'--------------------- THRESHOLD {labelled_cover*100} % ---------------------')
    print(f'{patches_kept_nb} patches kept, which is {patches_kept_nb * 100 / total_patches} % of the patches.')
    print('--------------------- LEVEL1 ---------------------')
    print('Heterogeneity of labelled patches at level 1:')
    print(l1_by_class)
    print('Heterogeneity of labelled patches at level 1 in %:')
    print(heterogeneity_l1_per)
    print('--------------------- LEVEL2 ---------------------')
    print('Heterogeneity of labelled patches at level 2 :')
    print(l2_by_class)
    print('Heterogeneity of labelled patches at level 2 in %:')
    print(heterogeneity_l2_per)