# load csv file in ../csv/coverage_patch/class_count_l2_256.csv
import pandas as pd
from pathlib import Path

path = Path('../csv/coverage_patch/class_count_l2_256.csv')
labelled_cover = 1
l2_dict = pd.read_csv(path)

l1_by_class = {k: 0 for k in range(1, 7)}
l2_by_class = {k: 0 for k in range(1, 27)}
count = 0

total_patches = l2_dict.shape[0]
pairs_kept = {'img_path': None, 'msk_path': None}

for mask_path in l2_dict["mask_path"].iloc[:10]:
    count += 1
    if count % 100 == 0:
        print(f'Processing {count} over {total_patches} patches.')

    l2_dict_temp = l2_dict[l2_dict["mask_path"] == mask_path]
    print(l2_dict_temp)
    labelled_pixels = l2_dict_temp[[str(i) for i in range(1, 27)]].sum(axis=1).iloc[0]
    nb_pixels = l2_dict_temp[[str(i) for i in range(0, 27)]].sum(axis=1).iloc[0]
    
    if labelled_pixels / nb_pixels >= labelled_cover:
        temp_df_2 = l2_dict_temp[[str(i) for i in range(1, 27)]]
        temp_df_2 = temp_df_2.loc[:, (temp_df_2 != 0).any() & temp_df_2.notna().any()]
        len_df2 = len(temp_df_2.columns)
        if len_df2 > 1: # more than one class in the patch
            pairs_kept['msk_path'] = l2_dict_temp['msk_path'].iloc[0]

# look for the image corresponding to the mask
#msk pah ../data/patch256/msk/l123/zone100_0_0/msk_zone100_0_0_patch_10_10.tif
#img path coresponding is ../data/patch256/img/zone100_0_0/img_zone100_0_0_patch_10_10.tif

# get the image path
img_path = pairs_kept['msk_path'].replace('msk', 'img').replace('l123', '')
pairs_kept['img_path'] = img_path

print(pairs_kept)