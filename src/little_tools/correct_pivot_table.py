# load the pivot table 0.99
# ../data/data_3/zone85/230617_Ecomed_15cm_L93_4canaux_zone85_0_1.tif

import pandas as pd
from pathlib import Path

pivot_table_path = Path('../../csv/tif_labelled/pivot_table.csv')
intersect_df = pd.read_csv(pivot_table_path)
# I have one column named tif_path, value sin it looks like ../data/data_3/zone85/230617_Ecomed_15cm_L93_4canaux_zone85_0_1.tif
# Add new column tif_path_name with the name of the tif file: img_zone85_0_1.tif
intersect_df['tif_path_name'] = intersect_df['tif_path'].apply(lambda x: x.split('_')[-3:])
intersect_df['tif_path_name'] = intersect_df['tif_path_name'].apply(lambda x: '_'.join(x))
#add img
intersect_df['tif_path_name'] = '../data/full_img_msk/img/img_' + intersect_df['tif_path_name']

print(intersect_df['tif_path_name'].head())

# save the new pivot table
intersect_df.to_csv(pivot_table_path, index=False)
