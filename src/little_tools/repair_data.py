# list all the files in new_data
import os
from pathlib import Path

'''data_dir_1 = Path('../new_data/')
tif_paths = list(data_dir_1.rglob('*.tif'))
data_dir_2 = Path('../original_data/')
#'230617_Ecomed_15cm_L93_4canaux_zone172_0_0.tif' in [tif_path.name for tif_path in tif_paths]. Extract zone172

for tif in tif_paths:
    zone = tif.name.split('_')[5]
    #remove zone
    zone_id = zone.replace('zone', '')
    if int(zone_id) > 0 and int(zone_id) < 41:
        sub_dir = 'data_1'
    elif int(zone_id) > 40 and int(zone_id) < 81:
        sub_dir = 'data_2'
    elif int(zone_id) > 80 and int(zone_id) < 121:
        sub_dir = 'data_3'
    elif int(zone_id) > 120:
        sub_dir = 'data_4'
    
    new_tif_path = Path(f'../original_data/{sub_dir}/{zone}/{tif.name}')
    if new_tif_path.exists():
        print(f'{new_tif_path} already exists')
    else:
        print(f'{new_tif_path} does not exist')
        os.rename(tif, new_tif_path)'''


# load tif images in data_1 and new_data_1 in original_data and print how many there are
data_dir_1 = Path('../original_data/data_1')
tif_paths = list(data_dir_1.rglob('*.tif'))
print(len(tif_paths))

data_dir_1 = Path('../original_data/new_data_1')
tif_paths = list(data_dir_1.rglob('*.tif'))
print(len(tif_paths))