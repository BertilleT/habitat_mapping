## make a list of all zones 
## list all folders name in patch256
## split at 8 and keep first name
## unique names

from pathlib import Path
import pandas as pd
import rasterio
import torch

masks_path = Path('../../data/patch256/msk/')
l1_classes_path = Path('../../csv/l1_dict.csv')
l2_classes_path = Path('../../csv/l2_dict.csv')
l3_classes_path = Path('../../csv/l3_dict.csv')
l1_nb_pixels_by_zone_path = Path('../../csv/l1_nb_pixels_by_zone.csv')
l2_nb_pixels_by_zone_path = Path('../../csv/l2_nb_pixels_by_zone.csv')
l3_nb_pixels_by_zone_path = Path('../../csv/l3_nb_pixels_by_zone.csv')

zones = []
for zone in masks_path.iterdir():
    zones.append(zone.name.split('_')[0])
zones = list(set(zones))
#print(zones)

#load available classes at level 1. Check int column in l1_dict.csv
l1_list = pd.read_csv(l1_classes_path)
l2_list = pd.read_csv(l2_classes_path)
l3_list = pd.read_csv(l3_classes_path)
#l1_dict to list from int column
l1_list = l1_list['int'].tolist()
l2_list = l2_list['int'].tolist()
l3_list = l3_list['int'].tolist()
#print(l1_list)

per_l1_cl_by_zone = pd.DataFrame(0, index=range(len(l1_list)), columns=zones)
per_l2_cl_by_zone = pd.DataFrame(0, index=range(len(l2_list)), columns=zones)
per_l3_cl_by_zone = pd.DataFrame(0, index=range(len(l3_list)), columns=zones)
#print(per_l1_cl_by_zone)

for zone in zones:
    print('--------------------------Processing ', zone, '--------------------------')
    #list all masks path from data/patch256/msk which containes the zone name   msk_zone102_0_0_etc.tif
    masks = list(masks_path.rglob(f'*_{zone}_*'))
    masks = [mask for mask in masks if mask.suffix == '.tif']
    # create an empty dict to store the number of pixels for each class
    l1_nb_pixels_bycl = {i: 0 for i in l1_list}
    l2_nb_pixels_bycl = {i: 0 for i in l2_list}
    l3_nb_pixels_bycl = {i: 0 for i in l3_list}

    #iterate over masks
    c = 0
    for mask_path in masks:
        print('Processing mask ', c, ' out of ', len(masks))
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
