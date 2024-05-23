## make a list of all zones 
## list all folders name in patch256
## split at 8 and keep first name
## unique names

from pathlib import Path
import pandas as pd

masks_path = Path('../../data/patch256/msk/')
l1_classes_path = Path('../../csv/l1_dict.csv')
l2_classes_path = Path('../../csv/l2_dict.csv')
l3_classes_path = Path('../../csv/l3_dict.csv')

zones = []
for zone in masks_path.iterdir():
    zones.append(zone.name.split('_')[0])
zones = list(set(zones))
print(zones)

#load available classes at level 1. Check int column in l1_dict.csv
l1_list = pd.read_csv(l1_classes_path)
#l1_dict to list from int column
l1_list = l1_list['int'].tolist()
print(l1_list)

per_l1_cl_by_zone = pd.DataFrame(0, index=range(len(l1_list)), columns=zones)

print(per_l1_cl_by_zone)

for zone in zones:
    #list all masks in zone
    masks = list(masks_path.glob(f'{zone}_*'))