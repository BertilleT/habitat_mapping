# load l1_nb_pixels_by_zone.csv
# do percentage by zone and percentage aggragated
import pandas as pd
from pathlib import Path

path = Path(f'../../data/patch256/msk/')
l1_classes_path = Path('../../csv/l2_nb_pixels_by_zone.csv')

l1_classes_by_zone_df = pd.read_csv(l1_classes_path)
l1_classes_by_zone_df = l1_classes_by_zone_df.set_index('int')
l1_classes_by_zone_df['all_zones'] = l1_classes_by_zone_df.sum(axis=1)
#for eahc column, trasnform the values in percentages
for col in l1_classes_by_zone_df.columns:
    # sum of values from one column
    total = l1_classes_by_zone_df[col].sum()
    l1_classes_by_zone_df[col] = l1_classes_by_zone_df[col] / total

print(l1_classes_by_zone_df)
l1_classes_by_zone_df.to_csv('../../csv/l2_per_nb_pixels_by_zone_per.csv', index=True)