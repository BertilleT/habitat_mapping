import geopandas as gpd
from shapely.geometry import Point
import sys
import numpy as np
import rasterio
from rasterio import features
from utils import *
from pathlib import Path
from rasterio.plot import show

# PATHS and DIRECTORIES
data_dir = Path('../data/')
full_img_dir = data_dir / 'full_img_msk' / 'img'
tif_paths = list(full_img_dir.rglob('*.tif'))
msk_dir = data_dir / 'full_img_msk' / 'msk'
msk_dir.mkdir(exist_ok=True)
csv_dir = Path('../csv/tif_labelled/')
shapefile_path = data_dir / 'shp/HABNATs.shp'
threshold = 0.99
pt_name = 'final_pivot_table.csv'
pivot_table_path = csv_dir / pt_name


# UNIQUE CLASSES AT 3 LEVELS FOR ALL MY POLYGONS

all_polygons, filtered_tif_paths = load_and_filter_shapefile(shapefile_path, pivot_table_path, None)

all_polygons['CDEUNIS_1_l1'] = all_polygons['CDEUNIS_1'].apply(lambda x: x[0] if isinstance(x, str) else x)
all_polygons['CDEUNIS_1_l2'] = all_polygons['CDEUNIS_1'].apply(lambda x: x[0:2] if isinstance(x, str) else x)
all_polygons['CDEUNIS_1_l3'] = all_polygons['CDEUNIS_1'].apply(lambda x: x[0:4] if isinstance(x, str) else x)
# if not in the well represented classes, replace by Z for Others
well_represented_classes = ['E', 'F', 'G', 'I', 'J']
all_polygons.loc[~all_polygons['CDEUNIS_1_l1'].isin(well_represented_classes), 'CDEUNIS_1_l1'] = 'Z'

# if not None, remove the . in the CDEUNIS_1_l3
all_polygons.loc[all_polygons['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'] = all_polygons.loc[all_polygons['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'].apply(lambda x: x.replace('.', ''))
# keep CDEUNIS_1_l2 value only if its starts by one value from CDEUNIS_1_l1
all_polygons.loc[all_polygons['CDEUNIS_1_l2'].notnull(), 'CDEUNIS_1_l2'] = all_polygons.loc[all_polygons['CDEUNIS_1_l2'].notnull(), 'CDEUNIS_1_l2'].apply(lambda x: x if x[0] in all_polygons['CDEUNIS_1_l1'].unique() else None)
# keep CDEUNIS_1_l3 value only if its starts by two values from CDEUNIS_1_l2
all_polygons.loc[all_polygons['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'] = all_polygons.loc[all_polygons['CDEUNIS_1_l3'].notnull(), 'CDEUNIS_1_l3'].apply(lambda x: x if x[0:2] in all_polygons['CDEUNIS_1_l2'].unique() else None)

print(all_polygons['CDEUNIS_1_l1'].unique())
print(all_polygons['CDEUNIS_1_l2'].unique())
print(all_polygons['CDEUNIS_1_l3'].unique())

# save to csv list of uniques class at level 1, level 2 and level 3
my_unique_classes_dict = {
    'l1': all_polygons['CDEUNIS_1_l1'].unique(),
    'l2': all_polygons['CDEUNIS_1_l2'].unique(),
    'l3': all_polygons['CDEUNIS_1_l3'].unique(),
}
series_list = [pd.Series(values, name=key) for key, values in my_unique_classes_dict.items()]
my_unique_classes_df = pd.concat(series_list, axis=1)
path_to_un_classes = csv_dir / 'classes_grouped_3l_new.csv'
if not path_to_un_classes.exists():
    my_unique_classes_df.to_csv(path_to_un_classes, index=False)