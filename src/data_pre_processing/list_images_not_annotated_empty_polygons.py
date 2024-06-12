import rasterio
from shapely.geometry import box
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

'''def generate_pivot_table_intersect(my_polygons, my_tif_paths, pivot_table_path):
    # This function aims at creating the pivot table to link the shapefiles and the tif images when they intersect
    intersect = []
    # check for valid polygons
    my_polygons = my_polygons[my_polygons['geometry'].is_valid]
    len_polygon_kept = 0
    count = 0
    for tif_path in my_tif_paths:
        count += 1
        print(f'Processing {count} tif files over {len(my_tif_paths)}')
        with rasterio.open(tif_path) as src:
            tif_bounds = src.bounds
            tif_box = box(*tif_bounds)  # Convert BoundingBox to shapely box
        
        # Check each polygon
        for index, polygon in my_polygons.iterrows():
            geom = polygon['geometry']
            if geom is not None and tif_box.intersects(geom):
                len_polygon_kept += 1
                intersect.append({
                    'tif_path': tif_path,
                    'polygon_index': polygon['index'],
                })            
    print(f'{len_polygon_kept} polygons are kept, it represents {round(len_polygon_kept * 100 / len(my_polygons))}% of the total number of valid polygons. ')
    intersect_df = pd.DataFrame(intersect)
    #intersect_df.to_csv(pivot_table_path, index=False)
    #print(f'The pivot table of the intersection between polygons and tif files saved.')
    return intersect_df


original_data_dir = Path('../../original_data/')
shapefile_path = original_data_dir / 'data_1/HABNAT/HABNATs.shp'
pivot_table_path = f'../../csv/final_pivot_table.csv'
data_dir = Path('../../data/')
msk_dir = data_dir / 'full_img_msk' / 'msk' 
full_img_dir = data_dir / 'full_img_msk' / 'img'


## POLYGONS

# Read shapefile
my_polygons = gpd.read_file(shapefile_path)
print(f'The original number of polygons is {len(my_polygons)}.')
# Add unique index
my_polygons['index'] = range(len(my_polygons))

### INVALID POLYGONS
invalid_polygons = my_polygons[~my_polygons['geometry'].is_valid]
print(f'{len(invalid_polygons)} polygons are invalid and will be removed.')
print(f'It represents {round(len(invalid_polygons) * 100 / len(my_polygons))}% of the total number of polygons.')
invalid_polygons.to_file('../../data/invalid_polygons.shp')

#print unique values in CDEUNIS_1
print(f'Unique values in CDEUNIS_1: {my_polygons["CDEUNIS_1"].unique()}')

### NOT ANNOTATED POLYGONS
not_annotated_polygons = my_polygons[my_polygons['CDEUNIS_1'].isnull()]
print(f'{len(not_annotated_polygons)} polygons are not annotated.')
print(f'It represents {round(len(not_annotated_polygons) * 100 / len(my_polygons))}% of the total number of polygons.')
# print 10 rows of not annotated polygons
print(not_annotated_polygons.head(10))
not_annotated_polygons.to_file('../../data/not_annotated_polygons.shp')'''


'''intersect = []
# check for valid polygons
my_polygons = my_polygons[my_polygons['geometry'].is_valid]
#check for polygons annotated
my_polygons = my_polygons[~my_polygons['CDEUNIS_1'].isnull()]
print(f'{len(my_polygons)} polygons are valid and annotated.')
len_polygon_kept = 0
count = 0
not_annotated_images = []
for tif_path in my_tif_paths:
    nb_annotated_polygons = 0
    count += 1
    print(f'Processing {count} tif files over {len(my_tif_paths)}')
    with rasterio.open(tif_path) as src:
        tif_bounds = src.bounds
        tif_box = box(*tif_bounds)  # Convert BoundingBox to shapely box
    
    # Check each polygon
    for index, polygon in my_polygons.iterrows():
        geom = polygon['geometry']
        if geom is not None and tif_box.intersects(geom):
            nb_annotated_polygons += 1
            len_polygon_kept += 1
            intersect.append({
                'tif_path': tif_path,
                'polygon_index': polygon['index'],
            })    
    if nb_annotated_polygons == 0:
        not_annotated_images.append(tif_path)

#save list of not annotated images
pd.DataFrame(not_annotated_images).to_csv('not_annotated_images.csv', index=False)
print(f'{len_polygon_kept} polygons are kept, it represents {round(len_polygon_kept * 100 / len(my_polygons))}% of the total number of valid polygons. ')
intersect_df = pd.DataFrame(intersect)
intersect_df.to_csv("intersect_df_ecomed", index=False)
print(f'The pivot table of the intersection between polygons and tif files saved.')'''