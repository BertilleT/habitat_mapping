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

original_data_dir = Path('../../original_data/')
shapefile_path = original_data_dir / 'data_1/HABNAT/HABNATs.shp'

## POLYGONS

# Read shapefile
my_polygons = gpd.read_file(shapefile_path)
print(f'The original number of polygons is {len(my_polygons)}.')
# Add unique index
my_polygons['index'] = range(len(my_polygons))

'''
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

'''my_tif_paths = list(original_data_dir.glob('**/*.tif')) # get all tif files in the original_data directory and subdirectories
print(f'There are {len(my_tif_paths)} tif files.')
#my_tif_paths = my_tif_paths[:10] # for testing purposes
intersect = []
# check for valid polygons
my_polygons = my_polygons[my_polygons['geometry'].is_valid]
#check for polygons annotated
my_polygons = my_polygons[~my_polygons['CDEUNIS_1'].isnull()]
print(f'{len(my_polygons)} polygons are valid and annotated.')
len_polygon_kept = 0
count = 0
not_annotated_images = []
polygons_matching_one_image = []

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
            polygons_matching_one_image.append(polygon['index'])
            nb_annotated_polygons += 1
            len_polygon_kept += 1
            intersect.append({
                'tif_path': tif_path,
                'polygon_index': polygon['index'],
            })    
    if nb_annotated_polygons == 0:
        print(f'No annotated polygon in {tif_path}.')
        not_annotated_images.append(tif_path)

# Missing images
polygons_matching_one_image = list(set(polygons_matching_one_image))
print(f'{len(polygons_matching_one_image)} polygons match at least one image.')
# get lists of polygpns matching 0 images
polygons_matching_zero_image = list(set(my_polygons['index']) - set(polygons_matching_one_image))
print('The indexes of the polygons matching no images are:', polygons_matching_zero_image)
# print how many polygons do not match images, over the total number of polygons
print(f'{len(polygons_matching_zero_image)} polygons do not match any image, it represents {round(len(polygons_matching_zero_image) * 100 / len(my_polygons))}% of the total number of polygons.')
# create a shpefile with polygons matching 0 images
polygons_matching_zero_image = my_polygons[my_polygons['index'].isin(polygons_matching_zero_image)]
#save shapefile
polygons_matching_zero_image.to_file('polygons_matching_no_image.shp')

# Missing annotated polygons in images
pd.DataFrame(not_annotated_images).to_csv('not_annotated_images.csv', index=False)
print(f'{len_polygon_kept} polygons are kept, it represents {round(len_polygon_kept * 100 / len(my_polygons))}% of the total number of valid polygons. ')
intersect_df = pd.DataFrame(intersect)
intersect_df.to_csv("intersect_df_ecomed", index=False)
print(f'The pivot table of the intersection between polygons and tif files saved.') 
'''

#load the tif_paths saved
tif_paths = pd.read_csv('not_annotated_images.csv')

print(tif_paths.head(100))
tif_paths = tif_paths.rename(columns={'0': 'images_without_annotated_polygons'})
print(tif_paths.head(10))
# Ihave one single col 0 with one item "../../original_data/data_1/zone27/230617_Ecomed_15cm_L93_4canaux_zone27_1_1.tif", "../../original_data/data_2/zone27/230617_Ecomed_15cm_L93_4canaux_zone27_1_1.tif"
# remove ../../original_data/data_1 or ../../original_data/data_2 from the begining of all rows
tif_paths['images_without_annotated_polygons'] = tif_paths['images_without_annotated_polygons'].apply(lambda x: x[27:])
print(tif_paths.head(100))
# unique keep
tif_paths = tif_paths.drop_duplicates(subset='images_without_annotated_polygons')
# get nb
print(f'{len(tif_paths)} images without annotated polygons.')
print('It represents', round(len(tif_paths) * 100 / 689), '% of the total number of images.')
# save to csv
tif_paths.to_csv('images_without_annotated_polygons.csv', index=False)