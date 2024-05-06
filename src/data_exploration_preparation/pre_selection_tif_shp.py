import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from shapely.geometry import box
import pandas as pd
from matplotlib.patches import Patch
import os

data_dir = Path('../original_data/')
# Load the shapefile
shapefile_path = data_dir / 'data_1/HABNAT/HABNATs.shp'
# Load the tif files in original_data
my_tif_paths = list(data_dir.rglob('*.tif'))

# Read shapefile
shapefile = gpd.read_file(shapefile_path)
# Add unique index
shapefile['index'] = range(len(shapefile))

# Store the shp id and the tif path when intersect > 0 in a csv

my_polygons = shapefile[['index', 'geometry']]

intersect = []
# check or valid polygons
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
pivot_table_path = f'../csv/tif_labelled/pivot_table.csv'
intersect_df = pd.DataFrame(intersect)
intersect_df.to_csv(pivot_table_path, index=False)
print(f'The pivot table of the intersection between polygons and tif files saved.')