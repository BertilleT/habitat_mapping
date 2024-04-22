# Pipeline to pre-process the shapefile
import geopandas as gpd
from pathlib import Path
import pandas as pd

# Define paths
data_dir = Path('../data/')
shapefile_path = data_dir / 'data_1/HABNAT/HABNATs.shp'
pivot_table_path = '../csv/intersection_shp_tif.csv'

# Read shapefile
shapefile = gpd.read_file(shapefile_path)
# Add unique index
shapefile['index'] = range(len(shapefile))
# Load pivot table
intersect_df = pd.read_csv(pivot_table_path)

# Filter shapefile and tif paths
filtered_shapefile = shapefile[shapefile['index'].isin(intersect_df['polygon_index'])]
filtered_tif_paths = intersect_df['tif_path'].unique()

# Correct TXRECOUV_1
filtered_shapefile.loc[(filtered_shapefile['TXRECOUV_1'] == 0) & pd.isnull(filtered_shapefile['CDEUNIS_2']), 'TXRECOUV_1'] = 100