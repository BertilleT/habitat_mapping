from pathlib import Path
import geopandas as gpd
import rasterio
from utils import *

# DEFINE VARIABLES
threshold = 0.75
correct_TXRECOUV_1 = False
add_date_column_to_shp = False
patch_size = 64
print('Variables defined. ')

# DEFINE PATHS
data_dir = Path('../data/')
csv_dir = Path('../csv/')
shapefile_path = data_dir / 'data_1/HABNAT/HABNATs.shp'
my_tif_paths = list(data_dir.rglob('*.tif'))
excel_date_path = data_dir / 'data_1/HABNAT/BDD_AJ_HABNAT_FINALE2.xlsx'
pt_name = 'intersection_shp_tif_' + str(threshold)[2:] + '.csv'
pivot_table_path = csv_dir / pt_name
print('Paths defined. ')
shapefile = gpd.read_file(shapefile_path)
print('HABNATs.shp loaded.')
shapefile['index'] = range(len(shapefile))

# GENERATE PIVOT TABLE
# The pivot table is a csv file that links the shapefile and the tif images based on a minimum intersection threshold
if not pivot_table_path.exists():
    print(str(pivot_table_path) + ' does not exist. Generating pivot table...')
    intersect_df, polygons_kept_per = generate_pivot_table_intersect(shapefile, my_tif_paths, pivot_table_path, threshold)
    print('Pivot table generated.')
    print(f'{round(polygons_kept_per, 2)}% of the polygons have been kept. ')
else:
    print('Pivot table already exists. ')
    intersect_df = pd.read_csv(pivot_table_path)
    polygons_kept_per = round(len(intersect_df['polygon_index'].unique()) * 100 / len(shapefile), 2)
    print(f'{polygons_kept_per}% of the polygons have been kept. ')

'''# Keep only the polygons and images present in the pivot table
filtered_shapefile = shapefile[shapefile['index'].isin(intersect_df['polygon_index'])]
filtered_shapefile.loc[filtered_shapefile['CDEUNIS_1'] == 'G1131', 'CDEUNIS_1'] = 'G1.131'
# Correct TXRECOUV_1(covery rate occupied by the first label). We consider it is 100% when it is 0 and CDEUNIS_2 is null
if correct_TXRECOUV_1:
    filtered_shapefile.loc[(filtered_shapefile['TXRECOUV_1'] == 0) & pd.isnull(filtered_shapefile['CDEUNIS_2']), 'TXRECOUV_1'] = 100

# Add column year to the shapefile
if add_date_column_to_shp:
    filtered_shapefile = add_date_column(filtered_shapefile, excel_date_path)
    filtered_shapefile['year'] = filtered_shapefile['date_habna'].apply(lambda x: str(x)[:4] if not pd.isnull(x) else x)
    filtered_shapefile['year'] = filtered_shapefile['year'].fillna(0)
    filtered_shapefile['year'] = filtered_shapefile['year'].apply(lambda x: int(x))'''

filtered_tif_paths = intersect_df['tif_path'].unique()

# Split the images into patches of size patch_size
for tif_path in filtered_tif_paths:
    split_image_into_patches(tif_path, patch_size)
