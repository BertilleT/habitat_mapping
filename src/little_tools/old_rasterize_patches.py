import geopandas as gpd
from shapely.geometry import Point
import sys
import numpy as np
import rasterio
from rasterio import features
from utils import *
from pathlib import Path
from rasterio.plot import show

def overwriteRaster(polygons, template_raster, output_raster):
    template_ds = rasterio.open(template_raster)
    template_meta = template_ds.meta.copy()
    
    with rasterio.open(output_raster, 'w+', **template_meta) as out:
        out_arr = out.read(1)
        shapes = ((geom,value) for geom, value in zip(polygons.geometry, polygons.iloc[:,1]))
        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)


data_file = 'data_1'
zone = 'zone26'
patch_size = 64
tif_id = '0_0'

threshold = 0.75
data_dir = Path('../original_data/')
csv_dir = Path('../csv/')
shapefile_path = data_dir / 'data_1/HABNAT/HABNATs.shp'
pt_name = 'intersection_shp_tif_' + str(threshold)[2:] + '.csv'
pivot_table_path = csv_dir / pt_name
patches_path = Path(f'../original_data/{data_file}/{zone}/patch_{str(patch_size)}_img_{tif_id}/')
# 230617_Ecomed_15cm_L93_4canaux_zone26_0_0.tif
tif_path = Path(f'../original_data/{data_file}/{zone}/230617_Ecomed_15cm_L93_4canaux_{zone}_{tif_id}.tif')

'''polygons_one_tif, _ = load_and_filter_shapefile(shapefile_path, pivot_table_path, tif_path)
polygons_one_tif['CDEUNIS_1_l1'] = polygons_one_tif['CDEUNIS_1'].apply(lambda x: x[0] if isinstance(x, str) else x)
well_represented_classes = ['E', 'F', 'G', 'I', 'J']
polygons_one_tif.loc[~polygons_one_tif['CDEUNIS_1_l1'].isin(well_represented_classes), 'CDEUNIS_1_l1'] = 'Z'
'''
int_classes = {
    'E': 1,
    'F': 2,
    'G': 3,
    'I': 4, 
    'J': 5,
    'Z': 0,
}
'''polygons_one_tif['CDEUNIS_1_l1'] = polygons_one_tif['CDEUNIS_1_l1'].map(int_classes)
polygons_one_tif = polygons_one_tif[['geometry', 'CDEUNIS_1_l1']]'''
patches_one_tif = list(patches_path.glob('*.tif'))
one_patch = patches_one_tif[2000] 
# open the patch and print its shape
with rasterio.open(one_patch) as src:
    print(src.shape)
one_patch_id = one_patch.stem.split('_')[-6:]
one_patch_id = '_'.join(one_patch_id)
#use one_patch.stem()
one_mask_path = Path(f'../original_data/{data_file}/{zone}/patch_{str(patch_size)}_msk_{tif_id}/mask_{one_patch_id}.tif')

#overwriteRaster(polygons_one_tif, one_patch, one_mask_path)

'''# Load the mask
mask = rasterio.open(one_mask_path)
show(mask, cmap='viridis')
dummy_plot = plt.imshow(mask.read(1), cmap='viridis')
plt.colorbar(dummy_plot, label='Class')
plt.show()'''

# get shape of mask, how many pixels
mask = rasterio.open(one_mask_path)
mask_shape = mask.shape
print(mask_shape)

# load 10 patches and print their shape
patches = patches_one_tif[:10]
shapes = []
for patch in patches:
    with rasterio.open(patch) as src:
        shapes.append(src.shape)
print(shapes)