from pathlib import Path
from utils import *

def percentage_patches_kept(polygons_one_tif, patches_one_tif, threshold):
    '''Return the percentage of patches kept after filtering the patches based on the threshold'''
    # count number of patches with at least one pixel laballed. A patch has one
    unlabelled_patches = 0
    fully_labelled_patches = 0
    incomplete_labelled_patches = 0
    labelled_area_by_patch = {}
    # for each patch
    for patch_path in patches_one_tif:
        # get patch_area
        patch_area = 0
        with rasterio.open(patch_path) as src:
            patch_area = src.width * src.height
            patch_labelled_area = 0
            # for each polygon
            for index, polygon in polygons_one_tif.iterrows():
                geom = polygon['geometry']
                if geom is not None:
                    # get intersection
                    patch_bounds = src.bounds
                    patch_box = box(*patch_bounds)
                    intersection = patch_box.intersection(geom)
                    intersection_percentage = intersection.area / geom.area
                    patch_labelled_area += intersection.area
            # if patch_labelled_area is 0
            if patch_labelled_area == 0:
                unlabelled_patches += 1
            # if patch_labelled_area is equal to patch_area
            elif patch_labelled_area == patch_area:
                fully_labelled_patches += 1
            # if patch_labelled_area is different from 0 and patch_area
            else:
                incomplete_labelled_patches += 1
            labelled_area_by_patch[patch_path] = patch_labelled_area

        all_labelled_patches = fully_labelled_patches + incomplete_labelled_patches
        # in labelled_area_by_patch, keep only the rows with patch_labelled_area > threshold
        threshold_labelled_patches = {k: v for k, v in labelled_area_by_patch.items() if v > threshold}
        # count number of rows in threshold_labelled_patches
        threshold_labelled_patches_count = len(threshold_labelled_patches)
        # percentage of patches kept over all labelled patches according to the threshold
        per_accepted_patches = threshold_labelled_patches_count * 100 / all_labelled_patches

    return unlabelled_patches, fully_labelled_patches, incomplete_labelled_patches, per_accepted_patches
        

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

polygons_one_tif, my_tif_path = load_and_filter_shapefile(shapefile_path, pivot_table_path, tif_path)
patches_one_tif = list(patches_path.glob('*.tif'))