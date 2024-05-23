import rasterio
from shapely.geometry import box
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from rasterio.plot import show
from rasterio import features

def generate_pivot_table_intersect(my_polygons, my_tif_paths, pivot_table_path):
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
    intersect_df.to_csv(pivot_table_path, index=False)
    print(f'The pivot table of the intersection between polygons and tif files saved.')
    return intersect_df

def load_all_shapefile(shp_path):
    shapefile = gpd.read_file(shp_path)
    print('HABNATs.shp loaded.')
    shapefile['index'] = range(len(shapefile))
    return shapefile

def correct_filter_shapefile(shapefile, intersect_df, one_tif_path):
    if one_tif_path is None: 
        # When one_tif_path path is None, we want to retrieve all the polygons id associated to all tif images written in the pivot table
        filtered_shapefile = shapefile[shapefile['index'].isin(intersect_df['polygon_index'])]
    else:
        # When one_tif_path path is not None, we want to retrieve all the polygons id associated to one tif image
        filtered_shapefile = shapefile[shapefile['index'].isin(intersect_df[intersect_df['tif_path_name'] ==  str(one_tif_path)]['polygon_index'])]
    filtered_shapefile.loc[filtered_shapefile['CDEUNIS_1'] == 'G1131', 'CDEUNIS_1'] = 'G1.131'
    # chen CDEUNIS_1 is None, replace its value by the value from CDEUNIS ...
    filtered_shapefile.loc[filtered_shapefile['CDEUNIS_1'].isnull(), 'CDEUNIS_1'] = filtered_shapefile.loc[filtered_shapefile['CDEUNIS_1'].isnull(), 'CDEUNIS']
    # Some polygons have no annotations. Remove them. 
    filtered_shapefile = filtered_shapefile[filtered_shapefile['CDEUNIS_1'].notnull()]
    filtered_tif_paths = intersect_df['tif_path_name'].unique()
    return filtered_shapefile, filtered_tif_paths

def rasterize_multichannel(polygons, template_raster, output_raster):
    template_ds = rasterio.open(template_raster)
    template_meta = template_ds.meta.copy()
    template_meta.update(count=3, dtype='uint8')

    with rasterio.open(output_raster, 'w+', **template_meta) as out:
        # Initialiser un tableau pour les trois canaux
        out_arr = np.zeros((3, out.height, out.width), dtype='uint8')
        # Créer des formes pour chaque niveau de classe
        shapes_l1 = ((geom, value) for geom, value in zip(polygons.geometry, polygons.iloc[:,1]))
        shapes_l2 = ((geom, value) for geom, value in zip(polygons.geometry, polygons.iloc[:,2]))
        shapes_l3 = ((geom, value) for geom, value in zip(polygons.geometry, polygons.iloc[:,3]))
        # Rasteriser chaque niveau dans le canal correspondant
        out_arr[0] = features.rasterize(shapes=shapes_l1, fill = 0, out=out_arr[0], transform=out.transform, dtype='uint8')
        out_arr[1] = features.rasterize(shapes=shapes_l2, fill = 0, out=out_arr[1], transform=out.transform, dtype='uint8')
        out_arr[2] = features.rasterize(shapes=shapes_l3, fill = 0, out=out_arr[2], transform=out.transform, dtype='uint8')
        print('Values available in channel 1:')
        print(np.unique(out_arr[0]))
        print('Values available in channel 2:')
        print(np.unique(out_arr[1]))
        print('Values available in channel 3:')
        print(np.unique(out_arr[2]))
        print('---------CHANGE from 0 to 255 and -1')
        # replace values of 0 by 255
        out_arr[out_arr == 0] = 255
        # less one to all values
        out_arr = out_arr - 1
        #print values in channel 1
        print('Values available in channel 1:')
        print(np.unique(out_arr[0]))
        print('Values available in channel 2:')
        print(np.unique(out_arr[1]))
        print('Values available in channel 3:')
        print(np.unique(out_arr[2]))
        out.write(out_arr[0], 1)
        out.write(out_arr[1], 2)
        out.write(out_arr[2], 3)

def split_img_msk_into_patches(tif_path, mask_path, patch_size):
    # Function inspired from the script available at 
    # https://github.com/bnsreenu/python_for_microscopists/blob/master/Tips_Tricks_5_extracting_patches_from_large_images_and_masks_for_semantic_segm.py
    
    patch_img_dir = f'../../data/patch{patch_size}/img/{str(tif_path).split("_")[3]}_{str(tif_path).split("_")[4]}_{str(tif_path).split("_")[5]}'
    patch_msk_dir = f'../../data/patch{patch_size}/msk/{str(tif_path).split("_")[3]}_{str(tif_path).split("_")[4]}_{str(tif_path).split("_")[5]}'
    patch_img_dir = Path(patch_img_dir[:-4])
    patch_msk_dir = Path(patch_msk_dir[:-4])
    if patch_img_dir.exists() and patch_msk_dir.exists():
        print(f'Patches for {tif_path} already created.')
        return None
    patch_img_dir.mkdir(parents=True, exist_ok=True)
    patch_msk_dir.mkdir(parents=True, exist_ok=True)

    img = tiff.imread(tif_path)
    msk = tiff.imread(mask_path)

    patches_mask = patchify(msk, (patch_size, patch_size, 3), step=patch_size)  
    indexes_do_not_save_empty_patches = []
    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            mask_path = patch_msk_dir / f'msk_{tif_path.stem[4:]}_patch_{i}_{j}.tif'
            if mask_path.exists():
                continue
            single_patch_mask = patches_mask[i,j,:,:]            
            if 255 in single_patch_mask[0, :, :, 0]:
                #print(f'Mask {mask_path.name} is not fully labelled.')
                indexes_do_not_save_empty_patches.append((i, j))
                continue
            tiff.imwrite(mask_path, single_patch_mask)
            #print(f'Mask {mask_path.name} created.')

    print(f'Masks created for {mask_path}.')

    patches_img = patchify(img, (patch_size, patch_size, 4), step=patch_size)  
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            patch_path = patch_img_dir / f'{tif_path.stem}_patch_{i}_{j}.tif'
            if patch_path.exists():
                continue
            if (i, j) in indexes_do_not_save_empty_patches:
                #print(f'Patch {patch_path.name} is empty.')
                continue
            single_patch_img = patches_img[i,j,:,:]
            tiff.imwrite(patch_path, single_patch_img)
            #print(f'Patch {patch_path.name} created.')
    print(f'Patches created for {tif_path}.')
    print('Uncompletely labelled masks and their corresponding images were not saved, there are ', len(indexes_do_not_save_empty_patches), ' of them, over ', patches_mask.shape[0] * patches_mask.shape[1], ' patches.')
    return None

def rescale_band(band, p_min=2, p_max=98):
    p2, p98 = np.percentile(band, (p_min, p_max))
    return np.clip((band - p2) / (p98 - p2), 0, 1)

def plot_tif_image(tif_path): 
    # Open the TIF image
    with rasterio.open(tif_path) as src:
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)
        red_rescaled = rescale_band(red)
        green_rescaled = rescale_band(green)
        blue_rescaled = rescale_band(blue)
        rgb_image = np.dstack((red_rescaled, green_rescaled, blue_rescaled))

        # Get image bounds
        tif_bounds = src.bounds
        # Plot the image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb_image, extent=[tif_bounds.left, tif_bounds.right, tif_bounds.bottom, tif_bounds.top])
    plt.show()

def plot_img_msk(img, msk):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    red = img[0]
    green = img[1]
    blue = img[2]
    red_rescaled = rescale_band(red)
    green_rescaled = rescale_band(green)
    blue_rescaled = rescale_band(blue)
    rgb_image = np.dstack((red_rescaled, green_rescaled, blue_rescaled))

    # Plot the image
    axes[0].imshow(rgb_image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    # Plot the mask
    axes[1].imshow(msk, cmap='viridis')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #path_tif_to_plot = '/media/bertille/My Passport/natural_habitats_mapping/data/data_1/zone10/patch_64_img_1_0/230617_Ecomed_15cm_L93_4canaux_zone10_1_0_patch_0_24.tif'
    #path_tif_to_plot = '/media/bertille/My Passport/natural_habitats_mapping/data/data_1/zone26/patch_64/230617_Ecomed_15cm_L93_4canaux_zone26_0_0_patch_0_64.tif'
    path_mask_to_plot = '/media/bertille/My Passport/natural_habitats_mapping/data/data_1/zone26/mask_230617_Ecomed_15cm_L93_4canaux_zone26_0_0.tif'
    #plot_tif_image(path_tif_to_plot)
    mask = rasterio.open(path_mask_to_plot)
    mask_shape = mask.shape
    print(mask_shape)
    show(mask, cmap='viridis')
    dummy_plot = plt.imshow(mask.read(1), cmap='viridis')
    plt.colorbar(dummy_plot, label='Class')
    plt.show()