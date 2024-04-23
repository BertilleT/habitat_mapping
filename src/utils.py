import rasterio
from shapely.geometry import box
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def generate_pivot_table_intersect(my_polygons, my_tif_paths, pivot_table_path, threshold):
    # This funcrion aims at creating the pivot table to link the shapefiles and the tif images based on a minimum intersection threshold
    intersect = []
    len_polygon_kept = 0
    for tif_path in my_tif_paths:
        with rasterio.open(tif_path) as src:
            tif_bounds = src.bounds
            tif_box = box(*tif_bounds)  # Convert BoundingBox to shapely box

        # Check each polygon
        for index, polygon in my_polygons.iterrows():
            geom = polygon['geometry']
            if geom is not None and tif_box.intersects(geom):
                intersection = tif_box.intersection(geom)
                intersection_percentage = intersection.area / geom.area

                # Check if the intersection is greater than threshold
                if intersection_percentage >= threshold:
                    #print(f'The polygon with index {polygon["index"]} intersects significantly with {tif_path.name}')
                    intersect.append({
                        'tif_path': tif_path,
                        'polygon_index': polygon['index'],
                        'intersection_percentage': intersection_percentage
                    })
                    len_polygon_kept += 1

    intersect_df = pd.DataFrame(intersect)
    intersect_df.to_csv(pivot_table_path, index=False)

    return intersect_df, len_polygon_kept * 100 / len(my_polygons)

def add_date_column(shapefile, intersect_df, excel_date_path):
    shapefile_zone_df = intersect_df.copy()
    shapefile_zone_df['zone'] = shapefile_zone_df['tif_path'].apply(lambda x: x.split('/')[-2])
    shapefile_zone_df = shapefile_zone_df[['polygon_index', 'zone']]
    shapefile = pd.merge(shapefile, shapefile_zone_df, left_on='index', right_on='polygon_index', how='left')
    shapefile.drop(columns=['polygon_index'], inplace=True)
    dates_df = pd.read_excel(excel_date_path)
    dates_df = dates_df[['zone_AJ', 'date_habna']].drop_duplicates(subset='zone_AJ')
    shapefile = pd.merge(shapefile, dates_df, left_on='zone', right_on='zone_AJ', how='left')
    shapefile.drop(columns=['zone_AJ'], inplace=True)
    return shapefile

def split_image_into_patches(tif_path, patch_size):
    print(f'Creating patches for {tif_path}.')
    tif_path = Path(tif_path)
    parent_dir = tif_path.parent
    patches_dir = parent_dir / f'patch_{patch_size}'
    patches_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1) # Move the channel axis to the last axis because rasterio reads the image in the form (bands, height, width)
        img_height, img_width, _ = img.shape
        for i in range(0, img_height, patch_size):
            for j in range(0, img_width, patch_size):
                patch = img[i:i+patch_size, j:j+patch_size]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    patch_path = patches_dir / f'{tif_path.stem}_patch_{i}_{j}.tif'
                    with rasterio.open(patch_path, 'w', **src.profile) as dst:
                        dst.write(patch.transpose(2, 0, 1))
                        print(f'Patch {patch_path.name} created.')
            print(f'{i} rows processed.')
    print(f'Patches created for {tif_path}.')
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

# if this file is main

if __name__ == '__main__':
    # plot /media/bertille/My Passport/natural_habitats_mapping/data/data_1/zone26/patch_64/230617_Ecomed_15cm_L93_4canaux_zone26_0_0_patch_0_64.tif
    plot_tif_image('/media/bertille/My Passport/natural_habitats_mapping/data/data_1/zone26/patch_64/230617_Ecomed_15cm_L93_4canaux_zone26_0_0_patch_0_64.tif')