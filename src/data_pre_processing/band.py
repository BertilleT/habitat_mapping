#load img_zone1_0_0_patch_2_12.tif and print metadata about channels rasterio
import rasterio

path = "../../data/patch256/img/zone1_0_0/img_zone1_0_0_patch_2_12.tif"
with rasterio.open(path) as src:
    # get metadata about channels
    print(src.profile)