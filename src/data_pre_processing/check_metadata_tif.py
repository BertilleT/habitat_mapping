# load one tif image from path = "/media/bertille/My Passport/natural_habitats_mapping/data/full_img_msk/img"
# print metadata from the tif image with rasterio

import rasterio
import os

path = "../../data/full_img_msk/img/img_zone12_1_1.tif"
my_images = os.listdir(path)