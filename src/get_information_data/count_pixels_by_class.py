#to_del

#loop on all masks from /media/bertille/My Passport/natural_habitats_mapping/data/full_img_msk/msk
#print possible values in each class and count
#mean

import os
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

#path to masks
path = "/media/bertille/My Passport/natural_habitats_mapping/data/full_img_msk/msk"

my_masks = os.listdir(path)
#remove /media/bertille/My Passport/natural_habitats_mapping/data/full_img_msk/msk/level123_old
my_masks.remove('level123_old')
i = 0
#change all 254 values to 255 in all masks channels
for mask in my_masks:
    i = i + 1
    print(i, '/', len(my_masks))
    if mask == "level123_old":
        continue
    with rasterio.open(os.path.join(path, mask)) as src:
        data = src.read()
        data[data == 254] = 255
        profile = src.profile
        with rasterio.open(os.path.join(path, mask), 'w', **profile) as dst:
            dst.write(data)

#loop on all masks
classes_l1 = {i:0 for i in range(1, 10)}
classes_l1[254] = 0
classes_l1[255] = 0
classes_l2 = {i:0 for i in range(1, 37)}
classes_l2[254] = 0
classes_l2[255] = 0
classes_l3 = {i:0 for i in range(1, 128)}
classes_l3[254] = 0
classes_l3[255] = 0

for mask in my_masks[:10]:
    if mask == "level123_old":
        continue
    #print possible values in each class and count for each channels, there are 3
    with rasterio.open(os.path.join(path, mask)) as src:
        data = src.read()
        print(np.unique(data[0], return_counts=True))
        print(np.unique(data[1], return_counts=True))
        print(np.unique(data[2], return_counts=True))
        #add to classes_l1, classes_l2, classes_l3
        for i in range(1, 10):
            classes_l1[i] += np.sum(data[0] == i)
        for i in range(1, 37):
            classes_l2[i] += np.sum(data[1] == i)
        for i in range(1, 128):
            classes_l3[i] += np.sum(data[2] == i)
        classes_l1[254] += np.sum(data[0] == 254)
        classes_l1[255] += np.sum(data[0] == 255)
        classes_l2[254] += np.sum(data[1] == 254)
        classes_l2[255] += np.sum(data[1] == 255)
        classes_l3[254] += np.sum(data[2] == 254)
        classes_l3[255] += np.sum(data[2] == 255)

# remove key values with 0 as value
classes_l1 = {k:v for k,v in classes_l1.items() if v != 0}
classes_l2 = {k:v for k,v in classes_l2.items() if v != 0}
classes_l3 = {k:v for k,v in classes_l3.items() if v != 0}

# transform to percentage
total = sum(classes_l1.values())
classes_l1 = {k:round(v*100/total, 2) for k,v in classes_l1.items()}
total = sum(classes_l2.values())
classes_l2 = {k:round(v*100/total, 2) for k,v in classes_l2.items()}
total = sum(classes_l3.values())
classes_l3 = {k:round(v*100/total, 2) for k,v in classes_l3.items()}
print(classes_l1)
print(classes_l2)
print(classes_l3)