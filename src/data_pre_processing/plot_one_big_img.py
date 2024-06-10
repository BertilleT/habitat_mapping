#plot one single image. 

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

'''img_path = '../../data/full_img_msk/img/img_zone1_0_0.tif'
with rasterio.open(img_path) as src:
    img = src.read()
    if self.channels == 3:
        img = img[:3]
    # turn to values betw 0 and 1 and to float
    # linear normalisation with p2 and p98
    #p2, p98 = np.percentile(img, (2, 98))
    #img = np.clip(img, p2, p98)
    #img = (img - p2) / (p98 - p2)
    #img = img.astype(np.float32)
    # Normalize each channel separately
    normalized_img = np.zeros_like(img, dtype=np.float32)
    for c in range(self.channels):
        channel = img[c, :, :]
        p2, p98 = np.percentile(channel, (2, 98))
        channel = np.clip(channel, p2, p98)
        channel = (channel - p2) / (p98 - p2)
        normalized_img[c, :, :] = channel.astype(np.float32)
    img = normalized_img
    img = img.transpose(1, 2, 0)
    plt.imshow(img)
    #save
    plt.savefig('one_full_img.png')'''

my_colors_map =  {
    0: '#789262',  # Vert olive
    1: '#555555',  # Gris
    2: '#006400',  # Vert foncé
    3: '#00ff00',  # Vert vif
    4: '#ff4500',  # Rouge
    5: '#8a2be2',  # Violet
    6: '#ffffff'  # Blanc
}

l1_dict_path = '../../csv/l1_dict.csv'
msk_path = '../../data/full_img_msk/msk/msk_zone1_0_0.tif'
#load l1_dict, keep columns int et int_grouped
group_under_represented_classes = {0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 2, 6: 5, 7: 3, 8: 4, 9: 5, 255: 6}
group_under_represented_classes_uint8 = {np.uint8(k): np.uint8(v) for k, v in group_under_represented_classes.items()}

#load msk
with rasterio.open(msk_path) as src:
    msk = src.read()
    msk = msk[0]
    print(msk)
    print(np.unique(msk))
    msk_mapped = np.vectorize(group_under_represented_classes_uint8.get)(msk)
    print(msk_mapped)
    classes_msk = np.unique(msk_mapped)
    print(classes_msk)
    legend_colors_msk = [my_colors_map[c] for c in classes_msk]
    print(legend_colors_msk)
    custom_cmap_msk = mcolors.ListedColormap(legend_colors_msk)
    plt.imshow(msk_mapped, cmap=custom_cmap_msk)
    plt.colorbar()
    #savefig
    plt.savefig('one_full_msk.png')