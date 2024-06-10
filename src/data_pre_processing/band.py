#plot one single image. 

import numpy as np
import rasterio
import matplotlib.pyplot as plt

img_path = '../../data/full_img_msk/img/img_zone1_0_0.tif'
with rasterio.open(img_path) as src:
    img = src.read()
    normalized_img = np.zeros_like(img, dtype=np.float32)
    for c in range(4):
        channel = img[c, :, :]
        p2, p98 = np.percentile(channel, (2, 98))
        channel = np.clip(channel, p2, p98)
        channel = (channel - p2) / (p98 - p2)
        normalized_img[c, :, :] = channel.astype(np.float32)
    normalized_img = normalized_img[:3, :, :]
    normalized_img = normalized_img.transpose(1, 2, 0)
    plt.imshow(normalized_img)
    #save
    plt.savefig('one_full_img.png')
