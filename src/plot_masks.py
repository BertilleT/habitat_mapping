import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import time

# PLOT MASKS L1, L2, AND L3
path_l1 = '/media/bertille/My Passport/natural_habitats_mapping/data/full_img_msk/msk/level1/l1_msk_zone111_0_0.tif'
path_l2 = '/media/bertille/My Passport/natural_habitats_mapping/data/full_img_msk/msk/level2/l2_msk_img_zone137_0_0.tif'
path_l3 = '/media/bertille/My Passport/natural_habitats_mapping/data/full_img_msk/msk/level3/l3_msk_img_zone137_0_0.tif'
path_l123 = '/media/bertille/My Passport/natural_habitats_mapping/data/full_img_msk/msk/level123/l123_msk_zone102_0_0.tif'

my_paths = [path_l123]#, path_l2, path_l3]
'''for p in my_paths:
    # get unique values from tif 
    mask = rasterio.open(p)
    mask_array = mask.read(1)
    print(set(mask_array.flatten()))
    

    #time
    start = time.time()
    mask = rasterio.open(p)

    show(mask, cmap='viridis')
    print(time.time() - start)
    #time
    start = time.time()
    mask_array = mask.read(1)
    print(time.time() - start)'''

levels_to_plot = [0, 1, 2]

for p in my_paths:
    # Open the multi-channel image
    mask = rasterio.open(p)
        
    # Plot each channel separately
    for i in levels_to_plot:
        # Read the channel data
        mask_array = mask.read(i + 1)
        #get unique values
        print(set(mask_array.flatten()))
        # Plot the channel
        '''plt.imshow(mask_array, cmap='viridis') 
        plt.title(f'Channel {i + 1}')  
        plt.axis('off')  
        plt.colorbar()  # Add colorbar
        plt.show()  '''

# PLOT MASKS L123

'''path_l123 = '/media/bertille/My Passport/natural_habitats_mapping/data/full_img_msk/msk/level123/msk_zone136_1_1.tif'
#timer
start = time.time()
mask = rasterio.open(path_l123)
print(time.time() - start)
#time
start = time.time()
mask_array = mask.read(1)
print(time.time() - start)
#timer
start = time.time()
mask_array = mask_array // 100000
print(time.time() - start)
# get unique values from mask_array
print(set(mask_array.flatten()))

show(mask_array, cmap='viridis')
plt.show()
# for each pixel value change it to (num // 100000)'''


