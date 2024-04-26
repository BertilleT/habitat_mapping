# check for corrupted tif images 
import rasterio
import pandas as pd
from pathlib import Path

data_dir = Path('../../data/full_img_msk/img')
my_tif_paths = list(data_dir.rglob('*.tif'))
corrupted_tif_list = []
for tif_file in my_tif_paths:
    try:
        with rasterio.open(tif_file) as src:
            print('hey')
            pass
    except Exception as e:
        #keep only the file name and its parent folder
        tif_file = str(tif_file).split('/')[-2] + '/' + str(tif_file).split('/')[-1]
        print(f"Failed to open {tif_file}: {e}")
        corrupted_tif_list.append(tif_file)
        continue

print(f'Corrupted TIF files: {corrupted_tif_list}')
# save the corrupted tif files into csv
corrupted_tif_df = pd.DataFrame(corrupted_tif_list, columns=['corrupted_tif_images'])
corrupted_tif_df.to_csv('../../csv/corrupted/corrupted_tif_images_26_04.csv', index=False)