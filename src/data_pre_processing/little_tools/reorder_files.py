from pathlib import Path
import pandas as pd
import shutil

threshold = 0.99
pivot_table_path = Path(f'../csv/intersection_shp_tif_{threshold}.csv')
#unique the images_paths from tif_path columns in the pivot table
intersect_df = pd.read_csv(pivot_table_path)
filtered_tif_paths = intersect_df['tif_path'].unique()

#create a folder in data called data
data_dir = Path('../new_data/')
data_dir.mkdir(exist_ok=True)
#create a directory called full_img in the data folder
full_img_dir = data_dir / 'full_img_msk'
full_img_dir.mkdir(exist_ok=True)
#create a directory called img in the full_img_msk folder
img_dir = full_img_dir / 'img'
img_dir.mkdir(exist_ok=True)

print(filtered_tif_paths)
# move all the tif files referenced in the pivot table to full_img
for tif_path in filtered_tif_paths:
    tif_path = Path(tif_path)  
    destination_path = img_dir / tif_path.name
    # copy past the tif image to the img_dir, witout removing it from original location
    #shutil.copy(tif_path, destination_path)
    if tif_path.exists():
        tif_path.rename(destination_path)

# check if the files are moved
print(list(img_dir.rglob('*.tif')))

for tif_path in list(img_dir.rglob('*.tif')):
    if 'img' in tif_path.name:
        continue
    #split the name of the file by "_"
    split_name = tif_path.stem.split('_')
    #get the last three elements of the split name
    new_name = '_'.join(split_name[-3:])
    #rename the file
    tif_path.rename(img_dir / f'img_{new_name}.tif')

#check if the files are renamed
print(list(img_dir.rglob('*.tif')))