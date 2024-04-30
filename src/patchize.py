from pathlib import Path
from utils import split_img_msk_into_patches

patch_size = 128
tif_paths = list(Path('../data/full_img_msk/img').glob('*.tif'))
print(len(tif_paths))  # 100
#list all directries name in Path('../data/patch64/msk/l123')

# Create patch 128 for the same img and msk already created for patch 64
#masks_id_64 = list(Path('../data/patch64/rest/msk/l123').glob('*'))
#masks_id_64 = [mask_id.stem for mask_id in masks_id_64]
#tif_paths = [tif_path for tif_path in tif_paths if '_'.join(tif_path.stem.split('_')[1:]) in masks_id_64]

for tif_path in tif_paths[80:180]:
    name = tif_path.stem
    mask_path = Path('../data/full_img_msk/msk/level123') / f'l123_msk_{name[4:]}.tif'
    split_img_msk_into_patches(tif_path, mask_path, patch_size)