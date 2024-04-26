from pathlib import Path
from utils import split_img_msk_into_patches

patch_size = 64
tif_paths = list(Path('../data/full_img_msk/img').glob('*.tif'))
 
for tif_path in tif_paths[30:50]:
    name = tif_path.stem
    mask_path = Path('../data/full_img_msk/msk/level1') / f'l1_msk_{name[4:]}.tif'
    split_img_msk_into_patches(tif_path, mask_path, patch_size)