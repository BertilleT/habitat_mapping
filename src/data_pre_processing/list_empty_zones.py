# list all subfolders whic are not empty in the img_folder

from pathlib import Path

path = Path('../../data/patch256/img/')
subfolders = [x for x in path.iterdir() if x.is_dir()]
empty_folders = [x for x in subfolders if not list(x.iterdir())] 
# get the last elment of the path, split 8 and keep the first
empty_zones = [str(x).split('/')[-1] for x in empty_folders]
#order 
empty_zones.sort()
print(empty_zones)