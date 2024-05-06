# list all folder names in /media/bertille/My Passport/natural_habitats_mapping/data/patch256/img
from pathlib import Path
import pandas as pd

# LOAD THE FOLDER NAMES OF THE PAIRS (IMGS, MSKS), WHICH CONTAINS THE ZONEID
img_path = Path('../data/patch256/img')
msk_path = Path('../data/patch256/msk/l123')
img_folders = [x for x in img_path.iterdir() if x.is_dir()]
#img_folders = [str(x).split('/')[-1] for x in img_folders]
img_folders = [str(x).split('_')[0] for x in img_folders]
img_folders = list(set(img_folders))

# LOAD THE DATES AND ITS ZONEID ASSOCIATED, KEEP ONLY VALID DATES (FROM 2021 TO 2024) AND ZONEID ASSOCIATED
dates_df = pd.read_excel('../data/shp/BDD_AJ_HABNAT_FINALE2.xlsx')
dates_df = dates_df[['zone_AJ', 'date_habna']].drop_duplicates(subset='zone_AJ')
dates_df = dates_df[dates_df['date_habna'].notna()]
dates_df['date_habna'] = pd.to_datetime(dates_df['date_habna'])
dates_df = dates_df[dates_df['date_habna'].dt.year.isin([2021, 2022, 2023, 2024])]
zone_valid_dates = dates_df['zone_AJ'].values.tolist()

# DROP THE FOLDERS OF IMGS AND MSKS WHICH HAVE ZONEID WITH INVALID DATES
'''for folder in img_folders:
    if folder not in zone_valid_dates:
        img_folders.remove(folder) # remove the folder name from the list
print(len(img_folders))'''

for path_folder in img_folders: 
    valid = False
    for valid_zone in zone_valid_dates:
        #if valid_zone string is in string path_folder
        if valid_zone in path_folder:
            valid = True
            break
    if not valid:
        #empty the folder and drop it
        print('not valid')
        print(path_folder)
        '''img_folder = img_path / path_folder
        msk_folder = msk_path / path_folder
        for file in img_folder.iterdir():
            file.unlink()
        for file in msk_folder.iterdir():
            file.unlink()
        img_folder.rmdir()
        msk_folder.rmdir()'''

#print(len([x for x in img_path.iterdir() if x.is_dir()]))