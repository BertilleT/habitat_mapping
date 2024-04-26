from pathlib import Path
import subprocess
import pandas as pd

'''This file is dedicated to unzipping the folders zone1.zip, zone2.zip, zone3.zip, etc., which are in data_dir and to save list of corrupted zip files into a csv file'''

data_dir = '../original_data/'

corrupted_zip_list = []
try:
    zip_files = list(Path(data_dir).rglob('*.zip'))
except Exception as e:
    print(f"Failed to list zip files: {e}")
    zip_files = []

for zip_file in zip_files:
    zip_file = str(zip_file)  # Ensure the path is a string
    output_folder = Path(data_dir) / Path(zip_file).stem  # Assuming extracted to a folder named like the ZIP file minus '.zip'

    # Check if the folder already exists and has content
    if output_folder.exists() and any(output_folder.iterdir()):
        print(f'Skipping {zip_file}, already unzipped.')
        continue

    print(f'Unzipping {zip_file}')
    try:
        # Use subprocess to run the unzip command
        subprocess.run(['unzip', '-q', '-o', zip_file, '-d', data_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f'Failed to unzip {zip_file}: {e}')
        corrupted_zip_list.append(zip_file)
        continue  # This will skip to the next file in the loop

print(f'Corrupted ZIP files: {corrupted_zip_list}')
# save the corrupted zip files into csv
corrupted_zip_df = pd.DataFrame(corrupted_zip_list, columns=['corrupted_zip_files'])
corrupted_zip_df.to_csv('../csv/corrupted_zip_files.csv', index=False)