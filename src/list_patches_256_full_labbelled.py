from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd

patch_size = 256

masks_path = Path(f'../data/patch{patch_size}/msk/l123')
csv1_name = f'../csv/coverage_patch/class_count_l1_{patch_size}.csv'

#Load the csv 
l1_dict = pd.read_csv(csv1_name)
print(len(l1_dict))
# select rows where column 0 is nan or 0
nan_rows = l1_dict[l1_dict['0'].isnull() | (l1_dict['0'] == 0)]
print(len(nan_rows))

#save csv with nan rows
nan_rows.to_csv(f'../csv/coverage_patch/l1_{patch_size}_100per_labelled.csv', index=False)