# open l2_dict.csv
from pathlib import Path
import pandas as pd

l1_path = Path('../../csv/l1_dict.csv')
l2_path = Path('../../csv/l2_dict.csv')
l1_dict = pd.read_csv(l1_path)
l2_dict = pd.read_csv(l2_path)
#geet columns names of l1   

#keep first letter of code
l2_dict['code'] = l2_dict['code'].str[0]
#add a column to l2_dict with the value of int_grouped from l1_dict wheeen same code
l2_dict['l1_int'] = l2_dict['code'].map(l1_dict.set_index('code')['int_grouped'])

#set nan to 254
l2_dict['l1_int'] = l2_dict['l1_int'].fillna(254)
# to int
l2_dict['l1_int'] = l2_dict['l1_int'].astype(int)
print(l1_dict)
print(l2_dict)

#save l2_dict to csv
l2_dict.to_csv('../../csv/l2_map_l1.csv', index=False)