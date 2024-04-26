import pandas as pd


# Load ../../csv/label_map_to_int_dict/old_l1.csv
# Load ../../csv/label_map_to_int_dict/old_l2.csv
# Load ../../csv/label_map_to_int_dict/old_l3.csv
old_l1 = pd.read_csv('../../csv/label_map_to_int_dict/old_l1.csv')
old_l2 = pd.read_csv('../../csv/label_map_to_int_dict/old_l2.csv')
old_l3 = pd.read_csv('../../csv/label_map_to_int_dict/old_l3.csv')
old_classes_l1_int = old_l1['class'].to_list()
old_classes_l2_int = old_l2['class'].to_list()
old_classes_l3_int = old_l3['class'].to_list()

# Load ../../csv/tif_labelled/classes_grouped_3l.csv
new_classes_grouped_3l = pd.read_csv('../../csv/tif_labelled/classes_grouped_3l.csv')
new_classes_l1 = new_classes_grouped_3l['l1'].dropna().unique()
new_classes_l2 = new_classes_grouped_3l['l2'].dropna().unique()
new_classes_l3 = new_classes_grouped_3l['l3'].dropna().unique()

# if there are classes on classes_grouped_3l['l1'] not present into old_l1['class'], add them to old_l1
# get the maximum value of int in old_l1
max_int_l1 = old_l1['int'].max()

for i in new_classes_l1:
    if i not in old_classes_l1_int:
        old_l1.loc[len(old_l1.index)] = [i, max_int_l1+1]
        print(f'{i} added to old_l1 with integer {max_int_l1+1}')
        max_int_l1 += 1

# remove 99 and 999 from old_l2 and old_l3
old_l2_temp = old_l2[old_l2['int'] != 99]
max_int_l2 = old_l2_temp['int'].max()
for i in new_classes_l2:
    if i not in old_classes_l2_int:
        old_l2.loc[len(old_l2.index)] = [i, max_int_l2+1]
        print(f'{i} added to old_l2 with integer {max_int_l2+1}')
        max_int_l2 += 1

old_l3_temp = old_l3[old_l3['int'] != 999]
max_int_l3 = old_l3_temp['int'].max()

for i in new_classes_l3:
    if i not in old_classes_l3_int:
        old_l3.loc[len(old_l3.index)] = [i, max_int_l3+1]
        print(f'{i} added to old_l3 with integer {max_int_l3+1}')
        max_int_l3 += 1