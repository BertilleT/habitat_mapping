import pandas as pd
import matplotlib.pyplot as plt

# Load CSV 1
data_csv1 = pd.read_csv('../../results/resnet18_64_l1/random_shuffling/all/resnet18_multi_label_64_random_60epochs_bs4096/metrics_test/F1s.csv')

# Load CSV 2
data_csv2 = pd.read_csv('../../results/resnet18_64_l2/random_shuffling/all/resnet18_multi_label_64_random_60epochs_bs4096_level2/metrics_test/F1s.csv')

level1_to_level2 = {
    0: [0, 1, 2],
    1: [3, 4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12],
    4: [13, 14, 15],
    5: [16],
    6: [17]
}

print(data_csv1)
print(data_csv2)

#remove mean
data_csv1 = data_csv1[data_csv1['class'] != 'mean']
data_csv2 = data_csv2[data_csv2['class'] != 'mean']
# add a column to data_csv2 with l2 classees based on mapping above
#rename F1 in l2_F1
data_csv2.rename(columns={'F1': 'l2_F1'}, inplace=True)

data_csv2['l1_F1'] = data_csv2['class'].apply(lambda x: next(iter([k for k, v in level1_to_level2.items() if int(x) in v]), None))
print(data_csv2)
# replace l1_F1 with values from data_csv1
data_csv2['l1_F1'] = data_csv2['l1_F1'].apply(lambda x: data_csv1[data_csv1['class'] == str(x)]['F1'].values[0])

print(data_csv2)

# plot add legend and dif color for both curve
plt.plot(data_csv2['class'], data_csv2['l1_F1'], label='level1', color='blue', linestyle='dashed')
plt.plot(data_csv2['class'], data_csv2['l2_F1'], label='level2', color='red', linestyle='dashed')
# x axis legend classes
plt.xlabel('Classes')
# y axis legend F1
plt.ylabel('F1')
# quadrillage
plt.grid()
plt.legend()
#save plot
plt.savefig('l1_vs_l2_F1.png')