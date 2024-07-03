# load excel from ../../original_data/data_1/HABNAT/BDD_AJ_HABNAT_FINALE2.xlsx
from pathlib import Path
import pandas as pd

path = Path('../../original_data/data_1/HABNAT/BDD_AJ_HABNAT_FINALE2.xlsx')
dates_df = pd.read_excel(path)
print(dates_df.head())

dates_df = dates_df[['zone_AJ', 'date_habna']].drop_duplicates(subset='zone_AJ')
# drop rows with null values
dates_df = dates_df.dropna()

dates_df['year'] = dates_df['date_habna'].apply(lambda x: str(x)[:4] if not pd.isnull(x) else x)
#drop date_hana column

# print all unique values from year 
print(dates_df['year'].unique())
dates_df = dates_df.drop(columns=['date_habna'])
print(dates_df.head(100))
dates_df['year'] = dates_df['year'].fillna(0)
dates_df['year'] = dates_df['year'].apply(lambda x: int(x))

print(dates_df.head())

# get number of eones
print(dates_df['zone_AJ'].nunique())
#keep only when year = 2023
dates_df = dates_df[dates_df['year'] == 2023]
print(dates_df.head())
# get number of zones
print(dates_df['zone_AJ'].nunique())

#store to a list the zonesid with year 2023
dates_df = dates_df['zone_AJ']
#drop null values
dates_df = dates_df.dropna()
#save to csv list of zonesid with year 2023
# dates_df.to_csv('../../csv/zones_2023.csv', index=False)


#load the list
zones_2023 = pd.read_csv('../../csv/zones_2023.csv')
print(zones_2023.head(100))