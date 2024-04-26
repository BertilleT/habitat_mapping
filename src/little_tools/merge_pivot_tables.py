from pathlib import Path
import pandas as pd

old_pivot_table = Path('../../csv/intersection99/intersection_shp_tif_0.99.csv')
new_pivot_table = Path('../../csv/tif_labelled/pivot_table.csv')
merged_pivot_table = Path('../../csv/tif_labelled/final_pivot_table.csv')
# keep only polygon_index and tif_path_name in both pivot tables
old_df = pd.read_csv(old_pivot_table)
old_df = old_df[['polygon_index', 'tif_path_name']]
new_df = pd.read_csv(new_pivot_table)
new_df = new_df[['polygon_index', 'tif_path_name']]

# concat the two pivot tables
pivot_table = pd.concat([old_df, new_df], ignore_index=True)

# save it to the new pivot table
pivot_table.to_csv(merged_pivot_table, index=False)