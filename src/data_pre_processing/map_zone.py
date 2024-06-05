#list all tif images path in the folder given in path
import os
from pathlib import Path
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import folium
import pandas as pd

strat = 'zone'
path_imgs_id = '../../unet_256_l1/stratified_shuffling_zone_mediteranean/seed2/img_ids_by_set.csv'
path_to_map = '../../train_val_test_maps/stratified_by_zone_mediteranean_seed2.html'

df = pd.read_csv(path_imgs_id)
train_img_ids = df.loc[df['set'] == 'train_img_ids', 'img_ids'].values[0]
val_img_ids = df.loc[df['set'] == 'val_img_ids', 'img_ids'].values[0]
test_img_ids = df.loc[df['set'] == 'test_img_ids', 'img_ids'].values[0]

#split at , and turn to list
train_img_ids = train_img_ids.split(',')
val_img_ids = val_img_ids.split(',')
test_img_ids = test_img_ids.split(',')
# st all elements as a string
print(train_img_ids)
print(val_img_ids)
print(test_img_ids)
path = '../../data/full_img_msk/img'
tif_files = list(Path(path).rglob('*.tif'))
zone_dict = {}

# EMPTY ZONE IMG IN DATA

empty_zones = ['zone102_1_1', 'zone103_0_0', 'zone103_0_1', 'zone114_0_0', 'zone118_0_0', 'zone120_0_1', 'zone120_1_1', 'zone121_0_0', 'zone121_0_1', 'zone121_1_1', 'zone124_0_0', 'zone125_0_0', 'zone128_0_0', 'zone12_2_3', 'zone12_3_4', 'zone12_4_3', 'zone136_0_0', 'zone138_0_0', 'zone139_0_0', 'zone139_0_2', 'zone140_0_0', 'zone140_1_0', 'zone141_0_0', 'zone141_0_1', 'zone141_1_1', 'zone142_0_1', 'zone144_0_0', 'zone145_0_1', 'zone147_0_0', 'zone147_1_0', 'zone147_1_1', 'zone147_1_2', 'zone148_0_0', 'zone14_2_0', 'zone153_0_0', 'zone153_1_0', 'zone156_0_1', 'zone15_0_0', 'zone15_0_1', 'zone15_1_1', 'zone15_1_2', 'zone162_0_2', 'zone165_0_0', 'zone168_0_0', 'zone168_0_1', 'zone169_0_0', 'zone16_2_0', 'zone170_0_5', 'zone170_1_4', 'zone170_2_0', 'zone170_2_4', 'zone170_2_6', 'zone170_3_0', 'zone170_3_1', 'zone170_3_2', 'zone170_3_3', 'zone170_3_5', 'zone170_3_6', 'zone170_3_8', 'zone171_0_0', 'zone171_1_2', 'zone171_1_3', 'zone17_0_3', 'zone17_1_0', 'zone17_1_1', 'zone22_0_1', 'zone22_1_0', 'zone22_1_1', 'zone23_0_0', 'zone23_0_1', 'zone25_0_2', 'zone25_1_2', 'zone27_1_0', 'zone30_1_0', 'zone37_1_0', 'zone39_0_1', 'zone3_0_1', 'zone41_0_0', 'zone41_0_1', 'zone44_2_2', 'zone44_3_2', 'zone44_4_2', 'zone44_4_4', 'zone47_1_1', 'zone47_1_2', 'zone48_1_0', 'zone50_1_0', 'zone50_1_1', 'zone52_0_0', 'zone52_0_1', 'zone54_0_0', 'zone54_1_1', 'zone56_0_1', 'zone58_0_0', 'zone58_1_1', 'zone59_0_1', 'zone5_1_1', 'zone63_1_0', 'zone63_1_1', 'zone65_0_0', 'zone65_1_1', 'zone65_2_0', 'zone66_1_0', 'zone66_1_1', 'zone66_1_2', 'zone66_2_2', 'zone70_0_0', 'zone74_0_1', 'zone74_1_0', 'zone74_2_2', 'zone75_1_2', 'zone75_1_4', 'zone75_2_3', 'zone76_0_0', 'zone76_0_2', 'zone78_0_0', 'zone78_10_7', 'zone78_10_8', 'zone78_11_8', 'zone78_12_8', 'zone78_12_9', 'zone78_13_9', 'zone78_14_10', 'zone78_14_9', 'zone78_16_10', 'zone78_1_0', 'zone78_1_1', 'zone78_2_1', 'zone78_2_2', 'zone78_3_2', 'zone78_3_3', 'zone78_4_4', 'zone78_5_5', 'zone78_6_5', 'zone78_6_6', 'zone78_7_6', 'zone78_8_6', 'zone78_8_7', 'zone80_1_0', 'zone93_0_0', 'zone94_0_0', 'zone94_0_1', 'zone94_1_1', 'zone94_2_1', 'zone94_2_2', 'zone94_3_2', 'zone94_3_3', 'zone94_4_2', 'zone94_4_3', 'zone96_1_0', 'zone96_2_1', 'zone97_1_1']

# print metadata of first tif image
for tif_file in tif_files:
    #get the zoneid by selecting 5 and 6th element of split
    zoneid_father = os.path.basename(tif_file).split('_')[1]
    #if strat == 'zone':
    #    zoneid = os.path.basename(tif_file).split('_')[1]
    #else:
    zoneid = os.path.basename(tif_file).split('_')[1] + '_' + os.path.basename(tif_file).split('_')[2] + '_' + os.path.basename(tif_file).split('_')[3]
        #remove the .tif extension
        
    zoneid = zoneid[:-4]
    with rasterio.open(tif_file) as src:
        #get the centroid
        centroid = src.transform * (src.width//2, src.height//2)

        zone_dict[zoneid] = centroid
# check zoneid is in train, val or test, if not rmeove it from zone_dict
for zoneid in list(zone_dict.keys()):
    print(zoneid)
    if strat == 'zone':
        zoneid_father = zoneid.split('_')[0]
    else:
        zoneid_father = zoneid
    print(zoneid_father)
    if zoneid_father not in train_img_ids and zoneid_father not in val_img_ids and zoneid_father not in test_img_ids: 
        print(zoneid, 'not in train, val or test')
        zone_dict.pop(zoneid)
    if zoneid in empty_zones:
        print(zoneid, 'is empty')
        if zoneid in zone_dict:
            zone_dict.pop(zoneid)
print(zone_dict)
print(len(zone_dict))
# project L93: EPSG:2154
centroid_list = [(zone_id, centroid) for zone_id, centroid in zone_dict.items()]
centroid_gdf = gpd.GeoDataFrame(centroid_list, columns=['zone_id', 'centroid'])
centroid_gdf['geometry'] = [Point(xy) for xy in centroid_gdf['centroid']]
centroid_gdf.crs = 'EPSG:2154'
# every point to lat long
centroid_gdf = centroid_gdf.to_crs('EPSG:4326')
# drop the centroid column
centroid_gdf.drop(columns=['centroid'], inplace=True)
# add a aoloumn color to centroid_gdf, one color if zone_id is in train, val or test
centroid_gdf['color'] = 'black'
for zone_id in train_img_ids:
    centroid_gdf.loc[centroid_gdf['zone_id'].str.contains(zone_id), 'color'] = 'red'
for zone_id in val_img_ids:
    centroid_gdf.loc[centroid_gdf['zone_id'].str.contains(zone_id), 'color'] = 'blue'
for zone_id in test_img_ids:
    centroid_gdf.loc[centroid_gdf['zone_id'].str.contains(zone_id), 'color'] = 'green'

m = folium.Map(location=[43.6, 3], zoom_start=7)

# Add CircleMarker for each point in centroid_gdf with popup
for i, row in centroid_gdf.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        color=row['color'],
        fill=True,
        fill_color=row['color'],
        fill_opacity=1
    ).add_child(folium.Popup(row['zone_id'])).add_to(m)
    # Add the popup to the CircleMarker
    folium.Popup(row['zone_id']).add_to(folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=0))
# Create the legend
legend_html = '''
<div style="position: fixed;
     bottom: 50px; left: 50px; width: 150px; height: 90px;
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: 0.8;">
     &nbsp; <b>Legend</b> <br>
     &nbsp; Train &nbsp; <i class="fa fa-circle fa-1x" style="color:red"></i><br>
     &nbsp; Validation &nbsp; <i class="fa fa-circle fa-1x" style="color:blue"></i><br>
     &nbsp; Test &nbsp; <i class="fa fa-circle fa-1x" style="color:green"></i>
</div>
'''

# Add the legend to the map
m.get_root().html.add_child(folium.Element(legend_html))
# Save the map to an HTML file
m.save(path_to_map)