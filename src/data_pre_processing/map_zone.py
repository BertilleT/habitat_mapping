#list all tif images path in the folder given in path
import os
from pathlib import Path
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import folium
import pandas as pd

#image_id_by_set_path = '../../unet_256_l1/stratified_shuffling_by_image_seed1/img_ids_by_set.csv'
#image_id_by_set_path = '../../csv/zones_stratified_split.csv'

# the csv file looks like this
#,set,img_ids
#0,train_img_ids,"zone84_0_0,zone76_1_2,zone104_0_0,zone14_0_0,zone12_4_2,zone54_0_1,zone24_0_0,zone30_0_1,zone37_0_0,zone17_1_2,zone16_0_0,zone156_1_1,zone25_1_3,zone45_1_2,zone169_0_1,zone68_1_0,zone21_0_0,zone116_0_0,zone147_2_2,zone28_0_0,zone75_0_3,zone47_1_0,zone74_1_2,zone75_1_3,zone39_0_0,zone78_5_4,zone27_0_3,zone34_0_0,zone90_0_0,zone45_0_2,zone6_0_0,zone74_0_2,zone50_0_1,zone25_3_0,zone44_5_3,zone77_1_1,zone158_1_0,zone115_0_0,zone142_1_0,zone155_1_0,zone3_1_0,zone170_1_5,zone5_1_0,zone117_0_0,zone74_1_1,zone165_0_1,zone136_1_1,zone17_0_2,zone77_1_0,zone145_0_0,zone51_0_1,zone121_1_0,zone25_2_1,zone96_1_1,zone10_0_1,zone85_0_0,zone74_2_0,zone4_0_0,zone129_0_1,zone75_2_2,zone15_1_0,zone147_3_4,zone45_0_1,zone133_1_0,zone85_0_1,zone2_0_0,zone19_1_0,zone25_0_3,zone44_5_2,zone120_1_0,zone78_4_3,zone51_1_1,zone139_1_2,zone97_0_1,zone66_2_4,zone25_2_0,zone33_1_0,zone41_1_1,zone68_1_1,zone136_1_2,zone67_0_0,zone63_0_1,zone143_0_1,zone76_1_1,zone133_0_0,zone17_0_1,zone50_0_0,zone127_0_1,zone67_0_1,zone114_1_0,zone12_2_1,zone28_0_1,zone27_0_2,zone80_1_1,zone95_0_1,zone139_1_0,zone78_9_7,zone102_0_1,zone10_1_1,zone132_0_0,zone19_2_0,zone112_0_0,zone139_1_1,zone112_0_1,zone172_0_0,zone74_1_3,zone142_0_0,zone162_1_1,zone144_1_1,zone74_0_3,zone59_1_2,zone147_4_5,zone158_0_0,zone96_0_0,zone102_1_0,zone12_3_3,zone12_1_0,zone45_1_0,zone171_1_1,zone7_0_0,zone154_0_0,zone38_0_0,zone170_3_4,zone3_0_0,zone26_0_0,zone68_0_1,zone22_0_0,zone7_0_1,zone65_1_0,zone77_0_1,zone30_0_0,zone120_0_0,zone148_0_1"
#1,val_img_ids,"zone12_0_1,zone97_0_0,zone63_0_0,zone73_0_0,zone56_0_0,zone73_0_1,zone165_1_1,zone156_1_0,zone17_1_3,zone161_0_0,zone33_1_1,zone162_0_1,zone44_4_3,zone7_1_0,zone27_0_0,zone1_0_0,zone170_0_6,zone12_0_0,zone165_1_0,zone28_1_0,zone14_1_0,zone16_0_1,zone27_0_1,zone66_2_3,zone5_0_0,zone53_1_2,zone12_1_1,zone74_2_1,zone137_0_0,zone129_0_0,zone33_0_0,zone57_0_0,zone51_1_0,zone147_4_4,zone72_0_0,zone41_1_0,zone71_0_0,zone101_0_0,zone76_0_1,zone159_0_0,zone155_0_0,zone147_3_3,zone45_1_1,zone157_0_0,zone98_0_0"
#2,test_img_ids,"zone78_9_8,zone30_1_1,zone59_0_0,zone164_0_0,zone134_0_0,zone106_0_0,zone59_1_0,zone95_0_0,zone90_1_0,zone93_1_0,zone48_0_0,zone28_1_1,zone160_0_0,zone88_0_0,zone77_0_0,zone78_16_11,zone100_0_0,zone5_0_1,zone139_0_1,zone12_2_2,zone17_0_0,zone78_15_10,zone123_0_0,zone144_0_1,zone170_3_7,zone69_0_0,zone19_0_0,zone127_0_0,zone102_0_0,zone12_3_2,zone113_0_0,zone59_1_1,zone167_0_0,zone47_0_0,zone20_0_0,zone145_1_0,zone162_1_2,zone78_17_11,zone143_0_0,zone10_1_0,zone126_0_0,zone11_0_0,zone147_3_5,zone136_1_0,zone47_0_1"

'''df = pd.read_csv(image_id_by_set_path)
train_img_ids = df.loc[df['set'] == 'train_img_ids', 'img_ids'].values[0].split(',')
val_img_ids = df.loc[df['set'] == 'val_img_ids', 'img_ids'].values[0].split(',')
test_img_ids = df.loc[df['set'] == 'test_img_ids', 'img_ids'].values[0].split(',')
'''

train_img_ids = ['zone171', 'zone116', 'zone106', 'zone67', 'zone169', 'zone12', 'zone44',
 'zone73', 'zone161', 'zone11', 'zone84', 'zone85', 'zone100', 'zone63',
 'zone148', 'zone129', 'zone133', 'zone65', 'zone22', 'zone78', 'zone120',
 'zone66', 'zone126', 'zone38', 'zone30', 'zone45', 'zone7', 'zone53', 'zone144',
 'zone59', 'zone76', 'zone156', 'zone115', 'zone97', 'zone165', 'zone3', 'zone21',
 'zone10', 'zone1', 'zone57', 'zone162', 'zone75', 'zone145', 'zone24', 'zone74',
 'zone113', 'zone77', 'zone132', 'zone4', 'zone39', 'zone14', 'zone72', 'zone159',
 'zone37', 'zone48', 'zone90', 'zone121', 'zone34', 'zone104', 'zone147',
 'zone143', 'zone114', 'zone16', 'zone112', 'zone134', 'zone19', 'zone71',
 'zone98', 'zone93', 'zone172', 'zone80']
val_img_ids = ['zone51', 'zone142', 'zone20', 'zone69', 'zone27', 'zone26', 'zone88', 'zone15',
 'zone25', 'zone56', 'zone47', 'zone170', 'zone137', 'zone127', 'zone158',
 'zone136', 'zone41', 'zone164', 'zone101', 'zone96', 'zone2', 'zone102',
 'zone33', 'zone155']
test_img_ids = ['zone167', 'zone54', 'zone154', 'zone68', 'zone28', 'zone139', 'zone50',
 'zone123', 'zone160', 'zone95', 'zone17']

path = '../../data/full_img_msk/img'
tif_files = list(Path(path).rglob('*.tif'))
zone_dict = {}

# print metadata of first tif image
for tif_file in tif_files:
    #get the zoneid by selecting 5 and 6th element of split
    zoneid_father = os.path.basename(tif_file).split('_')[1]
    zoneid = os.path.basename(tif_file).split('_')[1] + '_' + os.path.basename(tif_file).split('_')[2] + '_' + os.path.basename(tif_file).split('_')[3]
    #remove the .tif extension
    zoneid = zoneid[:-4]
    with rasterio.open(tif_file) as src:
        #get the centroid
        centroid = src.transform * (src.width//2, src.height//2)
        zone_dict[zoneid] = centroid

# check zoneid is in train, val or test, if not rmeove it from zone_dict
for zoneid in list(zone_dict.keys()):
    zoneid_father = zoneid.split('_')[0]
    if zoneid_father not in train_img_ids and zoneid_father not in val_img_ids and zoneid_father not in test_img_ids:
        print(zoneid, 'not in train, val or test')
        zone_dict.pop(zoneid)
# project L93: EPSG:2154
print(zone_dict)
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
    ).add_to(m)
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
m.save('../../data/train_val_test_maps/stratified_by_zone_seed42.html')