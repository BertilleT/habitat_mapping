# PIPELINE TO PRE-PROCESS THE DATA

## What we need
- pivot_table.csv: contains polygons paths and associated images paths when they intersect at more than 0%. 
- a dict for each level l1, l2, l3 with name class, code class and integer class

## STEPS
### STEP 0 DONE
To create pivot_table.csv  
What could be done: to keep only the shapefiles with CDEUNIS_2 None.  
- pre_selection_tif_shp.py  
- data_preparation_utils.py: generate_pivot_table_intersect()


### STEP 1
To create a dict for each label uniquely present in shapefile CDEUNIS_1 at each level  
- extract_unique_classes_l123.py  
- rasterize_polygons_3masks.py: see first part

### STEP 2
To create the masks keeping CDEUNIS1 = to rasterize polygons.  
- rasterize_polygons_3masks.py  
- from_3m_to_1m.py
    
### STEP 3
To create the patches, do not save patches with no labels at level 1  
- data_preparation_utils.py: split_img_msk_into_patches()
- patchize.py