import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd

## SETTINGS
# -------------------------------------------------------------------------------------------
test_existing_model = True
if test_existing_model: 
    name_setting = '0_random_shuffling_seed1'
    #laod all variables from csv best_epoch_to_test
    best_epoch = pd.read_csv(f'../../unet_256_l1/best_epoch_to_test.csv')
    #remov the space before alla values and name columns
    best_epoch.columns = best_epoch.columns.str.strip()
    best_epoch = best_epoch.apply(lambda x: x.str.strip() if x.dtype == "object" else x) 
    # take the row with same name_setting
    best_epoch = best_epoch[best_epoch['name_setting'] == name_setting]
    #turn each column and value into a variable
    for column in best_epoch.columns:
        if column not in ['name_setting', 'stratified']:
            exec(f'{column} = {best_epoch[column].values[0]}')
        else:
            exec(f'{column} = best_epoch[column].values[0]')

    # turn random_seed, bs and in_channels into int
    random_seed = int(random_seed)
    bs = int(bs)
    in_channels = int(in_channels)
    if pre_trained == True:
        encoder_weights = 'imagenet'
    else: 
        encoder_weights = None
else:
    stratified = 'zone_mediteranean' # 'random', 'zone', 'image', 'acquisition', 'zone_mediteranean'
    name_setting = '1_stratified_shuffling_zone_mediteranean_seed2'
    random_seed = 2
    data_augmentation = False
    encoder_weights = None #"imagenet" or None
    in_channels = 4
    training = True
    plot_test = True
    bs = 16
    nb_epochs = 50
    patience = 15
    best_epoch = 1

if stratified == 'random':
    parent = 'random_shuffling/'
elif stratified == 'zone':
    parent = 'stratified_shuffling_by_zone/'
elif stratified == 'image':
    parent = 'stratified_shuffling_by_image/'
elif stratified == 'acquisition':
    parent = 'stratified_shuffling_acquisition/'
elif stratified == 'zone_mediteranean':
    parent = 'stratified_shuffling_zone_mediteranean/'
config_name = 'unet_256_l1/' + parent + name_setting

# -------------------------------------------------------------------------------------------
seeds_splitting = {'zone1': [0.68, 0.15], 'image1': [0.55, 0.24], 'random1': [0.6, 0.2], 'zone3': [0.68, 0.14], 'image3': [0.55, 0.24], 'acquisition1': [0.6, 0.2], 'zone_mediteranean1': [0.63, 0.18], 'zone_mediteranean2': [0.5, 0.34]}
zoneseed = stratified + str(random_seed)
splitting = seeds_splitting[zoneseed]

Path(f'../../{config_name}/models').mkdir(parents=True, exist_ok=True)
Path(f'../../{config_name}/metrics_test').mkdir(exist_ok=True)
Path(f'../../{config_name}/metrics_train_val').mkdir(exist_ok=True)

patch_level_param = {
    'patch_size': 256, 
    'level': 2, 
}

data_loading_settings = {
    'img_folder' : Path(f'../../data/patch{patch_level_param["patch_size"]}/img/'),
    'msk_folder' : Path(f'../../data/patch{patch_level_param["patch_size"]}/msk/'),
    'stratified' : stratified, # 'random', 'zone', 'image'
    'random_seed' : random_seed, 
    'split' : splitting, 
    'msks_256_fully_labelled' : pd.read_csv('../../csv/coverage_patch/p256_100per_labelled.csv'), 
    'path_pixels_by_zone': Path(f'../../csv/l{patch_level_param["level"]}_nb_pixels_by_zone.csv'),
    'bs': bs,
    'classes_balance': Path(f'../../{config_name}/classes_balance.csv'),
    'img_ids_by_set': Path(f'../../unet_256_l1/{parent}seed{random_seed}/img_ids_by_set.csv'),
    'data_augmentation': data_augmentation,
}

model_settings = {
    'encoder_name': "efficientnet-b7",
    'encoder_weights': encoder_weights,
    'in_channels': in_channels,
    'classes': 6 if patch_level_param['level'] == 1 else 113, # 113 to be checked
    'path_to_intermed_model': f'../../{config_name}/models/unet_intermed',
    'path_to_intermed_optim': f'../../{config_name}/models/optim_intermed',
    'path_to_last_model': f'../../{config_name}/models/unet_last.pt',
    'path_to_last_optim': f'../../{config_name}/models/optim_last.pt',
    'path_to_best_model': f'../../{config_name}/models/unet_intermed_epoch{best_epoch}.pt'#f'../../{config_name}/models/unet_intermed_epoch34.pt',#f'../../{config_name}/models/unet_intermed_epoch10.pt',#f'../../{config_name}/models/unet_intermed_epoch3.pt',#f'../../{config_name}/models/unet_intermed_epoch63.pt',#f'../../{config_name}/models/unet_intermed_epoch35.pt',
    }

training_settings = {
    'training': training,
    'lr': 1e-4,
    'criterion': 'Dice', #Dice or CrossEntropy
    'optimizer': 'Adam',
    'nb_epochs': nb_epochs,
    'early_stopping': True,
    'patience': patience, 
    'restart_training': None, # 42 if you want to restart training from a certain epoch, put the epoch number here, else put 0
    'losses_mious_path': f'../../{config_name}/metrics_train_val/losses_mious.csv',
}

plotting_settings = {
    'plot_test': plot_test,
    'losses_path': f'../../{config_name}/metrics_train_val/losses.png',
    'mious_path': f'../../{config_name}/metrics_train_val/mious_train_val.png',
    'nb_plots': 6,
    #'my_colors_map': {0: '#87edc1', 1: '#789262', 2: '#006400', 3: '#00ff00', 4: '#ff4500', 5: '#555555'},
    'my_colors_map': {
        0: '#789262',  # Vert olive
        1: '#555555',  # Gris
        2: '#006400',  # Vert foncé
        3: '#00ff00',  # Vert vif
        4: '#ff4500',  # Rouge
        5: '#8a2be2',  # Violet
    }, 
    'habitats_dict' : {
        0: "Prairies terrains domines par des especes non graminoides \n des mousses ou des lichens",
        1: "Landes fourres et toundras",
        2: "Bois forets et autres habitats boises",
        3: "Habitats agricoles horticoles et domestiques régulierement \n ou recemment cultives",
        4: "Zones baties sites industriels et autres habitats artificiels",
        5: "Autre: Habitats marins, Habitats cotiers, Eaux de surfaces continentales, \n Habitats continentaux sans vegetation ou à vegetation clairsemee, Autres"
    }, 
    'confusion_matrix_path': f'../../{config_name}/metrics_test/confusion_matrix.png',
    'IoU_path': f'../../{config_name}/metrics_test/IoUs.csv',
    'F1_path': f'../../{config_name}/metrics_test/F1s.csv',
    'pred_plot_path': f'../../{config_name}/metrics_test/test_preds.png',
    'img_msk_plot_path': f'../../{config_name}/img_msk.png',
}

# Save important settings in a csv file: 
'patch_size', 'level', 'stratified', 'random_seed', 'split' 'bs' 'encoder_name' 'encoder_weights' 'in_channels' 'classes' 'training' 'lr' 'criterion' 'optimizer' 'nb_epochs' 'early_stopping' 'patience'
settings = {
    'patch_size': patch_level_param['patch_size'],
    'level': patch_level_param['level'],
    'stratified': data_loading_settings['stratified'],
    'random_seed': data_loading_settings['random_seed'],
    'split': data_loading_settings['split'],
    'bs': data_loading_settings['bs'],
    'encoder_name': model_settings['encoder_name'],
    'encoder_weights': model_settings['encoder_weights'],
    'in_channels': model_settings['in_channels'],
    'classes': model_settings['classes'],
    'training': training_settings['training'],
    'lr': training_settings['lr'],
    'criterion': training_settings['criterion'],
    'optimizer': training_settings['optimizer'],
    'nb_epochs': training_settings['nb_epochs'],
    'early_stopping': training_settings['early_stopping'],
    'patience': training_settings['patience'],
}

# save settings to csv if not already done

for key in settings.keys():
    settings[key] = str(settings[key])
settings_df = pd.DataFrame(settings, index=[0])
if not Path(f'../../{config_name}/settings.csv').exists():
    settings_df.to_csv(f'../../{config_name}/settings.csv', index=False)
