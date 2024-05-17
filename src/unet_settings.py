import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd

## SETTINGS
config_name = 'unet_256_l1/random_shuffling_pre_trained'

Path(f'../{config_name}/models').mkdir(parents=True, exist_ok=True)
Path(f'../{config_name}/figures').mkdir(exist_ok=True)

patch_level_param = {
    'patch_size': 256, 
    'level': 1, 
}

data_loading_settings = {
    'img_folder' : Path(f'../data/patch{patch_level_param["patch_size"]}/img/'),
    'msk_folder' : Path(f'../data/patch{patch_level_param["patch_size"]}/msk/l123/'),
    'msks_256_fully_labelled' : pd.read_csv('../csv/coverage_patch/p256_100per_labelled.csv'), 
    'stratified' : False,
    'random_seed' : 42,
    'splitting' : [0.6, 0.2, 0.2],
    'bs': 16,
}

model_settings = {
    'encoder_name': "efficientnet-b7",
    'encoder_weights': None, #"imagenet" or None
    'in_channels': 4,
    'classes': 6 if patch_level_param['level'] == 1 else 113, # 113 to be checked
    'path_to_intermed_model': f'../{config_name}/models/unet_intermed',
    'path_to_intermed_optim': f'../{config_name}/models/optim_intermed',
    'path_to_last_model': f'../{config_name}/models/unet_last.pt',
    'path_to_last_optim': f'../{config_name}/models/optim_last.pt',
    'path_to_best_model': f'../{config_name}/models/unet_intermed_epoch35.pt',#f'../{config_name}/models/unet_intermed_epoch3.pt',
}

training_settings = {
    'training': False,
    'lr': 1e-4,
    'criterion': 'Dice', #Dice or CrossEntropy
    'optimizer': 'Adam',
    'nb_epochs': 200, 
    'early_stopping': True,
    'patience': 30,
    'losses_mious_path': f'../{config_name}/losses_mious.csv',
    'restart_training': None, # 42 if you want to restart training from a certain epoch, put the epoch number here, else put 0
}

plotting_settings = {
    'plot_test': True,
    'pred_plot_path': f'../{config_name}/figures/test_preds.png',
    'losses_path': f'../{config_name}/figures/losses.png',
    'mious_path': f'../{config_name}/figures/mious.png',
    'nb_plots': 6,
    'my_colors_map': {1: '#789262', 2: '#ff4500', 3: '#006400', 4: '#00ff00', 5: '#555555', 6: '#8a2be2'},
}

# Save important settings in a csv file: 
'patch_size', 'level', 'stratified', 'random_seed', 'splitting' 'bs' 'encoder_name' 'encoder_weights' 'in_channels' 'classes' 'training' 'lr' 'criterion' 'optimizer' 'nb_epochs' 'early_stopping' 'patience'
settings = {
    'patch_size': patch_level_param['patch_size'],
    'level': patch_level_param['level'],
    'stratified': data_loading_settings['stratified'],
    'random_seed': data_loading_settings['random_seed'],
    'splitting': data_loading_settings['splitting'],
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
if not Path(f'../{config_name}/settings.csv').exists():
    settings_df.to_csv(f'../{config_name}/settings.csv', index=False)