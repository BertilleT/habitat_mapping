import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd

## SETTINGS
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
    'bs': 8,
}

model_settings = {
    'encoder_name': "efficientnet-b7",
    'encoder_weights': None,
    'in_channels': 4,
    'classes': 6 if patch_level_param['level'] == 1 else 113, # to check 113
    'path_to_intermed_model': '../unet256_randomshuffling/models/unet_intermed',
    'path_to_model': '../unet256_randomshuffling/models/unet_last.pt',
    'path_to_optim': '../unet256_randomshuffling/models/optim_last.pt',
}

training_settings = {
    'training': False,
    'lr': 1e-4,
    'criterion': 'CrossEntropy',
    'optimizer': 'Adam',
    'nb_epochs': 4,
}

plotting_settings = {
    'plot_test': True,
    'pred_plot_path': '../unet256_randomshuffling/figures/test_preds.png',
    'losses_path': '../unet256_randomshuffling/figures/losses.png',
    'nb_plots': 6,
    'my_colors_map': {1: '#789262', 2: '#ff4500', 3: '#006400', 4: '#00ff00', 5: '#555555', 6: '#8a2be2'},
}