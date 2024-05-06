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
    'bs': 16,
}

model_settings = {
    'encoder_name': "efficientnet-b7",
    'encoder_weights': None,
    'in_channels': 4,
    'classes': 6 if patch_level_param['level'] == 1 else 113, # to check 113
    'path_to_model': '../models/unet_last.pt',
    'path_to_optim': '../models/optim_last.pt',
}

training_settings = {
    'training': True,
    'lr': 1e-4,
    'criterion': nn.CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'nb_epochs': 10,
}

plotting_settings = {
    'plot_test': True,
    'pred_plot_path': '../figures/test_preds.png',
    'losses_path': '../figures/losses.png',
    'nb_plots': 10,
    'my_colors_map': {1: '#789262', 2: '#ff4500', 3: '#006400', 4: '#00ff00', 5: '#555555', 6: '#8a2be2'},
}