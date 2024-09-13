'''
This script contains all the settings to configure the model training and testing process.
Global Variables:
    - level: Level of the classification (1, 2)
    - model_name: Name of the model (UNet, Resnet18, Resnet34)
    - patch_size: Size of the image patches (256, 128, 64)
    - test_existing_model: Boolean indicating whether to test an existing model
    - model_type: Type of model based on the model name and patch size
    - name_setting: Name of the folder where the model is saved and results are stored
    - normalisation: Type of normalisation for the data (channel_by_channel, all_channels_together)
    - random_seed: Seed for the random number generator
    - data_augmentation: Boolean indicating whether to use data augmentation
    - pre_trained: Type of pre-trained model (IGN, imagenet)
    - year: Year of the data (2023, all)
    - in_channels: Number of input channels (3, 4)
    - training: Boolean indicating whether to train the model
    - testing: Boolean indicating whether to test the model
    - plot_test: Boolean indicating whether to plot the test results
    - plot_re_assemble: Boolean indicating whether to plot the reassembled patches
    - tune_alpha1: Boolean indicating whether to tune alpha1
    - tune_alpha2: Boolean indicating whether to tune alpha2
    - post_processing: Boolean indicating whether to use post-processing
    - nb_output_heads: Number of output heads (1, 2)
    - location: Location of the data (mediteranean, all)
    - seeds_splitting: Dictionary containing the seeds for different stratified cases
    - zoneseed: Seed for the stratified case
    - splitting: Splitting values for the stratified case
    - config_name: Path to the results based on the model type, parent, heterogeneity, and name setting
    - parent: Parent of the stratified case
    - heterogeneity_path: Path to the heterogeneity based on the heterogeneity value
'''

import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim

## ----------------------------------------------------------------------- SETTINGS ----------------------------------------------------------------------- ##
level = 1
model_name = 'Resnet18' # 'UNet', 'Resnet18', 'Resnet34'
patch_size = 256 # 256, 128 or 64
test_existing_model = True # True or False


# -------------------------------------------------------------------------------------------
model_type = f'{model_name.lower()}_{patch_size}_l{level}/'

# -------------------------------------------------------------------------------------------
# If you want to test an existing model
# -------------------------------------------------------------------------------------------
if test_existing_model: # if you want to test an existing model
    name_setting = '0_random_shuffling_seed1' # choose a name for the folder where the model is saved and where the results will be saved
    # load all variables from csv best_epoch_to_test
    csv_path = f'../../results/{model_type}best_epoch_to_test.csv' # look for epoch to test in this csv
    if Path(csv_path).exists():
        best_epoch_to_test = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    # remove the space before all values and name columns
    best_epoch_to_test.columns = best_epoch_to_test.columns.str.strip()
    best_epoch_to_test = best_epoch_to_test.apply(lambda x: x.str.strip() if x.dtype == "object" else x) 
    # take the row with same name_setting
    best_epoch_to_test = best_epoch_to_test[best_epoch_to_test['name_setting'] == name_setting]
    #turn each column and value into a variable
    for column in best_epoch_to_test.columns:
        if column not in ['name_setting', 'stratified', 'normalisation', 'year', 'location']:
            exec(f'{column} = {best_epoch_to_test[column].values[0]}')
        else:
            exec(f'{column} = best_epoch_to_test[column].values[0]')

    # turn random_seed, bs and in_channels into int
    random_seed = int(random_seed)
    bs = int(bs)
    in_channels = int(in_channels)
    print(f'pre_trained: {pre_trained}')
    if pre_trained.lower() == 'imagenet': # Imagenet or IGN
        if model_name == 'UNet':
            encoder_weights = pre_trained
        else:
            encoder_weights = None
    else: 
        encoder_weights = None

    task = "pixel_classif" # 'image_classif', 'pixel_classif'
    heterogeneity = 'all' # 'homogeneous', 'heterogeneous', 'all'
    lr = 1e-3 
    optimizer = 'Adam' # 'Adam', 'AdamW'
    labels = "single" # 'multi', 'single'
    loss = 'Dice' # 'Dice', 'CrossEntropy', 'BCEWithDigits'
    classes = 6 # 6
    testing = False # True, False
    plot_test = True 
    plot_re_assemble = False 
    tune_alpha1 = False 
    tune_alpha2 = False 
    post_processing = False 
    nb_output_heads = 1 # 1, 2
    location = "all" # 'mediteranean', 'all'
    normalisation = "channel_by_channel" # 'all_channels_together', 'channel_by_channel'


# -------------------------------------------------------------------------------------------
# If you want to train a new model
# -------------------------------------------------------------------------------------------
else:
    stratified = 'random' # 'random', 'zone', 'image', 'acquisition', 'zone_mediteranean', 'zone2023'
    name_setting = 'resnet64_multi_label_64_random_60epochs_bs4096' # choose a name for the folder where the model is saved and where the results will be saved
    normalisation = "channel_by_channel" # "all_channels_together" or "channel_by_channel"
    random_seed = 1 
    data_augmentation = False
    pre_trained = None # 'IGN' or 'imagenet'
    year = 'all'# '2023' or 'all'
    in_channels = 4 # 3 or 4
    training = True 
    testing = True

    plot_test = True
    plot_re_assemble = False
    post_processing = False

    tune_alpha1 = False
    tune_alpha2 = False

    bs = 4096 
    nb_epochs = 60 
    patience = 60 # early stopping
    best_epoch = 1
    task = "image_classif" # 'image_classif' or 'pixel_classif'
    labels = "multi" # 'multi' or 'single'
    heterogeneity = 'all' # 'homogeneous' or 'heterogeneous', 'all'
    location ='all' # 'mediteranean' or 'all'
    lr = 1e-3
    optimizer = 'Adam' # 'Adam' or 'AdamW'
    loss = 'BCEWithDigits' # 'Dice' or 'CrossEntropy' or 'BCEWithDigits'
    classes = 7 # 6, 7 
    nb_output_heads = 1

    if pre_trained:
        if model_name == 'UNet':
            encoder_weights = 'imagenet'
        else:
            encoder_weights = None
    else: 
        encoder_weights = None


# -------------------------------------------------------------------------------------------
# The path to the results is defined according to the following structure:
#   - results/model_type/parent/heterogeneity/name_setting
# -------------------------------------------------------------------------------------------
parent = stratified

if model_name in ['Resnet18', 'Resnet34']:
    heterogeneity_path = heterogeneity + '/'
else:
    heterogeneity_path = ''

config_name = 'results/'  + model_type + parent + heterogeneity_path + name_setting


# -------------------------------------------------------------------------------------------
# Splitting the seeds for the different stratified cases
# -------------------------------------------------------------------------------------------
if heterogeneity != 'homogeneous':
    seeds_splitting = {'zone1': [0.68, 0.2], 'image1': [0.55, 0.24], 'random1': [0.6, 0.2], 'zone3': [0.68, 0.14], 'image3': [0.55, 0.24], 'acquisition1': [0.6, 0.2], 'zone_mediteranean1': [0.63, 0.18], 'zone_mediteranean2': [0.5, 0.34], 'zone20231': [0.63, 0.14] }
else: 
    if location != 'mediteranean':
        seeds_splitting = {'zone1': [0.7, 0.2], 'random1': [0.6, 0.2]}
    else:
        if year == '2023':
            seeds_splitting = {'zone1': [0.71, 0.14], 'random1': [0.6, 0.2]}
        else: 
            seeds_splitting = {'zone1': [0.64, 0.19], 'random1': [0.6, 0.2]}
zoneseed = stratified + str(random_seed)
splitting = seeds_splitting[zoneseed]

Path(f'../../{config_name}/models').mkdir(parents=True, exist_ok=True)
Path(f'../../{config_name}/metrics_test').mkdir(exist_ok=True)
Path(f'../../{config_name}/metrics_train_val').mkdir(exist_ok=True)
Path(f'../../results/{model_type}{parent}/seed{random_seed}').mkdir(exist_ok=True)

# Load the JSON file
with open(f'habitat_dict_l{level}.json', 'r') as file:
    data = json.load(file)
habitats_dict = {int(k): v for k, v in data.items()}

with open(f'colors_map_l{level}.json', 'r') as file:
    data = json.load(file)
colors_map = {int(k): v for k, v in data.items()}

# -------------------------------------------------------------------------------------------
# The settings are stored in dictionaries named according to the following structure: 
#   - patch_level_param: contains the patch size and the level
#   - data_loading_settings: contains the settings for the data loading
#   - model_settings: contains the settings for the model
#   - training_settings: contains the settings for the training
#   - plotting_settings: contains the settings for the plotting
#   - settings: contains the most important settings
# -------------------------------------------------------------------------------------------

patch_level_param = {
    'patch_size': patch_size, 
    'level': level, 
}

data_loading_settings = {
    'img_folder' : Path(f'../../data/patch{patch_level_param["patch_size"]}/img/'),
    'msk_folder' : Path(f'../../data/patch{patch_level_param["patch_size"]}/msk/'),
    'stratified' : stratified, # 'random', 'zone', 'image'
    'random_seed' : random_seed, 
    'split' : splitting, 
    'bs': bs,
    'normalisation': normalisation,
    'classes_balance': Path(f'../../{config_name}/classes_balance.csv'),
    'img_ids_by_set': Path(f'../../results/{model_type}{parent}seed{random_seed}/img_ids_by_set.csv'),
    'data_augmentation': data_augmentation,
    'year': year, 
    '2023_zones': Path('../../csv/zones_2023.csv'),
    'heterogen_patches_path': Path(f'../../csv/heterogen_masks_{patch_size}.csv'),
    'patches' : heterogeneity, # 'homogeneous' or 'heterogeneous', 'all', 
    'location': location # 'mediteranean' or 'all'
}

model_settings = {
    'model': model_name, # UNet, Resnet18, Resnet34
    'encoder_name': "efficientnet-b7",
    'pre_trained': pre_trained, # True
    'encoder_weights': encoder_weights,
    'in_channels': in_channels,
    'classes': classes, # 113 to be checked
    'path_to_intermed_model': f'../../{config_name}/models/unet_intermed',
    'path_to_intermed_optim': f'../../{config_name}/models/optim_intermed',
    'path_to_last_model': f'../../{config_name}/models/unet_last.pt',
    'path_to_last_optim': f'../../{config_name}/models/optim_last.pt',
    'path_to_best_model': f'../../{config_name}/models/unet_intermed_epoch{best_epoch}.pt',#f'../../{config_name}/models/unet_intermed_epoch34.pt',#f'../../{config_name}/models/unet_intermed_epoch10.pt',#f'../../{config_name}/models/unet_intermed_epoch3.pt',#f'../../{config_name}/models/unet_intermed_epoch63.pt',#f'../../{config_name}/models/unet_intermed_epoch35.pt',
    'task': task, # image_classif, pixel_classif
    'labels': labels, # multi, single
    'nb_output_heads': nb_output_heads, # 1 or 2
    'weights_ign_path': '../../pre_trained_model_ign/FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth', 
    }

training_settings = {
    'training': training,
    'testing': testing,
    'lr': lr,
    'criterion': loss, #Dice or CrossEntropy
    'optimizer': optimizer,
    'nb_epochs': nb_epochs,
    'early_stopping': True,
    'patience': patience, 
    'restart_training': None, # if you want to restart training from a certain epoch, put the epoch number here, else put None
    'losses_metric_path': f'../../{config_name}/metrics_train_val/losses_metric.csv',
    'tune_alpha1': tune_alpha1,
    'tune_alpha2': tune_alpha2,
    'alpha1': 0.5, 
    'alpha2': 0.5,
    'beta': 0.5,
}

plotting_settings = {
    'post_processing': post_processing, # 'True' or 'False
    'plot_test': plot_test,
    'losses_path': f'../../{config_name}/metrics_train_val/losses.png',
    'metrics_path': f'../../{config_name}/metrics_train_val/metrics_train_val.png',
    'nb_plots': 16,
    'colors_map': colors_map,
    'habitats_dict' : habitats_dict,
    'confusion_matrix_path': f'../../{config_name}/metrics_test/confusion_matrix.png',
    'IoU_path': f'../../{config_name}/metrics_test/IoUs.csv',
    'F1_path': f'../../{config_name}/metrics_test/F1s.csv',
    'pred_plot_path': f'../../{config_name}/metrics_test/test_preds.png',
    'img_msk_plot_path': f'../../{config_name}/img_msk.png',
    're_assemble_patches_path': f'../../{config_name}/metrics_test/re_assemble_patches.png',
    'plot_re_assemble': plot_re_assemble,
}

# Save important settings in a csv file if not already done

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

for key in settings.keys():
    settings[key] = str(settings[key])
settings_df = pd.DataFrame(settings, index=[0])

if not Path(f'../../{config_name}/settings.csv').exists():
    settings_df.to_csv(f'../../{config_name}/settings.csv', index=False)