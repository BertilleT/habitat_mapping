import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd

## SETTINGS
# -------------------------------------------------------------------------------------------
test_existing_model = True
patch_size = 256
model_type = f'resnet18_{patch_size}_l1/' # resnet18_256_l1/ or  unet_256_l1/

if test_existing_model: 
    name_setting = 'resnet18_strat_zone1_homogene_lr3'
    #laod all variables from csv best_epoch_to_test
    best_epoch_to_test = pd.read_csv(f'../../results/{model_type}best_epoch_to_test.csv')
    #remov the space before alla values and name columns
    best_epoch_to_test.columns = best_epoch_to_test.columns.str.strip()
    best_epoch_to_test = best_epoch_to_test.apply(lambda x: x.str.strip() if x.dtype == "object" else x) 
    # take the row with same name_setting
    best_epoch_to_test = best_epoch_to_test[best_epoch_to_test['name_setting'] == name_setting]
    #turn each column and value into a variable
    print(best_epoch_to_test)
    print(type(best_epoch_to_test))

    for column in best_epoch_to_test.columns:
        print(f"Processing column: {column}")
        print(f"Type of best_epoch_to_test[{column}]: {type(best_epoch_to_test[column])}")
        print(f"First value of best_epoch_to_test[{column}]: {best_epoch_to_test[column].values[0]}")
            
        if column not in ['name_setting', 'stratified', 'normalisation', 'year']:
            exec(f'{column} = {best_epoch_to_test[column].values[0]}')
        else:
            exec(f'{column} = best_epoch_to_test[column].values[0]')

    # turn random_seed, bs and in_channels into int
    random_seed = int(random_seed)
    bs = int(bs)
    in_channels = int(in_channels)
    if pre_trained == True:
        encoder_weights = 'imagenet'
    else: 
        encoder_weights = None

    task = "image_classif"
    heterogeneity = 'homogeneous'
    lr = 1e-3
    bs = 16
    optimizer = 'Adam'

# ---------------------------------------

else:
    stratified = 'random' # 'random', 'zone', 'image', 'acquisition', 'zone_mediteranean', 'zone2023'
    name_setting = 'resnet18_random_homogene_70epochs' # 
    normalisation = "channel_by_channel" # "all_channels_together" or "channel_by_channel"
    random_seed = 1
    data_augmentation = False
    encoder_weights = None #"imagenet" or None
    year = 'all'# '2023' or 'all'
    in_channels = 4
    training = True
    plot_test = True
    bs = 16
    nb_epochs = 70
    patience = 70
    best_epoch = 1
    task = "image_classif"
    heterogeneity = 'homogeneous'
    location ='all' # 'mediteranean' or 'all'
    lr = 1e-4
    optimizer = 'Adam' # 'Adam' or 'AdamW'

if stratified == 'random':
    parent = 'random_shuffling/'
elif stratified == 'zone':
    parent = 'stratified_shuffling_by_zone/'
elif stratified == 'image':
    parent = 'stratified_shuffling_by_image/'
elif stratified == 'acquisition':
    parent = 'stratified_shuffling_acquisition/'
elif stratified == 'zone2023':
    parent = 'stratified_shuffling_zone2023/'

config_name = 'results/'  + model_type + parent + name_setting

# -------------------------------------------------------------------------------------------
if heterogeneity != 'homogeneous':
    seeds_splitting = {'zone1': [0.68, 0.15], 'image1': [0.55, 0.24], 'random1': [0.6, 0.2], 'zone3': [0.68, 0.14], 'image3': [0.55, 0.24], 'acquisition1': [0.6, 0.2], 'zone_mediteranean1': [0.63, 0.18], 'zone_mediteranean2': [0.5, 0.34], 'zone20231': [0.63, 0.14] }
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

patch_level_param = {
    'patch_size': patch_size, 
    'level': 1, 
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
    'model': 'Resnet18', # UNet, Resnet18
    'encoder_name': "efficientnet-b7",
    'encoder_weights': encoder_weights,
    'in_channels': in_channels,
    'classes': 6 if patch_level_param['level'] == 1 else 113, # 113 to be checked
    'path_to_intermed_model': f'../../{config_name}/models/unet_intermed',
    'path_to_intermed_optim': f'../../{config_name}/models/optim_intermed',
    'path_to_last_model': f'../../{config_name}/models/unet_last.pt',
    'path_to_last_optim': f'../../{config_name}/models/optim_last.pt',
    'path_to_best_model': f'../../{config_name}/models/unet_intermed_epoch{best_epoch}.pt',#f'../../{config_name}/models/unet_intermed_epoch34.pt',#f'../../{config_name}/models/unet_intermed_epoch10.pt',#f'../../{config_name}/models/unet_intermed_epoch3.pt',#f'../../{config_name}/models/unet_intermed_epoch63.pt',#f'../../{config_name}/models/unet_intermed_epoch35.pt',
    'task': task, # image_classif, pixel_classif
    }

training_settings = {
    'training': training,
    'lr': lr,
    'criterion': 'CrossEntropy', #Dice or CrossEntropy
    'optimizer': optimizer,
    'nb_epochs': nb_epochs,
    'early_stopping': True,
    'patience': patience, 
    'restart_training': None, # 42 if you want to restart training from a certain epoch, put the epoch number here, else put 0
    'losses_metric_path': f'../../{config_name}/metrics_train_val/losses_metric.csv',
}

plotting_settings = {
    'plot_test': plot_test,
    'losses_path': f'../../{config_name}/metrics_train_val/losses.png',
    'metrics_path': f'../../{config_name}/metrics_train_val/metrics_train_val.png',
    'nb_plots': 16,
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
    'l2_habitats_dict' : {
        0: "sédiment intertidal",
        1: "habitats pélagiques",
        2: "dunes côtières et rivages sableux",
        3: "galets côtiers",
        4: "falaises corniches et rivages \nrocheux incluant le supralittoral",
        5: "eaux dormantes de surface",
        6: "eaux courantes de surface",
        7: "zones littorales des eaux de \nsurface continentales",
        8: "pelouses sèches",
        9: "prairies mésiques",
        10: "prairies humides et prairies \nhumides saisonnières",
        11: "ourlets clairières forestières \n et peuplements de grandes herbacées non graminées",
        12: "steppes salées continentales",
        13: "fourrés tempérés et \nméditerranéo-montagnards",
        14: "landes arbustives tempérées",
        15: "maquis matorrals arborescents \net fourrés \nthermo-méditerranéens",
        16: "garrigues",
        17: "fourrés ripicoles et des \nbas-marais",
        18: "haies",
        19: "plantations d'arbustes",
        20: "forêts de feuillus caducifoliés",
        21: "forêts de feuillus sempervirents",
        22: "forêts de conifères",
        23: "formations mixtes d’espèces \ncaducifoliées et de conifères",
        24: "alignements d’arbres petits \nbois anthropiques boisements  récemment abattus stades initiaux de boisements et taillis",
        25: "éboulis",
        26: "falaises continentales pavements \nrocheux et affleurements rocheux",
        27: "habitats continentaux divers sans\n végétation ou à végétation clairsemée",
        28: "cultures et jardins maraîchers",
        29: "zones cultivées des jardins \net des parcs",
        30: "bâtiments des villes et des \nvillages",
        31: "constructions à faible densité",
        32: "sites industriels d’extraction",
        33: "réseaux de transport et autres \nzones de construction à surface dure",
        34: "plans d’eau construits très \nartificiels et structures connexes",
        35: "dépôts de déchets",
        255: "unknown"
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
