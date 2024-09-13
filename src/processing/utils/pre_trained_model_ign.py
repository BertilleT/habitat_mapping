# load IGN weights FLAIR-INC_rgb_12cl_resnet34-unet_weights.pth from:  
# https://huggingface.co/collections/IGNF/flair-models-landcover-semantic-segmentation-65bb67415a5dbabc819a95de

import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
from torchsummary import summary

# get information abour architecture of the pre_trained model
model = smp.Unet(
    encoder_name="resnet34",  # Encoder: ResNet34
    in_channels=4,            # Number of input channels
    classes=19               # Number of output classes
)
path_rgbi = "../../pre_trained_model_ign/FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth"
state_dict = torch.load(path_rgbi, map_location=torch.device('cpu'))

new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('model.seg_model.'):
        new_key = key[len('model.seg_model.'):]  # Remove prefix
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

unexpected_keys = ['criterion.weight']  # Add any other unexpected keys if necessary
new_state_dict_2 = {key: value for key, value in new_state_dict.items() if key not in unexpected_keys}

# Load modified state_dict into the model
model.load_state_dict(new_state_dict_2) 

summary(model, input_size=(4, 64, 64))

# ecoder weights have key starting by model.seg_model.encoder.
encoder_weights = {key: value for key, value in new_state_dict_2.items() if key.startswith('encoder.')}
# remove the "encoder." from the begining of the key 
encoder_weights = {key[len('encoder.'):] : value for key, value in encoder_weights.items()}

#load resnet34
resnet34 = models.resnet34(weights=False)
num_channels = 4
num_filters = resnet34.conv1.out_channels
kernel_size = resnet34.conv1.kernel_size
stride = resnet34.conv1.stride
padding = resnet34.conv1.padding
conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
resnet34.conv1 = conv1
resnet34.fc = nn.Identity()
resnet34.load_state_dict(encoder_weights)
resnet34.head = nn.Linear(512, 7)