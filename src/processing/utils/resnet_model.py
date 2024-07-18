import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp

class Ecomed_ResNet(nn.Module):
    def __init__(self, model_settings):
        super(Ecomed_ResNet, self).__init__()
        # Load pre-trained ResNet18 model
        self.pre_trained = model_settings['pre_trained']
        self.model_name = model_settings['model']
        if self.model_name == 'Resnet18':
            if self.pre_trained == 'imagenet':
                self.model = models.resnet18(pretrained=True)
            else:
                self.model = models.resnet18(pretrained=False)
        elif self.model_name == 'Resnet34':
            if self.pre_trained == 'imagenet':
                self.model = models.resnet34(pretrained=True)
            else: # includes the case when pre_trained == 'IGN'
                self.model = models.resnet34(pretrained=False)

        self.nb_output_heads = model_settings['nb_output_heads'] # 1 or 2
        self.weights_ign_path = model_settings['weights_ign_path']
        
        # Modify the first conv layer to accommodate different number of input channels
        num_channels = model_settings['in_channels']
        num_filters = self.model.conv1.out_channels
        kernel_size = self.model.conv1.kernel_size
        stride = self.model.conv1.stride
        padding = self.model.conv1.padding
        conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
        if self.pre_trained == 'imagenet':
            # Initialize the weights for the additional channels with the mean of the pre-trained weights
            mean_weights = self.model.conv1.weight.data.mean(dim=1, keepdim=True)
            conv1.weight.data = torch.cat([self.model.conv1.weight.data, mean_weights], dim=1)
            print('Pretrained imagenet (by default) weights loaded')

        elif self.pre_trained == 'IGN':
            unet = smp.Unet(
                encoder_name="resnet34",  # Encoder: ResNet34
                in_channels=4,            # Number of input channels
                classes=19               # Number of output classes
            )
            state_dict = torch.load(self.weights_ign_path, map_location=torch.device('cpu'))
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.seg_model.'):
                    new_key = key[len('model.seg_model.'):]  # Remove prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            unexpected_keys = ['criterion.weight']  # Add any other unexpected keys if necessary
            new_state_dict_2 = {key: value for key, value in new_state_dict.items() if key not in unexpected_keys}

            # ecoder weights have key starting by model.seg_model.encoder.
            encoder_weights = {key: value for key, value in new_state_dict_2.items() if key.startswith('encoder.')}
            # remove the "encoder." from the begining of the key 
            encoder_weights = {key[len('encoder.'):] : value for key, value in encoder_weights.items()}

        self.model.conv1 = conv1
        # Remove the original fully connected layer
        self.model.fc = nn.Identity()
        if self.pre_trained == 'IGN':
            self.model.load_state_dict(encoder_weights)
        # Define the different heads based on the configuration
        if self.nb_output_heads == 1:
            self.head = nn.Linear(512, 18)  # Single head for multi-label with 7 classes
        elif self.nb_output_heads == 2:
            self.head1 = nn.Linear(512, 6)  # Multi-label head with 6 classes
            self.head2 = nn.Linear(512, 1)  # Single class head

    def forward(self, x):
        x = self.model(x)
        if self.nb_output_heads == 1:
            x = self.head(x)
        elif self.nb_output_heads == 2:
            x1 = self.head1(x)
            x2 = self.head2(x)
            x = torch.cat([x1, x2], dim=1)
        return x