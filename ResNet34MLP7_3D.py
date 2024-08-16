import torch
import torch.nn as nn
from torchvision.models import resnet34

class ResNet34MLP7_3D(nn.Module):
    def __init__(self):
        super(ResNet34MLP7_3D, self).__init__()

        self.resnet = resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # MLP Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
    
    def forward(self, x):
        # Forward pass through ResNet
        features = self.resnet(x)
        # Flatten the features
        features = torch.flatten(features, start_dim=1)
        # Forward pass through the fully connected layers
        output = self.fc_layers(features)
        return output