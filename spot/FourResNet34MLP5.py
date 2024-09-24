import torch
import torch.nn as nn
from torchvision.models import resnet34

class FourResNet34MLP5(nn.Module):
    def __init__(self):
        super(FourResNet34MLP5, self).__init__()

        # ResNet1
        self.resnet1 = resnet34(pretrained=False)
        self.resnet1 = nn.Sequential(*list(self.resnet1.children())[:-1])
        # ResNet2
        self.resnet2 = resnet34(pretrained=False)
        self.resnet2 = nn.Sequential(*list(self.resnet2.children())[:-1])
        # ResNet3
        self.resnet3 = resnet34(pretrained=False)
        self.resnet3 = nn.Sequential(*list(self.resnet3.children())[:-1])
        # ResNet4
        self.resnet4 = resnet34(pretrained=False)
        self.resnet4 = nn.Sequential(*list(self.resnet4.children())[:-1])

        # MLP Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
    
    def forward(self, image1, image2, image3, image4):
        
        # Forward pass through ResNet
        embedding1 = self.resnet1(image1)
        embedding1 = torch.flatten(embedding1, start_dim=1)
        embedding2 = self.resnet2(image2)
        embedding2 = torch.flatten(embedding2, start_dim=1)
        embedding3 = self.resnet3(image3)
        embedding3 = torch.flatten(embedding3, start_dim=1)
        embedding4 = self.resnet4(image4)
        embedding4 = torch.flatten(embedding4, start_dim=1)

        # Concatenate the features
        features = torch.cat((embedding1, embedding2, embedding3, embedding4), dim=1)

        # Forward pass through the fully connected layers
        output = self.fc_layers(features)
        
        return output