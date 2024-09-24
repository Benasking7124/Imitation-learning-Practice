import torch
import torch.nn as nn
from torchvision.models import resnet18

class FourResNet18MLP5_r(nn.Module):
    def __init__(self):
        super(FourResNet18MLP5_r, self).__init__()

        # ResNet1
        self.resnet1 = resnet18(pretrained=False)
        self.resnet1 = nn.Sequential(*list(self.resnet1.children())[:-1])
        # ResNet2
        self.resnet2 = resnet18(pretrained=False)
        self.resnet2 = nn.Sequential(*list(self.resnet2.children())[:-1])
        # ResNet3
        self.resnet3 = resnet18(pretrained=False)
        self.resnet3 = nn.Sequential(*list(self.resnet3.children())[:-1])
        # ResNet4
        self.resnet4 = resnet18(pretrained=False)
        self.resnet4 = nn.Sequential(*list(self.resnet4.children())[:-1])

        # MLP Layers
        self.fc_layer1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU())
        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())
        self.fc_layer3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())
        self.fc_layer4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())
        self.fc_layer5 = nn.Linear(1024, 1)
    
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
        output1 = self.fc_layer1(features)
        output2 = self.fc_layer2(output1) + output1
        output3 = self.fc_layer3(output2) + output2
        output4 = self.fc_layer4(output3) + output3
        output = self.fc_layer5(output4)
        
        return output