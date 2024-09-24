import torch
import torch.nn as nn
from torchvision.models import resnet34

class FourResNet34MLP10_dr(nn.Module):
    def __init__(self):
        super(FourResNet34MLP10_dr, self).__init__()

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
        self.fc_layer1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU())
        
        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc_layer3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU())
        self.transform3 = nn.Linear(1024, 512)

        self.fc_layer4 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU())
        
        self.fc_layer5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU())
        self.transform5 = nn.Linear(512, 256)

        self.fc_layer6 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU())
        
        self.fc_layer7 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU())
        self.transform7 = nn.Linear(256, 64)

        self.fc_layer8 = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU())
        self.transform8 = nn.Linear(64, 16)

        self.fc_layer9 = nn.Sequential(
            nn.Linear(16, 4),
            nn.ReLU())
        self.transform9 = nn.Linear(16, 4)

        self.fc_layer10 = nn.Linear(4, 1)
    
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
        output3 = self.fc_layer3(output2) + self.transform3(output2)
        output4 = self.fc_layer4(output3) + output3
        output5 = self.fc_layer5(output4) + self.transform5(output4)
        output6 = self.fc_layer6(output5) + output5
        output7 = self.fc_layer7(output6) + self.transform7(output6)
        output8 = self.fc_layer8(output7) + self.transform8(output7)
        output9 = self.fc_layer9(output8) + self.transform9(output8)
        output = self.fc_layer10(output9)
        
        return output