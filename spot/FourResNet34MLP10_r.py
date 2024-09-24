import torch
import torch.nn as nn
from torchvision.models import resnet34
import cv2
import numpy as np

class FourResNet34MLP10_r(nn.Module):
    def __init__(self):
        super(FourResNet34MLP10_r, self).__init__()

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
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc_layer4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())
        
        self.fc_layer5 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc_layer6 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())
        
        self.fc_layer7 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc_layer8 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc_layer9 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc_layer10 = nn.Linear(1024, 1)
    
    def forward(self, image1, image2, image3, image4):

        # cv2_image1 = image1[0].cpu().numpy()
        # cv2_image1 = np.transpose(cv2_image1, (1, 2, 0))
        # cv2_image1 = (cv2_image1 * 255).astype(np.uint8)
        # cv2.imshow('image', cv2_image1)

        # cv2_image2 = image1[1].cpu().numpy()
        # cv2_image2 = np.transpose(cv2_image2, (1, 2, 0))
        # cv2_image2 = (cv2_image2 * 255).astype(np.uint8)
        # cv2.imshow('image2', cv2_image2)

        # cv2_image3 = image1[2].cpu().numpy()
        # cv2_image3 = np.transpose(cv2_image3, (1, 2, 0))
        # cv2_image3 = (cv2_image3 * 255).astype(np.uint8)
        # cv2.imshow('image3', cv2_image3)

        # cv2_image4 = image1[3].cpu().numpy()
        # cv2_image4 = np.transpose(cv2_image4, (1, 2, 0))
        # cv2_image4 = (cv2_image4 * 255).astype(np.uint8)
        # cv2.imshow('image4', cv2_image4)

        # cv2.waitKey(0)
        
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
        output5 = self.fc_layer5(output4) + output4
        output6 = self.fc_layer6(output5) + output5
        output7 = self.fc_layer7(output6) + output6
        output8 = self.fc_layer8(output7) + output7
        output9 = self.fc_layer9(output8) + output8
        output = self.fc_layer10(output9)
        
        return output