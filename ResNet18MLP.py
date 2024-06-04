import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18MLP(nn.Module):
    def __init__(self):
        super(ResNet18MLP, self).__init__()

        self.resnet = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # MLP Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
    
    def forward(self, x):
        # Forward pass through ResNet
        features = self.resnet(x)
        # Flatten the features
        features = torch.flatten(features, start_dim=1)
        # Forward pass through the fully connected layers
        output = self.fc_layers(features)
        return output
    
model = ResNet18MLP()

# Example input (random image tensor)
example_input = torch.randn(1, 3, 224, 224)

# Forward pass (get the predicted coordinates)
predicted_coordinates = model(example_input)
print(predicted_coordinates)