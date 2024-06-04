from ImagePositionDataset import ImagePositionDataset
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch
from ResNet18MLP import ResNet18MLP
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = '/home/ben/ml_project/dataset3/train/'
VALID_PATH = '/home/ben/ml_project/dataset3/valid/'

# MODEL_PATH = '/home/ben/ml_project/model'

train_dataset = ImagePositionDataset(TRAIN_PATH, (TRAIN_PATH + 'labels.npy'), cuda=True)
valid_dataset = ImagePositionDataset(VALID_PATH, (VALID_PATH + 'labels.npy'), cuda=True)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=True)

model = ResNet18MLP()
# model.load_state_dict(torch.load('/home/ben/ml_project/model/model_epoch_1803.pth'))
model.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# num_epochs = 20
epoch = 0
epoch_loss = float('inf')
epoch_losses = []
# epoch_losses = list(np.load('/home/ben/ml_project/model/epoch_losses.npy'))
# for epoch in range(num_epochs):
while epoch_loss > 1e-4:
    model.train()
    running_loss = 0.0
    for input, label in train_dataloader:
        optimizer.zero_grad()

        output = model(input)
        loss = loss_fn(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    epoch_losses.append(epoch_loss)
    # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.5f}')
    torch.save(model.state_dict(), f'/home/ben/ml_project/weights/model_epoch_{epoch+1}.pth')
    np.save('/home/ben/ml_project/weights/epoch_losses', epoch_losses)
    epoch += 1

print('Finished Training')
plt.plot(epoch_losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Epoch Loss")

for i in range(len(epoch_losses)):
    plt.annotate(str(round(epoch_losses[i], 4)), xy=(i, epoch_losses[i]))

plt.show()