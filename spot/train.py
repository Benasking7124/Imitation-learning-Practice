from SpotImagePositionDataset import SpotImagePositionDataset
from torch.utils.data import DataLoader
import torch
from FourResNet18MLP3 import FourResNet18MLP3
import matplotlib.pyplot as plt
import numpy as np
import cv2

TRAIN_PATH = '/home/ben/ml_project/spot/dataset/train/'
# VALID_PATH = '/home/ben/ml_project/spot/dataset/valid/'
WEIGHT_PATH = '/home/ben/ml_project/spot/weights/'
CONTINUE = 0   # Start from beginning, use 0

model = FourResNet18MLP3()

if torch.cuda.is_available():
    device = torch.device("cuda")
    train_dataset = SpotImagePositionDataset(TRAIN_PATH, (TRAIN_PATH + 'labels.npy'), 4, cuda=True)
    # valid_dataset = SpotImagePositionDataset(VALID_PATH, (VALID_PATH + 'labels.npy'), cuda=True)
    model.to(device)

else:
    train_dataset = SpotImagePositionDataset(TRAIN_PATH, (TRAIN_PATH + 'labels.npy'))
    # valid_dataset = SpotImagePositionDataset(VALID_PATH, (VALID_PATH + 'labels.npy'))

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=True)

if CONTINUE > 1:
    model.load_state_dict(torch.load(WEIGHT_PATH + 'epoch_' + str(CONTINUE) + '.pth'))

# Training Parameters
loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = CONTINUE + 1
epoch_loss = 1
epoch_losses = []
if CONTINUE > 1:
    epoch_losses = list(np.load(WEIGHT_PATH + 'epoch_losses.npy'))[:CONTINUE]

while epoch_loss > 1e-4:
    model.train()
    running_loss = 0.0
    for inputs, label in train_dataloader:
        # cv2_image1 = inputs[0][0].cpu().numpy()
        # cv2_image1 = np.transpose(cv2_image1, (1, 2, 0))
        # cv2_image1 = (cv2_image1 * 255).astype(np.uint8)
        # cv2.imshow('image', cv2_image1)

        # cv2_image2 = inputs[1][0].cpu().numpy()
        # cv2_image2 = np.transpose(cv2_image2, (1, 2, 0))
        # cv2_image2 = (cv2_image2 * 255).astype(np.uint8)
        # cv2.imshow('image2', cv2_image2)

        # cv2.waitKey(0)

        optimizer.zero_grad()

        output = model(inputs[:][0], inputs[:][1], inputs[:][2], inputs[:][3])
        loss = loss_fn(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    epoch_losses.append(epoch_loss)

    print(f'Epoch {epoch}, Loss: {epoch_loss:.6f}')
    np.save('/home/ben/ml_project/weights/epoch_losses', epoch_losses)

    if (epoch % 100) == 0:
        torch.save(model.state_dict(), (WEIGHT_PATH + 'epoch_' + str(epoch) + '.pth'))
        print('Save Weights')
    
    epoch += 1

print('Finished Training')
torch.save(model.state_dict(), (WEIGHT_PATH + 'epoch_' + str(epoch) + '.pth'))
print('Save Weights')
plt.plot(epoch_losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Epoch Loss")

for i in range(len(epoch_losses)):
    if (i % 500) == 0:
        plt.annotate(str(round(epoch_losses[i], 5)), xy=(i, epoch_losses[i]))

plt.show()