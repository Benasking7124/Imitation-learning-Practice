from ResNet18MLP import ResNet18MLP
from ImagePositionDataset import ImagePositionDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

TEST_PATH = '/home/ben/ml_project/dataset3/test/'

model = ResNet18MLP()
model.load_state_dict(torch.load('/home/ben/ml_project/weights/model_epoch_1803.pth'))

test_dataset = ImagePositionDataset(TEST_PATH, (TEST_PATH + 'labels.npy'))
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

loss_fn = torch.nn.MSELoss()
losses = []
count = 0
for image, label in test_dataloader:
    y_pred = model(image)
    loss = loss_fn(y_pred, label)
    losses.append(loss.item())
    print(count)
    print("Prediction: ", y_pred, "Label: ", label)
    print(loss.item())
    count += 1

plt.plot(losses)
plt.title("Testing Loss")
plt.xlabel("Image Number")
plt.ylabel("Loss")

for i in range(len(losses)):
    plt.annotate(str(round(losses[i], 2)), xy=(i, losses[i]))

average_loss = round((sum(losses) / len(losses)), 6)
plt.text(0, plt.gca().get_ylim()[1] - 0.05, f'Average Loss: {average_loss}')
print("Average Loss: ", average_loss)

plt.show()