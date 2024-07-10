from ResNet34MLP8 import ResNet34MLP8
from ImagePositionDataset import ImagePositionDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np

TEST_PATH = '/home/ben/ml_project/dataset3/test/'
WEIGHT_PATH = '/home/ben/ml_project/weights/'
RESULT_CSV_NAME = '/home/ben/ml_project/Results/Dataset3_ResNet34MLP8_Adam_1441.csv'
best_result = np.empty([0, 4])

def test_model(dataloader, weight_name, draw=False):
    model = ResNet34MLP8()
    model.load_state_dict(torch.load(WEIGHT_PATH + weight_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = torch.nn.MSELoss()
    losses = []
    count = 0

    global best_result
    for image, label in dataloader:
        y_pred = model(image)
        loss = loss_fn(y_pred, label)
        losses.append(loss.item())
        if draw is True:
            print(count)
            print("Prediction: ", y_pred.tolist(), "Label: ", label.tolist())
            print(loss.item())
            count += 1

            iteration_result = np.array([y_pred.tolist(), label.tolist()]).flatten()
            best_result = np.vstack([best_result, iteration_result])

    
    if draw is True:
        plt.plot(losses)
        plt.title("Testing Loss:" + weight_name)
        plt.xlabel("Image Number")
        plt.ylabel("Loss")
    
        for i in range(len(losses)):
            plt.annotate(str(round(losses[i], 4)), xy=(i, losses[i]))

    average_loss = round((sum(losses) / len(losses)), 5)
    print("Weight Name:", weight_name, " Average Loss: ", average_loss)

    if draw is True:
        plt.text(0, plt.gca().get_ylim()[1] - 0.05, f'Average Loss: {average_loss}')
        plt.show()
        np.savetxt(RESULT_CSV_NAME, best_result, delimiter=',')
    
    return average_loss

if torch.cuda.is_available():
    device = torch.device("cuda")
    test_dataset = ImagePositionDataset(TEST_PATH, (TEST_PATH + 'labels.npy'), cuda=True)

else:
    test_dataset = ImagePositionDataset(TEST_PATH, (TEST_PATH + 'labels.npy'))

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

average_losses = []
x_axis = []
# for i in range(100, 1400, 100):
#     weight_name = 'epoch_' + str(i) + '.pth'
#     x_axis.append(i)
#     average_losses.append(test_model(test_dataloader, weight_name))

# Last One
last_weights = 1441
x_axis.append(last_weights)
average_losses.append(test_model(test_dataloader, ('epoch_' + str(last_weights) + '.pth'), draw=True))

plt.plot(x_axis, average_losses)
plt.title("Average Testing Loss")
plt.xlabel("Epoch Number")
plt.ylabel("Average Loss")
for i in range(len(average_losses)):
        plt.annotate(str(round(average_losses[i], 5)), xy=(x_axis[i], average_losses[i]))

plt.show()