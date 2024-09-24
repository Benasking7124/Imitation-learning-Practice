from FourResNet34MLP10_r import FourResNet34MLP10_r
from SpotImagePositionDataset import SpotImagePositionDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

TEST_PATH = '/home/ben/ml_project/spot/dataset/test/'
# TEST_PATH = '/home/ben/ml_project/spot/dataset/valid/'
# TEST_PATH = '/home/ben/ml_project/spot/dataset/train/'
WEIGHT_PATH = '/home/ben/ml_project/spot/weights/'
AVERAGE_LOSS_CSV_NAME = '/home/ben/ml_project/spot/Results/average_loss.csv'
RESULT_CSV_NAME = '/home/ben/ml_project/spot/Results/epoch_630.csv'
best_result = np.empty([0, 2])

def test_model(dataloader, weight_name, draw=False):
    model = FourResNet34MLP10_r()
    model.load_state_dict(torch.load(WEIGHT_PATH + weight_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loss_fn = torch.nn.MSELoss()
    losses = []
    count = 0

    global best_result
    for images, label in dataloader:
        # image1 = images[:, 0, :, :]
        # cv2_image1 = image1[0].cpu().numpy()
        # cv2_image1 = np.transpose(cv2_image1, (1, 2, 0))
        # cv2_image1 = (cv2_image1 * 255).astype(np.uint8)
        # cv2.imshow('image', cv2_image1)

        # cv2.waitKey(0)

        y_pred = model(images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :], images[:, 3, :, :])
        loss = loss_fn(y_pred, label)
        losses.append(loss.item())
        if draw is True:
            print(count)
            print("Prediction: ", y_pred.tolist(), "Label: ", label.tolist())
            print(loss.item())
            count += 1

            iteration_result = np.array([y_pred.tolist()[0], label.tolist()]).flatten()
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
    test_dataset = SpotImagePositionDataset(TEST_PATH, (TEST_PATH + 'labels.npy'), 4, cuda=True)

else:
    test_dataset = SpotImagePositionDataset(TEST_PATH, (TEST_PATH + 'labels.npy'), 4)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

average_losses = []
x_axis = []
last_weights = 79
# for i in range(20, last_weights, 20):
#     weight_name = 'epoch_' + str(i) + '.pth'
#     x_axis.append(i)
#     average_losses.append(test_model(test_dataloader, weight_name))

# Last One
x_axis.append(last_weights)
average_losses.append(test_model(test_dataloader, ('epoch_' + str(last_weights) + '.pth'), draw=True))

# plt.plot(x_axis, average_losses)
# plt.title("Average Testing Loss")
# plt.xlabel("Epoch Number")
# plt.ylabel("Average Loss")
# for i in range(len(average_losses)):
#         plt.annotate(str(round(average_losses[i], 5)), xy=(x_axis[i], average_losses[i]))

# np.savetxt(AVERAGE_LOSS_CSV_NAME, np.array(average_losses), delimiter=',')

# plt.show()