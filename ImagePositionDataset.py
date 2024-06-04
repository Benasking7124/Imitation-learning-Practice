from torch.utils.data import Dataset
import numpy as np
import cv2, torch

class ImagePositionDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None, cuda=False):
        self.images_path = image_path
        self.labels = np.load(label_path).astype(np.float32)
        self.transform = transform
        self.cuda = cuda

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        image_path = self.images_path + format(index, '05d') + '.png'
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)
        image /= 255

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        if self.cuda is True:
            device = torch.device("cuda")
            image_tensor = torch.tensor(image).to(device)
            label_tensor = torch.tensor(label).to(device)
        else:
            image_tensor = torch.tensor(image)
            label_tensor = torch.tensor(label)

        return image_tensor, label_tensor