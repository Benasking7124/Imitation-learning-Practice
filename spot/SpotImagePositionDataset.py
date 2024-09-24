from torch.utils.data import Dataset
import numpy as np
import cv2, torch

class SpotImagePositionDataset(Dataset):
    def __init__(self, dataset_path, label_path, num_camera, transform=None, cuda=False):
        self.dataset_path = dataset_path
        self.labels = np.load(label_path).astype(np.float32)
        self.transform = transform
        self.cuda = cuda
        self.num_camera = num_camera

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        # print(index)
        time_stamp_path = self.dataset_path + format(index, '05d') + '/'

        image_pair = []
        for i in range(self.num_camera):
            image_path = time_stamp_path + str(i) + '.png'
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            image = image.transpose(2, 0, 1)
            image = image.astype(np.float32)
            image /= 255
            if self.transform:
                image = self.transform(image)
            
            image_pair.append(image)


        label = self.labels[index]

        if self.cuda is True:
            device = torch.device("cuda")
            image_tensor = torch.tensor(image_pair).to(device)
            label_tensor = torch.tensor(label).to(device)
        else:
            image_tensor = torch.tensor(image_pair)
            label_tensor = torch.tensor(label)

        return image_tensor, label_tensor