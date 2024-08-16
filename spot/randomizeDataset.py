import random, os
import numpy as np

DATASET_PATH = '/home/ben/ml_project/spot/dataset/'
TRAIN_PATH = DATASET_PATH + 'train/'
VALID_PATH = DATASET_PATH + 'valid/'
TEST_PATH = DATASET_PATH + 'test/'

DATA_NUMBER = 1000
TRAIN_NUMBER = int(DATA_NUMBER * 0.8)
VALID_NUMBER = int(TRAIN_NUMBER + DATA_NUMBER * 0.1)

if not os.path.exists(TRAIN_PATH):
    os.mkdir(TRAIN_PATH)

if not os.path.exists(VALID_PATH):
    os.mkdir(VALID_PATH)

if not os.path.exists(TEST_PATH):
    os.mkdir(TEST_PATH)

labels = np.load(DATASET_PATH + 'labels.npy')

order = list(range(1001))

random.shuffle(order)

train_labels = []
for i in range(TRAIN_NUMBER):
    index = order[i]
    folder_name = DATASET_PATH + format(index, '05d')
    new_folder_name = TRAIN_PATH + format(i, '05d')
    train_labels.append(labels[index])

    os.rename(folder_name, new_folder_name)   # Move the file to train folder

np.save((TRAIN_PATH + 'labels'), train_labels)


valid_labels = []
for i in range(VALID_NUMBER - TRAIN_NUMBER):
    index = order[i + TRAIN_NUMBER]
    folder_name = DATASET_PATH + format(index, '05d')
    new_folder_name = VALID_PATH + format(i, '05d')
    valid_labels.append(labels[index])

    os.rename(folder_name, new_folder_name)

np.save((VALID_PATH + 'labels'), valid_labels)


test_labels = []
for i in range(DATA_NUMBER - VALID_NUMBER + 1):
    index = order[i + VALID_NUMBER]
    folder_name = DATASET_PATH + format(index, '05d')
    new_folder_name = TEST_PATH + format(i, '05d')
    test_labels.append(labels[index])

    os.rename(folder_name, new_folder_name)

np.save((TEST_PATH + 'labels'), test_labels)