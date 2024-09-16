import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from time import time
from mpl_toolkits import mplot3d
import torch
from torch.utils.data import Dataset
import nibabel as nib
from torchvision.transforms import v2
import torchio as tio


def train_test_split(input_folder, train, set_size):
    filenames = sorted(os.listdir(input_folder))
    if train:
        dataset_list = [
            torch.load(os.path.join(input_folder, filename))
            for filename in filenames[:set_size]
        ]
    else:
        dataset_list = [
            torch.load(os.path.join(input_folder, filename))
            for filename in filenames[set_size : int(1.25 * set_size)]
        ]

    dataset = np.concatenate(dataset_list, axis=0)
    dataset = np.expand_dims(dataset, axis=1)
    return dataset
    # print(len(dataset))


class AutoDataset(Dataset):
    def __init__(self, input_folder, set_size, train=True, transform=None, num_aug=0):
        self.data = train_test_split(input_folder, train, set_size)
        self.transform = transform
        self.num_aug = num_aug

    # datasets require the __getitem__ method (return a Tensor)
    def __getitem__(self, index):
        if self.num_aug == 0:
            x = torch.from_numpy(self.data[index]).float()
            return x, x
        og_idx = index // self.num_aug
        x = torch.from_numpy(self.data[og_idx]).float()
        if self.transform:
            x = self.transform(x)
        return x, x

    # datasets requrie the __len__ method (returns number of data samples)
    def __len__(self):
        if self.num_aug == 0:
            return len(self.data)
        return len(self.data) * self.num_aug


transform = tio.Compose(
    [
        # tio.RandomAffine(),
        # tio.RandomFlip(),
        tio.RandomNoise(),
        # Add more transformations as needed
    ]
)

