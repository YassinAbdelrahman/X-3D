import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchio as tio

def train_test_split(input_folder, train, max_samples):
    filenames = sorted(os.listdir(input_folder))
    if train:
        max_samples = int(0.9*max_samples)
        dataset_list = [
            torch.load(os.path.join(input_folder, filename))
            for filename in filenames[:max_samples]
        ]
    else:
        min_samples = int(0.9*max_samples)
        dataset_list = [
            torch.load(os.path.join(input_folder, filename))
            for filename in filenames[min_samples:max_samples]
        ]

    dataset = np.concatenate(dataset_list, axis=0)
    dataset = np.expand_dims(dataset, axis=1)
    return dataset


class AEDataset(Dataset):
    def __init__(self, input_folder, set_size, train=True, transform=None, num_aug=0):
        self.data = train_test_split(input_folder, train, set_size)
        self.transform = transform
        self.num_aug = num_aug

    def __getitem__(self, index):
        if self.num_aug == 0:
            x = torch.from_numpy(self.data[index]).float()
            return x, x
        og_idx = index // self.num_aug
        x = torch.from_numpy(self.data[og_idx]).float()
        if self.transform:
            x = self.transform(x)
        return x, x

    def __len__(self):
        if self.num_aug == 0:
            return len(self.data)
        return len(self.data) * self.num_aug


transform = tio.Compose(
    [
        tio.RandomNoise(),    ]
)

