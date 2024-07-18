import os
import torch
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib


class BoneDataset(Dataset):
    def __init__(self, directory, max_samples=None):
        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".nii.gz")
        ]
        print(f"Found {len(self.files)} files")
        if max_samples is not None:
            self.files = self.files[:max_samples]
        self.max_shape = self._find_max_shape()
        print(f"Max shape determined as {self.max_shape}")

    def _find_max_shape(self):
        max_shape = [0, 0, 0]  # 3D images without channel dimension
        for file_path in self.files:
            image = nib.load(file_path).get_fdata()
            max_shape = np.maximum(max_shape, image.shape)
        return max_shape

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = nib.load(path).get_fdata()
        padded_image = np.zeros(self.max_shape, dtype=np.float32)
        padded_image[: image.shape[0], : image.shape[1], : image.shape[2]] = image
        tensor_image = torch.tensor(padded_image).unsqueeze(0)  # Add channel dimension
        return tensor_image, tensor_image
