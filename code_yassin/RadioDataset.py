import os
import torch
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from collections import defaultdict
import re


class RadioDataset(Dataset):
    def __init__(self, directory_2d, directory_latent, max_samples=None):
        self.directory_2d = directory_2d
        self.files = [
            os.path.join(directory_2d, f) for f in sorted(os.listdir(directory_2d))
        ]
        print(f"Found {len(self.files)} files")

        self.dataset_2d_files = defaultdict(list)
        for f in os.listdir(directory_2d):
            if os.path.isfile(os.path.join(directory_2d, f)):
                common_id = os.path.splitext(f)[0].rsplit("_", 1)[
                    0
                ]  # assuming format like 'id_1.png', 'id_2.png'
                # print(common_id)
                self.dataset_2d_files[common_id].append(f)

        self.latents = []
        for f in os.listdir(directory_latent):
            if os.path.isfile(os.path.join(directory_latent, f)):
                common_id = os.path.splitext(f)[0]
                print(common_id)
                # common_id = "img_" + common_id
                # print(common_id)

                if common_id in self.dataset_2d_files:
                    self.latents.append(f)
        print(self.latents)
        # print(self.dataset_2d_files)
        if max_samples is not None:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pathy = sorted(list(self.dataset_2d_files))[idx]
        print(pathy)
        # common_id = os.path.splitext(pathy)[0]
        # print(common_id)
        image_list = []
        for img_name_2d in self.dataset_2d_files[pathy]:
            # print(img_name_2d)
            # image_2d = Image.open(os.path.join(self.dataset_2d_dir, img_name_2d))
            image_2d = torch.load(os.path.join(self.directory_2d, img_name_2d))
            # image_2d = remove_transparency(image_2d).convert("L")
            # image_2d = transforms.functional.to_tensor(image_2d)
            image_list.append(image_2d)
        # print(len(image_2d_list))
        # sample = {"2d_images": image_2d_list, "3d_image": image_3d}
        sample = torch.stack(image_list)
        return sample, sample


a = RadioDataset(
    "/homes/yassin/E_ResearchData/DRR_tensors",
    "/homes/yassin/output/latent_vectors",
    20,
)
print(len(a))
print(len(a[0][0]))
