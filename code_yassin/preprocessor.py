import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from time import time
from mpl_toolkits import mplot3d
import torch
from torch.utils.data import Dataset
import sys
import nibabel as nib
from torchvision import transforms

folder_path = "Data/tensors_02"
input_folder_artemis = "Data/segmented_labels/artemis_femur"
input_folder_unk = "Data/segmented_labels/unk_femur"
# input_folder = "E_ResearchData/labels_not_geo"


# files = []
filenames = sorted(os.listdir(input_folder_unk))
s_counter = 0
for batch_thing in range(10):
    files = []
    for filename in filenames[s_counter:s_counter+30]:
        f = os.path.join(input_folder_unk, filename)
        img = nib.load(f).get_fdata()
        # print(filename, img.shape)
        files.append(img)

    # for filename in os.listdir(input_folder_unk):
    #     if filename.endswith(".nii.gz"):
    #         f = os.path.join(input_folder_unk, filename)
    #         img = nib.load(f).get_fdata()
    #         files.append(img)

    # files = [file for file in files if not any(value > 900 for value in file.shape)]


    arr = np.asarray(files, dtype="object")
    max_shape = np.array([arra.shape for arra in arr]).max(axis=0)
    print(max_shape)
    # Initialize an empty list to store the padded arrays
    padded_arrays = []
    print("ok1")
    for arra in arr:
        # Pad the current array and append it to the list
        padded_arr = np.pad(
            arra,
            [
                (0, max_shape[0] - arra.shape[0]),
                (0, max_shape[1] - arra.shape[1]),
                (0, max_shape[2] - arra.shape[2]),
            ],
            mode="edge",
        )
        padded_arrays.append(padded_arr)

    # Convert the list of padded arrays to a NumPy array
    padded_arrays = np.array(padded_arrays)
    print("ok2")
    padded_torch = np.expand_dims(padded_arrays, axis=1)
    padded_torch = torch.from_numpy(padded_torch).float()
    print(padded_torch.shape)

    # for sample in padded_torch:
    #     sample = torch.nn.functional.interpolate(sample, scale_factor=0.1, mode="nearest")
    #     print(sample.shape)
    #     torch.save(sample, os.path.join(folder_path, f"torch_sample_{s_counter}"))
    #     s_counter += 1
    padded_torch = torch.nn.functional.interpolate(
        padded_torch, size=(120, 72, 236), mode="nearest"
    )
    # print(padded_torch.shape)
    for sample in padded_torch:
        # print(sample.shape)
        outp_name = filenames[s_counter]
        outp_name = outp_name[:-7]
        torch.save(sample, os.path.join(folder_path, f"torch_Unk_{outp_name}.pt"))
        s_counter += 1


# # datasets inheret from torch.utils.data.Dataset
# class ExampleDataset(Dataset):
#     def __init__(self, split="train", transform=False, max_shape=None):

#         self.split = split
#         if transform:
#             self.transform = True
#         if max_shape:
#             self.max_shape = max_shape
#         if split == "train":
#             self.data = pre_pro("train", 30)
#         else:
#             self.data = pre_pro("test", 30, max_shape)
#         # print(np.shape(self.data))

#     # datasets require the __getitem__ method (return a Tensor)
#     def __getitem__(self, index):
#         x = torch.from_numpy(self.data[index]).float()

#         if self.transform:
#             x = torch.nn.functional.interpolate(
#                 x.unsqueeze(0), scale_factor=0.1, mode="nearest"
#             ).squeeze(0)
#         # print(x[0].shape)
#         return x

#     # datasets requrie the __len__ method (returns number of data samples)
#     def __len__(self):
#         return len(self.data)


# x = ExampleDataset(split="train", transform=True)
# xshape = x[0][0].shape
# x2 = ExampleDataset(split="test", transform=True, max_shape=xshape)
# for example in x:
#     print(example.shape)
# for example in x2:
#     print(example.shape)
# print(len(x))
# print(len(x2))

# # print(len(ExampleDataset(split="train", transform=True)))
# y = x[0][0].numpy()
# y2 = x2[0][0].numpy()
# ni_img = nib.Nifti1Image(y, np.eye(4))
# nib.save(ni_img, f"output_images/tester4.nii.gz")
# with np.printoptions(threshold=np.inf):
#     print(y)
# with np.printoptions(threshold=np.inf):
#     print(y2)
