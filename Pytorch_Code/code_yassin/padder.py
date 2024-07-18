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

folder_path = "E_ResearchData/femur/tensors_label"
input_folder_artemis = "E_ResearchData/labels/artemis_femur"
input_folder_unk = "E_ResearchData/labels/unk_femur"
input_folder = "E_ResearchData/labels_not_geo"


files = []
filenames = sorted(os.listdir(input_folder))


for filename in filenames:
    if filename.endswith(".nii.gz"):
        f = os.path.join(input_folder, filename)
        img = nib.load(f).get_fdata()
        files.append(img)

arr = np.asarray(files, dtype="object")
max_shape = np.array([arra.shape for arra in arr]).max(axis=0)
print(max_shape)
