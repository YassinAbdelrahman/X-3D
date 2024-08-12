import matplotlib.pyplot as plt# type: ignore
import torch# type: ignore
import torchio as tio# type: ignore
import os
import seaborn as sns #type: ignore
from tqdm import tqdm #type: ignore
import pandas as pd #type:ignore


import random
import math
from diffdrr.drr import DRR# type: ignore
from diffdrr.visualization import plot_drr # type: ignore
from diffdrr.data import read# type: ignore
from diffdrr.pose import convert # type: ignore
from diffdrr.registration import Registration # type: ignore
import numpy as np# type: ignore
from diffdrr.metrics import NormalizedCrossCorrelation2d # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_folder = "Data/XCT_pairs/ANON5UB95J1SC.nii.gz"
# output_folder = "/Data/segmented_data/artemis_femur"
filenames = sorted(os.listdir(input_folder))
plt.figure()
i = 0
rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
translations = torch.tensor([[0.0, 2300.0, -290.0]], device=device)
image = filenames[0]
img = tio.ScalarImage(os.path.join(input_folder, image))
bounds = img.get_bounds()
width = int(abs(bounds[1][1] - bounds[1][0])) + 120
height = int((abs(bounds[2][1] - bounds[2][0]) + 150) ) 
print(width, height)
sub = read(
    volume=img,
    orientation="AP",
    bone_attenuation_multiplier=9.0,
)

subject = tio.Subject(sub)

drr = DRR(
    subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
    sdd=2560,  # Source-to-detector distance (i.e., the C-arm's focal length)
    height=height,
    width=width,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    delx=2,  # Pixel spacing (in mm)
).to(device)
# Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)


# print(image)
    
    
    # rotations[0][1] =   random.uniform(-5, 5) * math.pi / 180
    # print(bounds)
    # width = int(
    #     (abs(bounds[0][1] - bounds[0][0])) + abs(bounds[1][1] - bounds[1][0] / 2)
    # )
    
    
img = drr(
    rotations,
    translations,
    parameterization="euler_angles",
    convention="ZXY",
)
# half_height = img.shape[2] // 2
# img = img[:, :, :half_height]
plot_drr(img, ticks=False)
# torch.save(img, os.path.join(output_folder, f"DRR_torch_{i}"))
# print(img)
# plt.show()
# pose = convert(
#     rotations, translations, parameterization="euler_angles", convention="ZXY"
# )
ground_truth = img
image = image.replace(".nii.gz", "")
# imgs = []
# n_points = [200, 400, 600, 800, 1000]
# for n in n_points:
#     img = drr(pose, n_points=n)
#     imgs.append(img)
# fig, axs = plt.subplots(1, 4, figsize=(14, 7), dpi=300, tight_layout=True)
# img = torch.concat(imgs)
# axs = plot_drr(img, ticks=False, title=[f"n_points={n}" for n in n_points], axs=axs)
plt.savefig(f"Data/test/img_{image}_{i}.png")
plt.show()
