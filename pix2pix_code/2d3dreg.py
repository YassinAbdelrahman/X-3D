# Make a random DRR


import matplotlib.pyplot as plt
import torch
import numpy as np
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from diffdrr.data import read
from diffdrr.pose import convert # type: ignore
import torchio as tio # type: ignore
import os

np.random.seed(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_folder = "Data/segmented_data/artemis_femur/"
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
SDD = 2560

sub = read(
    volume=img,
    orientation="AP",
    bone_attenuation_multiplier=9.0,
)

subject = tio.Subject(sub)




true_params = {
    "sdr": SDD,
    "alpha": 0.0,
    "beta": 0.0,
    "gamma": 0.0,
    "bx": 0.0,
    "by": 2300.0,
    "bz": -290.0,
}

def pose_from_carm(sid, tx, ty, alpha, beta, gamma):
    rot = torch.tensor([[alpha, beta, gamma]])
    xyz = torch.tensor([[tx, sid, ty]])
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")


def get_initial_parameters(true_params):
    alpha = true_params["alpha"] + np.random.uniform(-np.pi / 4, np.pi / 4)
    beta = true_params["beta"] + np.random.uniform(-np.pi / 4, np.pi / 4)
    gamma = true_params["gamma"] + np.random.uniform(-np.pi / 4, np.pi / 4)
    bx = true_params["bx"] + np.random.uniform(-30.0, 30.0)
    by = true_params["by"] + np.random.uniform(-30.0, 30.0)
    bz = true_params["bz"] + np.random.uniform(-30.0, 30.0)
    pose = pose_from_carm(by, bx, bz, alpha, beta, gamma).to(device)
    rotations, translations = pose.convert("euler_angles", "ZXY")
    return rotations, translations, pose


rotations, translations, pose = get_initial_parameters(true_params)
drr = DRR(subject, sdd=SDD, height=height,width=width, delx=2).to(device)
with torch.no_grad():
    est = drr(pose)
plot_drr(est)
plt.savefig(f"Data/test/img_moving.png")
plt.show()

rotations, translations