import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import matplotlib.pyplot as plt# type: ignore
import torch# type: ignore
import torchio as tio# type: ignore
import os
import seaborn as sns #type: ignore
from tqdm import tqdm #type: ignore
import pandas as pd #type:ignore
from diffdrr.drr import DRR# type: ignore
from diffdrr.visualization import plot_drr # type: ignore
from diffdrr.data import read# type: ignore
from diffdrr.pose import convert # type: ignore
from diffdrr.registration import Registration # type: ignore
import numpy as np# type: ignore
from diffdrr.metrics import NormalizedCrossCorrelation2d # type: ignore
# Read a PIL image

image = Image.open('Data/test/xraysplit/XrayANON5UB95J1SC_full.png')
image = ImageOps.grayscale(image) 

# Define a transform to convert PIL 
# image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor()
])

# Convert the PIL image to Torch tensor
img_tensor = transform(image).unsqueeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_folder = "Data/XCT_pairs"
filenames = sorted(os.listdir(input_folder))
plt.figure()
i = 0
rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
translations = torch.tensor([[-80.0, 2400.0, 0.0]], device=device)
image = filenames[0]
img = tio.ScalarImage(os.path.join(input_folder, image))

sub = read(
    volume=img,
    orientation="AP",
    bone_attenuation_multiplier=9.0,
)

subject = tio.Subject(sub)


img_tensor = img_tensor.float()

print("pt1 done")

SDD = 2560
height = 876
width = 456
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
    # half_height = est.shape[2] // 2
    # est = est[:, :, :half_height]
plot_drr(est)
plt.savefig(f"Data/test/img_moving.png")
# plt.show()

print("pt2 done")


criterion = NormalizedCrossCorrelation2d()
img_tensor = img_tensor.float()
# criterion(img_tensor, est).item()

def optimize(
    reg: Registration,
    ground_truth,
    lr_rotations=5e-2,
    lr_translations=1e2,
    momentum=0,
    dampening=0,
    n_itrs=500,
    optimizer="sgd",  # 'sgd' or `adam`
):
    # Initialize an optimizer with different learning rates
    # for rotations and translations since they have different scales
    if optimizer == "sgd":
        optim = torch.optim.SGD(
            [
                {"params": [reg._rotation], "lr": lr_rotations},
                {"params": [reg._translation], "lr": lr_translations},
            ],
            momentum=momentum,
            dampening=dampening,
            maximize=True,
        )
        optimizer = optimizer.upper()
    elif optimizer == "adam":
        optim = torch.optim.Adam(
            [
                {"params": [reg._rotation], "lr": lr_rotations},
                {"params": [reg._translation], "lr": lr_translations},
            ],
            maximize=True,
        )
        optimizer = optimizer.title()
    else:
        raise ValueError(f"Unrecognized optimizer {optimizer}")

    params = []
    losses = [criterion(ground_truth, reg()).item()]
    for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
        # Save the current set of parameters
        alpha, beta, gamma = reg.rotation.squeeze().tolist()
        bx, by, bz = reg.translation.squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])
        if itr % 5 == 2:
            df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
            df["loss"] = losses
            print(losses[-1])
            df.to_pickle("params_adam.pickle")

        # Run the optimization loop
        optim.zero_grad()
        estimate = reg()
        loss = criterion(ground_truth, estimate)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        pbar.set_description(f"NCC = {loss.item():06f}")

        # Stop the optimization if the estimated and ground truth images are 99.9% correlated
        if loss > 0.999:
            if momentum != 0:
                optimizer += " + momentum"
            if dampening != 0:
                optimizer += " + dampening"
            tqdm.write(f"{optimizer} converged in {itr + 1} iterations")
            break

    # Save the final estimated pose
    alpha, beta, gamma = reg.rotation.squeeze().tolist()
    bx, by, bz = reg.translation.squeeze().tolist()
    params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    return df

# Keyword arguments for diffdrr.drr.DRR
kwargs = {
    "subject": subject,
    "sdd": SDD,
    "height": height,
    "width" : width,
    "delx": 2,
    "stop_gradients_through_grid_sample": True,  # Enables faster optimization
}

# def optimize_lbfgs(
#     reg: Registration,
#     ground_truth,
#     lr,
#     line_search_fn=None,
#     n_itrs=500,
# ):
#     # Initialize the optimizer and define the closure function
#     optim = torch.optim.LBFGS(reg.parameters(), lr, line_search_fn=line_search_fn)

#     def closure():
#         if torch.is_grad_enabled():
#             optim.zero_grad()
#         estimate = reg()
#         loss = -criterion(ground_truth, estimate)
#         if loss.requires_grad:
#             loss.backward()
#         return loss

#     params = []
#     losses = [closure().abs().item()]
#     for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
#         # Save the current set of parameters
#         alpha, beta, gamma = reg.rotation.squeeze().tolist()
#         bx, by, bz = reg.translation.squeeze().tolist()
#         params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

#         # Run the optimization loop
#         optim.step(closure)
#         with torch.no_grad():
#             loss = closure().abs().item()
#             losses.append(loss)
#             pbar.set_description(f"NCC = {loss:06f}")

#         # Stop the optimization if the estimated and ground truth images are 99.9% correlated
#         if loss > 0.999:
#             if line_search_fn is not None:
#                 method = f"L-BFGS + strong Wolfe conditions"
#             else:
#                 method = "L-BFGS"
#             tqdm.write(f"{method} converged in {itr + 1} iterations")
#             break

#     # Save the final estimated pose
#     alpha, beta, gamma = reg.rotation.squeeze().tolist()
#     bx, by, bz = reg.translation.squeeze().tolist()
#     params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

#     df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
#     df["loss"] = losses
#     return df

drr = DRR(**kwargs).to(device)
reg = Registration(
    drr,
    rotations.clone(),
    translations.clone(),
    parameterization="euler_angles",
    convention="ZXY",
)
params_adam = optimize(reg, img_tensor, 1e-1, 5e0, optimizer="adam")
params_adam.to_pickle("params_adam.pickle")

# del drr