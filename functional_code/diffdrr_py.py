import matplotlib.pyplot as plt
import torch
import torchio as tio
import os
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from diffdrr.data import read

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_folder = "Data/segmented_data/artemis_femur/"
filenames = sorted(os.listdir(input_folder))
plt.figure()
i = 0
rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
translations = torch.tensor([[0.0, 2300.0, -290.0]], device=device)
image = filenames[0]
img = tio.ScalarImage(os.path.join(input_folder, image))
bounds = img.get_bounds()
width = int(abs(bounds[1][1] - bounds[1][0])) + 120
height = int(abs(bounds[2][1] - bounds[2][0])) + 150
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
img = drr(
    rotations,
    translations,
    parameterization="euler_angles",
    convention="ZXY",
)

half_height = img.shape[2] // 2
img = img[:, :, :half_height]
plot_drr(img, ticks=False)
image = image.replace(".nii.gz", "")
plt.savefig(f"Data/test/img_{image}_{i}.png")
plt.close()
i += 1
torch.cuda.empty_cache()