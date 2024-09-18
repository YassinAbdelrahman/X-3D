from matplotlib import pyplot as plt
import os
import random
import torch
import torchio as tio
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from diffdrr.data import read
import nibabel as nib
import torch.nn.functional as TF
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dir = "Data/segmented_data/unk_femur/"
augmentation_factor = 40  # Number of augmented images to generate per original image
translation_range = 30  # Translation range in mm
rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
translations = torch.tensor([[0.0, 2150.0, -300.0]], device=device)


# Load NIfTI files from input directory
input_files = sorted(os.listdir(input_dir))

# input_files = input_files[76:]


padding = (0, 0, 0, 0, 250, 250)
for input_file in input_files:
    i = 0
    # Load NIfTI file
    img = nib.load(os.path.join(input_dir, input_file))
    data = torch.tensor(img.get_fdata(), dtype=torch.float32)
    if "A" in input_file:
        data = torch.flip(data, dims=[0])  # Flip along the x-axis

    data = TF.pad(data, padding)
    aug_img = data[None, :, :, :]
    aug_img = tio.ScalarImage(tensor=aug_img)
    bounds = aug_img.get_bounds()
    width = int(abs(bounds[1][1] - bounds[1][0])) + 150
    height = int((abs(bounds[2][1] - bounds[2][0]) + 150) )
    sub = read(
        volume=aug_img,
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

    # Apply augmentation and save augmented images
    for i in range(augmentation_factor):
        random_value = 0.43 * (2 * torch.rand(1).item() - 1)
        rotations[0][2] =   random.uniform(-5, 5) * math.pi / 180
        rotations[0, 0] = random_value

        # *-300 makes it so the bone is still in the image
        translations[0, 0] = random_value * -300

        # Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
        plot_img = drr(
            rotations, translations, parameterization="euler_angles", convention="ZYX"
        )

        print(plot_img.shape[3], plot_img.shape[2]) 
        half_height = int(plot_img.shape[2] *0.6)
        plot_img = plot_img[:, :, :half_height]
        plot_drr(plot_img, ticks=False)
        input_file = input_file.replace(".nii.gz", "")
        plt.savefig(f"Data/DRRs_real/img_Unk_{input_file}_{i}.png")
        plt.close()

        i += 1
        torch.cuda.empty_cache()
        
