import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import torch
import torchio as tio
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from diffdrr.data import read
from diffdrr.pose import convert
import nibabel as nib
import torch.nn.functional as TF
import torchvision.transforms.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dir = "/homes/yassin/E_ResearchData/CT_Scans/scalar_images_Unk_B"
# output_dir = "/homes/yassin/E_ResearchData/CT_Scans/CT_Out_test"
augmentation_factor = 40  # Number of augmented images to generate per original image
translation_range = 30  # Translation range in mm
rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
translations = torch.tensor([[0.0, 875.0, 0.0]], device=device)


def apply_augmentation(image):

    # Random translation
    translation = [
        random.uniform(-translation_range, translation_range) for _ in range(2)
    ]
    image = F.affine(image, angle=0, translate=tuple(translation), scale=1, shear=0)
    return image.squeeze(0)


# os.makedirs(output_dir, exist_ok=True)
os.makedirs("/homes/yassin/output_images/DRRs_Unk_B", exist_ok=True)
# Load NIfTI files from input directory
input_files = sorted(os.listdir(input_dir))
padding = (0, 0, 0, 0, 250, 250)
for input_file in input_files:
    if input_file == "mirrored_Art_LOEX_010_femur_B.nii.gz":
        continue

    # Assign the random value to the first element of the tensor

    i = 0
    # Load NIfTI file
    img = nib.load(os.path.join(input_dir, input_file))
    data = torch.tensor(img.get_fdata(), dtype=torch.float32)
    data = torch.flip(data, dims=[0])  # Flip along the x-axis

    data = TF.pad(data, padding)
    print(input_file)
    # Apply augmentation and save augmented images
    for i in range(augmentation_factor):
        random_value = 0.43 * (2 * torch.rand(1).item() - 1)
        # random_value = -torch.rand(1).item() * torch.pi / 6
        print(random_value)
        rotations[0, 0] = random_value
        translations[0, 0] = random_value * -1020
        # translations[0, 0] = math.sin(random_value) * -600
        # print(rotations)
        # print(translations)
        # augmented_img = apply_augmentation(data)
        aug_img = data[None, :, :, :]
        # aug_img = data[None, :, :, :]
        aug_img = tio.ScalarImage(tensor=aug_img)
        bounds = aug_img.get_bounds()
        width = int(abs(bounds[1][1] - bounds[1][0])) + 120
        height = int((abs(bounds[2][1] - bounds[2][0]) + 150) / 2)
        sub = read(
            volume=aug_img,
            orientation="AP",
            bone_attenuation_multiplier=9.0,
        )

        subject = tio.Subject(sub)

        drr = DRR(
            subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
            sdd=1020,  # Source-to-detector distance (i.e., the C-arm's focal length)
            height=height,
            width=width,  # Height of the DRR (if width is not seperately provided, the generated image is square)
            delx=2,  # Pixel spacing (in mm)
        ).to(device)
        # Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)

        plot_img = drr(
            rotations, translations, parameterization="euler_angles", convention="ZYX"
        )
        print(plot_img.shape[3], plot_img.shape[2])

        plot_drr(plot_img, ticks=False)
        # plt.grid(color="white", linestyle="--", linewidth=0.5)
        # plt.xticks(
        #     np.arange(0, plot_img.shape[3], 15)
        # )  # Set x-ticks at intervals of 10
        # plt.yticks(
        #     np.arange(0, plot_img.shape[2], 15)
        # )  # Set y-ticks at intervals of 10
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        # torch.save(plot_img, os.path.join(output_dir, f"DRR_torch_{i}"))
        print(i)
        # plt.show()
        # pose = convert(
        #     rotations, translations, parameterization="euler_angles", convention="ZXY"
        # )
        input_file = input_file.replace(".nii.gz", "")
        # imgs = []
        # n_points = [200, 400, 600, 800, 1000]
        # for n in n_points:
        #     img = drr(pose, n_points=n)
        #     imgs.append(img)
        # fig, axs = plt.subplots(1, 4, figsize=(14, 7), dpi=300, tight_layout=True)
        # img = torch.concat(imgs)
        # axs = plot_drr(img, ticks=False, title=[f"n_points={n}" for n in n_points], axs=axs)
        plt.savefig(f"/homes/yassin/output_images/DRRs_Unk_B/img_{input_file}_{i}.png")
        plt.close()

        i += 1
        torch.cuda.empty_cache()
        # out_img = augmented_img.numpy()
        # # Save augmented image
        # output_filename = os.path.splitext(input_file)[0] + f"_aug_{i}.nii.gz"

        # nifti_img = nib.Nifti1Image(out_img, np.eye(4))
        # file_path = os.path.join(output_dir, output_filename)

        # Save the image to a file
        # nib.save(nifti_img, file_path)
