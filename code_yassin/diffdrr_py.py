import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchio as tio
import os
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from diffdrr.data import read
from diffdrr.pose import convert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_folder = "/homes/yassin/E_ResearchData/CT_Scans/scalar_images/"
output_folder = "/homes/yassin/E_ResearchData/DRR_tensors/"
filenames = sorted(os.listdir(input_folder))
plt.figure()
i = 0
rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
for image in filenames:
    print(image)
    img = tio.ScalarImage(os.path.join(input_folder, image))
    bounds = img.get_bounds()
    print(bounds)
    # width = int(
    #     (abs(bounds[0][1] - bounds[0][0])) + abs(bounds[1][1] - bounds[1][0] / 2)
    # )
    width = int(abs(bounds[1][1] - bounds[1][0])) + 20
    height = int((abs(bounds[2][1] - bounds[2][0]) + 150) / 2)
    sub = read(
        volume=img,
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

    img = drr(
        rotations,
        translations,
        parameterization="euler_angles",
        convention="ZXY",
    )
    plot_drr(img, ticks=False)
    # torch.save(img, os.path.join(output_folder, f"DRR_torch_{i}"))
    print(i)
    # plt.show()
    # pose = convert(
    #     rotations, translations, parameterization="euler_angles", convention="ZXY"
    # )
    i = i.replace(".nii.gz", "")
    # imgs = []
    # n_points = [200, 400, 600, 800, 1000]
    # for n in n_points:
    #     img = drr(pose, n_points=n)
    #     imgs.append(img)
    # fig, axs = plt.subplots(1, 4, figsize=(14, 7), dpi=300, tight_layout=True)
    # img = torch.concat(imgs)
    # axs = plot_drr(img, ticks=False, title=[f"n_points={n}" for n in n_points], axs=axs)
    plt.savefig(f"/homes/yassin/output_images/test/img_{i}.png")
    plt.close()
    i += 1
    torch.cuda.empty_cache()
    quit()
