from Autoencoder3D_co import Autoencoder3D
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from AE_Dataset import AutoDataset
from torch.nn.functional import interpolate


def save_nifti(data, filename):
    """Saves a numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, np.eye(4))  # no affine transformation
    nib.save(nifti_img, filename)


def main():
    print("Loading model...")
    model = Autoencoder3D()
    model.encoder.load_state_dict(torch.load("./encoder.pth"))
    model.decoder.load_state_dict(torch.load("./decoder.pth"))

    model.eval()

    print("Loading dataset...")
    # test_dataset = BoneDataset(
    #     "/homes/yassin/E_ResearchData/labels_not_geo", max_samples=20
    # )
    test_dataset = AutoDataset(
        "/homes/yassin/E_ResearchData/femur/tensors_label_02", train=False, set_size=200
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=0
    )

    print("checking if output folder exists...")
    if not os.path.exists("./output"):
        os.makedirs("./output")

    print("Processing data...")
    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        print(f"Processing batch {i+1}/{len(test_dataloader)}")
        with torch.no_grad():
            reconstructed = model(inputs)
        original_np = inputs.squeeze(1).cpu().numpy()
        reconstructed = interpolate(
            reconstructed, size=inputs.shape[2:], mode="nearest"
        )

        reconstructed_np = reconstructed.squeeze(1).cpu().numpy()
        print(reconstructed_np[0].shape)
        reconstructed_np[(reconstructed_np > 0.05) & (reconstructed_np < 1)] = 1
        # Saving the original and reconstructed as NIfTI files
        original_filename = f"./output/original_{i}.nii.gz"
        reconstructed_filename = f"./output/reconstructed_{i}.nii.gz"
        save_nifti(original_np[0], original_filename)
        save_nifti(reconstructed_np[0], reconstructed_filename)
        print(f"Saved original to {original_filename}")
        print(f"Saved reconstructed to {reconstructed_filename}")


if __name__ == "__main__":
    main()
