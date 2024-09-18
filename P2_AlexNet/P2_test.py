import nibabel as nib
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from P2_dataset import Custom2D3DDataset
from P2_network import RadioToEmb


def save_nifti(data, filename):
    """Saves a numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, np.eye(4))
    nib.save(nifti_img, filename)


def main():
    print("Loading model...")
    model = RadioToEmb((120,72,236))
    model.alex.load_state_dict(torch.load("/nethome/2514818/X-3D/output/radio_alex_epoch_14.pth"))
    model.decoder.load_state_dict(torch.load("/nethome/2514818/X-3D/AE_checkpoints/decoder_epoch_186.pth"))
    model.eval()

    print("Loading dataset...")
    test_dataset = Custom2D3DDataset(
        "/nethome/2514818/Data/final_data",
        max_samples=3,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print("checking if output folder exists...")
    if not os.path.exists("./output/radio"):
        os.makedirs("./output/radio")

    print("Processing data...")
    for i, data in enumerate(test_dataloader):
        inputs, img = data
        for j in range(inputs.size(1)):
            inp = inputs[0][j].unsqueeze(0)
            print(f"Processing batch {i+1}/{len(test_dataloader)}")
            with torch.no_grad():
                _, outputs = model(inp)
            original_np = img.squeeze(1).cpu().numpy()
            print(original_np.shape)
            reconstructed_np = outputs.squeeze(1).cpu().numpy()
            print(reconstructed_np.shape)
            original_filename = f"./output/radio/original_{i}.nii.gz"
            reconstructed_filename = f"./output/radio/reconstructed_{i}.nii.gz"
            save_nifti(original_np[0], original_filename)
            save_nifti(reconstructed_np[0], reconstructed_filename)
            print(f"Saved original to {original_filename}")
            print(f"Saved reconstructed to {reconstructed_filename}")

if __name__ == "__main__":
    main()
