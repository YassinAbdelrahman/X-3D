import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from P3_network import CombinedModel
from P3_dataset import Custom2D3DDataset


def save_nifti(data, filename):
    """Saves a numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, np.eye(4))  # no affine transformation
    nib.save(nifti_img, filename)


def main():
    print("Loading model...")
    model = CombinedModel((120,72,236))
    model.alexnet.load_state_dict(torch.load("./combined_alex.pth"))
    # model.alex.load_state_dict(torch.load("./combined_alex.pth"))
    model.encoder.load_state_dict(torch.load("./combined_encoder.pth"))
    model.decoder.load_state_dict(torch.load("./combined_decoder.pth"))

    model.eval()

    print("Loading dataset...")
    # test_dataset = BoneDataset(
    #     "/homes/yassin/E_ResearchData/labels_not_geo", max_samples=20
    # )
    test_dataset = Custom2D3DDataset(
        "/nethome/2514818/Data/final_data",
        max_samples=1,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print("checking if output folder exists...")
    if not os.path.exists("./output/combined"):
        os.makedirs("./output/combined")

    print("Processing data...")
    for i, data in enumerate(test_dataloader):
        radios, image = data
        print(radios.size())
        for j in range(2):
            # print(radios.size(0))
            radio = radios[0][j].unsqueeze(0)
            print(radio.shape)
            print(f"Processing batch {i+1}/{len(test_dataloader)}")
            with torch.no_grad():
                _, reconstructed = model(radio)
            original_np = image.squeeze(1).cpu().numpy()
            # print(original_np.shape)
            # print(reconstructed.shape)
            reconstructed_np = reconstructed.squeeze(1).cpu().numpy()
            print(reconstructed_np[0].shape)
            reconstructed_np[(reconstructed_np > 0.05) & (reconstructed_np < 1)] = 1

            # print(reconstructed_np.shape)
            # class_p = class_pred.cpu().numpy()
            # latent_vector = latent.cpu().numpy()
            # print(class_p, latent_vector)
            # Saving the original and reconstructed as NIfTI files
            # np.savetxt(f"./output/combined/class_p_{i}_{j}_comb.txt", class_p, fmt="%f")
            # np.savetxt(
            #     f"./output/combined/latent_{i}_{j}_comb.txt", latent_vector, fmt="%f"
            # )
            original_filename = f"./output/combined/original_{i}_{j}.nii.gz"
            reconstructed_filename = f"./output/combined/reconstructed_{i}_{j}.nii.gz"
            save_nifti(original_np[0], original_filename)
            save_nifti(reconstructed_np[0], reconstructed_filename)
            print(f"Saved original to {original_filename}")
            print(f"Saved reconstructed to {reconstructed_filename}")


if __name__ == "__main__":
    main()
