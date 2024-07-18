import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from combined_dataset import Custom2D3DDataset
from RadiotoEmbNetwork_co import RadioToEmb


# def save_nifti(data, filename):
#     """Saves a numpy array as a NIfTI file."""
#     nifti_img = nib.Nifti1Image(data, np.eye(4))  # no affine transformation
#     nib.save(nifti_img, filename)


def main():
    print("Loading model...")
    model = RadioToEmb()
    model.squeeze.load_state_dict(torch.load("./squeeze.pth"))
    model.encoder.load_state_dict(torch.load("./encoder.pth"))
    model.eval()

    print("Loading dataset...")
    # test_dataset = BoneDataset(
    #     "/homes/yassin/E_ResearchData/labels_not_geo", max_samples=20
    # )
    test_dataset = Custom2D3DDataset(
        "/homes/yassin/E_ResearchData/paired_tensors_cropped",
        max_samples=5,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print("checking if output folder exists...")
    if not os.path.exists("./output/radio"):
        os.makedirs("./output/radio")

    print("Processing data...")
    for i, data in enumerate(test_dataloader):
        inputs, nii = data
        for j in range(inputs.size(1)):
            # print(radios.size(0))
            inp = inputs[0][j].unsqueeze(0)
            print(f"Processing batch {i+1}/{len(test_dataloader)}")
            with torch.no_grad():
                preds, embs = model(inp, nii)

            # class_p = preds.cpu().numpy()
            # latent_vector = embs.cpu().numpy()
            # print(class_p, latent_vector)
            # Saving the original and reconstructed as NIfTI files
            # np.savetxt(f"./output/radio/class_p_{i}_{j}.txt", class_p, fmt="%f")
            # print(f"class_p_{i}_{j}")
            # np.savetxt(f"./output/radio/latent_{i}_{j}.txt", latent_vector, fmt="%f")
            # save_nifti(original_np[0], original_filename)
            # save_nifti(reconstructed_np[0], reconstructed_filename)
            # print(f"Saved original to {original_filename}")
            # print(f"Saved reconstructed to {reconstructed_filename}")


if __name__ == "__main__":
    main()
