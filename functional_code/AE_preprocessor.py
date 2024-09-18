import numpy as np
import os
import torch
import nibabel as nib

folder_path = "Data/tensors_02"
input_folder_artemis = "Data/segmented_labels/artemis_femur"
input_folder_unk = "Data/segmented_labels/unk_femur"

filenames = sorted(os.listdir(input_folder_unk))
s_counter = 0
for _ in range(10):
    files = [] 

    # in batches of 30 because of RAM
    for filename in filenames[s_counter:s_counter+30]:
        f = os.path.join(input_folder_unk, filename)
        img = nib.load(f).get_fdata()
        files.append(img)

    arr = np.asarray(files, dtype="object")
    max_shape = np.array([arra.shape for arra in arr]).max(axis=0)
    padded_arrays = []
    # padding needed for interpolate function
    for arra in arr:
        padded_arr = np.pad(
            arra,
            [
                (0, max_shape[0] - arra.shape[0]),
                (0, max_shape[1] - arra.shape[1]),
                (0, max_shape[2] - arra.shape[2]),
            ],
            mode="edge",
        )
        padded_arrays.append(padded_arr)

    # Convert the list of padded arrays to a NumPy array
    padded_arrays = np.array(padded_arrays)
    padded_torch = np.expand_dims(padded_arrays, axis=1)
    padded_torch = torch.from_numpy(padded_torch).float()

    padded_torch = torch.nn.functional.interpolate(
        padded_torch, size=(120, 72, 236), mode="nearest"
    )
    for sample in padded_torch:
        outp_name = filenames[s_counter]
        outp_name = outp_name[:-7]
        torch.save(sample, os.path.join(folder_path, f"torch_Unk_{outp_name}.pt"))
        s_counter += 1