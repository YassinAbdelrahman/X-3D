import torch
import nibabel as nib
import numpy as np

# Step 1: Load the PyTorch tensor from the .pt file
tensor_file = '/nethome/2514818/Data/final_data/img_LOEX_001_femur_A_batch_1_3d.pt'
tensor_data = torch.load(tensor_file)[0]
print(tensor_data.shape)
# Ensure the tensor is in CPU and float32 format (common in medical imaging)
tensor_data = tensor_data.float().cpu()

# Step 2: Convert the tensor to a NumPy array
numpy_data = tensor_data.numpy()

# Step 3: Create a NIfTI image from the NumPy array
# The affine matrix should define the transformation from voxel indices to world space.
# Here we use an identity matrix, but you should replace it with the appropriate matrix.
affine = np.eye(4)  
nifti_img = nib.Nifti1Image(numpy_data, affine)

# Step 4: Save the NIfTI image to a file
nifti_file = 'nifttest.nii'
nib.save(nifti_img, nifti_file)

print(f"NIfTI file saved as {nifti_file}")
