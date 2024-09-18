import torch
import nibabel as nib
import numpy as np

# Step 1: Load the PyTorch tensor from the .pt file
tensor_file = '/nethome/2514818/Data/final_data/img_LOEX_001_femur_A_batch_1_3d.pt'
tensor_data = torch.load(tensor_file)[0]
print(tensor_data.shape)
tensor_data = tensor_data.float().cpu()
numpy_data = tensor_data.numpy()
affine = np.eye(4)  
nifti_img = nib.Nifti1Image(numpy_data, affine)
nifti_file = 'nifttest.nii.gz'
nib.save(nifti_img, nifti_file)

print(f"NIfTI file saved as {nifti_file}")
