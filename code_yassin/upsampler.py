import nibabel as nib
import torch
import torch.nn.functional as F
import numpy as np

# Step 1: Load the NIfTI file
nifti_file = '/nethome/2514818/Data/segmented_labels/artemis_femur/LOEX_001_femur_A_Label.nii.gz'
img = nib.load(nifti_file)
data = img.get_fdata()

# Step 2: Convert the image data to a PyTorch tensor
tensor_data = torch.tensor(data, dtype=torch.float32)
tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)

downsampled_tensor = F.interpolate(tensor_data, scale_factor=0.2, mode='trilinear', align_corners=False)
downsampled_data = downsampled_tensor.squeeze().numpy()


downsampled_img = nib.Nifti1Image(downsampled_data, img.affine)
nib.save(downsampled_img, 'downsampled_image.nii.gz')
# Step 3: Upsample the tensor by 5 times
# Adding a dummy batch and channel dimension as interpolate expects a 5D tensor: (N, C, D, H, W)
tensor_data = downsampled_tensor
upsampled_tensor = F.interpolate(tensor_data, scale_factor=5, mode='trilinear', align_corners=False)

# Step 4: Remove the dummy dimensions and convert back to numpy
upsampled_data = upsampled_tensor.squeeze().numpy()
# print(upsampled_data.shape)
# Step 5: Convert the upsampled data back to NIfTI format and save it
upsampled_img = nib.Nifti1Image(upsampled_data, img.affine)
nib.save(upsampled_img, 'upsampled_image.nii.gz')
