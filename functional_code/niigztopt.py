import os
import nibabel as nib
import torch

def convert_nii_to_tensor(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all nii.gz files in the input folder
    nii_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]
    
    for nii_file in nii_files:
        # Load the NIfTI file
        nii_path = os.path.join(input_folder, nii_file)
        nii_data = nib.load(nii_path)
        image_array = nii_data.get_fdata()
        
        # Convert the numpy array to a torch tensor
        tensor = torch.tensor(image_array, dtype=torch.float32)
        
        # Save the tensor as a .pt file
        output_path = os.path.join(output_folder, nii_file.replace('.nii.gz', '.pt'))
        torch.save(tensor, output_path)
        print(f"Converted {nii_file} to {output_path}")

# Example usage
input_folder = 'Data/niigz_02'
output_folder = 'Data/tensors_02'
convert_nii_to_tensor(input_folder, output_folder)
