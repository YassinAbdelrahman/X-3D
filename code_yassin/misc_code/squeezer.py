import os
import torch

# Define the folder containing the tensors
folder_path = '/nethome/2514818/Data/final_data'

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('_2d.pt'):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Load the tensor
        tensor = torch.load(file_path)
        
        # Check the shape to ensure it matches [1, 10, 1, 388, 302]
        if tensor.shape == (1, 10, 1, 388, 302):
            # Remove the first dimension
            tensor = tensor.squeeze(0)
            
            # Save the tensor back to the same file
            torch.save(tensor, file_path)
            print(f'Successfully reshaped and saved: {filename}')
        else:
            print(f'Skipped {filename}: Unexpected shape {tensor.shape}')