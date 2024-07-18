import os
import torch

# Define the path to the folder containing the tensors
tensor_dir = "/homes/yassin/E_ResearchData/DRR_tensors_cropped"

# Iterate over each file in the tensor directory
for filename in os.listdir(tensor_dir):
    if filename.endswith(".pt"):
        # Load the tensor
        tensor_path = os.path.join(tensor_dir, filename)
        tensor = torch.load(tensor_path)

        # Print the size of the tensor
        print(f" {tensor.size()}")

print("Size check completed!")
