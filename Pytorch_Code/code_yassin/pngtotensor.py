import os
from PIL import Image
import torch
from torchvision import transforms

# Define the paths
input_dir = (
    "/homes/yassin/output_images/DRRs_Unk_B"  # Path to the folder containing images
)
output_dir = "/homes/yassin/E_ResearchData/DRR_tensors"  # Path to the folder where tensors will be saved

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the image transformation
transform = transforms.ToTensor()

# Iterate over each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # Load the image
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path).convert("L")  # Convert to grayscale if needed

        # Transform the image to a tensor
        tensor = transform(image)

        # Save the tensor
        tensor_filename = filename.replace(".png", ".pt")
        tensor_path = os.path.join(output_dir, tensor_filename)
        torch.save(tensor, tensor_path)
    print(f"converted {filename}")

print("Conversion completed!")
