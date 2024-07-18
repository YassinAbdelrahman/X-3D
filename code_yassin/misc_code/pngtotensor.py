import os
from PIL import Image
import torch
from torchvision import transforms

# Define the paths
input_dir = "/homes/yassin/output_images/DRRs_Art_B/cropped"  # Path to the folder containing images
output_dir = "/homes/yassin/E_ResearchData/DRR_tensors_cropped"  # Path to the folder where tensors will be saved

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the target size
target_size = (302, 388)

# Iterate over each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # Load the image
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path).convert("L")  # Convert to grayscale if needed

        # Calculate padding
        width, height = image.size
        pad_left = 0
        pad_top = 0
        pad_right = max(0, target_size[0] - width)
        pad_bottom = max(0, target_size[1] - height)
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        # Define the image transformation with padding
        transform = transforms.Compose(
            [transforms.Pad(padding, fill=0), transforms.ToTensor()]
        )

        # Transform the image to a tensor
        tensor = transform(image)

        # Save the tensor
        tensor_filename = filename.replace(".png", ".pt")
        tensor_path = os.path.join(output_dir, tensor_filename)
        torch.save(tensor, tensor_path)

        print(f"Converted {filename}")
print("Padding and conversion completed!")
