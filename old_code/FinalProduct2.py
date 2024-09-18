from FinalNetwork import FinalNet
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import nibabel as nib
import os

model = FinalNet()
model.eval()


# Preprocessing function for input images
def preprocess_image(image_path, target_size=(302, 380)):
    image = Image.open(image_path).convert("L")

    width, height = image.size
    pad_left = 0
    pad_top = 0
    pad_right = max(0, target_size[0] - width)
    pad_bottom = max(0, target_size[1] - height)
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    # Define the image transformation with padding
    transform = transforms.Compose(
        [
            transforms.Pad(padding, fill=0),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension


# Function to save the output tensor as nii.gz
def save_as_nii(tensor, output_path):
    array = (
        tensor.squeeze().detach().cpu().numpy()
    )  # Remove batch and channel dimensions
    nii_image = nib.Nifti1Image(array, np.eye(4))
    nib.save(nii_image, output_path)


# Example usage
input_image_path = (
    "/homes/yassin/output_images/DRRs_Art_A/cropped/img_Art_LOEX_003_femur_A_0.png"
)
output_nii_path = "/homes/yassin/final_output/FinalOne.nii.gz"

# Load and preprocess the input image
input_tensor = preprocess_image(input_image_path)
# input_tensor = torch.load(
#     "/homes/yassin/E_ResearchData/paired_tensors_cropped/img_Art_LOEX_002_femur_A_2d.pt"
# )
print(input_tensor.shape)
model.squeeze.load_state_dict(torch.load("./combined_squeeze.pth"))
model.decoder.load_state_dict(torch.load("./combined_decoder.pth"))
# Run the model
with torch.no_grad():
    output_tensor = model(input_tensor)

# Save the output tensor as a nii.gz file
save_as_nii(output_tensor[0][0], output_nii_path)
print("Conversion completed!")
