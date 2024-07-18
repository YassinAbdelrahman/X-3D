from PIL import Image, ImageChops
import os


def remove_whitespace(image):
    # Convert the image to grayscale
    gray_image = image.convert("L")

    # Create a binary mask where white pixels are 255 and non-white are 0
    bg = Image.new("L", gray_image.size, 255)
    diff = ImageChops.difference(gray_image, bg)
    bbox = diff.getbbox()

    if bbox:
        return image.crop(bbox)
    else:
        return image


def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            cropped_image = remove_whitespace(image)

            output_path = os.path.join(output_folder, filename)
            cropped_image.save(output_path)
        print(f"Processed and saved: {filename}")


# Example usage
input_folder = "/homes/yassin/output_images/DRRs_Unk_B"
output_folder = "/homes/yassin/output_images/DRRs_Unk_B/cropped"

process_images(input_folder, output_folder)
