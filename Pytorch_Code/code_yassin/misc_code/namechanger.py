import os
import re

# Define the directory containing the files
directory = "/homes/yassin/output_images/DRRs_Unk_B"

# Define the regex pattern to match the files you want to rename
pattern = re.compile(r"^img_mirrored_LOEX_(\d+)_femur_B_(\d+).png$")


# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the filename matches the pattern
    match = pattern.match(filename)
    if match:
        # Extract the YYY part
        yyy_part = match.group(1)
        id_part = match.group(2)
        # Create the new filename

        new_filename = f"img_Unk_LOEX_{yyy_part}_femur_A_{id_part}.png"
        # Construct full file paths
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f"Renamed: {filename} -> {new_filename}")

print("Renaming completed.")
