import os
import shutil


def move_non_mirrored_files(src_folder, dst_folder):
    # Ensure the destination folder exists
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Iterate over all files in the source folder
    for filename in os.listdir(src_folder):
        # Check if "mirrored" is not in the filename
        if "mirrored" not in filename:
            # Construct full file paths
            src_file = os.path.join(src_folder, filename)
            dst_file = os.path.join(dst_folder, filename)

            # Move the file
            shutil.move(src_file, dst_file)
            print(f"Moved: {src_file} -> {dst_file}")


# Define the source and destination folders
src_folder = "/homes/yassin/output_images/DRRs_Art_B"
dst_folder = "/homes/yassin/output_images/DRRs_Unk_A"

# Call the function to move the files
move_non_mirrored_files(src_folder, dst_folder)
