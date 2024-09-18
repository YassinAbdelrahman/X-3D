import os
import numpy as np
import nibabel as nib

def mirror_nifti_files(input_directory, output_directory):
    # Create output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.startswith("LOEX_") and filename.endswith("_femur_B.nii.gz"):
            # Load the NIfTI file
            filepath = os.path.join(input_directory, filename)

            nifti_img = nib.load(filepath)
            nifti_data = nifti_img.get_fdata()

            # Mirror the data along the desired axis (example: flip along x-axis)
            mirrored_data = np.flip(nifti_data, axis=0)  # Adjust axis as needed

            # Save the mirrored data as a new NIfTI file
            mirrored_img = nib.Nifti1Image(
                mirrored_data, nifti_img.affine, nifti_img.header
            )
            new_filename = "mirrored_Unk_" + filename
            output_filepath = os.path.join(output_directory, new_filename)
            nib.save(mirrored_img, output_filepath)

            # Also save the mirrored data as a numpy array
            print(f"Processed and saved: {filename}")
            os.remove(filepath)
            print(f"removed {filename}")


# Example usage
input_directory = "/homes/yassin/E_ResearchData/CT_Scans/scalar_images_Unk_A"
output_directory = "/homes/yassin/E_ResearchData/CT_Scans/scalar_images_Unk_B"
mirror_nifti_files(input_directory, output_directory)
