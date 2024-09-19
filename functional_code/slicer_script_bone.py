import os
import slicer  # type: ignore
import numpy as np

input_folder = "/path/to/your/input/folder"
output_folder = "/path/to/your/output/folder"

for file_name in os.listdir(input_folder):
    if file_name.startswith("LOEX_") and file_name.endswith(".nii.gz"):
        
        subject_id = file_name.split("_")[1][:3]
        img = slicer.util.loadVolume(os.path.join(input_folder, f"LOEX_{subject_id}_0000.nii.gz"))
        img_array = slicer.util.arrayFromVolume(img)
        segmentation_node = slicer.util.loadVolume(os.path.join(input_folder, file_name))

        binary_labelmap_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')

        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentation_node, binary_labelmap_node, img_array)

        # 1 and 3 should be changed to 2 and 4 if the labels contain all other lower extremity bones as well
        binary_labelmap_array = slicer.util.arrayFromVolume(binary_labelmap_node)
        binary_labelmap_array_femur_A = (binary_labelmap_array == 1).astype(int)
        binary_labelmap_array_femur_B = (binary_labelmap_array == 3).astype(int)

        img_array_femur_A = np.multiply(img_array, binary_labelmap_array_femur_A)
        img_new_femur_A = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        slicer.util.updateVolumeFromArray(img_new_femur_A, img_array_femur_A)
        output_volume_path_femur_A = os.path.join(output_folder, f"LOEX_{subject_id}_femur_A.nii.gz")
        slicer.util.saveNode(img_new_femur_A, output_volume_path_femur_A)
        print(f"Saved femur_A for subject {subject_id} to {output_volume_path_femur_A}")

        img_array_femur_B = np.multiply(img_array, binary_labelmap_array_femur_B)
        img_new_femur_B = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        slicer.util.updateVolumeFromArray(img_new_femur_B, img_array_femur_B)
        output_volume_path_femur_B = os.path.join(output_folder, f"LOEX_{subject_id}_femur_B.nii.gz")
        slicer.util.saveNode(img_new_femur_B, output_volume_path_femur_B)
        print(f"Saved femur_B for subject {subject_id} to {output_volume_path_femur_B}")
