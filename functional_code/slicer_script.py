import numpy as np

seg_node = slicer.util.getNode(NODELABEL)
seg_arr = slicer.util.arrayFromVolume(seg_node)
img_node = slicer.util.getNode(IMGNODE)
img_arr = slicer.util.arrayFromVolume(img_node)
img_bone_arr = np.zeros_like(img_arr)
img_bone_arr = np.multiply(img_arr, seg_arr)
slicer.util.updateVolumeFromArray(img_node,img_bone_arr)