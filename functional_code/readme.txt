radio_preprocessor: creates DRRs and 40 augmentations of each DRR
upsampler: upsamples a downsampled .nii.gz file (like the output of our network)
remove_white: removes the white borders from the DRRs
pttoniigz:convert .pt files into .nii.gz files
AE_preprocessor: preprocesses the CT .nii.gz: pads them all to the same shape and downsamples them to 120x72x236, which is about 0.2x (in batches of 30 because of memory issues
)
pngtotensor: converts DRRs from .png to .pt 
niigztopt: converts .nii.gz files to .pt
mirror: mirrors .nii.gz files, used for flipping bones to the same orientation
diffdrr_py: creates singular DRRs
batcher: creates batches of DRRs in .pt format, since we had 40 augmentations of each DRR we did 4 batches of 10 images
