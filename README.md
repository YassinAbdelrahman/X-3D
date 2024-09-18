# X-3D
This project is meant to generate 3D models from a singular 2D radiograph. For this, an autoencoder (AE) is trained on 3D labels of bones generated from real CT scans. Following this, a second network takes Digitally Reconstructed Radiographs which are created from the previous CT scans (CT-DRRs) and generates a 3D model of the same bone. A third network combines the two previous network to increase the accuracy, resulting in 3D models generated from singular 2D radiographs.

Following is summation of the required libraries and a step-by-step breakdown of preprocessing and using data for the X-3D networks. The bone we focussed on in this project was the thighbone (femur). 

## Required libraries
The following libraries are required for running all the code:

```
nibabel
torch
numpy
matplotlib
torchio
diffdrr
nnunet
PIL
```

## AE Preprocessing
The preprocessing steps for the AE data were as follows:
1. Converting a DICOM file of a CT scan into a .nii.gz file.
2. Generating 3D labels of each bone in a CT scan.
3. Creating a separate 3D label file of the needed bone (femur) from the rest of the bones (which, in our case, were 2 femurs per CT scan of the lower extremities). 
<!-- This was done with both the 3D labels and the original 3D models. -->
4. Mirroring the 3D labels of femurs of the right leg so all bones have the same orientation.
5. Ensuring all 3D label files were of the same size using padding.
6. Using interpolation to make the data to be processed by the network smaller.
7. Converting the 3D label files to .pt files to be used in the network.




### Step 1: DICOM to .nii.gz

We start with DICOM files of CT scans. In our case, these CT scans were of the lower extremities of the body. These DICOM files were loaded into [3DSlicer](https://www.slicer.org/) and saved as .nii.gz files.

### Step 2: Generating 3D labels
We have used [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for automatically generating 3D labels.  To run the network, execute the following command:

```
placeholder command
```



### Step 3: 3DSlicer script for separating bone labels
Because the lower extremities contain 12(? double-check) different bones, the two femurs need to be separated and saved separately for proper training of the network. In 3DSlicer, run the following command:

```
placeholder command 2, replace path with /path/to/slicer /path/to/script
```

### Step 4: Mirroring
Using `mirror.py`, we mirrored the femurs of the right leg.

### Step 5-7: Readying for network
Running `AE_preprocessor.py`, steps 5 to 7 will be done. The preprocessing in the code is done in batches of 30 because lack of memory kills the process.
The interpolation shape is 120x72x236 because that is around 0.2x the resolution of the biggest shape in the data.

## Preprocessing for the second and third networks

The preprocessing steps for the other two networks were as follows:
1. Creating a separate 3D model file of the needed bone (femur) from the rest of the bones (which, in our case, were 2 femurs per CT scan of the lower extremities). 
2. Geometrically normalizing the 3D model files.
3. Generating CT-DRRs and 40 augmentations of each of them.
4. Removing the whitespace around the CT-DRRS 
5. Padding and converting the .png files to .pt files 
6. Batching 10 CT-DRRs of the same bone for training.


### Step 1: 3DSlicer script for separating bone models
Separating the bone models was also done with a 3DSlicer script:

```
placeholder command 3, replace path with /path/to/slicer /path/to/script
```


### Step 2: Geometric normalization
Using Presurgeo (LINK), we generated files that contained landmarks of the femurs. 
Because there were too many of these landmarks which made geometric normalization too hard for 3DSlicer, we picked 5 landmarks all around the bone.
The following step was loading in one of these sets of landmarks in 3DSlicer as a template and used the Fiducial Registration Wizard module with all of the other bones and their landmarks to geometrically normalize the bones on top of each other.

### Step 3: CT-DRRS 
We use [DiffDRR](https://github.com/eigenvivek/DiffDRR) for the creation of CT-DRRs.
Running `radio_preprocessor.py` generates 40 CT-DRRs of each 3D bone model. We used 2560 mm as the source-to-detector distance to replicate the way radiographs are made at the UMCU[^1]. 

Padding was added so that rotation did not put the bone behind the detector. 

[^1]: The UMCU uses this machine for creating radiographs: https://www.philips.nl/healthcare/product/HC712220/digitaldiagnost-digital-radiography-system#documents

### Step 4: Removing whitespace
Around the CT-DRRs there is a good amount of whitespace, which should be removed by executing `remove_white.py`.

### Step 5: Padding and converting 
`pngtopt.py` is used to pad the CT-DRRs to 302x388, which was the biggest shape of all the DRRs, and to convert the .png files to .pt files.

### Step 6: Batching
Because loading in one CT-DRR and one 3D label each time the network runs a forward pass would take too much time, we batched the CT-DRRs into 4x10 CT-DRRs using `batcher.py`.





## Networks

###  Network folders

All three folders consist of 4(5) files:
1. PX_dataset: This handles loading the dataset used for training the network of part X.
2. PX_network: The network corresponding to part X.
3. PX_train: The training file for the network of part X.
4. PX_test: The test file for the network of part X.
5. (PX_best_epoch: The file used to figure out the best performing network weights of part X).

### Part 1: Autoencoder

The Autoencoder was trained on 250 3D labels of the femur as input and output, with Gaussian noise added to the input to suppress overfitting. The rest of the 3D labels were used for testing.


### Part 2: AlexNet - Decoder

A predictor (AlexNet) is trained with an input of 250 CT-DRRs augmented 40 times in batches of 10 images. These were encoded into a latent space which was decoded using the decoder of Part 1 with fixed weights. The output was compared to the 3D label which the CT-DRRs were based on.

### Part 3: T-Network

This network replicates Part 2 but compares the encoding of the output of Part 2 with the encoding of the 3D label the CT-DRRs were based on.


