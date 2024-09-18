# X-3D
This project is meant to generate 3D models from a singular 2D radiograph. For this, an autoencoder is trained on 3D labels of bones generated from real CT scans. Following this, a second network takes Digitally Reconstructed Radiographs which are created from the previous CT scans (CT-DRRs) and generates a 3D model of the same bone. A third network combines the two previous network to increase the accuracy, resulting in 3D models generated from singular 2D radiographs.

Following is a step-by-step breakdown of preprocessing and using data for the X-3D networks. The bone we focussed on in this project was the thighbone (femur). 

## Step 1: preprocessing
The preprocessing steps for the autoencoder were as follows:
1. Converting a DICOM file of a CT scan into a .nii.gz file.
2. Generating 3D labels of each bone in a CT scan.
3. Creating a separate 3D label file of the needed bone (femur) from the rest of the bones (in our case, 2 femurs per CT scan of the lower extremities). 
<!-- This was done with both the 3D labels and the original 3D models. -->
4. Ensuring all 3D label files were of the same size using padding.
5. Using interpolation to make the data to be processed by the 

### DICOM to .nii.gz

We start with DICOM files of CT scans. In our case, these CT scans were of the lower extremities of the body. These DICOM files were loaded into 3DSlicer (https://www.slicer.org/)

### creating labels
nnUnet

###





### Step X: Networks

All three folders consist of 4(5) files:
PX_dataset: This handles loading the dataset used for training the network of part X.
PX_network: The network corresponding to part X.
PX_train: The training file for the network of part X.
PX_test: The test file for the network of part X.
(PX_best_epoch: The file used to figure out the best performing network weights of part X).

### Part 1: Autoencoder

The Autoencoder was trained on 250 3D labels of the femur as input and output, with Gaussian noise added to the input to suppress overfitting. The rest of the 3D labels were used for testing.


### Part 2: AlexNet - Decoder

A predictor (AlexNet) is trained with an input of 250 CT-DRRs augmented 40 times in batches of 10 images. These were encoded into a latent space which was decoded using the decoder of Part 1 with fixed weights. The output was compared to the 3D label which the CT-DRRs were based on.

### Part 3: T-Network

This network replicates Part 2 but compares the encoding of the output of Part 2 with the encoding of the 3D label the CT-DRRs were based on.
