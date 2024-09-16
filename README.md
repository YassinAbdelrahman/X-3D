# X-3D

## Network folders

All three folders consist of 4(5) files:
PX_dataset: This handles loading the dataset used for training the network of part X
PX_network: The network corresponding to part X
PX_train: The training file for the network of part X
PX_test: The test file for the network of part X
(PX_best_epoch: The file used to figure out the best performing network weights of part X)

### Part 1: Autoencoder

The Autoencoder was trained on 250 3D labels of the femur as input and output, with Gaussian noise added to the input to suppress overfitting. The rest of the 3D labels were used for testing.


### Part 2: AlexNet - Decoder

A predictor (AlexNet) is trained with an input of 250 CT-DRRs augmented 40 times in batches of 10 images. These were encoded into a latent space which was decoded using the decoder of Part 1 with fixed weights. The output was compared to the 3D label which the CT-DRRs were based on.

### Part 3: T-Network

This network replicates Part 2 but compares the encoding of the output of Part 2 with the encoding of the 3D label the CT-DRRs were based on.
