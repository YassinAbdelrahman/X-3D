import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Define the AlexNetEmbedding model
# class AlexNetEmbedding(nn.Module):
#     def __init__(self):
#         super(AlexNetEmbedding, self).__init__()
#         # Load pre-trained AlexNet
#         self.alexnet = models.alexnet()

#         # Modify the first convolutional layer to accept 1 channel input
#         self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

#         # Remove the last fully connected layer (fc8) of AlexNet
#         self.alexnet.classifier = nn.Sequential(
#             *list(self.alexnet.classifier.children())[:-1]
#         )

#         # Add a 64D fully connected layer (fc8) for embedding

#     def forward(self, x):
#         # Forward pass through AlexNet
#         x = self.alexnet(x)
#         # Forward pass through the additional fc8 layer for embedding
#         return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.Flatten(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),  # Using sigmoid to normalize the output
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class SqueezeNetEmbedding(nn.Module):
    def __init__(self):
        super(SqueezeNetEmbedding, self).__init__()

        self.squeezenet = models.squeezenet1_1()
        # Modify the first convolutional layer to accept 1 channel input
        self.squeezenet.features[0] = nn.Conv2d(
            1, 64, kernel_size=3, stride=2, padding=1
        )

        self.squeezenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(
                (15, 9 * 30)
            ),  # Adjust pooling layer to match output size
        )

    def forward(self, x):
        # Forward pass through SqueezeNet
        x = self.squeezenet(x)
        # Reshape the output to (-1, 64, 8, 5, 15)
        x = x.view(-1, 64, 15, 9, 30)
        return x


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        # Autoencoder
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.alex = AlexNetEmbedding()
        # CNN for classification
        self.flatten = nn.Flatten()

        self.squeezenet = SqueezeNetEmbedding()

    def forward(self, x, radio):
        encoded = self.encoder(x)
        # z = encoded.view(-1, 64, 15, 9, 30)
        x_recon = self.decoder(encoded)
        # encoded = self.flatten(encoded)
        x_pred = self.squeezenet(radio)
        # x_pred = self.flatten(x_pred)
        return x_recon, x_pred, encoded
