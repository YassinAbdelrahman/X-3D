import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary


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


class FinalNet(nn.Module):
    def __init__(self):
        super(FinalNet, self).__init__()
        self.squeeze = SqueezeNetEmbedding()
        # self.alex = AlexNetEmbedding()
        # self.fc8 = nn.Linear(4096, 1024)
        self.decoder = Decoder()

    def forward(self, radio):
        encoded = self.squeeze(radio)
        # pred = self.alex(radio)

        decoded = self.decoder(encoded)
        # pred = self.fc8(pred)
        # emb = self.fc9(emb)
        return decoded
