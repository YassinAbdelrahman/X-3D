import torch

from torch import nn
from torch import Tensor


class TEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.avg = nn.AdaptiveAvgPool3d(input_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(3, 2)

        self.conv1 = nn.Conv3d(1, 96, 5, 1, padding="same")
        self.norm1 = nn.BatchNorm3d(96)

        self.conv2 = nn.Conv3d(96, 256, 5, padding="same")
        self.norm2 = nn.BatchNorm3d(256)

        self.conv3 = nn.Conv3d(256, 384, 3, 1, padding="same")
        self.conv4 = nn.Conv3d(384, 256, 3, 1, padding="same")
        self.dropout = nn.Dropout3d()
        # self.dropout2 = nn.Dropout()
        # self.fc = nn.Linear(14 * 8 * 28, 4096)
        # self.fc = nn.LazyLinear(64)

    def forward(self, x: Tensor):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        print(x.shape)
        x = self.dropout(x)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        x = x.flatten(1)

        return x


class TDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.relu = nn.ReLU()

        self.fc = nn.Linear(64, 11 * 7 * 29)

        self.conv1 = nn.ConvTranspose3d(1, 256, 3, 1)
        self.conv2 = nn.ConvTranspose3d(256, 384, 3, 1)
        self.conv3 = nn.ConvTranspose3d(384, 256, 5, 1)
        self.conv4 = nn.ConvTranspose3d(256, 96, 7, 1)
        self.conv5 = nn.ConvTranspose3d(96, 1, 1, 1)

        # self.pool = nn.AdaptiveAvgPool3d(input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1, 11, 7, 29)

        x = self.relu(x)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.sigmoid(x)

        return x


class TNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = TEncoder()
        self.decoder = TDecoder()

    def forward(self, x):
        x = self.encoder(x)
        logits = self.decoder(x)

        return logits
