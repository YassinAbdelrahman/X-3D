import torch

from torch import nn
from torch import Tensor


class TEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(3, 2)

        self.conv1 = nn.Conv3d(1, 96, 5, stride=(2, 2, 3))
        self.norm1 = nn.BatchNorm3d(96)

        self.conv2 = nn.Conv3d(96, 256, 5, padding="same")
        self.norm2 = nn.BatchNorm3d(256)

        self.conv3 = nn.Conv3d(256, 384, 3, 1, padding="same")
        self.norm3 = nn.BatchNorm3d(384)

        self.conv4 = nn.Conv3d(384, 256, 3, 1, padding="same")

        # self.final_pool = nn.AvgPool3d(kernel_size=(2, 1, 6))

        self.fc = nn.LazyLinear(216)

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
        x = self.pool(x)
        x = self.norm3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        # x = self.final_pool(x)
        # print(x.shape)

        x = x.flatten(1)

        x = self.fc(x)

        return x


class TDecoder(nn.Module):
    def __init__(self, input_size: int | tuple) -> None:
        super().__init__()

        self.relu = nn.ReLU()

        self.fc = nn.Linear(216, 256 * 6 * 3 * 10)

        # self.conv1 = nn.ConvTranspose3d(1, 256, 3, 1)
        self.conv1 = nn.ConvTranspose3d(256, 384, 3, 2, 1, (0, 1, 1))
        self.conv2 = nn.ConvTranspose3d(384, 256, 3, 2, 1, (0, 1, 1))
        self.conv3 = nn.ConvTranspose3d(256, 96, 5, 2, 2, 1)
        self.conv4 = nn.ConvTranspose3d(96, 1, 5, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool3d(input_size)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 6, 3, 10)

        x = self.relu(x)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.sig(x)

        x = self.pool(x)

        return x


class TNetwork(nn.Module):
    def __init__(self, input_size: int | tuple) -> None:
        super().__init__()

        self.encoder = TEncoder()
        self.decoder = TDecoder(input_size)

    def forward(self, x):
        logits = self.encoder(x)
        x = self.decoder(logits)

        return logits, x


# x = torch.randn(3, 1, 120, 72, 236).to("cuda")

# model = TNetwork((120, 72, 236)).to("cuda")
# print(model(x)[1].shape)
