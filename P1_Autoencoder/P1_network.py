
from torch import nn
from torch import Tensor
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.conv3d(1, 48, 3, padding='same')
        self.normfirst = nn.BatchNorm3d(48)

        self.conv2 = nn.conv3d(48, 96, 3, padding='same')
        self.norm0 = nn.BatchNorm3d(96)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(2, 2)

        self.conv3 = nn.conv3d(96, 128, 5, stride=(2, 2, 3), padding=2)
        self.norm1 = nn.BatchNorm3d(128)

        self.conv4 = nn.conv3d(128, 256, 5, padding="same")
        self.norm2 = nn.BatchNorm3d(256)

        self.conv5 = nn.conv3d(256, 384, 3, 1, padding="same")
        self.norm3 = nn.BatchNorm3d(384)

        self.conv6 = nn.conv3d(384, 256, 3, 1, padding="same")


        self.fc = nn.LazyLinear(216)

    def forward(self, x: Tensor):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.normfirst(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm0(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm1(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.norm2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm3(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.flatten(1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size: int | tuple) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.input_size = input_size

        self.conv1 = nn.ConvTranspose3d(256, 384, 3,  stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(384) 
        self.conv2 = nn.ConvTranspose3d(384, 256, 3, stride=1, padding=1 )
        self.norm2 = nn.BatchNorm3d(256)
        self.conv3 = nn.ConvTranspose3d(256, 128, 5,  stride=1, padding=2)
        self.norm3 = nn.BatchNorm3d(128)
        self.conv4 = nn.ConvTranspose3d(128, 96, 5, (2,2,3), padding=2)
        self.norm4 = nn.BatchNorm3d(96)
        self.conv5 = nn.ConvTranspose3d(96, 48, 3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm3d(48)
        self.conv6 = nn.ConvTranspose3d(48, 1, 3, stride=1, padding=1)
        self.sig = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool3d(input_size)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 3, 2, 4)

        x = self.conv1(x)
        x = self.relu(x)
        x = F.interpolate(x,size=(7,4,9))
        x = self.norm1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x,size=(15,9,19))
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(30,18,39))
        x = self.norm3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = F.interpolate(x,size=(60,36,79))
        x = self.norm4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = F.interpolate(x,size=self.input_size)
        x = self.norm5(x)

        x = self.conv6(x)
        x = self.sig(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int | tuple) -> None:
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(input_size)

    def forward(self, x):
        logits = self.encoder(x)
        x = self.decoder(logits)

        return logits, x


# x = torch.randn(1, 1, 120, 72, 236).to("cpu")
# # x = torch.randn(1, 216).to("cpu")

# model = TNetwork((120, 72, 236)).to('cpu')
# # model = TNetwork((120, 72, 236)).decoder.to("cpu")
# print(model(x)[1].shape)
