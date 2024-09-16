
from torch import nn
from torch import Tensor
import torch.nn.functional as F

class TEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.convfirst = nn.Conv3d(1, 48, 3, padding='same')
        self.normfirst = nn.BatchNorm3d(48)

        self.conv0 = nn.Conv3d(48, 96, 3, padding='same')
        self.norm0 = nn.BatchNorm3d(96)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(2, 2)

        self.conv1 = nn.Conv3d(96, 128, 5, stride=(2, 2, 3), padding=2)
        self.norm1 = nn.BatchNorm3d(128)

        self.conv2 = nn.Conv3d(128, 256, 5, padding="same")
        self.norm2 = nn.BatchNorm3d(256)

        self.conv3 = nn.Conv3d(256, 384, 3, 1, padding="same")
        self.norm3 = nn.BatchNorm3d(384)

        self.conv4 = nn.Conv3d(384, 256, 3, 1, padding="same")

        # self.final_pool = nn.AvgPool3d(kernel_size=(2, 1, 6))

        self.fc = nn.LazyLinear(216)

    def forward(self, x: Tensor):
        # print('input', x.shape)

        x = self.convfirst(x)
        # print('convfirst', x.shape)
        x = self.relu(x)
        x = self.normfirst(x)

        x = self.conv0(x)
        # print('conv0', x.shape)
        x = self.relu(x)
        x = self.norm0(x)

        x = self.conv1(x)
        # print('conv1', x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print(' pool1', x.shape)
        x = self.norm1(x)

        x = self.conv2(x)
        # print('conv2', x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print('pool2', x.shape)

        x = self.norm2(x)

        x = self.conv3(x)
        # print('conv3', x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print('pool3', x.shape)

        x = self.norm3(x)

        x = self.conv4(x)
        # print('conv4', x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print('pool4', x.shape)


        # x = self.final_pool(x)
        # print(x.shape)
        x = x.flatten(1)
        # print('flatten',x.shape)
        x = self.fc(x)
        # print('fc',x.shape)
        return x


class TDecoder(nn.Module):
    def __init__(self, input_size: int | tuple) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.input_size = input_size
        self.fc = nn.Linear(216, 6144)


        # self.conv1 = nn.ConvTranspose3d(1, 256, 3, 1)
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

        # x = self.relu(x)
        # x = self.unpool(x)
        x = self.conv1(x)
        # print('conv1',x.shape)
        x = self.relu(x)
        x = F.interpolate(x,size=(7,4,9))
        x = self.norm1(x)
        

        # x = self.unpool(x)
        x = self.conv2(x)
        # print('conv2',x.shape)
        x = self.relu(x)
        x = F.interpolate(x,size=(15,9,19))
        x = self.norm2(x)

        x = self.conv3(x)
        # print('conv3',x.shape)
        x = self.relu(x)
        x = F.interpolate(x, size=(30,18,39))
        x = self.norm3(x)

        
        # print('interp3',x.shape)

        # x = self.unpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = F.interpolate(x,size=(60,36,79))
        # print('conv4',x.shape)

        x = self.norm4(x)


        x = self.conv5(x)
        x = self.relu(x)
        x = F.interpolate(x,size=self.input_size)

        x = self.norm5(x)
        # print('conv5',x.shape)


        x = self.conv6(x)
        # print(x.shape)

        x = self.sig(x)
        # print(x.shape)

        # x = self.pool(x)

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


# x = torch.randn(1, 1, 120, 72, 236).to("cpu")
# # x = torch.randn(1, 216).to("cpu")

# model = TNetwork((120, 72, 236)).to('cpu')
# # model = TNetwork((120, 72, 236)).decoder.to("cpu")
# print(model(x)[1].shape)
