import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor


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


class AlexNetEmbedding(nn.Module):
    def __init__(self):
        super(AlexNetEmbedding, self).__init__()
        # Load pre-trained AlexNet
        self.alexnet = models.alexnet()

        # Modify the first convolutional layer to accept 1 channel input
        self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

        # Remove the last fully connected layer (fc8) of AlexNet
        self.alexnet.classifier = nn.Sequential(
            *list(self.alexnet.classifier.children())[:-1]
        )
        self.fc8 = nn.LazyLinear(216)
        # Add a 64D fully connected layer (fc8) for embedding

    def forward(self, x):
        # Forward pass through AlexNet
        x = self.alexnet(x)
        x = self.fc8(x)
        # Forward pass through the additional fc8 layer for embedding
        return x


class CombinedModel(nn.Module):
    def __init__(self,input_size: int | tuple):
        super(CombinedModel, self).__init__()
        # Autoencoder
        self.encoder = TEncoder()
        self.decoder = TDecoder(input_size)
        # self.alex = AlexNetEmbedding()
        # CNN for classification
        self.flatten = nn.Flatten()

        self.alexnet = AlexNetEmbedding()

    def forward(self, radio):
        # z = encoded.view(-1, 64, 15, 9, 30)
        x_pred = self.alexnet(radio)
        x_recon = self.decoder(x_pred)
        # encoded = self.flatten(encoded)
        
        # x_pred = self.flatten(x_pred)
        return x_pred,x_recon


# x = torch.randn(1, 1, 308,388).to("cpu")
# # x = torch.randn(1, 216).to("cpu")

# model = CombinedModel((120, 72, 236)).to('cpu')
# # model = TNetwork((120, 72, 236)).decoder.to("cpu")
# print(model(x)[1].shape)