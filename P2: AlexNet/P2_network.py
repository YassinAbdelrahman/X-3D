import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

# Define the AlexNetEmbedding model
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

        # Add a 64D fully connected layer (fc8) for embedding
        self.fc8 = nn.LazyLinear(216)
        for param in self.alexnet.classifier[-1].parameters():
            param.requires_grad = False
        
        for param in self.fc8.parameters():
            param.requires_grad = False
    def forward(self, x):
        # Forward pass through AlexNet
        x = self.alexnet(x)
        x = self.fc8(x)
        # Forward pass through the additional fc8 layer for embedding
        return x




# class SqueezeNetEmbedding(nn.Module):
#     def __init__(self):
#         super(SqueezeNetEmbedding, self).__init__()

#         self.squeezenet = models.squeezenet1_1()
#         # Modify the first convolutional layer to accept 1 channel input
#         self.squeezenet.features[0] = nn.Conv2d(
#             1, 64, kernel_size=3, stride=2, padding=1
#         )

#         self.squeezenet.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Conv2d(512, 64, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(
#                 (15, 9 * 30)
#             ),  # Adjust pooling layer to match output size
#         )

#     def forward(self, x):
#         # Forward pass through SqueezeNet
#         x = self.squeezenet(x)
#         # Reshape the output to (-1, 64, 8, 5, 15)
#         x = x.view(-1, 64, 15, 9, 30)
#         return x


# class TDecoder(nn.Module):
#     def __init__(self, input_size: int | tuple) -> None:
#         super().__init__()

#         self.relu = nn.ReLU()

#         self.fc = nn.Linear(216, 256 * 6 * 3 * 10)

#         # self.conv1 = nn.ConvTranspose3d(1, 256, 3, 1)
#         self.conv1 = nn.ConvTranspose3d(256, 384, 3, 2, 1, (0, 1, 1))
#         self.conv2 = nn.ConvTranspose3d(384, 256, 3, 2, 1, (0, 1, 1))
#         self.conv3 = nn.ConvTranspose3d(256, 96, 5, 2, 2, 1)
#         self.conv4 = nn.ConvTranspose3d(96, 1, 5, 3, 1, 1)
#         self.sig = nn.Sigmoid()
#         self.pool = nn.AdaptiveAvgPool3d(input_size)

#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(-1, 256, 6, 3, 10)

#         x = self.relu(x)

#         x = self.conv1(x)
#         x = self.relu(x)

#         x = self.conv2(x)
#         x = self.relu(x)

#         x = self.conv3(x)
#         x = self.relu(x)

#         x = self.conv4(x)
#         x = self.sig(x)

#         x = self.pool(x)

#         return x


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

class RadioToEmb(nn.Module):
    def __init__(self, input_size: int | tuple):
        super(RadioToEmb, self).__init__()
        self.alex = AlexNetEmbedding()

        self.decoder = TDecoder(input_size)

    def forward(self, radio):
        pred = self.alex(radio)
        # print(pred.shape)
        # pred = self.alex(radio)
        outp = self.decoder(pred)
        # print(emb.shape)
        # pred = self.flatten(pred)
        # emb = self.flatten(emb)
        # emb = self.fc9(emb)
        # emb = emb.view(emb.size(0), -1)
        # emb = self.fc(emb)
        # print(emb.shape)

        return pred, outp


# x = torch.randn(1, 1, 302,388).to("cpu")

# model = RadioToEmb((120, 72, 236)).to("cpu")
# print(model(x)[1].shape)