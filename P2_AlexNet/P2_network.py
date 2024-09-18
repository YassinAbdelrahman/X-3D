import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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
        x = self.alexnet(x)
        x = self.fc8(x)
        return x


class TDecoder(nn.Module):
    def __init__(self, input_size: int | tuple) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.input_size = input_size
        self.fc = nn.Linear(216, 6144)
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

class RadioToEmb(nn.Module):
    def __init__(self, input_size: int | tuple):
        super(RadioToEmb, self).__init__()
        self.alex = AlexNetEmbedding()

        self.decoder = TDecoder(input_size)

    def forward(self, radio):
        pred = self.alex(radio)
        outp = self.decoder(pred)
        return pred, outp