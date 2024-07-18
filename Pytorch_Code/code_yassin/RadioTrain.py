import os
import torch
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import interpolate

# from RadiotoEmbNetwork import RadioToEmb
from RadiotoEmbNetwork_co import RadioToEmb
from combined_dataset import Custom2D3DDataset

# dataset = BoneDataset("/homes/yassin/E_ResearchData/labels_not_geo", max_samples=50)
dataset = Custom2D3DDataset(
    "data/tensors",
    max_samples=20,
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initializing the model
model = RadioToEmb((120, 72, 236))

# print(model)

criterion = nn.BCELoss()
criterion_2 = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
# weights_encoder = torch.load("./encoder.pth")
# model.encoder.load_state_dict(weights_encoder, strict=False)
weights_decoder = torch.load("./decoder.pth")
model.decoder.load_state_dict(weights_decoder, strict=False)

# for param in model.encoder.parameters():
#     param.requires_grad = False


def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, nii in dataloader:
            # inputs = inputs.to(device)
            nii = nii.to(device)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            # print(inputs.size())
            (
                preds,
                outputs,
            ) = model(inputs)
            outputs = interpolate(outputs, size=inputs.shape[2:], mode="nearest")
            l1_loss = 1e-4 * criterion_2(preds, torch.zeros_like(preds))
            for i in range(outputs):

                #
                loss += criterion(outputs[i], nii)
            loss = loss + l1_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print(f"Batch {i}, Loss: {total_loss / (len(dataloader))}")
        #     print(f"Image {i}, Loss: {total_loss / (len(dataloader))}")
        print(f"Epoch {epoch+1}, Loss: {total_loss / (len(dataloader))}")


# GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training the model
train(model, dataloader, criterion, optimizer)
torch.save(model.squeeze.state_dict(), "./squeeze.pth")
# torch.save(model.alex.state_dict(), "./alexnet.pth")
# torch.save(optimizer.state_dict(), "./radioopt.pth")
