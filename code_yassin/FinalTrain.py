import os
import torch
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from Autoencoder3D import Autoencoder3D
from misc_code.AE_Dataset import AutoDataset
import torch.optim as optim
from torch.nn.functional import interpolate


# dataset = BoneDataset("/homes/yassin/E_ResearchData/labels_not_geo", max_samples=50)
dataset = AutoDataset(
    "/homes/yassin/E_ResearchData/femur/tensors_label_02", set_size=50
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initializing the model
model = Autoencoder3D()
# print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)

            # Resize output to match labels
            outputs = interpolate(outputs, size=labels.shape[2:], mode="nearest")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")


# GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training the model
train(model, dataloader, criterion, optimizer)
torch.save(model.state_dict(), "./autoencoder3d.pth")
torch.save(optimizer.state_dict(), "./optimizer3d.pth")
