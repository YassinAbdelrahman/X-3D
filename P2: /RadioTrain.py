import os
import torch
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
from P2_network import RadioToEmb
from combined_dataset import Custom2D3DDataset

# dataset = BoneDataset("/homes/yassin/E_ResearchData/labels_not_geo", max_samples=50)
train_dataset = Custom2D3DDataset(
    "/nethome/2514818/Data/final_data",
    max_samples=1000,val=False
)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = Custom2D3DDataset(
    "/nethome/2514818/Data/final_data",
    max_samples=1000,val=True
)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Initializing the model
model = RadioToEmb((120, 72, 236))

# print(model)

criterion = nn.BCELoss()
criterion_2 = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=1e-5)
# weights_encoder = torch.load("./encoder.pth")
# model.encoder.load_state_dict(weights_encoder, strict=False)
# weights_decoder = torch.load("./decoder.pth")
model.decoder.load_state_dict(torch.load("/nethome/2514818/X-3D/AE_checkpoints/decoder_epoch_186.pth"))

for param in model.decoder.parameters():
    param.requires_grad = False

def validate(model, val_dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for inputs, nii in val_dataloader:
            loss = 0
            nii = nii.to(device)
            inputs = inputs.squeeze(0).to(device)
            
            # Forward pass
            preds, outputs = model(inputs)
            
            # Resize the output to match the target size
            outputs = interpolate(outputs, size=nii.shape[2:], mode="nearest")
            
            l1_loss = 1e-4 * criterion_2(preds, torch.zeros_like(preds))
            for i in range(outputs.shape[0]):
                loss += criterion(outputs[i], nii.squeeze(0))
            loss = loss + l1_loss
            
            val_loss += loss.item()
    
    # Calculate the average validation loss
    avg_val_loss = val_loss / len(val_dataloader)
    return avg_val_loss


def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=200):
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        j = 1
        for inputs, nii in train_dataloader:
            
            loss = 0
            # inputs = inputs.to(device)
            nii = nii.to(device)
            inputs = inputs.squeeze(0).to(device)
            optimizer.zero_grad()
            # print(inputs.size())
            (
                preds,
                outputs,
            ) = model(inputs)
            # print(outputs.shape)
            outputs = interpolate(outputs, size=nii.shape[2:], mode="nearest")
            l1_loss = 1e-4 * criterion_2(preds, torch.zeros_like(preds))
            for i in range(outputs.shape[0]):

                loss += criterion(outputs[i], nii.squeeze(0))
            loss = loss + l1_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(loss.item(), j, len(train_dataloader))
            j = j+ 1
            # print(f"Batch {i}, Loss: {total_loss / (len(dataloader))}")
        #     print(f"Image {i}, Loss: {total_loss / (len(dataloader))}")
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        avg_val_loss = validate(model, val_dataloader, criterion)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_train_loss}")
        if (epoch + 1) % 2 == 0:
            torch.save(model.alex.state_dict(), f"./output/radio_alex_epoch_{epoch + 1}.pth")
            # torch.save(optimizer.state_dict(), f"./Radio_checkpoints_new/radio_optimizer_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved for epoch {epoch + 1}")
    return train_losses, val_losses


# GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training the model
train_losses,val_losses = train(model, train_dataloader,val_dataloader, criterion, optimizer)
torch.save(model.alex.state_dict(), "./alex.pth")
plt.figure()
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.savefig("radio_losses.png")
plt.show()
# torch.save(model.alex.state_dict(), "./alexnet.pth")
# torch.save(optimizer.state_dict(), "./radioopt.pth")
