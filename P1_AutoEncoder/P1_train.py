import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import time
from P1_network import TNetwork
from P1_dataset import AutoDataset
import torch.optim as optim
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
import torchio as tio

transform = tio.Compose(
    [
        # tio.RandomAffine(),
        # tio.RandomFlip(),
        tio.RandomNoise(),
        # Add more transformations as needed
    ]
)

train_dataset = AutoDataset("/nethome/2514818/Data/tensors_02", set_size=220,transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

val_dataset = AutoDataset("/nethome/2514818/Data/tensors_02", train=False, set_size=220,transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=True)

# Initializing the model
model = TNetwork((120, 72, 236))




criterion = nn.BCELoss()
criterion_2 = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            latent, outputs = model(inputs)
            outputs = interpolate(outputs, size=labels.shape[2:], mode="nearest")
            l1_loss = 1e-4 * criterion_2(latent, torch.zeros_like(latent))
            loss = criterion(outputs, labels) + l1_loss
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=200):
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        tic = time.perf_counter()
        print(f"epoch {epoch}")
        total_loss = 0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            latent, outputs = model(inputs)
            # print(outputs.shape)
            # Resize output to match labels

            outputs = interpolate(outputs, size=labels.shape[2:], mode="nearest")
            # print(outputs.shape)
            l1_loss = 1e-4 * criterion_2(latent, torch.zeros_like(latent))
            loss = criterion(outputs, labels) + l1_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_train_loss}")

        avg_val_loss = validate(model, val_dataloader, criterion)
        val_losses.append(avg_val_loss)
        toc = time.perf_counter()
        print(f"this epoch was {toc-tic:0.4f} sec long")
        # Save checkpoints every 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save(model.encoder.state_dict(), f"./AE_checkpoints/encoder_epoch_{epoch + 1}.pth")
            torch.save(model.decoder.state_dict(), f"./AE_checkpoints/decoder_epoch_{epoch + 1}.pth")
            torch.save(optimizer.state_dict(), f"./AE_checkpoints/optimizer_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved for epoch {epoch + 1}")
            
    return train_losses, val_losses


# GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training the model
train_losses, val_losses = train(
    model, train_dataloader, val_dataloader, criterion, optimizer
)

# Saving the model
torch.save(model.encoder.state_dict(), "./encoder.pth")
torch.save(model.decoder.state_dict(), "./decoder.pth")
torch.save(optimizer.state_dict(), "./optimizer3d.pth")
print("saved encoder.pth,decoder.pth and optimizer3d.pth")

# Plotting the training and validation loss
plt.figure()
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.savefig("losses.png")
plt.show()
