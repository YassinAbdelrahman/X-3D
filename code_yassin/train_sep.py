import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn

from t_network import TNetwork
from AE_Dataset import AutoDataset
import torch.optim as optim
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt

train_dataset = AutoDataset("/home/yabdelrahman/data/labels", set_size=26)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = AutoDataset("/home/yabdelrahman/data/labels", train=False, set_size=26)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

# Initializing the model
model = TNetwork((120, 72, 236))


criterion = nn.BCELoss()
criterion_2 = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)


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


def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=2):
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
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
