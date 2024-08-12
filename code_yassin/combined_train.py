import os
import torch
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn

# from combined_network import CombinedModel
from combined_network_co import CombinedModel

import torch.optim as optim
from torch.nn.functional import interpolate

# from combined_dataset import Custom2D3DDataset
from combined_dataset import Custom2D3DDataset

# dataset = BoneDataset("/homes/yassin/E_ResearchData/labels_not_geo", max_samples=20)
# dataset_radio = RadioDataset(
#     "/homes/yassin/E_ResearchData/CT_Scans/CT_Out_test", max_samples=20
# )

dataset = Custom2D3DDataset(
    "/nethome/2514818/Data/final_data", max_samples=200
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# radio_dataset =
# Initializing the model
model = CombinedModel((120,72,236))
# print(model)

weights_encoder = torch.load("./encoder.pth")
weights_decoder = torch.load("./decoder.pth")
weights_alex = torch.load("./alex.pth")
# weights_alex = torch.load("./alexnet.pth")
model.encoder.load_state_dict(weights_encoder, strict=False)
model.decoder.load_state_dict(weights_decoder, strict=False)
model.alexnet.load_state_dict(weights_alex, strict=False)
# model.alex.load_state_dict(weights_decoder, strict=False)


# criterion = nn.MSELoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
# prev_optimizer = torch.load("./optimizer3d.pth")
# optimizer.load_state_dict(prev_optimizer, strict=False)
criterion_1 = nn.BCELoss()
criterion_2 = nn.L1Loss()
criterion_3 = nn.MSELoss()


print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def train(
    model,
    dataloader,
    criterion_reconstruction,
    criterion_classification,
    optimizer,
    num_epochs=20,
):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for radios, image in dataloader:
            image = image.to(device)
            loss = 0
            # print(radios.size())
            og_latent = model.encoder(image)
            for i in range(radios.size(0)):
                # print(i)
                radio = radios[0][i].unsqueeze(0)
                radio = radio.to(device)
                optimizer.zero_grad()
                preds, reconstructed = model(radio)
                fake_latent = model.encoder(reconstructed)
                reconstructed = interpolate(
                    reconstructed, size=image.shape[2:], mode="nearest"
                )
                
                # Compute losses
                loss_reconstruction = criterion_reconstruction(reconstructed, image)
                l1_loss = 1e-4 * criterion_2(preds, torch.zeros_like(preds))
                loss_classification = criterion_classification(og_latent, fake_latent)
                print("reconstruction loss: ", loss_reconstruction.item())
                print("classification loss: ", loss_classification.item())
                loss += loss_reconstruction + loss_classification + l1_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")


# GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training the model
train(
    model,
    dataloader,
    criterion_1,
    criterion_3,
    optimizer,
)
torch.save(model.encoder.state_dict(), "./combined_encoder.pth")
torch.save(model.decoder.state_dict(), "./combined_decoder.pth")
torch.save(model.alexnet.state_dict(), "./combined_alex.pth")
# torch.save(model.alex.state_dict(), "./combined_alex.pth")
torch.save(optimizer.state_dict(), "./combined_optimizer.pth")
