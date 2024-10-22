import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from P3_network import CombinedModel
import torch.optim as optim
from torch.nn.functional import interpolate
from P3_dataset import Custom2D3DDataset

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

model = CombinedModel((120,72,236))

weights_encoder = torch.load("./encoder.pth")
weights_decoder = torch.load("./decoder.pth")
weights_alex = torch.load("./alex.pth")
model.encoder.load_state_dict(weights_encoder, strict=False)
model.decoder.load_state_dict(weights_decoder, strict=False)
model.alexnet.load_state_dict(weights_alex, strict=False)

for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion_1 = nn.BCELoss()
criterion_2 = nn.L1Loss()
criterion_3 = nn.MSELoss()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def validate(model, val_dataloader, criterion_reconstruction, criterion_classification):
    model.eval()  
    total_val_loss = 0

    with torch.no_grad():  
        for radios, image in val_dataloader:
            image = image.to(device)
            loss = 0
            og_latent = model.encoder(image)
            for i in range(radios.size(0)):
                radio = radios[0][i].unsqueeze(0)
                radio = radio.to(device)
                preds, reconstructed = model(radio)
                fake_latent = model.encoder(reconstructed)
                reconstructed = interpolate(
                    reconstructed, size=image.shape[2:], mode="nearest"
                )

                loss_reconstruction = criterion_reconstruction(reconstructed, image)
                l1_loss = 1e-4 * criterion_2(preds, torch.zeros_like(preds))
                loss_classification = 1e-4 * criterion_classification(og_latent, fake_latent)

                loss += loss_reconstruction + loss_classification + l1_loss

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss

def train(
    model,
    train_dataloader,
    val_dataloader,
    criterion_reconstruction,
    criterion_classification,
    optimizer,
    num_epochs=200,
):
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for radios, image in train_dataloader:
            image = image.to(device)
            loss = 0
            og_latent = model.encoder(image)
            for i in range(radios.size(0)):
                radio = radios[0][i].unsqueeze(0)
                radio = radio.to(device)
                optimizer.zero_grad()
                preds, reconstructed = model(radio)
                fake_latent = model.encoder(reconstructed)
                reconstructed = interpolate(
                    reconstructed, size=image.shape[2:], mode="nearest"
                )
                
                loss_reconstruction = criterion_reconstruction(reconstructed, image)
                l1_loss = 1e-4 * criterion_2(preds, torch.zeros_like(preds))
                loss_classification = 1e-4 * criterion_classification(og_latent, fake_latent)
                print("reconstruction loss: ", loss_reconstruction.item())
                print("classification loss: ", loss_classification.item())
                loss += loss_reconstruction + loss_classification + l1_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        avg_val_loss = validate(model, val_dataloader, criterion_reconstruction,criterion_classification)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")
        if (epoch + 1) % 2 == 0:
            torch.save(model.decoder.state_dict(), f"./combined_checkpoints/combined_decoder_epoch_{epoch + 1}.pth")
            torch.save(model.alex.state_dict(), f"./combined_checkpoints/combined_alex_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved for epoch {epoch + 1}")
    return train_losses, val_losses


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train(
    model,
    train_dataloader,
    val_dataloader,
    criterion_1,
    criterion_3,
    optimizer)
torch.save(model.encoder.state_dict(), "./combined_encoder.pth")
torch.save(model.decoder.state_dict(), "./combined_decoder.pth")
torch.save(model.alexnet.state_dict(), "./combined_alex.pth")
torch.save(optimizer.state_dict(), "./combined_optimizer.pth")
