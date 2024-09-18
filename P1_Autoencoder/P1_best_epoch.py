from P1_network import TNetwork
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from P1_dataset import AEDataset
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt  # Import for plotting


def save_nifti(data, filename):
    """Saves a numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, np.eye(4))  # no affine transformation
    nib.save(nifti_img, filename)

    
def main():
    best_loss = float('inf')  
    best_epoch = None  
    all_losses = []  
    
    for epoch in range(0, 201):  
        model = TNetwork((120,72,236))
        
        encoder_path = f"/nethome/2514818/X-3D/AE_checkpoints/encoder_epoch_{epoch}.pth"
        decoder_path = f"/nethome/2514818/X-3D/AE_checkpoints/decoder_epoch_{epoch}.pth"

        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            print(f"Skipping epoch {epoch}, checkpoint files not found.")
            all_losses.append(float('inf'))  
            continue  

        model.encoder.load_state_dict(torch.load(encoder_path))
        model.decoder.load_state_dict(torch.load(decoder_path))

        bce_loss = torch.nn.BCELoss()
        model.eval()

        test_dataset = AEDataset(
            "/nethome/2514818/Data/tensors_02", train=False, set_size=200
        )

        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=True, num_workers=0
        )



        epoch_loss_sum = 0 
        batch_count = 0 

        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            print(f"Batch {i+1}/{len(test_dataloader)}")
            with torch.no_grad():
                _, reconstructed = model(inputs)
            
            _ = inputs.squeeze(1).cpu().numpy()
            reconstructed = interpolate(
                reconstructed, size=labels.shape[2:], mode="nearest"
            )
            loss = bce_loss(reconstructed, inputs)
            epoch_loss_sum += loss.item()  
            batch_count += 1  
            print(f"BCE Loss for batch {i+1}: {loss.item()}")
            
            reconstructed_np = reconstructed.squeeze(1).cpu().numpy()
            reconstructed_np[(reconstructed_np > 0.05) & (reconstructed_np < 1)] = 1
            


        # Calculate average loss for the epoch
        avg_loss = epoch_loss_sum / batch_count if batch_count > 0 else float('inf')
        all_losses.append(avg_loss)  
        print(f"Average BCE Loss for epoch {epoch}: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            print(f"New best model found at epoch {epoch} with loss {best_loss}")

    
    print(f"\nBest model was found at epoch {best_epoch} with a BCE loss of {best_loss:.6f}")


    # Plotting the loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, 201), all_losses, marker='o', color='b', label='BCE Loss')
    plt.title('BCE Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig("./output/loss_graph.png")
    plt.show()  

if __name__ == "__main__":
    main()
