from P1_network import TNetwork
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from P1_dataset import AutoDataset
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt  # Import for plotting


def save_nifti(data, filename):
    """Saves a numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, np.eye(4))  # no affine transformation
    nib.save(nifti_img, filename)

    
def main():
    best_loss = float('inf')  # Initialize best loss as infinity
    best_epoch = None  # Track the epoch with the best loss
    all_losses = []  # To store the losses for all epochs
    
    for epoch in range(0, 201):  # Looping through epochs 20 to 200
        print(f"Loading model for epoch {epoch}...")
        model = TNetwork((120,72,236))
        
        encoder_path = f"/nethome/2514818/X-3D/AE_checkpoints/encoder_epoch_{epoch}.pth"
        decoder_path = f"/nethome/2514818/X-3D/AE_checkpoints/decoder_epoch_{epoch}.pth"

        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            print(f"Skipping epoch {epoch}, checkpoint files not found.")
            all_losses.append(float('inf'))  # Append infinity if no file is found
            continue  # Skip the current epoch if either checkpoint is missing

        model.encoder.load_state_dict(torch.load(encoder_path))
        model.decoder.load_state_dict(torch.load(decoder_path))

        bce_loss = torch.nn.BCELoss()
        model.eval()

        print("Loading dataset...")
        test_dataset = AutoDataset(
            "/nethome/2514818/Data/tensors_02", train=False, set_size=170
        )

        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=True, num_workers=0
        )



        epoch_loss_sum = 0  # Track cumulative loss for the epoch
        batch_count = 0  # Count the number of batches

        print("Processing data...")
        exputs = None
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            print(f"Processing batch {i+1}/{len(test_dataloader)}")
            with torch.no_grad():
                _, reconstructed = model(inputs)
            
            original_np = inputs.squeeze(1).cpu().numpy()
            print(inputs.shape[2:])
            reconstructed = interpolate(
                reconstructed, size=labels.shape[2:], mode="nearest"
            )
            print(reconstructed.shape, inputs.shape)
            loss = bce_loss(reconstructed, inputs)
            epoch_loss_sum += loss.item()  # Add the batch loss to epoch total
            batch_count += 1  # Increment batch counter
            print(f"BCE Loss for batch {i+1}: {loss.item()}")
            
            if exputs is not None:
                loss_2 = bce_loss(exputs, inputs)
                print(f"BCE Loss for RECENT {i+1}: {loss_2.item()}")
            
            reconstructed_np = reconstructed.squeeze(1).cpu().numpy()
            print(reconstructed_np[0].shape)
            reconstructed_np[(reconstructed_np > 0.05) & (reconstructed_np < 1)] = 1
            
            # Saving the original and reconstructed as NIfTI files
            # original_filename = f"{output_dir}/original_{i}.nii.gz"
            # reconstructed_filename = f"{output_dir}/reconstructed_{i}.nii.gz"
            # save_nifti(original_np[0], original_filename)
            # save_nifti(reconstructed_np[0], reconstructed_filename)
            # print(f"Saved original to {original_filename}")
            # print(f"Saved reconstructed to {reconstructed_filename}")
            exputs = inputs

        # Calculate average loss for the epoch
        avg_loss = epoch_loss_sum / batch_count if batch_count > 0 else float('inf')
        all_losses.append(avg_loss)  # Append the average loss for the epoch
        print(f"Average BCE Loss for epoch {epoch}: {avg_loss}")

        # Check if this is the best loss so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            print(f"New best model found at epoch {epoch} with loss {best_loss}")

    # After all epochs are processed, display the best one
    if best_epoch is not None:
        print(f"\nBest model was found at epoch {best_epoch} with a BCE loss of {best_loss:.6f}")
    else:
        print("No valid model was found in the specified epoch range.")

    # Plotting the loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, 201), all_losses, marker='o', color='b', label='BCE Loss')
    plt.title('BCE Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig("./output/loss_graph.png")  # Save the graph as an image file
    plt.show()  # Display the graph

if __name__ == "__main__":
    main()
