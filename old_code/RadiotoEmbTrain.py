from t_network import TNetwork
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
import os
from tqdm import tqdm
from torchvision.transforms import v2
import torch
from RadiotoEmbNetwork import AlexNetEmbedding
import nibabel as nib


output_dir = "output_images"


def train_epoch(model, train_loader, criterion, optimizer, args):
    # set model to training mode
    model.train()

    pbar = tqdm(train_loader)
    for batch in pbar:
        # reset stored gradients
        optimizer.zero_grad()

        # send data to device
        batch = batch.to(args.device)

        # get model outputs (not interested in latents here)
        logits, latents = model(batch)

        # calculate loss
        loss = criterion(logits, batch)

        # backprop.
        loss.backward()

        # apply gradients
        optimizer.step()

        # tensor.item() returns just the number, so we can print it
        pbar.set_description(f"Training - loss: {loss.item():.4f}")


# during testing we do not need to calculate gradients
# you can also use within functions:
# with torch.no_grad():
#     ... # no gradients will be calculated here
#


@torch.no_grad()
def test_epoch(model, test_loader, criterion, args):
    # set model to evaluation mode
    model.eval()

    losses = []

    pbar = tqdm(test_loader)
    for batch in pbar:
        batch = batch.to(args.device)

        logits, latents = model(batch)

        loss = criterion(logits, batch).item()

        losses.append(loss)

        pbar.set_description(f"Testing - loss: {loss:.4f}")

    out_images = logits.detach().cpu().numpy()
    # ni_img = nib.Nifti1Image(out_images[0][0], np.eye(4))
    # nib.save(ni_img, f"{output_dir}/tester{inter}.nii.gz")
    # print(out_images[0][0].shape)

    print(f"Average test loss: {sum(losses) / len(losses):.4f}")
    return out_images[0][0]


def train(model, train_loader, test_loader, criterion, optimizer, args):
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch: {epoch}/{args.epochs}")

        test_epoch(model, test_loader, criterion, args)
        train_epoch(model, train_loader, criterion, optimizer, args)

    # test one final time
    ret = test_epoch(model, test_loader, criterion, args)
    # normalized_arr = (ret - np.min(ret)) / (np.max(ret) - np.min(ret))
    # normalized_arr[normalized_arr < 1] = 0
    # # normalized_arr[normalized_arr >= 0.5] = 1
    # ret[ret < 0] = 0
    # ret[ret > 0] = 1

    return ret


def main(args):
    # set seeds for deterministic output
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # find best kernels for your hardware, may speed up training
    torch.cuda.benchmark = True

    # load data
    print("Loading data...")
    train_set = RadioDataset(100, train=True)
    input_shape = train_set[0][0].shape
    # print(train_set[0].shape)
    test_set = RadioDataset(100, train=False)
    # print(test_set[0].shape)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # load model
    model = AlexNetEmbedding()

    # loss function
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    print("Starting training...")
    ret = train(model, train_loader, test_loader, criterion, optimizer, args)

    ni_img = nib.Nifti1Image(ret, np.eye(4))
    nib.save(ni_img, f"{output_dir}/tester_zer3.nii.gz")
    # with np.printoptions(threshold=np.inf):
    #     print(ret)

    # ni_img = nib.Nifti1Image(normi, np.eye(4))
    # nib.save(ni_img, f"{output_dir}/tester_norm2.nii.gz")
    # with np.printoptions(threshold=np.inf):
    #     print(normi)
    unique, counts = np.unique(ret, return_counts=True)
    print(dict(zip(unique, counts)))
    # unique, counts = np.unique(normi, return_counts=True)
    # print(dict(zip(unique, counts)))


if __name__ == "__main__":
    parser = ArgumentParser("T-Network train script")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    print("Train arguments:")
    print(args)

    main(args)
