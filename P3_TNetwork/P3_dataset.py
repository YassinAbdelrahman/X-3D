import torch
from torch.utils.data import Dataset
import os

class Custom2D3DDataset(Dataset):
    def __init__(self, directory, max_samples=None,val=False):
        """
        Args:
            directory (str): Path to the directory containing paired tensor files.
        """
        self.directory = directory
        self.tensor_files_2d = sorted(
            [f for f in os.listdir(directory) if f.endswith("_2d.pt")]
        )
        self.tensor_files_3d = sorted(
            [f for f in os.listdir(directory) if f.endswith("_3d.pt")]
        )

        if max_samples != None and val==False:
            max_samples = int(0.9*max_samples)
            self.tensor_files_2d = self.tensor_files_2d[:max_samples]
            self.tensor_files_3d = self.tensor_files_3d[:max_samples]
        if max_samples != None and val==True:
            min_samples = int(0.9*max_samples)
            self.tensor_files_2d = self.tensor_files_2d[min_samples:max_samples]
            self.tensor_files_3d = self.tensor_files_3d[min_samples:max_samples]

    def __len__(self):
        return len(self.tensor_files_2d)

    def __getitem__(self, idx):
        tensor_2d_path = os.path.join(self.directory, self.tensor_files_2d[idx])
        tensor_3d_path = os.path.join(self.directory, self.tensor_files_3d[idx])

        sample_2d = torch.load(tensor_2d_path)
        sample_3d = torch.load(tensor_3d_path)

        return sample_2d, sample_3d


# Example usage
# directory = "/nethome/2514818/Data/final_data"
# dataset = Custom2D3DDataset(directory)

# # Accessing a sample
# sample_2d, sample_3d = dataset[0]
# print(sample_2d.shape, sample_3d.shape)  # The shapes depend on the actual tensor files
