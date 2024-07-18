import torch
from torch.utils.data import Dataset
import os


class Custom2D3DDataset(Dataset):
    def __init__(self, directory, max_samples=None):
        """
        Args:
            directory (str): Path to the directory containing paired tensor files.
        """
        self.directory = directory

        # Assuming the files are named consistently, e.g., 'sample_0_2d.pt' and 'sample_0_3d.pt'
        self.tensor_files_2d = sorted(
            [f for f in os.listdir(directory) if f.endswith("_2d.pt")]
        )
        self.tensor_files_3d = sorted(
            [f for f in os.listdir(directory) if f.endswith("_3d.pt")]
        )
        print(len(self.tensor_files_2d),len(self.tensor_files_3d))
        # Ensure the number of 2D and 3D files match
        # assert len(self.tensor_files_2d) == len(
        #     self.tensor_files_3d
        # ), "Number of 2D and 3D tensor files must match"

        if max_samples != None:
            self.tensor_files_2d = self.tensor_files_2d[:max_samples]
            self.tensor_files_3d = self.tensor_files_3d[:max_samples]

    def __len__(self):
        return len(self.tensor_files_2d)

    def __getitem__(self, idx):
        tensor_2d_path = os.path.join(self.directory, self.tensor_files_2d[idx])
        tensor_3d_path = os.path.join(self.directory, self.tensor_files_3d[idx])

        sample_2d = torch.load(tensor_2d_path)
        sample_3d = torch.load(tensor_3d_path)

        return sample_2d, sample_3d


# Example usage
# directory = "/homes/yassin/E_ResearchData/paired_tensors"
# dataset = Custom2D3DDataset(directory)

# # Accessing a sample
# sample_2d, sample_3d = dataset[0]
# print(sample_2d.shape, sample_3d.shape)  # The shapes depend on the actual tensor files
