import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from collections import defaultdict
import nibabel as nib
import gzip
import pickle





class Custom2D3DDataset(Dataset):
    def __init__(self, dataset_2d_dir, dataset_3d_dir, max_samples=50):
        """
        Args:
            dataset_2d_dir (str): Directory with all the 2D images.
            dataset_3d_dir (str): Directory with all the 3D images (e.g., numpy arrays).
        """
        self.dataset_2d_dir = dataset_2d_dir
        self.dataset_3d_dir = dataset_3d_dir

        # List and group 2D images by their common identifier
        self.dataset_2d_files = defaultdict(list)
        for f in os.listdir(dataset_2d_dir):
            if os.path.isfile(os.path.join(dataset_2d_dir, f)):
                common_id = os.path.splitext(f)[0].rsplit("_", 1)[
                    0
                ]  # assuming format like 'id_1.png', 'id_2.png'
                # print(common_id)
                self.dataset_2d_files[common_id].append(f)
        # print(common_id)
        # print(self.dataset_2d_files)
        # print("HEHERERERERE")
        # print(self.dataset_2d_files)
        # print(common_id)
        # List 3D images and match them with 2D image groups
        self.dataset_3d_files = []
        for f in os.listdir(dataset_3d_dir):
            if os.path.isfile(os.path.join(dataset_3d_dir, f)):
                common_id = os.path.splitext(f)[0]
                common_id = common_id.replace("torch_", "")
                common_id = common_id.replace("_Label.nii", "")
                # common_id = common_id.replace("torch_mirrored_Unk_", "")
                # common_id = common_id.replace("torch_mirrored_Art_", "")
                # common_id = common_id.replace("torch_Unk_", "")
                common_id = "img_" + common_id
                # print(common_id)

                if common_id in self.dataset_2d_files:
                    self.dataset_3d_files.append(f)
        # print(common_id)
        if max_samples is not None:
            self.dataset_3d_files = self.dataset_3d_files[:max_samples]

    def __len__(self):
        return len(self.dataset_3d_files)

    def __getitem__(self, idx):
        # Get the 3D image filename
        img_name_3d = self.dataset_3d_files[idx]
        common_id = os.path.splitext(img_name_3d)[0]
        common_id = common_id.replace("torch_Art_", "")
        common_id = common_id.replace("_Label.nii", "")
        common_id = common_id.replace("torch_Unk_", "")
        common_id = common_id.replace("torch_mirrored_Unk_", "")
        common_id = common_id.replace("torch_mirrored_Art_", "")
        common_id = "img_" + common_id
        # Load 3D image
        image_3d = torch.load(os.path.join(self.dataset_3d_dir, img_name_3d))
        # Load corresponding 2D images
        image_2d_list = []
        # if common_id in self.dataset_2d_files:
        #     for img_name_2d in self.dataset_2d_files[common_id]:
        #         with gzip.open(
        #             os.path.join(self.dataset_2d_dir, img_name_2d), "rb"
        #         ) as f:
        #             image_2d = pickle.load(f)
        #             image_2d = self.remove_transparency(image_2d).convert("L")
        #             image_2d = transforms.functional.to_tensor(image_2d)
        #             image_2d_list.append(image_2d)

        for img_name_2d in self.dataset_2d_files[common_id]:
            # print(img_name_2d)
            # image_2d = Image.open(os.path.join(self.dataset_2d_dir, img_name_2d))
            image_2d = torch.load(os.path.join(self.dataset_2d_dir, img_name_2d))
            # image_2d = remove_transparency(image_2d).convert("L")
            # image_2d = transforms.functional.to_tensor(image_2d)
            image_2d_list.append(image_2d)
        # print(len(image_2d_list))
        # sample = {"2d_images": image_2d_list, "3d_image": image_3d}
        sample = torch.stack(image_2d_list)
        # print(sample.shape)
        return sample, image_3d


# Example usage
if __name__ == "__main__":
    # Define transforms

    dataset_2d_dir = "/homes/yassin/E_ResearchData/DRR_tensors"
    dataset_3d_dir = "/homes/yassin/E_ResearchData/femur/tensors_label_02"

    # Create dataset
    custom_dataset = Custom2D3DDataset(dataset_2d_dir, dataset_3d_dir)
    print(len(custom_dataset))
    print(len(custom_dataset[0][0]))
    # Create DataLoader
