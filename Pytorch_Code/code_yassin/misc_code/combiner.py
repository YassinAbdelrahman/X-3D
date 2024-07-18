import os
import torch
from collections import defaultdict
from PIL import Image


def remove_transparency(im, bg_colour=(255, 255, 255)):
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        alpha = im.convert("RGBA").split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


def load_and_save_2d_3d_images(
    dataset_2d_dir, dataset_3d_dir, save_dir, max_samples=None
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_2d_files = defaultdict(list)
    for f in sorted(os.listdir(dataset_2d_dir)):
        if os.path.isfile(os.path.join(dataset_2d_dir, f)):
            common_id = os.path.splitext(f)[0].rsplit("_", 1)[0]
            dataset_2d_files[common_id].append(f)
    dataset_3d_files = []
    for f in sorted(os.listdir(dataset_3d_dir)):
        if os.path.isfile(os.path.join(dataset_3d_dir, f)):
            common_id = (
                os.path.splitext(f)[0].replace("torch_", "").replace("_Label.nii", "")
            )
            common_id = "img_" + common_id
            if common_id in dataset_2d_files:
                dataset_3d_files.append(f)
    print(len(dataset_3d_files))

    if max_samples is not None:
        dataset_3d_files = dataset_3d_files[:max_samples]

    paired_images = []
    for img_name_3d in dataset_3d_files:
        common_id = (
            os.path.splitext(img_name_3d)[0]
            .replace("torch_", "")
            .replace("_Label.nii", "")
        )
        common_id = "img_" + common_id
        print(common_id)

        image_3d = torch.load(os.path.join(dataset_3d_dir, img_name_3d))
        image_2d_list = [
            torch.load(os.path.join(dataset_2d_dir, img_name_2d))
            for img_name_2d in dataset_2d_files[common_id]
        ]
        print(len(image_2d_list))

        sample = torch.stack(image_2d_list)
        paired_images.append((sample, image_3d))

        # Save the paired images
        torch.save(sample, os.path.join(save_dir, f"{common_id}_2d.pt"))
        torch.save(image_3d, os.path.join(save_dir, f"{common_id}_3d.pt"))
    return paired_images


# Example usage
if __name__ == "__main__":
    dataset_2d_dir = "/homes/yassin/E_ResearchData/DRR_tensors_cropped"
    dataset_3d_dir = "/homes/yassin/E_ResearchData/femur/tensors_label_02"
    save_dir = "/homes/yassin/E_ResearchData/paired_tensors_cropped"

    paired_images = load_and_save_2d_3d_images(dataset_2d_dir, dataset_3d_dir, save_dir)
    print(len(paired_images))
    if len(paired_images) > 0:
        print(len(paired_images[0][0]))
