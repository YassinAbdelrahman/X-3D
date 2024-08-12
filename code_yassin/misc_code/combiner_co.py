import os
from collections import defaultdict
import torch
def load_and_save_2d_3d_images(dataset_2d_dir, dataset_3d_dir, save_dir, max_samples=None):
    batch_am = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_2d_files = defaultdict(list)
    for f in sorted(os.listdir(dataset_2d_dir)):
        if os.path.isfile(os.path.join(dataset_2d_dir, f)):
            common_id = os.path.splitext(f)[0].rsplit("_", 1)[0]
            
            dataset_2d_files[common_id].append(f)
    print(len(dataset_2d_files))
    dataset_3d_files = []
    for f in sorted(os.listdir(dataset_3d_dir)):
        if os.path.isfile(os.path.join(dataset_3d_dir, f)):
            common_id = os.path.splitext(f)[0].replace("torch_", "").replace("_Label", "")
            common_id = "img_" + common_id
            print(common_id)
            if common_id in dataset_2d_files:
                dataset_3d_files.append(f)

    print(len(dataset_3d_files))

    if max_samples is not None:
        dataset_3d_files = dataset_3d_files[:max_samples]

    paired_images = []
    for img_name_3d in dataset_3d_files:
        common_id = os.path.splitext(img_name_3d)[0].replace("torch_", "").replace("_Label", "")
        common_id = "img_" + common_id
        print(common_id)

        image_3d = torch.load(os.path.join(dataset_3d_dir, img_name_3d))
        image_2d_list = [
            torch.load(os.path.join(dataset_2d_dir, img_name_2d))
            for img_name_2d in dataset_2d_files[common_id]
        ]
        print(len(image_2d_list))

        # Batch the 2D images into sets of 10 tensors each
        for i in range(0, len(image_2d_list), 10):
            batch_2d_images = torch.stack(image_2d_list[i:i + 10])
            batch_id = f"{common_id}_batch_{i // 10 + 1}"
            paired_images.append((batch_2d_images, image_3d))

            # Save the batches of paired images
            torch.save(batch_2d_images, os.path.join(save_dir, f"{batch_id}_2d.pt"))
            torch.save(image_3d, os.path.join(save_dir, f"{batch_id}_3d.pt"))

            print(f"saved {batch_id}_2d.pt")
            batch_am += 1
    print(batch_am, int(batch_am/4))
    return paired_images


# Example usage
if __name__ == "__main__":
    dataset_2d_dir = "Data/DRR_tensors_cropped"
    dataset_3d_dir = "Data/tensors_02"
    save_dir = "Data/final_data"

    paired_images = load_and_save_2d_3d_images(dataset_2d_dir, dataset_3d_dir, save_dir)
    print(len(paired_images))
    if len(paired_images) > 0:
        print(len(paired_images[0][0]))
