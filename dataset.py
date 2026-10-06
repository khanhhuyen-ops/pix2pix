import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import config


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = [
            os.path.join(self.root_dir, f)
            for f in sorted(os.listdir(self.root_dir))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not self.list_files:
            raise ValueError(f"No images found in dataset folder: {self.root_dir}")

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_path = self.list_files[index]
        image = np.array(Image.open(img_path).convert("RGB"))

        if image.shape[1] < 2:
            raise ValueError(f"Image width too small to split input/output: {img_path}")

        split_idx = image.shape[1] // 2
        input_image = image[:, :split_idx, :]
        target_image = image[:, split_idx:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image