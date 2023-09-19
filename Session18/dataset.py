#!/usr/bin/env python3
"""
DataSet class for training UNet
Author: Shilpaj Bhalerao
Date: Sep 18, 2023
"""
# Standard Library Imports
import os

# Third-Party Imports
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class UNetDataset(Dataset):
    def __init__(self, img_dir_path, mask_dir_path, transforms):
        """
        Constructor
        """
        self.transforms = transforms

        self.img_dir_path = img_dir_path
        self.mask_dir_path = img_dir_path

        self.images_path, self.mask_path = self.get_image_mask_address()

    def get_image_mask_address(self):
        """
        Method to get the paths of all the images and masks for UNet training
        """
        # Read the images folder like a list
        image_dataset = os.listdir(self.img_dir_path)
        mask_dataset = os.listdir(self.mask_dir_path)

        # Make a list for images and masks filenames
        orig_img = []
        mask_img = []
        for file in image_dataset:
            orig_img.append(file)
        for file in mask_dataset:
            mask_img.append(file)

        # Sort the lists to get both of them in same order (the dataset has exactly the same name for images and corresponding masks)
        orig_img.sort()
        mask_img.sort()

        return orig_img, mask_img

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        """
        Returns Image and it's corresponding mask
        """
        # Load the image and mask
        path = os.path.join(self.img_dir_path, self.images_path[index])
        sample_img = Image.open(path).convert('RGB')

        path = os.path.join(self.mask_dir_path, self.mask_path[index])
        sample_mask = UNetDataset.preprocess_mask(Image.open(path))

        # Transform the raw data
        processed_img = self.train_transform(sample_img)
        processed_mask = self.test_transform(sample_mask)
        processed_mask -= 1

        return processed_img, processed_mask

    @staticmethod
    def preprocess_mask(mask):
        """
        Extract the mask from the trimap image
        """
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask
