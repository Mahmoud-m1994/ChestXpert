import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class MultiSourceChestXRayDataset(Dataset):
    def __init__(self, dataframe, image_dirs, category_mapping, transform=None, augment_transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image metadata (e.g., 'id', 'finding_encoded', 'is_augmented')
            image_dirs (dict): Dictionary with directories for 'train' and 'augmented' images
            category_mapping (dict): Dictionary mapping category names to indices
            transform (callable, optional): Transform for non-augmented images
             (callable, optional): Transform for augmented images
        """
        print(augment_transform)
        self.dataframe = dataframe
        self.image_dirs = image_dirs
        self.category_mapping = category_mapping
        self.transform = transform
        self.augment_transform = augment_transform
        
        self.categories = list(category_mapping.keys())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract metadata
        row = self.dataframe.iloc[idx]
        img_id = row['id']
        finding_encoded = row['finding_encoded']
        is_augmented = row['is_augmented']

        # Convert `finding_encoded` to a list of integers
        if isinstance(finding_encoded, str) and finding_encoded.strip():
            finding_encoded = [int(i) for i in finding_encoded.split(',') if i.strip()]
        else:
            finding_encoded = []

        # One-hot encode the labels
        label = np.zeros(len(self.categories))
        for finding in finding_encoded:
            if finding in self.category_mapping.values():
                label[finding] = 1
        label = torch.tensor(label, dtype=torch.float)

        # Determine the image directory
        source_dir = self.image_dirs['augmented'] if is_augmented else self.image_dirs['train']
        img_path = os.path.join(source_dir, img_id)

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Apply the appropriate transformation
        if is_augmented and self.augment_transform:
            image = self.augment_transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, label, img_id
