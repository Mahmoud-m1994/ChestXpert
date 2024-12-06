import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, image_dir, category_mapping, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.category_mapping = category_mapping  # Mapping from category name to index

        # Create a list of all possible categories for one-hot encoding
        self.categories = list(category_mapping.keys())

        #print(self.categories)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image ID and finding_encoded from the DataFrame
        img_id = self.dataframe.iloc[idx]['id']
        finding_encoded = self.dataframe.iloc[idx]['finding_encoded']  # Comma-separated string of labels

        #print(finding_encoded)

        # Check if the finding_encoded string is not empty and handle invalid or empty values
        if isinstance(finding_encoded, str) and finding_encoded.strip():  # Check if it's a non-empty string
            # Convert the finding_encoded string to a list of integers (e.g., "5,13,15" -> [5, 13, 15])
            finding_encoded = [int(i) for i in finding_encoded.split(',') if i.strip()]
        else:
            # If the finding_encoded is empty or invalid, set it to an empty list
            finding_encoded = []

        # Create a one-hot encoded tensor for the labels
        label = np.zeros(len(self.categories))  # Start with an array of zeros
        for finding in finding_encoded:
            if finding in self.category_mapping.values():
                label[finding] = 1  # Set the index corresponding to the finding to 1

        label = torch.tensor(label, dtype=torch.float)  # Use torch.float for multi-label classification

        # Load the image
        img_path = os.path.join(self.image_dir, f"{img_id}")  # Adjust extension if necessary
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Final, return image with its label(s) and id (to debug or print)
        return image, label, img_id

