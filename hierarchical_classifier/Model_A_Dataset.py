import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Model_A_Dataset(Dataset):
    def __init__(self, dataframe, image_dir, category_mapping, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the image ids and their corresponding targets
            image_dir (str): Directory where images are stored
            category_mapping (dict): Dictionary mapping label names to indices (e.g., { 'No Finding': 0, 'Finding': 1 })
            transform (callable, optional): Optional transformation to be applied on a sample
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.category_mapping = category_mapping

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image ID and target label from the DataFrame
        img_id = self.dataframe.iloc[idx]['id']  # Assuming 'id' contains the image filename
        target_value = self.dataframe.iloc[idx]['No Finding']  # 'No Finding' is binary target

        # Convert the target value to the mapped label using category_mapping
        label = self.category_mapping['No Finding'] if target_value else self.category_mapping['Finding']

        # Load the image
        img_path = os.path.join(self.image_dir, f"{img_id}")  # Ensure image ID matches filename format
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return the image, label, and image ID for tracking
        return image, label, img_id
