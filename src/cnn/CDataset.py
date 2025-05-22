import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomArrayDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Combine x and y arrays as input features
        # Method 1: Concatenate them (assuming same length)
        combined_features = np.concatenate([sample.x, sample.y])
        
        # Or Method 2: Stack them as separate channels
        # combined_features = np.stack([sample.x, sample.y])
        
        input_tensor = torch.tensor(combined_features, dtype=torch.float32)
        
        # Add channel dimension if needed (for Method 1)
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, length*2]
        
        labels_tensor = torch.tensor(sample.labels, dtype=torch.float32)
        
        return input_tensor, labels_tensor