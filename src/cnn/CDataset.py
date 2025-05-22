import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomArrayDataset(Dataset):
    def __init__(self, data_array):
        self.data_array = data_array

    def __len__(self):
        # The total number of samples in your dataset
        return len(self.data_array)

    def __getitem__(self, idx):
        # Retrieve the 'x' and 'y' for the given index
        x_data = self.data_array[idx]['x']
        y_data = self.data_array[idx]['y']

        # Convert NumPy arrays to PyTorch tensors
        # For 1D CNN, 'x' should be (channels, length). If your 'x_data' is just (length,),
        # add an extra dimension for channels (e.g., 1 channel).
        x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(0) # Adds a channel dimension
        
        # 'y' should be a float tensor for BCEWithLogitsLoss or BCELoss
        y_tensor = torch.tensor(y_data, dtype=torch.float32)

        return x_tensor, y_tensor