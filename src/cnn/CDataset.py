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
        # Convert spectrum to tensor [length] -> [1, length]
        input_tensor = torch.tensor(sample.y, dtype=torch.float32).unsqueeze(0)
        labels_tensor = torch.tensor(sample.labels, dtype=torch.float32)
        weight = torch.tensor(sample.weight, dtype=torch.float32)
        return input_tensor, labels_tensor, weight