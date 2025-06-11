import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom dataset class for handling array-like input samples
class CustomArrayDataset(Dataset):
    def __init__(self, samples):
        # Store the list of samples (assumed to be objects with `.y` and `.labels` attributes)
        self.samples = samples

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve the sample at the specified index
        sample = self.samples[idx]

        # Convert the input spectrum `y` to a float tensor and add a channel dimension: [length] -> [1, length]
        input_tensor = torch.tensor(sample.y, dtype=torch.float32).unsqueeze(0)

        # Convert the labels to a float tensor (e.g., for multi-label classification)
        labels_tensor = torch.tensor(sample.labels, dtype=torch.float32)
<<<<<<< HEAD

        # Return a tuple of (input tensor, label tensor)
        return input_tensor, labels_tensor
=======
        weight = torch.tensor(sample.weight, dtype=torch.float32)
        return input_tensor, labels_tensor, weight
>>>>>>> c7aa17afb60b930db23a99a32aa3b39f18ba1c2f
