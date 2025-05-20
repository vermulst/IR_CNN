import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from jcamp import jcamp_readfile

def load_jcamp_as_tensor(filepath, input_length=1024):
    data = jcamp_readfile(filepath)
    y = np.array(data['y'], dtype=np.float32)

    # Resize to expected input length
    if len(y) < input_length:
        # Pad with zeros
        padded = np.zeros(input_length, dtype=np.float32)
        padded[:len(y)] = y
        y = padded
    else:
        y = y[:input_length]

    # Normalize (optional, but recommended)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Reshape to [batch, channels, length]
    tensor = torch.tensor(y).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, input_length]
    return tensor


class BasicCNN1D(nn.Module):
    def __init__(self, input_length, num_classes=10):
        super(BasicCNN1D, self).__init__()
        
        # Convolutional layers (adapted for 1D data)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer (adapted for 1D)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate flattened dimension
        self.flattened_size = 64 * (input_length // 8)  # 3 pooling layers reduce length by 2^3
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output
        x = x.view(-1, self.flattened_size)
        
        # Fully connected layers with dropout
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Path to your JCAMP file
# Relative to script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, "..", "data", "public", "samples", "00a0ee9a-ac02-4c9e-96a6-656d069fb80a")
filepath = os.path.normpath(relative_path)
# Step 1: Load data
input_length = 1024
input_tensor = load_jcamp_as_tensor(filepath, input_length=input_length)

# Step 2: Create model
model = BasicCNN1D(input_length=input_length, num_classes=10)

# Step 3: Set model to eval mode and test
model.eval()
with torch.no_grad():
    output = model(input_tensor)

print("Model output:", output)
print("Predicted class:", torch.argmax(output, dim=1).item())
