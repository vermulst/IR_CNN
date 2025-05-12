import torch
import torch.nn as nn
import torch.nn.functional as F

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