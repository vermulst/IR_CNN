import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN1D(nn.Module):
    def __init__(self, input_length, num_classes=3):
        super(BasicCNN1D, self).__init__()
        
        # Convolutional layers (adapted for 1D data)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)

        # Compute the output size after convolution and pooling layers to define the input to the fully connected layer
        self.output_size = self.calculate_output_size(input_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.output_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        # Dropout layer
        self.dropout = nn.Dropout(0.25)

    def calculate_output_size(self, input_length):
        # Simulate a forward pass with dummy input to determine the output feature size
        with torch.no_grad():
            x = torch.zeros(1, 1, input_length)  # Dummy input tensor
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.numel()  # Return the total number of features

    def forward(self, x):
        # Apply convolution + ReLU activation + pooling in sequence
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layer with ReLU and dropout
        x = self.dropout(F.relu(self.fc1(x)))

        # Final output layer (before activation)
        x = self.fc2(x)

        # Apply sigmoid activation for output (suitable for multi-label classification)
        return torch.sigmoid(x)
