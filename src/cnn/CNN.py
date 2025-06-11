import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN1D(nn.Module):
    def __init__(self, input_length, num_classes=3):
        super(BasicCNN1D, self).__init__()
<<<<<<< HEAD

        # Convolutional layers: (1, 64, 128) input channel, (64, 128, 256) output channels
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Max pooling layer with kernel size 2 for downsampling
=======
        
        # Convolutional layers (adapted for 1D data)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
>>>>>>> c7aa17afb60b930db23a99a32aa3b39f18ba1c2f
        self.pool = nn.MaxPool1d(2)

        # Compute the output size after convolution and pooling layers to define the input to the fully connected layer
        self.output_size = self.calculate_output_size(input_length)
<<<<<<< HEAD

        # First fully connected layer with 512 hidden units
        self.fc1 = nn.Linear(self.output_size, 512)

        # Second fully connected layer for classification into `num_classes` categories
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout layer to prevent overfitting during training
=======
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        # Dropout layer
>>>>>>> c7aa17afb60b930db23a99a32aa3b39f18ba1c2f
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
