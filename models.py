## CNN architecture for facial keypoint detection

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layers
        # 1 input image channel (grayscale), 32 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        # 32 input channels, 64 output channels, 3x3 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 3)
        # 64 input channels, 128 output channels, 3x3 square convolution kernel
        self.conv3 = nn.Conv2d(64, 128, 3)
        # 128 input channels, 256 output channels, 2x2 square convolution kernel
        self.conv4 = nn.Conv2d(128, 256, 2)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layers to prevent overfitting
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)

        # Fully connected layers
        # The input size to the first fully connected layer depends on the output size of the last convolutional layer
        # For a 224x224 input image, after the convolutional and pooling layers, the size would be:
        # 224 -> 220 (after conv1) -> 110 (after pool) -> 108 (after conv2) -> 54 (after pool) -> 
        # 52 (after conv3) -> 26 (after pool) -> 25 (after conv4) -> 12 (after pool)
        # So the input to the first fully connected layer would be 256 * 12 * 12 = 36864
        # However, we'll use a dynamic calculation in the forward method to handle different input sizes

        self.fc1 = nn.Linear(256 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)  # 136 values: 2 for each of the 68 keypoint (x, y) pairs

        # Initialize weights using Xavier initialization
        I.xavier_uniform_(self.conv1.weight)
        I.xavier_uniform_(self.conv2.weight)
        I.xavier_uniform_(self.conv3.weight)
        I.xavier_uniform_(self.conv4.weight)
        I.xavier_uniform_(self.fc1.weight)
        I.xavier_uniform_(self.fc2.weight)
        I.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # Convolutional layers with ReLU activation, max pooling, and dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)

        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(x)

        # Flatten the output for the fully connected layers
        # Get the dimensions of the current tensor
        batch_size = x.size(0)
        shape = x.size()[1:]
        flattened_size = 1
        for s in shape:
            flattened_size *= s

        x = x.view(batch_size, flattened_size)

        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)

        x = F.relu(self.fc2(x))
        x = self.dropout5(x)

        x = self.fc3(x)

        return x
