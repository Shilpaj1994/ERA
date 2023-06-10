#!/usr/bin/env python3
"""
Script containing model architectures
"""
# Third-Party Imports
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This defines the structure of the NN.
    """
    def __init__(self):
        """
        Constructor
        """
        # Initialize the Module class
        super(Net, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)

        # Fully Connected Layers
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Forward pass for model training
        :param x: Input layer
        :return: Output of the model
        """
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Network(nn.Module):
    """
    Increasing number of channels with each block strategy for the model
    """
    def __init__(self):
        """
        Constructor
        """
        # Convolution Block-1
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, bias=False)    # Input - 28    Output - 26    Receptive Field - 3
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False)   # Input - 26    Output - 24    Receptive Field - 5
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False)   # Input - 24    Output - 22    Receptive Field - 7
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False)   # Input - 22    Output - 20    Receptive Field - 9
        self.batch_norm4 = nn.BatchNorm2d(16)

        # Transition-1
        self.point_1 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=1, bias=False)  # Input - 20    Output - 20    Receptive Field - 9
        self.batch_norm_point1 = nn.BatchNorm2d(12)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                # Input - 20    Output - 10    Receptive Field - 10

        # Convolution Block-2
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, bias=False)  # Input - 10    Output - 8   Receptive Field - 14
        self.batch_norm5 = nn.BatchNorm2d(20)
        self.conv6 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, bias=False)  # Input - 8    Output - 6    Receptive Field - 18
        self.batch_norm6 = nn.BatchNorm2d(20)
        self.conv7 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, bias=False)  # Input - 6    Output - 4    Receptive Field - 22
        self.batch_norm7 = nn.BatchNorm2d(20)
        self.conv8 = nn.Conv2d(in_channels=20, out_channels=16, kernel_size=3, bias=False)  # Input - 4    Output - 2    Receptive Field - 26
        self.batch_norm8 = nn.BatchNorm2d(16)

        # Transition-2
        self.point_2 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False)  # Input - 2    Output - 2    Receptive Field - 26
        self.batch_norm_point2 = nn.BatchNorm2d(10)

        self.gap = nn.AvgPool2d(2)                                                            # Input - 2    Output - 1    Receptive Field - 28
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        """
        Forward Pass
        """
        # Convolution Block-1
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(self.dropout(x))

        x = F.relu(self.conv2(x))
        x = self.batch_norm2(self.dropout(x))

        x = F.relu(self.conv3(x))
        x = self.batch_norm3(self.dropout(x))

        x = F.relu(self.conv4(x))
        x = self.batch_norm4(self.dropout(x))

        # Transition Block-1
        x = self.max_pool1(F.relu(self.point_1(x)))
        x = self.batch_norm_point1(self.dropout(x))

        # Convolution Block-2
        x = F.relu(self.conv5(x))
        x = self.batch_norm5(self.dropout(x))

        x = F.relu(self.conv6(x))
        x = self.batch_norm6(self.dropout(x))

        x = F.relu(self.conv7(x))
        x = self.batch_norm7(self.dropout(x))

        x = F.relu(self.conv8(x))
        x = self.batch_norm8(self.dropout(x))

        # Transition Block-2
        x = F.relu(self.point_2(x))
        x = self.batch_norm_point2(self.dropout(x))

        x = self.gap(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=0)