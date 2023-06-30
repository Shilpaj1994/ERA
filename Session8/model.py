#!/usr/bin/env python3
"""
Script containing model architectures
"""
# Third-Party Imports
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------- Session-8 --------------------------------
class Session8(nn.Module):
    """
    Model for Session-9 CIFAR10 dataset
    """
    def __init__(self, normalization='batch'):
        """
        Constructor
        """
        # Initialize the Module class
        super(Session8, self).__init__()

        # Dropout value of 10%
        self.dropout_value = 0.1

        # Convolutional Block-1
        self.conv_block1 = self.standard_conv_layer(in_channels=3, out_channels=20, kernel_size=3, padding=1, normalization=normalization)
        self.conv_block2 = self.standard_conv_layer(in_channels=20, out_channels=20, kernel_size=3, padding=1, normalization=normalization)

        # Transitional Block-1
        self.conv_point1 = self.standard_conv_layer(in_channels=20, out_channels=16, kernel_size=1, padding=0, normalization=normalization)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block-2
        self.conv_block3 = self.standard_conv_layer(in_channels=16, out_channels=26, kernel_size=3, padding=1, normalization=normalization)
        self.conv_block4 = self.standard_conv_layer(in_channels=26, out_channels=26, kernel_size=3, padding=1, normalization=normalization)
        self.conv_block5 = self.standard_conv_layer(in_channels=26, out_channels=26, kernel_size=3, padding=1, normalization=normalization)

        # Transitional Block-2
        self.conv_point2 = self.standard_conv_layer(in_channels=26, out_channels=16, kernel_size=1, padding=0, normalization=normalization)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block-3
        self.conv_block7 = self.standard_conv_layer(in_channels=16, out_channels=32, kernel_size=3, padding=1, normalization=normalization)
        self.conv_block8 = self.standard_conv_layer(in_channels=32, out_channels=32, kernel_size=3, padding=2, normalization=normalization)
        self.conv_block9 = self.standard_conv_layer(in_channels=32, out_channels=32, kernel_size=3, padding=1, normalization=normalization)

        # Global Average Pooling
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1))

        # Output Layer
        self.conv_block10 = self.standard_conv_layer(in_channels=32, out_channels=10, kernel_size=1, padding=0, last_layer=True, normalization=normalization)

    def forward(self, x):
        """
        Forward pass for model training
        :param x: Input layer
        :return: Model Prediction
        """
        # Convolutional Block-1
        x = self.conv_block1(x)
        x = x + self.conv_block2(x)

        # Transitional Block-1
        x = self.conv_point1(x)
        x = self.pool1(x)

        # Convolutional Block-2
        x = self.conv_block3(x)
        x = x + self.conv_block4(x)
        x = x + self.conv_block5(x)

        # Transitional Block-2
        x = self.conv_point2(x)
        x = self.pool2(x)

        # Convolutional Block-3
        x = self.conv_block7(x)
        x = x + self.conv_block8(x)
        x = x + self.conv_block9(x)

        x = self.gap(x)
        x = self.conv_block10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

    def standard_conv_layer(self, in_channels: int,
                            out_channels: int,
                            kernel_size: int = 3,
                            padding: int = 0,
                            stride: int = 1,
                            normalization: str = "batch",
                            last_layer: bool = False):
        """
        Method to return a standard convolution block
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel used in the layer
        :param padding: Padding used in the layer
        :param stride: Stride used for convolution
        :param normalization: Type of normalization technique used
        :param last_layer: Flag to indicate if the layer is last convolutional layer of the network
        """
        # Select normalization type
        if normalization == "layer":
            _norm_layer = nn.GroupNorm(1, out_channels)
        elif normalization == "group":
            if not self.group:
                raise ValueError("Value of group is not defined")
            _norm_layer = nn.GroupNorm(self.groups, out_channels)
        else:
            _norm_layer = nn.BatchNorm2d(out_channels)

        # For last layer only return the convolution output
        if last_layer:
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, bias=False, padding=padding))
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, bias=False, padding=padding),
            nn.ReLU(),
            _norm_layer,
            nn.Dropout(self.dropout_value)
        )


# -------------------------------- Session-7 --------------------------------
class Session7_1(nn.Module):
    """
    Model for Session7 first iteration of MNIST dataset
    """
    def __init__(self):
        super(Session7_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(256, 512, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(512, 1024, 3) # 5 > 3 | 32 | 3*3*1024 | 3x3x1024x10 |
        self.conv7 = nn.Conv2d(1024, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)


class Session7_5(nn.Module):
    """
    Session 7, iteration 5 GAP model for MNIST
    """
    def __init__(self):
        super(Session7_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 3, padding=0, bias=False)       # 28 > 26 | 3
        self.batch_norm1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, 3, padding=0, bias=False)      # 26 > 24 | 5
        self.batch_norm2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 12, 3, padding=0, bias=False)      # 24 > 22 | 7
        self.batch_norm3 = nn.BatchNorm2d(12)
        self.conv4 = nn.Conv2d(12, 12, 3, padding=0, bias=False)      # 22 > 20 | 9
        self.batch_norm4 = nn.BatchNorm2d(12)

        self.pool1 = nn.MaxPool2d(2, 2)                               # 20 > 10 | 10

        self.conv5 = nn.Conv2d(12, 16, 3, padding=0, bias=False)      # 10 > 8 | 14
        self.batch_norm5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 10, 3, padding=0, bias=False)      #  8 > 6 | 18
        self.batch_norm6 = nn.BatchNorm2d(10)

        self.gap = nn.AvgPool2d(kernel_size=6)                        # 6 > 1 | 28

    def forward(self, x):
        x = self.batch_norm1(F.relu(self.conv1(x)))
        x = self.batch_norm2(F.relu(self.conv2(x)))
        x = self.batch_norm3(F.relu(self.conv3(x)))
        x = self.batch_norm4(self.conv4(x))

        x = self.pool1(x)

        x = self.batch_norm5(F.relu(self.conv5(x)))
        x = self.batch_norm6(F.relu(self.conv6(x)))
        x = self.gap(x)

        x = x.view(-1, 10)


class Session7Final(nn.Module):
    """
    Session7 Iteration 7 model for MNIST dataset
    """
    def __init__(self):
        super(Session7Final, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=0, bias=False)       # 28 > 26 | 3
        self.batch_norm1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=0, bias=False)      # 26 > 24 | 5
        self.batch_norm2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0, bias=False)      # 24 > 22 | 7
        self.batch_norm3 = nn.BatchNorm2d(10)
        self.conv4 = nn.Conv2d(10, 10, 3, padding=0, bias=False)      # 22 > 20 | 9
        self.batch_norm4 = nn.BatchNorm2d(10)

        self.point1 = nn.Conv2d(10, 10, 1, padding=0, bias=False)     # 20 > 20 | 9
        self.batch_normp1 = nn.BatchNorm2d(10)
        self.pool1 = nn.MaxPool2d(2, 2)                               # 20 > 10 | 10

        self.conv5 = nn.Conv2d(10, 12, 3, padding=0, bias=False)      # 10 > 8 | 14
        self.batch_norm5 = nn.BatchNorm2d(12)
        self.conv6 = nn.Conv2d(12, 12, 3, padding=0, bias=False)      #  8 > 6 | 18
        self.batch_norm6 = nn.BatchNorm2d(12)
        self.conv7 = nn.Conv2d(12, 12, 3, padding=0, bias=False)      #  6 > 4 | 22
        self.batch_norm7 = nn.BatchNorm2d(12)

        self.gap = nn.AdaptiveAvgPool2d(1)                            # 4 > 1 | 28

        self.point2 = nn.Conv2d(12, 10, 1, padding=0, bias=False)     # 1 > 1 | 28
        self.batch_normp2 = nn.BatchNorm2d(10)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.batch_norm4(self.conv4(x))

        x = self.pool1(self.batch_normp1(self.point1(x)))

        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = F.relu(self.batch_norm6(self.conv6(x)))
        x = self.batch_norm7(self.conv7(x))

        x = self.gap(x)

        x = self.batch_normp2(self.point2(x))

        x = x.view(-1, 10)                                             # 1x1x10 > 10
        return F.log_softmax(x, dim=-1)


# -------------------------------- Session-6 --------------------------------

class Session6(nn.Module):
    """
    This defines the structure of the NN.
    """
    def __init__(self):
        """
        Constructor
        """
        # Initialize the Module class
        super(Session6, self).__init__()

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


# -------------------------------- Session-5 --------------------------------


class Session5(nn.Module):
    """
    This defines the structure of the NN.
    """
    def __init__(self):
        """
        Constructor
        """
        # Initialize the Module class
        super(Session5, self).__init__()

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