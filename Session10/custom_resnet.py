#!/usr/bin/env python3
"""
Script containing model architectures
Author: Shilpaj Bhalerao
"""
# Third-Party Imports
import torch.nn as nn
import torch.nn.functional as F


class Session10Net(nn.Module):
    """
    David's Model Architecture for Session-10 CIFAR10 dataset
    """
    def __init__(self):
        """
        Constructor
        """
        # Initialize the Module class
        super(Session10Net, self).__init__()

        # Dropout value of 10%
        self.dropout_value = 0.1

        # Prep Layer
        self.prep_layer = self.standard_conv_layer(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)

        # Convolutional Block-1
        self.custom_block1 = Session10Net.custom_block(input_channels=64, output_channels=128)
        self.resnet_block1 = Session10Net.resnet_block(channels=128)

        # Convolutional Block-2
        self.custom_block2 = Session10Net.custom_block(input_channels=128, output_channels=256)

        # Convolutional Block-3
        self.custom_block3 = Session10Net.custom_block(input_channels=256, output_channels=512)
        self.resnet_block3 = Session10Net.resnet_block(channels=512)

        # MaxPool Layer
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=2)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=512, out_features=10, bias=False)

    def forward(self, x):
        """
        Forward pass for model training
        :param x: Input layer
        :return: Model Prediction
        """
        # Prep Layer
        x = self.prep_layer(x)

        # Convolutional Block-1
        x = self.custom_block1(x)
        r1 = self.resnet_block1(x)
        x = x + r1

        # Convolutional Block-2
        x = self.custom_block2(x)

        # Convolutional Block-3
        x = self.custom_block3(x)
        r2 = self.resnet_block3(x)
        x = x + r2

        # MaxPool Layer
        x = self.pool4(x)

        # Fully Connected Layer
        x = x.view(-1, 512)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

    def standard_conv_layer(self, in_channels: int,
                            out_channels: int,
                            kernel_size: int = 3,
                            padding: int = 0,
                            stride: int = 1,
                            dilation: int = 1,
                            normalization: str = "batch",
                            last_layer: bool = False,
                            conv_type: str = "standard"):
        """
        Method to return a standard convolution block
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel used in the layer
        :param padding: Padding used in the layer
        :param stride: Stride used for convolution
        :param dilation: Dilation for Atrous convolution
        :param normalization: Type of normalization technique used
        :param last_layer: Flag to indicate if the layer is last convolutional layer of the network
        :param conv_type: Type of convolutional layer
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

        # Select the convolution layer type
        if conv_type == "standard":
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, bias=False, padding=padding)
        elif conv_type == "depthwise":
            conv_layer = Session10Net.depthwise_conv(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding)
        elif conv_type == "dilated":
            conv_layer = Session10Net.dilated_conv(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding, dilation=dilation)

        # For last layer only return the convolution output
        if last_layer:
            return nn.Sequential(conv_layer)
        return nn.Sequential(
            conv_layer,
            _norm_layer,
            nn.ReLU(),
            # nn.Dropout(self.dropout_value)
        )

    @staticmethod
    def resnet_block(channels):
        """
        Method to create a RESNET block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, stride=1, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, stride=1, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    @staticmethod
    def custom_block(input_channels, output_channels):
        """
        Method to create a custom configured block
        :param input_channels: Number of input channels
        :param output_channels: Number of output channels
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, stride=1, kernel_size=3, bias=False, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    @staticmethod
    def depthwise_conv(in_channels, out_channels, stride=1, padding=0):
        """
        Method to return the depthwise separable convolution layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param padding: Padding used in the layer
        :param stride: Stride used for convolution
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=stride, groups=in_channels, kernel_size=3, bias=False, padding=padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1, bias=False, padding=0)
        )

    @staticmethod
    def dilated_conv(in_channels, out_channels, stride=1, padding=0, dilation=1):
        """
        Method to return the dilated convolution layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride used for convolution
        :param padding: Padding used in the layer
        :param dilation: Dilation value for a kernel
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, bias=False,
                      padding=padding, dilation=dilation)
        )
