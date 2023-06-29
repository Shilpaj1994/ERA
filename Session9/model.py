#!/usr/bin/env python3
"""
Script containing model architectures
Author: Shilpaj Bhalerao
"""
# Third-Party Imports
import torch.nn as nn
import torch.nn.functional as F


class Session9Net(nn.Module):
    """
    Model for Session-9 CIFAR10 dataset
    """
    def __init__(self):
        """
        Constructor
        """
        # Initialize the Module class
        super(Session9Net, self).__init__()

        # Dropout value of 10%
        self.dropout_value = 0.1

        # Convolutional Block-1
        self.conv_block1 = self.standard_conv_layer(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_block2 = self.standard_conv_layer(in_channels=32, out_channels=32, kernel_size=3, padding=2, dilation=2, conv_type="dilated")
        self.conv_block3 = self.standard_conv_layer(in_channels=32, out_channels=32, kernel_size=3, padding=1, dilation=2, stride=2, conv_type="dilated")

        # Convolutional Block-2
        self.conv_block4 = self.standard_conv_layer(in_channels=32, out_channels=38, kernel_size=3, padding=1)
        self.conv_block5 = self.standard_conv_layer(in_channels=38, out_channels=38, kernel_size=3, padding=2, dilation=2, conv_type="dilated")
        self.conv_block6 = self.standard_conv_layer(in_channels=38, out_channels=38, kernel_size=3, padding=1, dilation=2, stride=2, conv_type="dilated")

        # Convolutional Block-3
        self.conv_block7 = self.standard_conv_layer(in_channels=38, out_channels=40, kernel_size=3, padding=1, conv_type="depthwise")
        self.conv_block8 = self.standard_conv_layer(in_channels=40, out_channels=40, kernel_size=3, padding=2, dilation=2, conv_type="dilated")
        self.conv_block9 = self.standard_conv_layer(in_channels=40, out_channels=40, kernel_size=3, padding=1, dilation=2, stride=2, conv_type="dilated")

        # Convolutional Block-4
        self.conv_block10 = self.standard_conv_layer(in_channels=40, out_channels=64, kernel_size=3, padding=1)
        self.conv_block11 = self.standard_conv_layer(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2, conv_type="dilated")
        self.conv_block12 = self.standard_conv_layer(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2, conv_type="dilated")

        # Global Average Pooling
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1))

        # Output Layer
        self.conv_block14 = self.standard_conv_layer(in_channels=64, out_channels=10, kernel_size=1, padding=0, last_layer=True)

    def forward(self, x):
        """
        Forward pass for model training
        :param x: Input layer
        :return: Model Prediction
        """
        # Convolutional Block-1
        x = self.conv_block1(x)
        x = x + self.conv_block2(x)
        x = self.conv_block3(x)

        # Convolutional Block-2
        x = self.conv_block4(x)
        x = x + self.conv_block5(x)
        x = self.conv_block6(x)

        # Convolutional Block-3
        x = self.conv_block7(x)
        x = x + self.conv_block8(x)
        x = self.conv_block9(x)

        # Convolutional Block-4
        x = self.conv_block10(x)
        x = x + self.conv_block11(x)
        x = x + self.conv_block12(x)

        x = self.gap(x)

        x = self.conv_block14(x)

        x = x.view(-1, 10)
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
            conv_layer = Session9Net.depthwise_conv(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding)
        elif conv_type == "dilated":
            conv_layer = Session9Net.dilated_conv(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding, dilation=dilation)

        # For last layer only return the convolution output
        if last_layer:
            return nn.Sequential(conv_layer)
        return nn.Sequential(
            conv_layer,
            nn.ReLU(),
            _norm_layer,
            nn.Dropout(self.dropout_value)
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
