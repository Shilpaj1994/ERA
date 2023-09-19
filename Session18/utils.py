#!/usr/bin/env python3
"""
DataSet class for training UNet
Author: Shilpaj Bhalerao
Date: Sep 19, 2023
"""
# Standard Library Imports
from typing import NoReturn

# Third-Party Imports
import torch
import matplotlib.pyplot as plt


def tensor_trimap(t):
    """
    Create a tensor for a segmentation trimap.
    Input: Float tensor with values in [0.0 .. 1.0]
    Output: Long tensor with values in {0, 1, 2}
    """
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x


def args_to_dict(**kwargs):
    """
    Input arguments and return dictionary
    """
    return kwargs


def display_loss_and_accuracies(train_losses: list,
                                train_acc: list,
                                test_losses: list,
                                test_acc: list,
                                plot_size: tuple = (10, 10)) -> NoReturn:
    """
    Function to display training and test information(losses and accuracies)
    :param train_losses: List containing training loss of each epoch
    :param train_acc: List containing training accuracy of each epoch
    :param test_losses: List containing test loss of each epoch
    :param test_acc: List containing test accuracy of each epoch
    :param plot_size: Size of the plot
    """
    # Create a plot of 2x2 of size
    fig, axs = plt.subplots(2, 2, figsize=plot_size)

    # Plot the training loss and accuracy for each epoch
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")

    # Plot the test loss and accuracy for each epoch
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
