#!/usr/bin/env python3
"""
Function used for visualization of data and results
Author: Shilpaj Bhalerao
Date: Jun 21, 2023
"""
# Standard Library Imports
import math
from typing import NoReturn

# Third-Party Imports
import torch
import numpy as np
import matplotlib.pyplot as plt


# def display_mnist_data_samples(dataloader: 'DataLoader object', number_of_samples: int, dataset: str="CIFAR") -> NoReturn:
#     """
#     Function to display samples for dataloader
#     :param dataloader: Train or Test dataloader
#     :param number_of_samples: Number of samples to be displayed
#     """
#     # Get batch from the dataloader
#     batch_data, batch_label = next(iter(dataloader))
#
#     # Plot the samples from the batch
#     fig = plt.figure()
#     for i in range(number_of_samples):
#         plt.subplot(3, 4, i + 1)
#         plt.tight_layout()
#         if dataset == "MNIST":
#             plt.imshow(batch_data[i].squeeze(0), cmap='gray')
#         else:
#             plt.imshow(np.transpose(batch_data[i].squeeze(), (1, 2, 0)))
#         plt.title(batch_label[i].item())
#         plt.xticks([])
#         plt.yticks([])

def display_data_samples(data_set, number_of_samples: int, classes: list, dataset: str="CIFAR"):
    """
    Function to display samples for data_set
    :param data_set: Train or Test data_set
    :param number_of_samples: Number of samples to be displayed
    """
    # Get batch from the data_set
    batch_data = []
    batch_label = []
    for count, item in enumerate(data_set):
        if not count <= number_of_samples:
            break
        batch_data.append(item[0])
        batch_label.append(item[1])
    batch_data = torch.stack(batch_data, dim=0).numpy()

    # Plot the samples from the batch
    fig = plt.figure()
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples/x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        plt.tight_layout()
        if dataset == "MNIST":
            plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        else:
            plt.imshow(np.transpose(batch_data[i].squeeze(), (1, 2, 0)))
        plt.title(classes[batch_label[i]])
        plt.xticks([])
        plt.yticks([])


def plot_data(data, classes, inv_normalize, number_of_samples=10, dataset="CIFAR"):
    """
    Function to plot images with labels
    :param data: List[Tuple(image, label)]
    :param number_of_samples: Number of images to print
    """
    fig = plt.figure(figsize=(8, 5))

    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples/x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        if dataset == "MNIST":
            plt.imshow(data[i][0].squeeze(0).to('cpu'), cmap='gray')
        else:
            img = data[i][0].squeeze().to('cpu')
            img = inv_normalize(img)
            plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])


# def visualize_augmentation(data_loader, data_transforms):
#     """
#     Function to visualize the augmented data
#     :param data_loader: Train Dataloader to visualize the augmentations
#     :param data_transforms: Dictionary of transforms
#     """
#     # Get a batch from the data-loader
#     batch = next(iter(data_loader))
#
#     # Extract images and labels from a batch
#     images, labels = batch[0], batch[1]
#
#     # Get single image and label from the batch
#     orig_img, label = images[0], labels[0]
#
#     # Add batch dimension to the image
#     orig_img = orig_img.unsqueeze(0)
#
#     # List of data to plot and display
#     images_to_plot = [(orig_img, "Original")]
#
#     for key, trans in data_transforms.items():
#         out = trans(orig_img)
#         images_to_plot.append((out, key))
#
#     plot_data(images_to_plot)

def visualize_augmentation(data_set, data_transforms):
    """
    Function to visualize the augmented data
    :param data_set: Train Dataloader to visualize the augmentations
    :param data_transforms: Dictionary of transforms
    """
    # Get the batch from dataset
    batch = next(iter(data_set))

    # Extract images and labels from a batch
    images, labels = batch[0], batch[1]

    # Get single image and label from the batch
    orig_img, label = images[0], labels[0]

    # List of data to plot and display
    images_to_plot = [(orig_img, "Original")]

    for key, trans in data_transforms.items():
        out = trans(orig_img)
        images_to_plot.append((out, key))

    plot_data(images_to_plot)


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


