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
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------- DATA SAMPLES ----------------------------
def display_mnist_data_samples(dataset: 'DataLoader object', number_of_samples: int) -> NoReturn:
    """
    Function to display samples for dataloader
    :param dataset: Train or Test dataset transformed to Tensor
    :param number_of_samples: Number of samples to be displayed
    """
    # Get batch from the data_set
    batch_data = []
    batch_label = []
    for count, item in enumerate(dataset):
        if not count <= number_of_samples:
            break
        batch_data.append(item[0])
        batch_label.append(item[1])

    # Plot the samples from the batch
    fig = plt.figure()
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples/x_count)

    # Plot the samples from the batch
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(), cmap='gray')
        plt.title(batch_label[i])
        plt.xticks([])
        plt.yticks([])


def display_cifar_data_samples(data_set, number_of_samples: int, classes: list):
    """
    Function to display samples for data_set
    :param data_set: Train or Test data_set transformed to Tensor
    :param number_of_samples: Number of samples to be displayed
    :param classes: Name of classes to be displayed
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
        plt.imshow(np.transpose(batch_data[i].squeeze(), (1, 2, 0)))
        plt.title(classes[batch_label[i]])
        plt.xticks([])
        plt.yticks([])


# ---------------------------- MISCLASSIFIED DATA ----------------------------
def display_cifar_misclassified_data(data: list,
                                     classes: list[str],
                                     inv_normalize: transforms.Normalize,
                                     number_of_samples: int = 10):
    """
    Function to plot images with labels
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param number_of_samples: Number of images to print
    """
    fig = plt.figure(figsize=(8, 5))

    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples/x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        img = data[i][0].squeeze().to('cpu')
        img = inv_normalize(img)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])


def display_mnist_misclassified_data(data: list,
                                     number_of_samples: int = 10):
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
        img = data[i][0].squeeze(0).to('cpu')
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        plt.title(r"Correct: " + str(data[i][1].item()) + '\n' + 'Output: ' + str(data[i][2].item()))
        plt.xticks([])
        plt.yticks([])


# ---------------------------- AUGMENTATION SAMPLES ----------------------------
def visualize_cifar_augmentation(data_set, data_transforms):
    """
    Function to visualize the augmented data
    :param data_set: Dataset without transformations
    :param data_transforms: Dictionary of transforms
    """
    sample, label = data_set[6]
    total_augmentations = len(data_transforms)

    fig = plt.figure(figsize=(10, 5))
    for count, (key, trans) in enumerate(data_transforms.items()):
        if count == total_augmentations - 1:
            break
        plt.subplot(math.ceil(total_augmentations / 5), 5, count + 1)
        augmented = trans(image=sample)['image']
        plt.imshow(augmented)
        plt.title(key)
        plt.xticks([])
        plt.yticks([])


def visualize_mnist_augmentation(data_set, data_transforms):
    """
    Function to visualize the augmented data
    :param data_set: Dataset to visualize the augmentations
    :param data_transforms: Dictionary of transforms
    """
    sample, label = data_set[6]
    total_augmentations = len(data_transforms)

    fig = plt.figure(figsize=(10, 5))
    for count, (key, trans) in enumerate(data_transforms.items()):
        if count == total_augmentations - 1:
            break
        plt.subplot(math.ceil(total_augmentations / 5), 5, count + 1)
        img = trans(sample).to('cpu')
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        plt.title(key)
        plt.xticks([])
        plt.yticks([])


# ---------------------------- LOSS AND ACCURACIES ----------------------------
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


