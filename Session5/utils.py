#!/usr/bin/env python3
"""
Utility Script containing functions to be used for training
"""
# Standard Library Imports
from typing import NoReturn

# Third-Party Imports
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from tqdm import tqdm


def get_summary(model: 'object of model architecture', input_size: tuple) -> NoReturn:
    """
    Function to get the summary of the model architecture
    :param model: Object of model architecture class
    :param input_size: Input data shape (Channels, Height, Width)
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    summary(network, input_size=input_size)


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


def display_data_samples(dataloader: 'DataLoader object', samples: int) -> NoReturn:
    """
    Function to display samples for dataloader
    :param dataloader: Train or Test dataloader
    :param samples: Number of samples to be displayed
    """
    # Get batch from the dataloader
    batch_data, batch_label = next(iter(dataloader))

    # Plot the samples from the batch
    fig = plt.figure()
    for i in range(samples):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def GetCorrectPredCount(pPrediction, pLabels):
    """
    Function to return total number of correct predictions
    :param pPredictions: Model predictions on a given sample of data
    :param pLabels: Correct labels of a given sample of data
    :return: Number of correct predictions
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def train(model, device, train_loader, optimizer, criterion):
    """
    Function to train model on the training dataset
    :param model: Model architecture
    :param device: Device on which training is to be done (GPU/CPU)
    :param train_loader: DataLoader for training dataset
    :param optimizer: Optimization algorithm to be used for updating weights
    :param criterion: Loss function for training
    """
    # Enable layers like Dropout for model training
    model.train()

    # Utility to display training progress
    pbar = tqdm(train_loader)

    # Variables to track loss and accuracy during training
    train_loss = 0
    correct = 0
    processed = 0

    # Iterate over each batch and fetch images and labels from the batch
    for batch_idx, (data, target) in enumerate(pbar):

        # Put the images and labels on the selected device
        data, target = data.to(device), target.to(device)

        # Reset the gradients for each batch
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Get total number of correct predictions
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        # Display the training information
        pbar.set_description(
            desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

    return correct, processed, train_loss


def test(model, device, test_loader, criterion):
    """
    Function to test the model training progress on the test dataset
    :param model: Model architecture
    :param device: Device on which training is to be done (GPU/CPU)
    :param test_loader: DataLoader for test dataset
    :param criterion: Loss function for test dataset
    """
    # Disable layers like Dropout for model inference
    model.eval()

    # Variables to track loss and accuracy
    test_loss = 0
    correct = 0

    # Disable gradient updation
    with torch.no_grad():
        # Iterate over each batch and fetch images and labels from the batch
        for batch_idx, (data, target) in enumerate(test_loader):

            # Put the images and labels on the selected device
            data, target = data.to(device), target.to(device)

            # Pass the images to the output and get the model predictions
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            # Sum up batch correct predictions
            correct += GetCorrectPredCount(output, target)

    # Calculate test loss for a epoch
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct, test_loss
