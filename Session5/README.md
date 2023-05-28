# PyTorch - Model Training and Evaluation

- [1. Setup](#1-setup)
  * [1.1 Colab](#11-colab)
  * [1.1.2 Local System](#112-local-system)

- [2. Code Details](#2-code-details)

  - [2.1 model.py](#21-modelpy)

  * [2.2 utils.py](#utilspy)
    + [get_summary](#get-summary)
    + [display_loss_and_accuracies](#display-loss-and-accuracies)
    + [GetCorrectPredCount](#getcorrectpredcount)
    + [train](#train)
    + [test](#test)

  * [2.3 S5.ipynb](#23-s5ipynb)
    + [Code Block 1 - Importing Required Modules](#code-block-1---importing-required-modules)
    - [Code Block 2 - Checking GPU Availability](#code-block-2---checking-gpu-availability)
    - [Code Block 3 - Data Transformations](#code-block-3---data-transformations)
    - [Code Block 4 - Loading the Dataset](#code-block-4---loading-the-dataset)
    - [Code Block 5 - Creating Data Loaders](#code-block-5---creating-data-loaders)
    - [Code Block 6 - Visualizing Sample Data](#code-block-6---visualizing-sample-data)
    - [Code Block 7 - Importing the Model](#code-block-7---importing-the-model)
    - [Code Block 8 - Initializing Variables for Accuracy and Loss Graphs](#code-block-8---initializing-variables-for-accuracy-and-loss-graphs)
    - [Code Block 9 - Importing Utility Functions](#code-block-9---importing-utility-functions)
    - [Code Block 10 - Model Training and Evaluation](#code-block-10---model-training-and-evaluation)
    - [Code Block 11 - Displaying Loss and Accuracy](#code-block-11---displaying-loss-and-accuracy)

- [3. Run](#3-run)
  * [3.1 Run on Colab](#31-run-on-colab)
  * [3.2 Run on Local System](#32-run-on-local-system)



- This repository contains following files
  1. `README.md`
  2. `model.py`
  3. `utils.py`
  4. `S5.ipynb`

- The `S5.ipynb` file serves as the main file and contains the code for training a Deep Learning model
- The code from `model.py` and `utils.py` can be included in `S5.ipynb` but is separated for modularity and ease of experimentation

- Below are the sections that provide information on setting up the environment and training the model, covering both Colab and local system:

  **Section 1: Setup**

  - This section explains the steps required to set up the environment.

  **Section 2: Code Details**

  - This section provides detailed information about the code implementation.

  **Section 3: Run**

  - In this section, you will find instructions on how to execute the code.

---



## 1. Setup

### 1.1 Colab

- All the necessary dependencies are pre-installed on Google Colab for seamless usage.

- To import the code from `model.py` and `utils.py`, you can follow these steps within your Colab session:

  ```python
  if 'google.colab' in sys.modules:
      # Download the repository from GitHub
      print("Downloading repository on Colab...")
      !git clone https://github.com/Shilpaj1994/ERA.git
      
      # Add the downloaded repository to the system path
      sys.path.insert(0,'./ERA/Session5/')
  ```

- When the notebook is running in a Colab session, this code block will download the repository and add it to the system path

- As a result, you will be able to import functions and classes from `model.py` and `utils.py` within your Colab environment



### 1.1.2 Local System

- On the local system, we need to create an environment and install all the dependencies

- Create an environment using following commands:

  ```bash
  # Install pip
  $ sudo apt-get install python-pip
  
  # Install virtualenv
  $ pip install virtualenv
  
  # Create a virtual environment
  $ virtualenv virtualenv_name
  
  # Activate environment
  $ source virtualenv_name/bin/activate
  ```

- Once the environment is ready, we can start installing all the dependencies

  ```bash
  $ pip install torch
  $ pip install torchsummary
  $ pip install torchvision
  $ pip install tqdm
  $ pip install matplotlib
  $ pip install jupyterlab
  ```

  > Note: Based on the system configuration and Operating system, install the PyTorch version
  >
  > Reference Link - https://pytorch.org/get-started/locally/

- Once the environment is ready with all the dependencies, download the repository

  ```bash
  $ git clone https://github.com/Shilpaj1994/ERA.git
  ```

  

---



## 2. Code Details

This section contains detail information about on the file contents 



### 2.1 model.py

- This script contains the definition of a neural network model architecture
- It includes the implementation of the `Net` class, which defines the structure of the neural network.

```python
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

        # Convolutional Layer-1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
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
```

- The `Net` class inherits from the `nn.Module` class provided by PyTorch and represents a convolutional neural network (CNN) architecture. The architecture consists of several layers, including convolutional layers and fully connected layers.
- The `Net()` class is a child class of `nn.Module`
- `super(Net, self).__init__()` will inherit all the properties of `nn.Module` to our class
- Four convolutional and two fully connected layers are defined in the constructor
- The `forward(self, x)` method defines the forward pass for model training
- It takes an input tensor `x` and applies the defined layers to compute the output of the model
- The activation function used between the layers is the rectified linear unit (ReLU). The output is passed through a log softmax function to obtain the final magnitude of each class





### 2.2 utils.py

This script contains utility functions that can be used for training a model. It includes functions for model summary, displaying loss and accuracies, displaying data samples, and training/testing the model.


#### get summary

- This function prints the summary of the model architecture
- It takes an object of the model architecture and the input data shape as input parameters.

```python
def get_summary(model: object, input_size: tuple):
    """
    Function to get the summary of the model architecture
    :param model: Object of model architecture class
    :param input_size:
    :return:
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    summary(network, input_size=input_size)
```



#### display loss and accuracies

- This function displays the training and test information such as losses and accuracies
- It takes lists containing training losses, training accuracies, test losses, and test accuracies as input parameters
- Additionally, it accepts an optional parameter `plot_size` to specify the size of the plot.

```python
def display_loss_and_accuracies(train_losses: list, train_acc: list, test_losses: list, test_acc: list) -> NoReturn:
    """
    Function to display training and test losses and accuracies
    :param train_losses:
    :param train_acc:
    :param test_losses:
    :param test_acc:
    :return:
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")

    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")

    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")

    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
```





#### GetCorrectPredCount

- This function returns the total number of correct predictions
- It takes model predictions and correct labels of a given sample of data as input parameters and returns the number of correct predictions.

```python
def GetCorrectPredCount(pPrediction, pLabels):
    """
    Function to return total number of correct predictions
    :param pPredictions: Model predictions on a given sample of data
    :param pLabels: Correct labels of a given sample of data
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
```



#### train

- This function is used to train the model on the training dataset
- It takes the model architecture, device (GPU or CPU), training data loader, optimizer, and loss criterion as input parameters
- It returns the number of correct predictions, the number of processed samples, and the total training loss

```python
def train(model, device, train_loader, optimizer, criterion):
    """
    Function to train model on the training dataset
    :param model: Model architecture
    :param device: Device on which training is to be done (GPU/CPU)
    :param train_loader: DataLoader for training dataset
    :param optimizer: Optimization algorithm to be used for updating weights
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(
            desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

    return correct, processed, train_loss
```



#### test

- This function is used to test the model's training progress on the test dataset
- It takes the model architecture, device (GPU or CPU), test data loader, and loss criterion as input parameters
- It returns the number of correct predictions and the average test loss

```python
def test(model, device, test_loader, criterion):
    """
    Function to test the model training progress on the test dataset
    :param model: Model architecture
    :param device: Device on which training is to be done (GPU/CPU)
    :param test_loader: DataLoader for test dataset
    """
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct, test_loss
```





### 2.3 S5.ipynb

This file contains code for training and evaluating a neural network model on the MNIST dataset. The code is divided into several code blocks, each serving a specific purpose. Below is an overview of each code block:

#### Code Block 1 - Importing Required Modules

- In this code block, the necessary modules for the project are imported. These include `sys`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, and `torchvision`.

- If the code is being executed in Google Colab, the repository is downloaded from GitHub, and the required files are imported.

```python
# Import all the required modules
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

if 'google.colab' in sys.modules:
    # Download the repo from GitHub
    print("Downloading repository on Colab...")
    !git clone https://github.com/Shilpaj1994/ERA.git
    
    # Import files from the downloaded repository
    sys.path.insert(0,'./ERA/Session5/')
```



#### Code Block 2 - Checking GPU Availability

- This code block checks if a GPU is available and sets the device accordingly
- If a GPU is available, the device is set to "cuda"; otherwise, it is set to "cpu".

```python
# Check if GPU is available
# Set device as GPU if available else CPU
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```



#### Code Block 3 - Data Transformations

- This code block defines the transformations to be applied to the training and test data
- The transformations include random cropping, resizing, random rotation, tensor conversion, and normalization.

```python
# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1325,), (0.3104,))
    ])
```



#### Code Block 4 - Loading the Dataset

- In this code block, the MNIST dataset is downloaded and loaded using the `datasets.MNIST` class from torchvision
- The dataset is divided into training and test sets, and the defined transformations are applied
- This block performs the extract and transform part of the data pipeline

```python
train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
```



#### Code Block 5 - Creating Data Loaders

- Data loaders are created for both the training and test datasets using `torch.utils.data.DataLoader`
- The loaders handle batching, shuffling, and parallel data loading.

```python
batch_size = 512

kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
```



#### Code Block 6 - Visualizing Sample Data

This code block visualizes a batch of sample images and their corresponding labels using matplotlib.

```python
import matplotlib.pyplot as plt

batch_data, batch_label = next(iter(train_loader)) 

fig = plt.figure()

for i in range(12):
  plt.subplot(3,4,i+1)
  plt.tight_layout()
  plt.imshow(batch_data[i].squeeze(0), cmap='gray')
  plt.title(batch_label[i].item())
  plt.xticks([])
  plt.yticks([])
```

![Display Samples](./assets/display_image.jpg)



#### Code Block 7 - Importing the Model

The model architecture is imported from the `model` file.

```python
from model import Net
```

The `get_summary` function from the `utils` module is used to print a summary of the model architecture.

```python
# Model Summary
from utils import get_summary

get_summary(Net(), (1, 28, 28))
```

```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```



#### Code Block 8 - Initializing Variables for Accuracy and Loss Graphs

Empty lists are initialized to store the training and test losses, as well as the training and test accuracies. Additionally, a dictionary is created to store incorrectly predicted samples during testing.

```python
# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}
```



#### Code Block 9 - Importing Utility Functions

Utility functions for training and testing the model are imported from the `utils` file.

```python
from utils import train, test
```



#### Code Block 10 - Model Training and Evaluation

- In this code block, the model is trained and evaluated
- The model is instantiated, an optimization algorithm (SGD) is defined, and a learning rate scheduler is set
- The criterion for loss calculation is defined as the negative log-likelihood loss.
- The training loop runs for a specified number of epochs
- In each epoch, the model is trained on the training dataset, and the training loss and accuracy are recorded
- Then, the model's performance is evaluated on the test dataset, and the test loss and accuracy are recorded
- The learning rate scheduler is also updated after specific number of epochs

```python
# Put the model on selected device
model = Net().to(device)

# Optimization algorithm to update the weights
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Scheduler to change the learning rate after specific number of epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)

# New Line
criterion = F.nll_loss

# Number of epochs for which model is to be trained
num_epochs = 20

# For each epoch
for epoch in range(1, num_epochs+1):
    print(f'Epoch {epoch}')

    # Train the model on training dataset and append the training loss and accuracy
    correct, processed, train_loss = train(model, device, train_loader, optimizer, criterion)
    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))

    # Test the model's performance on test dataset and append the training loss and accuracy
    correct, test_loss = test(model, device, test_loader, criterion)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)
    
    # Update the learning rate after specified number of epochs
    scheduler.step()
```



#### Code Block 11 - Displaying Loss and Accuracy

This code block uses the `display_loss_and_accuracies` function from the `utils` module to plot the training and test loss curves and display the training and test accuracies.

```python
# Print loss and accuracy
from utils import display_loss_and_accuracies
display_loss_and_accuracies(train_losses, train_acc, test_losses, test_acc)
```



This code provides a complete pipeline for training and evaluating a neural network model on the MNIST dataset. It can be used as a starting point for developing and experimenting with different architectures and training strategies. Feel free to modify the code to suit your specific needs.



---



## 3. Run

### 3.1 Run on Colab

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session5/S5.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Click on the above button to open the notebook and Colab and execute the code
- Make a copy of this file in your Google Colab to ensure your changes are saved



### 3.2 Run on Local System

- Activate the environment

  ```bash
  $ source virtualenv_name/bin/activate
  ```

- Open Jupyterlab

  ```bash
  $ jupyter-lab
  ```

- Open and run the notebook

