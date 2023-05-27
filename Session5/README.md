# Session-5

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session5/S5.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


This repository contains following files

```bash
|- README.md
|- model.py
|- utils.py
|_ S5.ipynb
```



---



## 1. Setup

### 1.1 Install Dependencies

#### Colab

- All the dependencies are pre-installed on the Colab



#### Local System



```bash
$ pip install torch
$ pip install torchsummary
$ pip install torchvision
$ pip install tqdm
```





### 1.2 Download the source code

#### Colab

```bash
! git clone 
```



#### Local System

```bash
$ git clone 
```





## 2. Usage

This section contains detail information about on the file contents and how to run the pipeline



### 2.1 `model.py`

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

- This file contains the model architecture
- The `Net()` class is a child class of `nn.Module`
- `super(Net, self).__init__()` will inherit all the properties of `nn.Module` to our class
- All the layers in the architecture are defined in the constructor
- The `forward(self, x)` method defines the forward pass for model training
- It takes in the input data and return the prediction of the model on the input data
- This denotes how the data will flow through the network while training





### 2.2 `utils.py`

#### 2.2.1 `get_summary(model, input_size)`

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



- This function is used to print the summary of the model architecture
- 





#### 2.2.2 `display_loss_and_accuracies`

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





#### 2.2.3 `GetCorrectPredCount`

```python
def GetCorrectPredCount(pPrediction, pLabels):
    """
    Function to return total number of correct predictions
    :param pPredictions: Model predictions on a given sample of data
    :param pLabels: Correct labels of a given sample of data
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
```





#### 2.2.4 `train`

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





#### 2.2.5 `test`

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





### 2.3 `S5.ipynb`

