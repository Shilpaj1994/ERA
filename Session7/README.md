# Model Training

### Problem Statement:

- Achieve 99.4 % test accuracy on MNIST
- Achieve it under 15 epochs
- Achieve it under 8000 parameters





## Code-1: The Setup

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session7/F1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Target:
- Get the set-up right
- Set Extract, Transform and Load pipeline
- Visualize the transformed images
- Set Data Loader
- Set Basic Working Code
- Set Basic Training  & Test Loop

### Results:
- Parameters: 6.3M
- Best Training Accuracy: 98.72
- Best Test Accuracy: 99.29

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 28, 28]             320
            Conv2d-2           [-1, 64, 28, 28]          18,496
         MaxPool2d-3           [-1, 64, 14, 14]               0
            Conv2d-4          [-1, 128, 14, 14]          73,856
            Conv2d-5          [-1, 256, 14, 14]         295,168
         MaxPool2d-6            [-1, 256, 7, 7]               0
            Conv2d-7            [-1, 512, 5, 5]       1,180,160
            Conv2d-8           [-1, 1024, 3, 3]       4,719,616
            Conv2d-9             [-1, 10, 1, 1]          92,170
================================================================
Total params: 6,379,786
Trainable params: 6,379,786
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.51
Params size (MB): 24.34
Estimated Total Size (MB): 25.85
----------------------------------------------------------------
```



![image-20230616100115981](Data/image-20230616100115981.png)

![image-20230616100203798](Data/image-20230616100203798.png)



### Analysis:
- Extremely Heavy Model for such a problem
- Basic setup is completed
- Only basic augmentations is done
- Need to change the model



---



## Code-2: The Skeleton

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session7/F2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Target:

- Get the basic model skeleton right. I have to avoid changing this skeleton as much as possible
- No fancy stuff
- No BatchNorm layers, No Dropout, No Padding
- Use squeeze and expand kind of model architecture
- At least get 99% on test data

### Results:

- Parameters: 261K
- Best Training Accuracy: 98.49
- Best Test Accuracy: 99.02

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
            Conv2d-2           [-1, 64, 24, 24]          18,432
            Conv2d-3          [-1, 128, 22, 22]          73,728
         MaxPool2d-4          [-1, 128, 11, 11]               0
            Conv2d-5             [-1, 32, 9, 9]          36,864
            Conv2d-6             [-1, 64, 7, 7]          18,432
            Conv2d-7            [-1, 128, 5, 5]          73,728
            Conv2d-8             [-1, 32, 3, 3]          36,864
            Conv2d-9             [-1, 10, 1, 1]           2,880
================================================================
Total params: 261,216
Trainable params: 261,216
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.11
Params size (MB): 1.00
Estimated Total Size (MB): 2.11
----------------------------------------------------------------
```



![image-20230616123527020](Data/image-20230616123527020.png)

![image-20230616123601137](Data/image-20230616123601137.png)

### Analysis:

- The model is working but the number of parameters are still large
- Test accuracy is higher than training accuracy which shows that the model has the potential to reach the target



---



## Code-3: The Lighter Model

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session7/F3.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Target:

- Reduce the number of parameters in the model

### Results:

- Parameters: 14.4K
- Best Training Accuracy: 97.34
- Best Test Accuracy: 98.58

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 12, 26, 26]             108
            Conv2d-2           [-1, 16, 24, 24]           1,728
            Conv2d-3           [-1, 18, 22, 22]           2,592
            Conv2d-4           [-1, 12, 20, 20]           1,944
         MaxPool2d-5           [-1, 12, 10, 10]               0
            Conv2d-6             [-1, 16, 8, 8]           1,728
            Conv2d-7             [-1, 18, 6, 6]           2,592
            Conv2d-8             [-1, 12, 4, 4]           1,944
            Conv2d-9             [-1, 12, 2, 2]           1,296
           Conv2d-10             [-1, 10, 1, 1]             480
================================================================
Total params: 14,412
Trainable params: 14,412
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.26
Params size (MB): 0.05
Estimated Total Size (MB): 0.32
----------------------------------------------------------------
```



![image-20230616125831725](Data/image-20230616125831725.png)

![image-20230616125855444](Data/image-20230616125855444.png)



### Analysis:

- The model performed terribly after reducing the number of parameters
- Hence, the architecture was modified by shifting a layer from block 2 to block1
- The receptive field of the model is increased to 30
- The model still has potential since the test accuracy is better than the training accuracy



---



## Code-4: The Batch Normalization

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session7/F4.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Target:

- Add Batch Normalization to increase the model efficiency
- Try reducing batch size to 128

### Results:

- Parameters: 13.9K

- Best Training Accuracy: 99.24

- Best Test Accuracy: 99.53

  ```python
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Conv2d-1           [-1, 12, 26, 26]             108
         BatchNorm2d-2           [-1, 12, 26, 26]              24
              Conv2d-3           [-1, 12, 24, 24]           1,296
         BatchNorm2d-4           [-1, 12, 24, 24]              24
              Conv2d-5           [-1, 12, 22, 22]           1,296
         BatchNorm2d-6           [-1, 12, 22, 22]              24
              Conv2d-7           [-1, 12, 20, 20]           1,296
         BatchNorm2d-8           [-1, 12, 20, 20]              24
           MaxPool2d-9           [-1, 12, 10, 10]               0
             Conv2d-10             [-1, 16, 8, 8]           1,728
        BatchNorm2d-11             [-1, 16, 8, 8]              32
             Conv2d-12             [-1, 16, 6, 6]           2,304
        BatchNorm2d-13             [-1, 16, 6, 6]              32
             Conv2d-14             [-1, 10, 1, 1]           5,760
  ================================================================
  Total params: 13,948
  Trainable params: 13,948
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.00
  Forward/backward pass size (MB): 0.42
  Params size (MB): 0.05
  Estimated Total Size (MB): 0.48
  ----------------------------------------------------------------
  ```
  
  
  
  ![image-20230616143449285](Data/image-20230616143449285.png)

![image-20230616143524582](Data/image-20230616143524582.png)

### Analysis:

- By reducing the batch size to 128, observed quicker convergence
- To get a scope of reducing parameters, removed the last 2 layers capping the receptive field at 28
- Larger kernel of size 6x6 is used in the last layer



---



## Code-5: The Global Average Pooling

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session7/F5.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Target:

- Reduce the number of parameters are try to train the model
- Try reducing batch size to 64

### Results:

- Parameters: 7.3K
- Best Training Accuracy: 98.57
- Best Test Accuracy: 99.31

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 12, 26, 26]             108
       BatchNorm2d-2           [-1, 12, 26, 26]              24
            Conv2d-3           [-1, 12, 24, 24]           1,296
       BatchNorm2d-4           [-1, 12, 24, 24]              24
            Conv2d-5           [-1, 12, 22, 22]           1,296
       BatchNorm2d-6           [-1, 12, 22, 22]              24
            Conv2d-7           [-1, 12, 20, 20]           1,296
       BatchNorm2d-8           [-1, 12, 20, 20]              24
         MaxPool2d-9           [-1, 12, 10, 10]               0
           Conv2d-10             [-1, 16, 8, 8]           1,728
      BatchNorm2d-11             [-1, 16, 8, 8]              32
           Conv2d-12             [-1, 10, 6, 6]           1,440
      BatchNorm2d-13             [-1, 10, 6, 6]              20
        AvgPool2d-14             [-1, 10, 1, 1]               0
================================================================
Total params: 7,312
Trainable params: 7,312
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.42
Params size (MB): 0.03
Estimated Total Size (MB): 0.45
----------------------------------------------------------------
```



![image-20230616152812396](Data/image-20230616152812396.png)

![image-20230616152843461](Data/image-20230616152843461.png)

### Analysis:

- After using the Global Average Pooling layer, the number of parameters are reduced to 7,312
- Slight improvement in the accuracy when batch size of 64 was used instead of 128
- The model is training but failing to achieve the target of 99.4 %
- It seems like the model has reached its capacity
- Need to merge features effectively to get more output and increase the accuracy



---



## Code-6: Use 1x1 layers

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session7/F6.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Target:

- Use 1x1 layers for merging the features to increase model performance
- Add one more convolutional layer in block-2

### Results:

- Parameters: 6,874
- Best Training Accuracy: 98.49
- Best Test Accuracy: 99.38

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
       BatchNorm2d-2           [-1, 10, 26, 26]              20
            Conv2d-3           [-1, 10, 24, 24]             900
       BatchNorm2d-4           [-1, 10, 24, 24]              20
            Conv2d-5           [-1, 10, 22, 22]             900
       BatchNorm2d-6           [-1, 10, 22, 22]              20
            Conv2d-7           [-1, 10, 20, 20]             900
       BatchNorm2d-8           [-1, 10, 20, 20]              20
            Conv2d-9           [-1, 10, 20, 20]             100
      BatchNorm2d-10           [-1, 10, 20, 20]              20
        MaxPool2d-11           [-1, 10, 10, 10]               0
           Conv2d-12             [-1, 12, 8, 8]           1,080
      BatchNorm2d-13             [-1, 12, 8, 8]              24
           Conv2d-14             [-1, 12, 6, 6]           1,296
      BatchNorm2d-15             [-1, 12, 6, 6]              24
           Conv2d-16             [-1, 12, 4, 4]           1,296
      BatchNorm2d-17             [-1, 12, 4, 4]              24
AdaptiveAvgPool2d-18             [-1, 12, 1, 1]               0
           Conv2d-19             [-1, 10, 1, 1]             120
      BatchNorm2d-20             [-1, 10, 1, 1]              20
================================================================
Total params: 6,874
Trainable params: 6,874
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.42
Params size (MB): 0.03
Estimated Total Size (MB): 0.45
----------------------------------------------------------------
```

![image-20230618163005779](Data/image-20230618163005779.png)

![image-20230618163113875](Data/image-20230618163113875.png)

### Analysis:
- One more convolutional layer and two more layers of 1x1 are added to achieve this and reducing number of parameters
- Next, I will try to increase the capacity of the model by increasing the number of parameters



---



## Code-7: Increase the Capacity

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session7/F7.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Target:

- Increase the capacity of the model

### Results:

- Parameters: 7,884
- Best Training Accuracy: 98.67
- Best Test Accuracy: 99.38

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
       BatchNorm2d-2           [-1, 10, 26, 26]              20
            Conv2d-3           [-1, 10, 24, 24]             900
       BatchNorm2d-4           [-1, 10, 24, 24]              20
            Conv2d-5           [-1, 10, 22, 22]             900
       BatchNorm2d-6           [-1, 10, 22, 22]              20
            Conv2d-7           [-1, 10, 20, 20]             900
       BatchNorm2d-8           [-1, 10, 20, 20]              20
            Conv2d-9           [-1, 10, 20, 20]             100
      BatchNorm2d-10           [-1, 10, 20, 20]              20
        MaxPool2d-11           [-1, 10, 10, 10]               0
           Conv2d-12             [-1, 14, 8, 8]           1,260
      BatchNorm2d-13             [-1, 14, 8, 8]              28
           Conv2d-14             [-1, 14, 6, 6]           1,764
      BatchNorm2d-15             [-1, 14, 6, 6]              28
           Conv2d-16             [-1, 13, 4, 4]           1,638
      BatchNorm2d-17             [-1, 13, 4, 4]              26
AdaptiveAvgPool2d-18             [-1, 13, 1, 1]               0
           Conv2d-19             [-1, 10, 1, 1]             130
      BatchNorm2d-20             [-1, 10, 1, 1]              20
================================================================
Total params: 7,884
Trainable params: 7,884
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.42
Params size (MB): 0.03
Estimated Total Size (MB): 0.45
----------------------------------------------------------------
```

![image-20230618163334375](Data/image-20230618163334375.png)

![image-20230618165646627](Data/image-20230618165646627.png)

### Analysis:
- Model is consistently achieving 99.3+ accuracy in the last few epochs
- Need to push little bit by using learning rate scheduler



---



## Code-8: The LR Scheduler

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session7/F8.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Target:

- Use OneCycle LR scheduler to achieve the target

### Results:

- Parameters: 7,884
- Best Training Accuracy: 98.88
- Best Test Accuracy: 99.44

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
       BatchNorm2d-2           [-1, 10, 26, 26]              20
            Conv2d-3           [-1, 10, 24, 24]             900
       BatchNorm2d-4           [-1, 10, 24, 24]              20
            Conv2d-5           [-1, 10, 22, 22]             900
       BatchNorm2d-6           [-1, 10, 22, 22]              20
            Conv2d-7           [-1, 10, 20, 20]             900
       BatchNorm2d-8           [-1, 10, 20, 20]              20
            Conv2d-9           [-1, 10, 20, 20]             100
      BatchNorm2d-10           [-1, 10, 20, 20]              20
        MaxPool2d-11           [-1, 10, 10, 10]               0
           Conv2d-12             [-1, 14, 8, 8]           1,260
      BatchNorm2d-13             [-1, 14, 8, 8]              28
           Conv2d-14             [-1, 14, 6, 6]           1,764
      BatchNorm2d-15             [-1, 14, 6, 6]              28
           Conv2d-16             [-1, 13, 4, 4]           1,638
      BatchNorm2d-17             [-1, 13, 4, 4]              26
AdaptiveAvgPool2d-18             [-1, 13, 1, 1]               0
           Conv2d-19             [-1, 10, 1, 1]             130
      BatchNorm2d-20             [-1, 10, 1, 1]              20
================================================================
Total params: 7,884
Trainable params: 7,884
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.42
Params size (MB): 0.03
Estimated Total Size (MB): 0.45
----------------------------------------------------------------
```

![image-20230618163643353](Data/image-20230618163643353.png)

![image-20230618163710558](Data/image-20230618163710558.png)

### Analysis:
- With the OneCycle LR scheduler, the target of 99.40% is achieved consistently in the last 3 epochs
