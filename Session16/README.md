# Transformer - Language Translation

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session16/LitTransformer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This repository contains following files:

- `config.py`: Configuration data for Transformer model training
- `dataset.py`: Contains dataset class to create data for English to Italian translation 
- `light.py`: Contains lightning module to train the transformer model
- `model.py`: Contains building blocks of the transformer model
- `train.py`: Contain utilities for model training
- `LitTransformer.ipynb`: Notebook with model training details



## Faster Transformer Training

- The model was trained on dataset from HuggingFace

- The transformer model is trained to translate the English to French

- The model is trained for 20 epochs to achieve loss under 1.8
  ![Training Log](data/train_log.JPG)

- Below is the sample output after training

   ```commandline
    SOURCE: Then they stand up, and are surprised.
    TARGET: Allora esse si levano in piedi, sorprese.
    PREDICTED: Poi si e si .
   ```

- To train the model faster, following strategies are used



### 1. Dynamic Padding

Padding of the training and validation dataset is done based on the data in the batch



### 2. Parameter Sharing

Multiple encoder blocks share the parameters which improves to speed up training. Similarly, sharing of decoder blocks parameters is also implemented



### 3. One Cycle Policy

It is used to converge the model faster



### 4. Data Filtering

Input data is filtered out based on the length of the English and French sentence length



### 5. Automatic Mixed Precision [AMP]

Mixed precision is used for faster training



### 6. Feed Forward Network

Hidden layer of the Feed Forward Network is used with only 128 neurons instead of 256

 
