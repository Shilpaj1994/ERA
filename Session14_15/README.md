# Transformer - Language Translation

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session14_15/Lit_S14_15.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This repository contains following files:

- `config.py`: Configuration data for Transformer model training
- `dataset.py`: Contains dataset class to create data for English to Italian translation 
- `light.py`: Contains lightning module to train the transformer model
- `model.py`: Contains building blocks of the transformer model
- `train.py`: Contain utilities for model training
- `Lit_S14_15.ipynb`: Notebook with model training details






## Training Details

- The model was trained on dataset from HuggingFace

- The transformer model is trained to translate the English to Italian

- Model is trained using Google Colab Notebook

- The model is trained for 10 epochs
  ![Training Log](data/train_log.JPG)

- Below is the sample output after training

   ```commandline
    SOURCE: Then they stand up, and are surprised.
    TARGET: Allora esse si levano in piedi, sorprese.
    PREDICTED: Poi si e si .
   ```
