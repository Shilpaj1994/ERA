# NanoGPT

<a target="_blank" href="https://colab.research.google.com/github/Shilpaj1994/ERA/blob/master/Session21/NanoGPT.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- This application is based on the nanoGPT model trained in this [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy) by Andrej Karpathy
- Model is trained to output Shakesphere text
- It is the first step in the training ChatGPT where the aim of the model is to generate text
- This model is a Decoder only model
- The model prints out next character based on the previous characters

## Training Logs
```commandline
step 0: train loss 4.2221, val loss 4.2306
step 500: train loss 1.7600, val loss 1.9146
step 1000: train loss 1.3903, val loss 1.5987
step 1500: train loss 1.2644, val loss 1.5271
step 2000: train loss 1.1835, val loss 1.4978
step 2500: train loss 1.1233, val loss 1.4910
step 3000: train loss 1.0718, val loss 1.4804
step 3500: train loss 1.0179, val loss 1.5127
step 4000: train loss 0.9604, val loss 1.5102
step 4500: train loss 0.9125, val loss 1.5351
step 4999: train loss 0.8589, val loss 1.5565
```

## Sample Output
```commandline
But with prison, I will steal for the fimker.

KING HENRY VI:
To prevent it, as I love this country's cause.

HENRY BOLINGBROKE:
I thank bhop my follow. Walk ye were so?

NORTHUMBERLAND:
My lord, I hearison! Who may love me accurse
Some chold or flights then men shows to great the cur
Ye cause who fled the trick that did princely action?
Take my captiving sound, althoughts thy crown.

RICHMOND NE:
God neit will he not make it wise this!

DUKE VINCENTIO:
Worthy Prince forth from Lord Claudio!
```

## [App Link](https://huggingface.co/spaces/Shilpaj/nanoGPT)