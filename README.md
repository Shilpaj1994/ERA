# ERA



## Session-5 [Basic MNIST Model Training and Evaluation]

- Setup
- Import Dataset and create Dataloader
- Perform Transforms
- Visualize Model loss and accuracy on training and test dataset



## Session-6 [BackPropagation]

- Explanation of Backpropagation algorithm
- Example - simple network
- Calculation of all the gradients and performing backpropagation
- Weight update steps and model training explanation
- All the calculations in Excel file
- Effect of learning rate on loss and weight update



## Session-7 [Model Creation Process]

- All the steps to be followed for model creation
- 8 steps followed to create a model to achieve 99.4% accuracy under 15 epochs within 8000 parameters



## Session-8 [Normalization and Regularization]

- Basic Concepts
- Normalization and types of normalizations used in DL (Batch, Layer and Group)
- Regularization (L1 & L2)
- Implementation of normalization and Regularization in PyTorch
- Comparison of model with different types of normalization techniques
- Simple model to achieve 70% accuracy on CIFAR10 dataset



## Session-9 [Advance Convolutions]

- Different types of convolutions used in the DL models
- Implementation of these convolutions in PyTorch
- Model to achieve 85% on CIFAR under 50,000 parameters



## Session-10 [One Cycle Policy]

- Implementation of One Cycle Policy for faster training of models
- Model to achieve 90%+ accuracy on CIFAR10 in 24 epochs



## Session-11 [GradCam]

- Train ResNet18 for 20 epochs with 85+% accuracy
- Implemented GradCam to visualize the network activations for four block of the ResNet on the misclassified images



## Session-12 [PyTorch Lightning and Gradio App]

- Trained Session-10 model with PyTorch Lightning
- Created a Gradio app for display misclassified images, GradCam output, Feature maps and Kernel from conv layers




## Session-13 [Yolov3 Lightning and Gradio App]
- Trained Yolov3 model for 40 epochs on PASCAL VOC dataset
- Created a Gradio app to showcase the model output



## Session-14_15 [Lightning Transformer]

- Trained a transformer model from scratch to translate English to Italian
- Coded the model in PyTorch Lightning
- Trained for 10 epochs to validate if the model is learning the language



## Session-16 [Training Faster Transformer]



## Session-17 [LLMs]
- Trained BERT, GPT and ViT models
- Code for all 3 model architecture, dataset and training notebooks are included in the repo




## Session-18 [UNet and Variational AutoEncoders]
- Trained UNet mask of pets from OxfordPets dataset
- Trained Variational AutoEncoder on MNIST to regenerate the data
- Trained Variational AutoEncoder on CIFAR10 to regenerate the data
- In both the VAE models, label data is also passed to the model along with the images



## Session-19 [FastSAM]

- Gradio App for FastSAM implementation



## Session-20 [Stable Diffusion]


## Session-21 [NanoGPT]
- NanoGPT Transformer model [Decoder Only] is trained from the Andrej Karpathy's video
- Gradio App is created to generate Shakespeare style text character-by-character