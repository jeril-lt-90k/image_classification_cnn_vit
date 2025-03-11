
# Image Classification

This repository contains two image classification models: one based on a **Convolutional Neural Network (CNN)** and the other based on a **Vision Transformer (ViT)**. Both models are designed to classify images into two categories (e.g., Cats and Dogs) using PyTorch.

---

## 1. CNN-based Image Classifier

### Overview
The CNN model is a basic Convolutional Neural Network designed to classify images into two classes (e.g., Cat and Dog). The architecture includes several convolutional layers followed by fully connected layers.

### Code Walkthrough:
- **Data Preprocessing**: The dataset is loaded using `torchvision.datasets.ImageFolder` and transformed with resizing, normalization, and data augmentation.
- **CNN Model**: The model consists of convolutional layers, activation functions, and a final fully connected layer for classification.
- **Training**: The model is trained using the `CrossEntropyLoss` loss function and an `Adam` optimizer, using **Mixed Precision Training** to optimize memory usage and speed.
- **Evaluation**: The model is evaluated using accuracy and loss metrics.
- **Inference**: The model predicts whether an image is of a cat or dog.

---

## 2. Vision Transformer(ViT)-based Classifier

### Overview
The ViT model leverages a Vision Transformer architecture, a cutting-edge model for image classification tasks. It divides an image into patches and processes them with self-attention mechanisms.

### Code Walkthrough:
- **Data Preprocessing**: Similar to the CNN model, the dataset is transformed using standard ImageNet preprocessing (resize, center crop, normalize).
- **ViT Model**: The model uses a pre-trained Vision Transformer (ViT-B/16) from the `torchvision.models` library and modifies the output layer to classify binary images.
- **Training**: The model is trained using the `CrossEntropyLoss` loss function and an `Adam` optimizer, using **Mixed Precision Training** to optimize memory usage and speed.
- **Evaluation**: Validation is performed after each epoch to assess the model's performance.
- **Inference**: The model predicts whether an image is of a cat or dog.

---

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL
- tqdm

## Installation

1. Clone the repository or download the code.
2. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a CUDA-compatible GPU and necessary drivers installed for GPU acceleration. If you don't have a GPU, the script will automatically run on CPU.

---

## Training and Validation Data Folder Structure

images/
├── train/
│   ├── Cat/
│   └── Dog/
├── validate/
│   ├── Cat/
│   └── Dog/
├── [cat_test.jpg | dog_test.jpg]


## Training

- For both models, the training process involves:
  - Loading images with `torchvision.datasets.ImageFolder`.
  - Applying transformations to resize, normalize, and augment images.
  - Training for a set number of epochs using the Adam optimizer.
  - Saving the trained model after training is complete.

---

## Inference

- After training, you can perform inference on new images using the trained model.
- The `CNNModel` and `ViTModel` classes predicts whether the input image belongs to the class of "Cat" or "Dog".

---

## Licensing

### CNN Model

This CNN model is implemented from scratch using PyTorch. It is a custom design and is not based on any pre-trained weights.

### Vision Transformer (ViT)

The Vision Transformer model is built upon a pre-trained model from the official PyTorch torchvision library. The model is based on research from the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. You can find the official implementation at:

- **Paper**: [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)


