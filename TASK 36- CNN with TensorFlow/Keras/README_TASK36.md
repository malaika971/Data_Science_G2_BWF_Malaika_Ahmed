

# **TASK 36 - CNN WITH TENSORFLOW/KERAS**

This project demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow and Keras for classifying images as dogs or cats from the "Dogs vs Cats" dataset available on Kaggle.

## Dataset

The dataset contains images of dogs and cats. It is split into training and validation sets:
- **Training Set**: 20,000 images
- **Validation Set**: 5,000 images

### Downloading the Dataset

To download the dataset from Kaggle, use the following commands:

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d salader/dogs-vs-cats
```

## Setup

Make sure you have TensorFlow installed:

```bash
pip install tensorflow
```

## Model Overview

The CNN architecture is as follows:
1. **Convolutional Layers**: Three layers with ReLU activation to extract features.
2. **MaxPooling Layers**: To reduce the spatial dimensions.
3. **Fully Connected Layers**: To perform classification.

### Key Steps:

- **Data Preprocessing**: Load and normalize images.
- **Model Building**: Construct the CNN.
- **Training**: Use binary cross-entropy loss and Adam optimizer.
  

## Results and Insights

After training the model, the performance is analyzed using accuracy and loss plots. The model shows good accuracy on the training set but struggles with overfitting, indicated by a gap between training and validation accuracy.

### Potential Improvements:
- **Data Augmentation**
- **Dropout Layers** to reduce overfitting
- **Batch Normalization**

.
