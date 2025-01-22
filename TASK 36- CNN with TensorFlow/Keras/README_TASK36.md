

# **TASK 36 - CNN WITH TENSORFLOW/KERAS

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
  
## Example Code

Here is the essential code for building and training the model:

```python
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Load and preprocess data
train_ds = keras.utils.image_dataset_from_directory('/content/train', image_size=(256, 256), batch_size=32)
validation_ds = keras.utils.image_dataset_from_directory('/content/test', image_size=(256, 256), batch_size=32)

def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# Build the CNN model
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

## Results and Insights

After training the model, the performance is analyzed using accuracy and loss plots. The model shows good accuracy on the training set but struggles with overfitting, indicated by a gap between training and validation accuracy.

### Potential Improvements:
- **Data Augmentation**
- **Dropout Layers** to reduce overfitting
- **Batch Normalization**

.
