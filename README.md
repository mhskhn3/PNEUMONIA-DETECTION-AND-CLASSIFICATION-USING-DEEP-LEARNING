# Pneumonia Detection Using CNN on Chest X-ray Images

## Overview
This project aims to diagnose pneumonia using a convolutional neural network (CNN) applied to a dataset of pediatric chest X-ray images. The dataset is structured into training, validation, and testing sets, and includes images categorized as either pneumonia or normal.

## Key Features

1. **Dataset**: 
   - Contains 5,863 pediatric X-ray images.
   - Organized into three folders: Train, Test, Val.
   - Images resized to a static size.

2. **Data Pre-processing**:
   - Rescaling images to values between 0 and 1.
   - Applying data augmentation techniques like shear transformation, zoom, horizontal flip, etc.

3. **CNN Architecture**:
   - Initial filter value: 32, increasing layer-wise.
   - Layers: Conv2D, max-pooling, ReLU activation, and dropout.
   - Flattening after CNN layers followed by dense layers for classification.
   - SoftMax activation for multi-class classification; Sigmoid for binary classification.

4. **Models Evaluated**:
   - AlexNet
   - ResNet-50
   - VGG16

5. **Implementation**:
   - Data augmentation using `tensorflow.keras.preprocessing.image.ImageDataGenerator`.
   - Confusion matrix for evaluating model performance.
   - Flask used for developing a simple web interface.

6. **Results**:
   - Achieved an accuracy rate of 92.07% and a precision rate of 91.41%.

## Future Work
- Further optimization of the CNN model.
- Exploration of additional data augmentation techniques.
- Integration with other machine learning frameworks.
- Development of a more robust user interface.

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
