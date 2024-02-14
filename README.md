# SkinScan Overview

Skin cancer in Mexico represents a growing public health concern, with an alarming increase in incidence. 
This study focuses on addressing this urgency using convolutional neural networks, a prominent technique in 
machine learning and computer vision, known for it is high accuracy and similarity to human visual processing. 
Our objective is to iteratively develop and analyze machine learning models based on computer vision techniques, 
evaluating their accuracy and the variations in each iteration. Our objective is to propose an architecture that 
excels in both accuracy for classifying new image instances and efficiency for seamless integration into a mobile application. 

This project uses the extensive HAM10000 dataset, which includes 10,015 dermatoscopic images, we implement a 
convolutional neural network model to effectively classify various skin lesions.

## Model Architecture Proposal

Our current proposed model incorporates concepts from the ResNet Image Classification Model created by Google. The advantage of 
this model lies in the shortcuts found within the residual blocks, allowing most of the original image's information to pass 
through the block and removing any noise that random weight initialization might introduce.

Our previous models also included Depth-Separable Convolutional Layers, enabling efficient convolution operations and the 
extraction of essential features from the image. However, this came at the cost of accuracy, leading us to remove such layers.

```
Input (224x224x3)
|
|-- Conv2D (3x3, 32 filters, stride 2, relu)
|-- BatchNormalization
|-- Conv2D (3x3, 64 filters, relu)
|-- BatchNormalization
|-- Depthwise Separable Convolution (3x3, 128 filters, relu)
|-- BatchNormalization
|
|-- Residual Block 1 (64 filters)
|   |-- Conv2D (1x1, 64 filters, relu)
|   |-- BatchNormalization
|   |-- Conv2D (3x3, 64 filters, relu)
|   |-- BatchNormalization
|   |-- Conv2D (1x1, 128 filters, stride 2, linear)
|   |-- BatchNormalization
|
|-- Residual Block 2 (128 filters)
|   |-- Conv2D (1x1, 128 filters, relu)
|   |-- BatchNormalization
|   |-- Conv2D (3x3, 128 filters, relu)
|   |-- BatchNormalization
|   |-- Conv2D (1x1, 256 filters, stride 2, linear)
|   |-- BatchNormalization
|
|-- Residual Block 3 (256 filters)
|   |-- Conv2D (1x1, 256 filters, relu)
|   |-- BatchNormalization
|   |-- Conv2D (3x3, 256 filters, relu)
|   |-- BatchNormalization
|   |-- Conv2D (1x1, 512 filters, linear)
|   |-- BatchNormalization
|
|-- Global Average Pooling
|-- Dense (Number of classes, softmax activation)
```
