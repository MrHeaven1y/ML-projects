# Plant Disease Classification

## Overview

This project implements a deep learning-based plant disease classification system using transfer learning with the VGG16 architecture. The model is trained to identify 38 different classes of plant diseases across various crops, providing a tool for early detection and diagnosis of plant diseases.

## Dataset

The project uses the "New Plant Diseases Dataset" from Kaggle, which contains over 87,000 RGB images of healthy and diseased plant leaves, categorized into 38 different classes. The dataset includes:

- Training set: 70,295 images
- Validation set: 17,572 images
- Test set: Various plant disease images for testing

## Model Architecture

The classification model uses transfer learning with the following architecture:

- Base model: Pre-trained VGG16 (without top layers)
- Custom classification head:
    - Flatten layer
    - Dense layer with 256 units and ReLU activation
    - Dropout layer (0.5)
    - Output layer with softmax activation (38 classes)

## Implementation Details

- **Framework**: TensorFlow/Keras
- **Image Size**: 224×224 pixels (VGG16 input size)
- **Data Augmentation**:
    - Rotation
    - Width/height shifts
    - Shear transformation
    - Zoom
    - Horizontal flips
- **Training**:
    - Optimizer: Adam (learning rate: 1e-4)
    - Loss function: Categorical crossentropy
    - Batch size: 32
    - Epochs: 10

## Performance

The model achieved approximately 90.7% validation accuracy after 10 epochs of training, demonstrating its effectiveness in classifying plant diseases from leaf images.

## Disease Classes

The model can identify the following plant diseases and healthy plants:

1. Potato - Healthy
2. Raspberry - Healthy
3. Soybean - Healthy
4. Potato - Late Blight
5. Strawberry - Leaf Scorch
6. Apple - Cedar Apple Rust
7. Potato - Early Blight
8. Tomato - Leaf Mold
9. Cherry - Powdery Mildew
10. Peach - Bacterial Spot
11. Tomato - Mosaic Virus
12. Cherry - Healthy
13. Peach - Healthy
14. Tomato - Spider Mites
15. Apple - Black Rot
16. Corn - Common Rust
17. Apple - Scab
18. Corn - Healthy
19. Squash - Powdery Mildew
20. Grape - Leaf Blight
21. Corn - Northern Leaf Blight
22. Tomato - Septoria Leaf Spot
23. Grape - Healthy
24. Bell Pepper - Bacterial Spot
25. Corn - Gray Leaf Spot
26. Grape - Black Rot
27. Blueberry - Healthy
28. Tomato - Yellow Leaf Curl Virus
29. Tomato - Bacterial Spot
30. Tomato - Healthy
31. Tomato - Early Blight
32. Apple - Healthy
33. Tomato - Target Spot
34. Grape - Black Measles
35. Strawberry - Healthy
36. Orange - Citrus Greening
37. Tomato - Late Blight
38. Bell Pepper - Healthy

## Usage

To use the model for prediction:

```python
def predict_image(img_path, model):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return predicted_class, confidence
```

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib (for visualization)
- KaggleHub (for dataset access)

## Future Improvements

- Fine-tuning the base model layers
- Testing with more recent architectures (EfficientNet, ResNet)
- Implementing explainability techniques (Grad-CAM)
- Creating a web or mobile application interface

## Acknowledgements

The dataset used in this project is the ["New Plant Diseases Dataset"](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle, created by Vipool Jolly.

***

### preview:
![Image](https://github.com/user-attachments/assets/a140013a-817b-4159-88ee-798043eea356)
![Image](https://github.com/user-attachments/assets/844c53f3-12c2-41c9-84f3-46b22d5ddd8e)
![Image](https://github.com/user-attachments/assets/090d62f7-7af1-41cb-9772-d44aad51f95c)
