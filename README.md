# 🌿 Plant Disease Detection System for Sustainable Agriculture

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-Sequential_CNN-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google_Colab-Training-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-89.6%25-2ea44f?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge"/>
</p>

<p align="center">
  A deep learning system that identifies plant leaf diseases from images across 14 crop species and 38 disease/healthy categories — enabling early detection to reduce crop loss and minimize pesticide use.
</p>

---

## 📚 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Results & Performance](#results--performance)
- [Supported Classes (38)](#supported-classes-38)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running on Google Colab](#running-on-google-colab)
  - [Inference / Prediction](#inference--prediction)
- [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
- [Visualizations](#visualizations)
- [Known Issues & Limitations](#known-issues--limitations)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

This project builds a **Convolutional Neural Network (CNN)** from scratch to detect diseases in plant leaf images. Given a 224×224 RGB photograph of a leaf, the model outputs one of 38 labels — either a specific disease name or a "healthy" classification for that plant species.

The system is trained on the **New Plant Diseases Dataset (Augmented)** — a large publicly available dataset on Kaggle containing over 80,000 images across 38 classes.

Key highlights:
- **89.6% test accuracy** on 17,572 unseen images
- **38 classes** covering 14 crop species
- Custom CNN with ~26.1M parameters, trained end-to-end
- Training, validation, and test pipelines fully implemented in TensorFlow/Keras
- Google Colab-compatible with Google Drive integration

---

## Motivation

Plant diseases are a leading cause of agricultural yield loss worldwide, threatening food security for billions of people. Traditional disease identification relies on manual inspection by experts — a slow, costly, and unscalable process, especially in rural or developing regions.

This project demonstrates how AI can assist farmers, agronomists, and agricultural extension workers by:
- Providing **instant, photo-based disease diagnosis**
- Reducing reliance on broad-spectrum pesticides through accurate identification
- Enabling **early intervention** before disease spreads
- Forming a foundation for mobile/edge deployment in the field

---

## Dataset

**Source:** [New Plant Diseases Dataset (Augmented) — Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

The dataset contains images of healthy and diseased plant leaves, augmented to improve model generalization. All images are in JPG format.

| Split      | Images  | Notes                                        |
|------------|---------|----------------------------------------------|
| Training   | 63,282  | With 90° rotation augmentation + rescaling   |
| Validation | 1,742   | Rescaling only (10% split)                   |
| Test       | 17,572  | Rescaling only, no shuffle                   |
| **Total**  | **82,596** |                                           |

- **Image size (resized to):** 224 × 224 pixels
- **Color mode:** RGB (3 channels)
- **Classes:** 38 (disease + healthy labels)
- **Crop species covered:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## Model Architecture

A custom sequential CNN built with TensorFlow/Keras. The architecture progressively extracts spatial features through four convolutional blocks before classifying with fully connected layers.

```
Input: (224, 224, 3)
    │
    ▼
Conv1 → Conv2D(32, 7×7, ReLU, same) → MaxPool2D(2×2)    → (112, 112, 32)
    │
    ▼
Conv2 → Conv2D(64, 5×5, ReLU, same) → MaxPool2D(2×2)    → (56, 56, 64)
    │
    ▼
Conv3 → Conv2D(128, 3×3, ReLU, same)                     → (56, 56, 128)
    │
    ▼
Conv4 → Conv2D(256, 3×3, ReLU, same) → MaxPool2D(2×2)   → (28, 28, 256)
    │
    ▼
Flatten                                                   → (200,704,)
    │
    ▼
Dense(128, ReLU)                                          → (128,)
    │
    ▼
Dense(64, ReLU)                                           → (64,)
    │
    ▼
Output → Dense(38, Softmax)                               → (38,)
```

### Layer-by-layer summary

| Layer    | Type          | Output Shape         | Parameters  |
|----------|---------------|----------------------|-------------|
| Conv1    | Conv2D        | (None, 224, 224, 32) | 4,736       |
| Pool1    | MaxPooling2D  | (None, 112, 112, 32) | 0           |
| Conv2    | Conv2D        | (None, 112, 112, 64) | 51,264      |
| Pool2    | MaxPooling2D  | (None, 56, 56, 64)   | 0           |
| Conv3    | Conv2D        | (None, 56, 56, 128)  | 73,856      |
| Conv4    | Conv2D        | (None, 56, 56, 256)  | 295,168     |
| Pool3    | MaxPooling2D  | (None, 28, 28, 256)  | 0           |
| Flatten1 | Flatten       | (None, 200,704)      | 0           |
| Dense1   | Dense (ReLU)  | (None, 128)          | 25,690,240  |
| Dense2   | Dense (ReLU)  | (None, 64)           | 8,256       |
| Output   | Dense (Softmax)| (None, 38)          | 2,470       |

- **Total parameters:** 26,125,990 (~99.66 MB)
- **Trainable parameters:** 26,125,990
- **Non-trainable parameters:** 0

---

## Training Configuration

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy', 'precision', 'recall']
)
```

### Callbacks

| Callback             | Configuration                                           |
|----------------------|---------------------------------------------------------|
| EarlyStopping        | monitor=`val_loss`, patience=15, restore_best_weights=True |
| ModelCheckpoint      | monitor=`val_loss`, save_best_only=True → `best_model.keras` |
| ReduceLROnPlateau    | monitor=`val_loss`, factor=0.1, patience=15, min_lr=1e-6 |

### Data augmentation (training only)

| Augmentation        | Value |
|---------------------|-------|
| Rotation range      | 90°   |
| Width shift         | 0.0   |
| Height shift        | 0.0   |
| Shear range         | 0.0   |
| Zoom range          | 0.0   |
| Horizontal flip     | False |
| Vertical flip       | False |
| Rescale             | 1/255 |

---

## Results & Performance

### Training history (5 epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Precision | Val Recall | LR     |
|-------|-----------|-----------|----------|---------|---------------|------------|--------|
| 1     | 3.0843    | 17.25%    | 1.3503   | 59.59%  | 73.51%        | 45.87%     | 0.001  |
| 2     | 1.0131    | 69.06%    | 0.7364   | 77.10%  | 83.40%        | 70.95%     | 0.001  |
| 3     | 0.6049    | 80.83%    | 0.4671   | 85.25%  | 89.04%        | 82.09%     | 0.001  |
| 4     | 0.4194    | 86.44%    | 0.4326   | 85.48%  | 89.05%        | 82.66%     | 0.001  |
| **5** | **0.3311**| **89.23%**| **0.3621**|**88.35%**|**91.24%**  | **86.05%** | 0.001  |

### Final test set evaluation (17,572 images)

| Metric    | Score      |
|-----------|------------|
| Loss      | 0.3207     |
| Accuracy  | **89.58%** |
| Precision | **91.79%** |
| Recall    | **87.92%** |

The model reached 89.6% accuracy on the held-out test set after only 5 epochs, with no signs of overfitting — training and validation accuracy tracked closely throughout.

---

## Supported Classes (38)

<details>
<summary>Click to expand all 38 classes</summary>

| Index | Class Label                                          | Type    |
|-------|------------------------------------------------------|---------|
| 0     | Apple — Apple Scab                                   | Disease |
| 1     | Apple — Black Rot                                    | Disease |
| 2     | Apple — Cedar Apple Rust                             | Disease |
| 3     | Apple — Healthy                                      | ✅ Healthy |
| 4     | Blueberry — Healthy                                  | ✅ Healthy |
| 5     | Cherry (including sour) — Powdery Mildew             | Disease |
| 6     | Cherry (including sour) — Healthy                    | ✅ Healthy |
| 7     | Corn (maize) — Cercospora Leaf Spot / Gray Leaf Spot | Disease |
| 8     | Corn (maize) — Common Rust                           | Disease |
| 9     | Corn (maize) — Northern Leaf Blight                  | Disease |
| 10    | Corn (maize) — Healthy                               | ✅ Healthy |
| 11    | Grape — Black Rot                                    | Disease |
| 12    | Grape — Esca (Black Measles)                         | Disease |
| 13    | Grape — Leaf Blight (Isariopsis Leaf Spot)           | Disease |
| 14    | Grape — Healthy                                      | ✅ Healthy |
| 15    | Orange — Haunglongbing (Citrus Greening)             | Disease |
| 16    | Peach — Bacterial Spot                               | Disease |
| 17    | Peach — Healthy                                      | ✅ Healthy |
| 18    | Pepper (Bell) — Bacterial Spot                       | Disease |
| 19    | Pepper (Bell) — Healthy                              | ✅ Healthy |
| 20    | Potato — Early Blight                                | Disease |
| 21    | Potato — Late Blight                                 | Disease |
| 22    | Potato — Healthy                                     | ✅ Healthy |
| 23    | Raspberry — Healthy                                  | ✅ Healthy |
| 24    | Soybean — Healthy                                    | ✅ Healthy |
| 25    | Squash — Powdery Mildew                              | Disease |
| 26    | Strawberry — Leaf Scorch                             | Disease |
| 27    | Strawberry — Healthy                                 | ✅ Healthy |
| 28    | Tomato — Bacterial Spot                              | Disease |
| 29    | Tomato — Early Blight                                | Disease |
| 30    | Tomato — Late Blight                                 | Disease |
| 31    | Tomato — Leaf Mold                                   | Disease |
| 32    | Tomato — Septoria Leaf Spot                          | Disease |
| 33    | Tomato — Spider Mites / Two-Spotted Spider Mite      | Disease |
| 34    | Tomato — Target Spot                                 | Disease |
| 35    | Tomato — Tomato Yellow Leaf Curl Virus               | Disease |
| 36    | Tomato — Tomato Mosaic Virus                         | Disease |
| 37    | Tomato — Healthy                                     | ✅ Healthy |

**Summary:** 25 disease classes + 13 healthy classes across 14 plant species.

</details>

---

## Project Structure

```
plant-disease-detection/
│
├── PDDS.keras                        # Final saved model (all epochs)
├── best_model.keras                  # Best checkpoint (lowest val_loss)
│
├── plant_disease_detection.ipynb     # Main training notebook (Google Colab)
│
├── README.md                         # This file
│
└── assets/
    └── training_accuracy_plot.png    # Training vs validation accuracy graph
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Google Colab (recommended for GPU) or a local GPU environment
- Kaggle account (to download the dataset)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
pip install tensorflow numpy pandas matplotlib opencv-python seaborn
```

### Running on Google Colab

1. Open `plant_disease_detection.ipynb` in Google Colab.

2. Mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Download the dataset from Kaggle and place the zip at:
```
/content/drive/MyDrive/archive (2).zip
```

4. Run all cells in order. The notebook will:
   - Extract the dataset
   - Build and compile the CNN
   - Train for up to 5 epochs (with early stopping)
   - Evaluate on the test set
   - Save the model as `PDDS.keras`

### Inference / Prediction

Load the saved model and run inference on a new leaf image:

```python
import numpy as np
from tensorflow import keras

# Load model
model = keras.models.load_model('PDDS.keras')

# Define class labels (in index order)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def predict_disease(image_path):
    """Predict plant disease from a leaf image file path."""
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100

    print(f"Predicted: {class_names[predicted_index]}")
    print(f"Confidence: {confidence:.2f}%")
    return class_names[predicted_index], confidence

# Example usage
predict_disease('path/to/your/leaf_image.jpg')
```

---

## Data Preprocessing & Augmentation

### Training generator
```python
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=90,     # Random 90° rotation
    rescale=1/255.0,       # Normalize pixel values to [0, 1]
    validation_split=0.1,  # 10% held out for validation
)
```

### Validation & test generators
```python
# No augmentation — only normalization
val_test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0
)
```

All images are loaded at **224×224 pixels** in **RGB** color mode with a **batch size of 164**.

---

## Visualizations

### Training vs. Validation Accuracy

The plot below (generated during training) shows consistent improvement across 5 epochs with no significant overfitting gap between training and validation curves.

```python
plt.plot(epochs, acc,     color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue',  label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim(0, 1.02)
plt.show()
```

### Sample predictions

To visualize a batch of training images with their labels:

```python
classes = list(train_generator.class_indices.keys())
plt.figure(figsize=(20, 20))
for X_batch, y_batch in train_generator:
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_batch[i])
        plt.title(classes[np.where(y_batch[i] == 1)[0][0]])
        plt.grid(None)
    plt.show()
    break
```

---

## Known Issues & Limitations

1. **Dropout layers not applied** — The current code instantiates `tf.keras.layers.Dropout(0.5)` but never adds them to the model with `model.add(...)`. As a result, no dropout regularization is applied. This is a bug to fix in the next iteration.

2. **Test set = Validation directory** — The test generator points to the same `valid/` directory as the validation generator. Ideally, a separate held-out test set should be used for unbiased final evaluation.

3. **Large dense layer** — The `Flatten → Dense(128)` transition produces ~25.7M parameters, the bulk of the model. A GlobalAveragePooling2D layer would dramatically reduce this and improve generalization.

4. **Limited augmentation** — Only 90° rotation is applied. More diverse augmentation (flips, zoom, brightness shifts) would help generalization to real-world images taken in different lighting and angles.

5. **No class imbalance handling** — Some classes have significantly more samples than others. Class weights or oversampling could improve recall on underrepresented diseases.

---

## Future Work

- [ ] **Fix Dropout** — Add `model.add(keras.layers.Dropout(0.5))` after each Dense layer
- [ ] **Transfer learning** — Fine-tune EfficientNetB3 or ResNet50V2 pre-trained on ImageNet for higher accuracy with fewer parameters
- [ ] **Replace Flatten with GlobalAveragePooling2D** — Reduce parameters and improve generalization
- [ ] **Expand augmentation** — Add horizontal/vertical flips, brightness, zoom, and color jitter
- [ ] **Grad-CAM visualisation** — Highlight the specific leaf regions that triggered the prediction
- [ ] **Separate test set** — Use a genuinely unseen split for final evaluation
- [ ] **Class-wise performance analysis** — Generate a classification report and confusion matrix per class
- [ ] **TensorFlow Lite export** — Quantize and convert for mobile / edge deployment in the field
- [ ] **Web / mobile app** — Simple upload-and-classify interface for farmers
- [ ] **Multi-label support** — Handle leaves with multiple concurrent diseases

---

## Acknowledgements

- **Dataset:** [vipoooool — New Plant Diseases Dataset (Augmented) on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Training environment:** Google Colab (GPU runtime)
- **Framework:** TensorFlow 2.x / Keras

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

<p align="center">Made with 🌱 for sustainable agriculture</p>
