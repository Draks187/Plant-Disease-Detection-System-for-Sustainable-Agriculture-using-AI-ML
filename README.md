# 🌿 Plant Disease Detection System for Sustainable Agriculture

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-89.6%25-green)
![Classes](https://img.shields.io/badge/Classes-38-teal)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A deep learning system that classifies plant leaf diseases across 14 crop species into 38 categories. Built to support early disease detection in agriculture, reducing crop loss and minimizing unnecessary pesticide use.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Supported Classes](#supported-classes)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

---

## Overview

This project uses a custom CNN trained on the [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle. The model takes a 224×224 RGB leaf image as input and outputs one of 38 disease or healthy-plant labels.

The goal is to assist farmers and agronomists in identifying diseases early, enabling timely intervention and contributing to more sustainable crop management.

---

## Dataset

| Split      | Images  | Notes                          |
|------------|---------|--------------------------------|
| Training   | 63,282  | 90° rotation augmentation      |
| Validation | 1,742   | Rescale only                   |
| Test       | 17,572  | Rescale only, no shuffle       |

- **Input size:** 224 × 224 × 3 (RGB)
- **Classes:** 38 (disease + healthy labels across 14 crop types)
- **Source:** New Plant Diseases Dataset (Augmented) on Kaggle

---

## Model Architecture

A custom sequential CNN with ~26.1M trainable parameters.

| Layer   | Type       | Output Shape       | Params     |
|---------|------------|--------------------|------------|
| Conv1   | Conv2D     | (None, 224, 224, 32) | 4,736    |
| Pool1   | MaxPool2D  | (None, 112, 112, 32) | 0        |
| Conv2   | Conv2D     | (None, 112, 112, 64) | 51,264   |
| Pool2   | MaxPool2D  | (None, 56, 56, 64)   | 0        |
| Conv3   | Conv2D     | (None, 56, 56, 128)  | 73,856   |
| Conv4   | Conv2D     | (None, 56, 56, 256)  | 295,168  |
| Pool3   | MaxPool2D  | (None, 28, 28, 256)  | 0        |
| Flatten | Flatten    | (None, 200704)       | 0        |
| Dense1  | Dense(ReLU)| (None, 128)          | 25,690,240|
| Dense2  | Dense(ReLU)| (None, 64)           | 8,256    |
| Output  | Dense(Softmax)| (None, 38)        | 2,470    |

**Total parameters:** 26,125,990 (~99.7 MB)

**Training config:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

---

## Results

### Training history (5 epochs)

| Epoch | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| 1     | 17.3%     | 59.6%   | 1.3503   |
| 2     | 69.1%     | 77.1%   | 0.7364   |
| 3     | 80.8%     | 85.3%   | 0.4671   |
| 4     | 86.4%     | 85.5%   | 0.4326   |
| **5** | **89.2%** |**88.4%**|**0.3621**|

### Test set evaluation (17,572 images)

| Metric    | Score  |
|-----------|--------|
| Loss      | 0.3207 |
| Accuracy  | 89.58% |
| Precision | 91.79% |
| Recall    | 87.92% |

---

## Supported Classes

38 classes across 14 plant species. Healthy classes are marked ✅.

<details>
<summary>Click to expand all 38 classes</summary>

- Apple — Apple Scab
- Apple — Black Rot
- Apple — Cedar Apple Rust
- Apple — Healthy ✅
- Blueberry — Healthy ✅
- Cherry — Powdery Mildew
- Cherry — Healthy ✅
- Corn — Cercospora / Gray Leaf Spot
- Corn — Common Rust
- Corn — Northern Leaf Blight
- Corn — Healthy ✅
- Grape — Black Rot
- Grape — Esca (Black Measles)
- Grape — Leaf Blight (Isariopsis Leaf Spot)
- Grape — Healthy ✅
- Orange — Haunglongbing (Citrus Greening)
- Peach — Bacterial Spot
- Peach — Healthy ✅
- Pepper (Bell) — Bacterial Spot
- Pepper (Bell) — Healthy ✅
- Potato — Early Blight
- Potato — Late Blight
- Potato — Healthy ✅
- Raspberry — Healthy ✅
- Soybean — Healthy ✅
- Squash — Powdery Mildew
- Strawberry — Leaf Scorch
- Strawberry — Healthy ✅
- Tomato — Bacterial Spot
- Tomato — Early Blight
- Tomato — Late Blight
- Tomato — Leaf Mold
- Tomato — Septoria Leaf Spot
- Tomato — Spider Mites / Two-Spotted Spider Mite
- Tomato — Target Spot
- Tomato — Yellow Leaf Curl Virus
- Tomato — Mosaic Virus
- Tomato — Healthy ✅

</details>

---

## Getting Started

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib opencv-python seaborn
```

### Load the saved model
```python
from tensorflow import keras
import numpy as np

model = keras.models.load_model('PDDS.keras')

# Preprocess and predict
img = keras.preprocessing.image.load_img('leaf.jpg', target_size=(224, 224))
img_array = keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
```

### Train from scratch
Run the notebook in Google Colab. Mount your Drive, extract the dataset zip, and execute all cells in order.

---

## Project Structure
