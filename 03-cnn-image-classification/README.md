# CIFAR-10 Image Classification with CNN

Convolutional Neural Network for classifying 32x32 colour images into 10 categories.

## Overview

This project implements a CNN from scratch to classify small images into 10 categories. It demonstrates the effectiveness of deep convolutional architectures with batch normalization, dropout, and data augmentation for image classification tasks.

## Dataset

**CIFAR-10** (built into TensorFlow/Keras)
- **Total images:** 60,000 (32x32 RGB)
- **Training:** 50,000 images (5,000 per class)
- **Test:** 10,000 images (1,000 per class)
- **Classes:** 10, perfectly balanced
- **Image size:** 32×32×3 (RGB)

### Classes
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Model Architecture
Input (32×32×3) ↓ 
    Block 1: Conv2D(32) 
        → BatchNorm 
        → Conv2D(32)
        → BatchNorm
        → MaxPool
        → Dropout(0.25)
    Block 2: Conv2D(64) 
        → BatchNorm 
        → Conv2D(64) 
        → BatchNorm 
        → MaxPool 
        → Dropout(0.25)
    Block 3: Conv2D(128) 
        → BatchNorm
        → Conv2D(128)
        → BatchNorm
        → MaxPool
        → Dropout(0.25)
    Flatten 
        → Dense(256)
        → BatchNorm
        → Dropout(0.5)
        → Dense(10, softmax)

| Component | Details |
|-----------|---------|
| Total parameters | 816,938 |
| Trainable parameters | 815,530 |
| Non-trainable parameters | 1,408 |
| Model size | 3.12 MB |

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping | patience=10, restore best weights |
| Learning rate reduction | factor=0.5, patience=5 |
| Data augmentation | rotation ±15°, shift ±10%, horizontal flip |

## Results

### Overall Performance
| Metric | Score |
|--------|-------|
| **Test Accuracy** | **77.83%** |
| Epochs trained | 18 (early stopped) |
| Best epoch | 8 |

### Per-Class Results
| Class | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| airplane | 0.74 | 0.85 | 0.79 | 84.6% |
| automobile | 0.83 | 0.94 | 0.88 | 94.2% |
| bird | 0.84 | 0.57 | 0.68 | 57.2% |
| cat | 0.70 | 0.52 | 0.60 | 51.9% |
| deer | 0.75 | 0.80 | 0.77 | 79.6% |
| dog | 0.83 | 0.53 | 0.65 | 52.8% |
| frog | 0.73 | 0.92 | 0.82 | 91.9% |
| horse | 0.77 | 0.86 | 0.81 | 86.4% |
| ship | 0.85 | 0.89 | 0.87 | 89.2% |
| truck | 0.78 | 0.91 | 0.84 | 90.5% |

**Best class:** automobile (94.2%)  
**Worst class:** cat (51.9%)

### Most Confused Pairs
| Actual | Predicted as | Count |
|--------|-------------|-------|
| dog | cat | 158 |
| bird | frog | 103 |
| bird | airplane | 101 |
| dog | horse | 98 |
| cat | frog | 95 |

### Confidence Analysis
| Confidence Level | Accuracy | Samples |
|-----------------|----------|---------|
| 0–50% | 36.5% | 1,253 |
| 50–70% | 53.5% | 1,590 |
| 70–80% | 70.4% | 814 |
| 80–90% | 77.2% | 1,044 |
| 90–95% | 89.2% | 845 |
| 95–100% | 97.5% | 4,451 |

**Correct predictions avg confidence:** 87.6%  
**Wrong predictions avg confidence:** 59.8%

## Visualizations

| Plot | Description |
|------|-------------|
| `01_sample_per_class.png` | One sample image per class |
| `02_random_samples.png` | Random training samples |
| `03_class_distribution.png` | Class balance in train/test |
| `04_average_images.png` | Average image per class (class prototypes) |
| `05_augmentation_examples.png` | Data augmentation examples |
| `06_training_history.png` | Accuracy and loss curves |
| `07_confusion_matrix.png` | 10×10 confusion matrix |
| `08_correct_predictions.png` | Sample correct predictions with confidence |
| `09_wrong_predictions.png` | Sample wrong predictions with confidence |
| `10_per_class_accuracy.png` | Accuracy bar chart per class |
| `11_confidence_analysis.png` | Confidence distribution and accuracy by confidence |

## Tech Stack

- **Python 3.10+**
- **TensorFlow 2.15** — Model building and training
- **Keras** — High-level neural network API
- **scikit-learn** — Classification report, confusion matrix
- **numpy** — Array operations
- **matplotlib** — Plotting
- **seaborn** — Confusion matrix heatmap

## Project Structure
03-cnn-image-classification/ 
    ├── cnn_classifier.py # Main pipeline script 
    ├── requirements.txt # Python dependencies 
    ├── README.md # This file 
    └── plots/ # Generated visualizations 
        ├── 01_sample_per_class.png 
        ├── 02_random_samples.png 
        ├── 03_class_distribution.png 
        ├── 04_average_images.png 
        ├── 05_augmentation_examples.png 
        ├── 06_training_history.png 
        ├── 07_confusion_matrix.png 
        ├── 08_correct_predictions.png
        ├── 09_wrong_predictions.png
        ├── 10_per_class_accuracy.png 
        └── 11_confidence_analysis.png

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python cnn_classifier.py