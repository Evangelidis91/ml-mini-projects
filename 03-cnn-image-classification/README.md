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

The 3 convolutional blocks share the same internal pattern (Conv-BN-Conv-BN-Pool-Dropout) and are built via a small `conv_block(filters)` helper for readability.

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
| **Test Accuracy** | **88.07%** |
| Test Loss | 0.3600 |
| Epochs trained | 50 (full schedule) |
| Best epoch | 49 |
| Train-Val gap | 0.006 (✅ good generalization) |

> The learning rate reduction kicked in 3 times during training (at epochs 18, 37, and 45), each cut helping the model squeeze out more validation accuracy without overfitting.

### Per-Class Results
| Class | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| airplane | 0.91 | 0.90 | 0.91 | 89.8% |
| automobile | 0.91 | 0.97 | 0.94 | 96.7% |
| bird | 0.87 | 0.84 | 0.86 | 84.0% |
| cat | 0.83 | 0.69 | 0.75 | 69.1% |
| deer | 0.87 | 0.88 | 0.88 | 88.4% |
| dog | 0.87 | 0.76 | 0.81 | 75.5% |
| frog | 0.81 | 0.97 | 0.88 | 97.3% |
| horse | 0.91 | 0.93 | 0.92 | 92.8% |
| ship | 0.94 | 0.93 | 0.93 | 92.8% |
| truck | 0.89 | 0.94 | 0.91 | 94.3% |

**Best class:** frog (97.3%)  
**Worst class:** cat (69.1%)

> The model excels on classes with distinctive shapes/colours (vehicles, frogs, horses) and struggles with visually similar animal classes (cats, dogs, birds).

### Most Confused Pairs
| Actual | Predicted as | Count |
|--------|--------------|-------|
| dog | cat | 96 |
| cat | frog | 77 |
| cat | dog | 72 |
| bird | frog | 52 |
| deer | frog | 46 |
| truck | automobile | 41 |
| cat | bird | 40 |

> Cats ↔ dogs is the classic CIFAR-10 confusion: similar shapes, similar fur, often similar poses. Truck ↔ automobile is the other intuitive pair (both 4-wheeled vehicles).

### Confidence Analysis
| Confidence Level | Accuracy | Samples |
|------------------|----------|---------|
| 0–50% | 38.7% | 445 |
| 50–70% | 56.0% | 887 |
| 70–80% | 67.8% | 515 |
| 80–90% | 77.8% | 679 |
| 90–95% | 86.0% | 600 |
| 95–100% | 98.1% | 6,637 |

**Correct predictions avg confidence:** 94.0%  
**Wrong predictions avg confidence:** 67.1%

> The model is well-calibrated: when it's confident (>95%), it's right 98% of the time. About 66% of test samples land in this high-confidence bucket.

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
- **TensorFlow / Keras** — Model building and training
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