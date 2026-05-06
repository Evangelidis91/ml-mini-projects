"""
CIFAR-10 Image Classification with CNN
======================================
Convolutional Neural Network for classifying images into 10
categories using the CIFAR-10 dataset.

Dataset: CIFAR-10 (60,000 32x32 colour images, 10 classes)
Architecture: 3 Conv blocks + BatchNorm + Dropout + Dense layers
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # This hides the GPU from TensorFlow
import keras
from keras import layers, models, callbacks
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os

warnings.filterwarnings("ignore")

# Create output folder for plots
os.makedirs("plots", exist_ok=True)

# Class names for CIFAR-10
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data():
    """Load and preprocess CIFAR-10 dataset."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU available: {len(gpus) > 0}")
    if gpus:
        for gpu in gpus:
            print(f"  GPU: {gpu.name}")

    (X_train, y_train),(X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to 0-1
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Flatten labels for easier use
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print(f"Pixel range: {X_train.min():.1f} - {X_train.max():.1f}")
    print(f"Image size: {X_train.shape[1]}x{X_train.shape[2]}x{X_train.shape[3]}")

    return X_train, X_test, y_train, y_test, y_train_flat, y_test_flat


# ============================================================
# 2. VISUALIZE SAMPLES
# ============================================================
def plot_samples(X_train, y_train_flat):
    """Plot sample images from each class."""
    print("\n" + "=" * 60)
    print("SAMPLE IMAGES")
    print("=" * 60)

    # Show one image per class
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axes.flatten()):
        # Find first image of class i
        idx = np.where(y_train_flat == i)[0][0]
        ax.imshow(X_train[idx])
        ax.set_title(CLASS_NAMES[i], fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle("One Sample Per Class — CIFAR-10", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/01_sample_per_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/01_sample_per_class.png")

    # Show random samples
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))

    for ax in axes.flatten():
        idx = np.random.randint(0, len(X_train))
        ax.imshow(X_train[idx])
        ax.set_title(CLASS_NAMES[y_train_flat[idx]], fontsize=10)
        ax.axis("off")

    plt.suptitle("Random Training Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/02_random_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/02_random_samples.png")


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================
def plot_class_distribution(y_train_flat, y_test_flat):
    """Plot class distribution in train and test sets."""
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training set
    unique, counts = np.unique(y_train_flat, return_counts=True)
    axes[0].bar(
        [CLASS_NAMES[i] for i in unique], counts, color="steelblue", alpha=0.8
    )
    axes[0].set_title("Training Set Distribution")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Test set
    unique, counts = np.unique(y_test_flat, return_counts=True)
    axes[1].bar(
        [CLASS_NAMES[i] for i in unique], counts, color="coral", alpha=0.8
    )
    axes[1].set_title("Test Set Distribution")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("plots/03_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/03_class_distribution.png")

    print("\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        train_count = np.sum(y_train_flat == i)
        test_count = np.sum(y_test_flat == i)
        print(f"  {name:12s}: Train={train_count:,}, Test={test_count:,}")

    print(f"\nDataset is perfectly balanced (5,000 per class in train)")


# ============================================================
# 4. PIXEL ANALYSIS
# ============================================================
def plot_pixel_analysis(X_train, y_train_flat):
    """Analyze average pixel values per class."""
    print("\n" + "=" * 60)
    print("PIXEL ANALYSIS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axes.flatten()):
        class_images = X_train[y_train_flat == i]
        avg_image = class_images.mean(axis=0)
        ax.imshow(avg_image)
        ax.set_title(f"Avg: {CLASS_NAMES[i]}", fontsize=11)
        ax.axis("off")

    plt.suptitle("Average Image Per Class", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/04_average_images.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/04_average_images.png")

    # Mean pixel values per channel
    print("\nMean pixel values per channel:")
    print(f"  Red:   {X_train[:, :, :, 0].mean():.4f}")
    print(f"  Green: {X_train[:, :, :, 1].mean():.4f}")
    print(f"  Blue:  {X_train[:, :, :, 2].mean():.4f}")


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================
def create_data_augmentation(X_train):
    """Create and visualize data augmentation."""
    print("\n" + "=" * 60)
    print("DATA AUGMENTATION")
    print("=" * 60)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(X_train)

    # Visualize augmented images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    sample_img = X_train[0:1]

    axes[0][0].imshow(sample_img[0])
    axes[0][0].set_title("Original", fontweight="bold")
    axes[0][0].axis("off")

    for i, ax in enumerate(axes.flatten()[1:]):
        augmented = datagen.flow(sample_img, batch_size=1)
        ax.imshow(next(augmented)[0])
        ax.set_title(f"Augmented {i + 1}")
        ax.axis("off")

    plt.suptitle("Data Augmentation Examples", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/05_augmentation_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/05_augmentation_examples.png")

    print("\nAugmentation configuration:")
    print("  - Rotation: ±15°")
    print("  - Width shift: ±10%")
    print("  - Height shift: ±10%")
    print("  - Horizontal flip: Yes")

    return datagen


# ============================================================
# 6. BUILD MODEL
# ============================================================
def build_model():
    """Build the CNN model architecture."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)

    model = models.Sequential(
        [
            # Block 1: 32 filters
            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                input_shape=(32, 32, 3),
            ),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 2: 64 filters
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 3: 128 filters
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),

        ]
    )

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print model summary
    model.summary()

    # Count parameters by type
    total_params = model.count_params()
    trainable_params = sum(
        keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    print(f"\nTotal parameters:         {total_params:,}")
    print(f"Trainable parameters:     {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

    return model


# ============================================================
# 7. TRAIN MODEL
# ============================================================
def train_model(model, datagen, X_train, y_train, X_test, y_test):
    """Train the model with data augmentation and callbacks."""
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    batch_size = 64
    max_epochs = 50

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Early stopping patience: 10")
    print(f"  LR reduction patience: 5")
    print(f"  Optimizer: Adam")
    print(f"  Loss: Sparse Categorical Crossentropy")
    print(f"\nTraining with data augmentation...")
    print("-" * 60)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=max_epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    return history


# ============================================================
# 8. TRAINING HISTORY
# ============================================================
def plot_training_history(history):
    """Plot training and validation accuracy/loss curves."""
    print("\n" + "=" * 60)
    print("TRAINING HISTORY")
    print("=" * 60)

    epochs_trained = len(history.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/06_training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/06_training_history.png")

    # Print stats
    best_epoch = np.argmax(history.history["val_accuracy"])
    print(f"\nEpochs trained: {epochs_trained}")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"  Train accuracy:      {history.history['accuracy'][best_epoch]:.4f}")
    print(f"  Validation accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
    print(f"  Train loss:          {history.history['loss'][best_epoch]:.4f}")
    print(f"  Validation loss:     {history.history['val_loss'][best_epoch]:.4f}")

    # Check for overfitting
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    gap = final_train_acc - final_val_acc

    print(f"\nOverfitting check:")
    print(f"  Train-Val accuracy gap: {gap:.4f}")
    if gap > 0.1:
        print(f"  ⚠️  Significant overfitting detected")
    elif gap > 0.05:
        print(f"  ⚠️  Mild overfitting")
    else:
        print(f"  ✅ Good generalization")

# ============================================================
# 9. EVALUATE MODEL
# ============================================================
def evaluate_model(model, X_test, y_test, y_test_flat):
    """Evaluate model on test set."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"Test Loss:     {test_loss:.4f}")

    # Get predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\nClassification Report:")
    print(
        classification_report(
            y_test_flat, y_pred_classes, target_names=CLASS_NAMES
        )
    )

    return y_pred, y_pred_classes, test_accuracy, test_loss

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(y_test_flat, y_pred_classes, test_accuracy):
    """Plot confusion matrix."""
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(y_test_flat, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix (Accuracy: {test_accuracy:.2%})")
    plt.tight_layout()
    plt.savefig("plots/07_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/07_confusion_matrix.png")

    # Find most confused pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)

    print("\nMost confused pairs:")
    top_confused = []
    for i in range(10):
        for j in range(10):
            if cm_copy[i][j] > 0:
                top_confused.append((CLASS_NAMES[i], CLASS_NAMES[j], cm_copy[i][j]))

    top_confused.sort(key=lambda x: x[2], reverse=True)
    for actual, predicted, count in top_confused[:10]:
        print(f"  {actual:12s} → {predicted:12s}: {count} times")

# ============================================================
# 11. SAMPLE PREDICTIONS
# ============================================================
def plot_sample_predictions(X_test, y_test_flat, y_pred, y_pred_classes):
    """Plot sample predictions with confidence scores."""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    # Correct predictions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    correct_indices = np.where(y_pred_classes == y_test_flat)[0]

    for i, ax in enumerate(axes.flatten()):
        if i < len(correct_indices):
            idx = correct_indices[np.random.randint(0, len(correct_indices))]
            ax.imshow(X_test[idx])
            confidence = np.max(y_pred[idx]) * 100
            ax.set_title(
                f"{CLASS_NAMES[y_test_flat[idx]]} ({confidence:.0f}%)",
                color="green",
                fontsize=10,
            )
        ax.axis("off")

    plt.suptitle("Correct Predictions ✅", fontsize=14, color="green")
    plt.tight_layout()
    plt.savefig("plots/08_correct_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/08_correct_predictions.png")

    # Wrong predictions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    wrong_indices = np.where(y_pred_classes != y_test_flat)[0]

    for i, ax in enumerate(axes.flatten()):
        if i < len(wrong_indices):
            idx = wrong_indices[np.random.randint(0, len(wrong_indices))]
            ax.imshow(X_test[idx])
            confidence = np.max(y_pred[idx]) * 100
            true_label = CLASS_NAMES[y_test_flat[idx]]
            pred_label = CLASS_NAMES[y_pred_classes[idx]]
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label} ({confidence:.0f}%)",
                color="red",
                fontsize=9,
            )
        ax.axis("off")

    plt.suptitle("Wrong Predictions ❌", fontsize=14, color="red")
    plt.tight_layout()
    plt.savefig("plots/09_wrong_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/09_wrong_predictions.png")

    total = len(y_test_flat)
    correct = len(correct_indices)
    wrong = len(wrong_indices)
    print(f"\nTotal predictions: {total}")
    print(f"Correct: {correct} ({correct/total:.2%})")
    print(f"Wrong:   {wrong} ({wrong/total:.2%})")

# ============================================================
# 12. PER-CLASS ACCURACY
# ============================================================
def plot_per_class_accuracy(y_test_flat, y_pred_classes):
    """Plot accuracy for each class."""
    print("\n" + "=" * 60)
    print("PER-CLASS ACCURACY")
    print("=" * 60)

    class_accuracies = []
    for i in range(10):
        mask = y_test_flat == i
        class_acc = np.mean(y_pred_classes[mask] == i)
        class_accuracies.append(class_acc)

    # Sort by accuracy for better visualization
    sorted_indices = np.argsort(class_accuracies)
    sorted_names = [CLASS_NAMES[i] for i in sorted_indices]
    sorted_accs = [class_accuracies[i] for i in sorted_indices]

    print("\nAccuracy per class (sorted):")
    for name, acc in zip(sorted_names, sorted_accs):
        bar = "█" * int(acc * 30)
        print(f"  {name:12s}: {acc:.2%} {bar}")

    # Plot
    colors = [
        "green" if acc > 0.85 else "orange" if acc > 0.75 else "red"
        for acc in sorted_accs
    ]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_names, sorted_accs, color=colors, alpha=0.8)
    plt.xlabel("Accuracy")
    plt.title("Per-Class Accuracy (Green > 85%, Orange > 75%, Red ≤ 75%)")
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (acc, name) in enumerate(zip(sorted_accs, sorted_names)):
        plt.text(acc + 0.01, i, f"{acc:.1%}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("plots/10_per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/10_per_class_accuracy.png")

    # Best and worst classes
    best_class = CLASS_NAMES[np.argmax(class_accuracies)]
    worst_class = CLASS_NAMES[np.argmin(class_accuracies)]
    print(f"\nBest class:  {best_class} ({max(class_accuracies):.2%})")
    print(f"Worst class: {worst_class} ({min(class_accuracies):.2%})")

    return class_accuracies

# ============================================================
# 13. CONFIDENCE ANALYSIS
# ============================================================
def plot_confidence_analysis(y_test_flat, y_pred, y_pred_classes):
    """Analyze prediction confidence for correct vs wrong predictions."""
    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS")
    print("=" * 60)

    max_confidences = np.max(y_pred, axis=1) * 100

    correct_mask = y_pred_classes == y_test_flat
    correct_conf = max_confidences[correct_mask]
    wrong_conf = max_confidences[~correct_mask]

    print(f"Correct predictions — Avg confidence: {correct_conf.mean():.1f}%")
    print(f"Wrong predictions   — Avg confidence: {wrong_conf.mean():.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram comparison
    axes[0].hist(
        correct_conf, bins=30, alpha=0.7, color="green", label="Correct", density=True
    )
    axes[0].hist(
        wrong_conf, bins=30, alpha=0.7, color="red", label="Wrong", density=True
    )
    axes[0].set_xlabel("Confidence (%)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Confidence Distribution: Correct vs Wrong")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy by confidence bucket
    buckets = [(0, 50), (50, 70), (70, 80), (80, 90), (90, 95), (95, 100)]
    bucket_labels = []
    bucket_accuracies = []

    for low, high in buckets:
        mask = (max_confidences >= low) & (max_confidences < high)
        if mask.sum() > 0:
            acc = correct_mask[mask].mean()
            bucket_labels.append(f"{low}-{high}%")
            bucket_accuracies.append(acc)
            print(f"  Confidence {low:3d}-{high:3d}%: Accuracy={acc:.2%} (n={mask.sum()})")

    axes[1].bar(bucket_labels, bucket_accuracies, color="steelblue", alpha=0.8)
    axes[1].set_xlabel("Confidence Bucket")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy by Confidence Level")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("plots/11_confidence_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/11_confidence_analysis.png")

# ============================================================
# 14. FINAL SUMMARY
# ============================================================
def print_summary(model, test_accuracy, test_loss, class_accuracies, history):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    epochs_trained = len(history.history["accuracy"])
    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    best_class = CLASS_NAMES[np.argmax(class_accuracies)]
    worst_class = CLASS_NAMES[np.argmin(class_accuracies)]

    print(f"""
Dataset: CIFAR-10
Training:   50,000 images
Test:       10,000 images
Classes:    {len(CLASS_NAMES)}
Image size: 32x32x3

Model Architecture:
- 3 Convolutional blocks (32 → 64 → 128 filters)
- BatchNormalization after each Conv layer
- MaxPooling + Dropout(0.25) after each block
- Dense(256) + BatchNorm + Dropout(0.5)
- Softmax output (10 classes)
- Total parameters: {model.count_params():,}

Training Configuration:
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Data augmentation: rotation, shift, flip
- Early stopping (patience=10)
- Learning rate reduction (patience=5)

Results:
- Epochs trained: {epochs_trained}
- Best epoch: {best_epoch}
- Test Accuracy: {test_accuracy:.2%}
- Test Loss: {test_loss:.4f}
- Best class:  {best_class} ({max(class_accuracies):.2%})
- Worst class: {worst_class} ({min(class_accuracies):.2%})
""")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================
def main():
    """Run the complete CNN classification pipeline."""

    # 1. Load data
    X_train, X_test, y_train, y_test, y_train_flat, y_test_flat = load_data()

    # 2. Visualize samples
    plot_samples(X_train, y_train_flat)

    # 3. Class distribution
    plot_class_distribution(y_train_flat, y_test_flat)

    # 4. Pixel analysis
    plot_pixel_analysis(X_train, y_train_flat)

    # 5. Data augmentation
    datagen = create_data_augmentation(X_train)

    # 6. Build model
    model = build_model()

    # 7. Train model
    history = train_model(model, datagen, X_train, y_train, X_test, y_test)

    # 8. Training history
    plot_training_history(history)

    # 9. Evaluate
    y_pred, y_pred_classes, test_accuracy, test_loss = evaluate_model(
        model, X_test, y_test, y_test_flat
    )

    # 10. Confusion matrix
    plot_confusion_matrix(y_test_flat, y_pred_classes, test_accuracy)

    # 11. Sample predictions
    plot_sample_predictions(X_test, y_test_flat, y_pred, y_pred_classes)

    # 12. Per-class accuracy
    class_accuracies = plot_per_class_accuracy(y_test_flat, y_pred_classes)

    # 13. Confidence analysis
    plot_confidence_analysis(y_test_flat, y_pred, y_pred_classes)

    # 14. Summary
    print_summary(model, test_accuracy, test_loss, class_accuracies, history)

    print("\nAll plots saved in /plots folder")
    print("Done! ✅")




if __name__ == "__main__":
    main()