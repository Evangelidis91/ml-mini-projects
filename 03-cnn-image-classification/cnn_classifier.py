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
import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import warnings

from helper_methods import section, save_plot, plot_image_grid

warnings.filterwarnings("ignore")

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
    section("LOADING DATA")

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

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
    section("SAMPLE IMAGES")

    # One image per class
    images = [X_train[np.where(y_train_flat == i)[0][0]] for i in range(10)]
    titles = CLASS_NAMES
    plot_image_grid(
        images,
        titles,
        "01_sample_per_class.png",
        suptitle="One Sample Per Class — CIFAR-10",
    )

    # Random samples (3×6)
    indices = np.random.randint(0, len(X_train), size=18)
    images = [X_train[i] for i in indices]
    titles = [CLASS_NAMES[y_train_flat[i]] for i in indices]
    plot_image_grid(
        images,
        titles,
        "02_random_samples.png",
        suptitle="Random Training Samples",
        rows=3,
        cols=6,
        figsize=(18, 9),
    )


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================
def plot_class_distribution(y_train_flat, y_test_flat):
    section("CLASS DISTRIBUTION")

    sets = [
        ("Training Set Distribution", y_train_flat, "steelblue"),
        ("Test Set Distribution", y_test_flat, "coral"),
    ]

    with save_plot("03_class_distribution.png"):
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, (title, y, color) in zip(axes, sets):
            unique, counts = np.unique(y, return_counts=True)

            ax.bar(
                [CLASS_NAMES[i] for i in unique],
                counts,
                color=color,
                alpha=0.8,
            )
            ax.set_title(title)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)

    print("\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        train_count = np.sum(y_train_flat == i)
        test_count = np.sum(y_test_flat == i)
        print(f"  {name:12s}: Train={train_count:,}, Test={test_count:,}")


# ============================================================
# 4. PIXEL ANALYSIS
# ============================================================
def plot_pixel_analysis(X_train, y_train_flat):
    """Analyze average pixel values per class."""
    section("PIXEL ANALYSIS")

    avg_images = [X_train[y_train_flat == i].mean(axis=0) for i in range(10)]
    titles = [f"Avg: {name}" for name in CLASS_NAMES]
    plot_image_grid(avg_images, titles, "04_average_images.png",
                    suptitle="Average Image Per Class")

    print("\nMean pixel values per channel:")
    for i, ch in enumerate(["Red", "Green", "Blue"]):
        print(f"  {ch:6s}: {X_train[:, :, :, i].mean():.4f}")

# ============================================================
# 5. DATA AUGMENTATION
# ============================================================
def create_data_augmentation(X_train):
    """Create and visualize data augmentation."""
    section("DATA AUGMENTATION")

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(X_train)

    sample_img = X_train[0:1]

    with save_plot("05_augmentation_examples.png"):
        _, axes = plt.subplots(2, 5, figsize=(15, 6))

        axes[0][0].imshow(sample_img[0])
        axes[0][0].set_title("Original", fontweight="bold")
        axes[0][0].axis("off")

        for i, ax in enumerate(axes.flatten()[1:]):
            augmented = datagen.flow(sample_img, batch_size=1)
            ax.imshow(next(augmented)[0])
            ax.set_title(f"Augmented {i + 1}")
            ax.axis("off")

        plt.suptitle("Data Augmentation Examples", fontsize=14)

    print("\nAugmentation configuration:")
    print("  - Rotation: ±15°")
    print("  - Width shift: ±10%")
    print("  - Height shift: ±10%")
    print("  - Horizontal flip: Yes")

    return datagen


# ============================================================
# 6. BUILD MODEL
# ============================================================
def conv_block(filters, dropout=0.25):
    """Return a list of layers for a Conv-BN-Conv-BN-Pool-Dropout block."""
    return [
        layers.Conv2D(filters, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(filters, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout),
    ]


def build_model():
    section("MODEL ARCHITECTURE")

    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        *conv_block(32),
        *conv_block(64),
        *conv_block(128),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


# ============================================================
# 7. TRAIN MODEL
# ============================================================
def train_model(model, datagen, X_train, y_train, X_test, y_test):
    """Train the model with data augmentation and callbacks."""
    section("TRAINING")

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

    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print("  Early stopping patience: 10")
    print("  LR reduction patience: 5")
    print("  Optimizer: Adam")
    print("  Loss: Sparse Categorical Crossentropy")
    print("\nTraining with data augmentation...")
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
    section("TRAINING HISTORY")

    epochs_trained = len(history.history["accuracy"])

    with save_plot("06_training_history.png"):
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, metric, title in [(axes[0], "accuracy", "Model Accuracy"),
                                  (axes[1], "loss", "Model Loss")]:
            ax.plot(history.history[metric], label="Train", linewidth=2)
            ax.plot(history.history[f"val_{metric}"], label="Validation", linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Print stats
    best_epoch = np.argmax(history.history["val_accuracy"])
    print(f"\nEpochs trained: {epochs_trained}")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"  Train accuracy:      {history.history['accuracy'][best_epoch]:.4f}")
    print(f"  Validation accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
    print(f"  Train loss:          {history.history['loss'][best_epoch]:.4f}")
    print(f"  Validation loss:     {history.history['val_loss'][best_epoch]:.4f}")

    # Overfitting check
    gap = history.history["accuracy"][-1] - history.history["val_accuracy"][-1]
    print(f"\nOverfitting check:")
    print(f"  Train-Val accuracy gap: {gap:.4f}")
    if gap > 0.1:
        print("  ⚠️  Significant overfitting detected")
    elif gap > 0.05:
        print("  ⚠️  Mild overfitting")
    else:
        print("  ✅ Good generalization")


# ============================================================
# 9. EVALUATE MODEL
# ============================================================
def evaluate_model(model, X_test, y_test, y_test_flat):
    """Evaluate model on test set."""
    section("EVALUATION")

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
    section("CONFUSION MATRIX")

    cm = confusion_matrix(y_test_flat, y_pred_classes)

    with save_plot("07_confusion_matrix.png"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title(f"Confusion Matrix (Accuracy: {test_accuracy:.2%})")

    # Find most confused pairs
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    pairs = [(CLASS_NAMES[i], CLASS_NAMES[j], cm_off[i, j])
             for i, j in zip(*np.where(cm_off > 0))]
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nMost confused pairs:")
    for actual, predicted, count in pairs[:10]:
        print(f"  {actual:12s} → {predicted:12s}: {count} times")


# ============================================================
# 11. SAMPLE PREDICTIONS
# ============================================================
def plot_sample_predictions(X_test, y_test_flat, y_pred, y_pred_classes):
    section("SAMPLE PREDICTIONS")

    correct_mask = y_pred_classes == y_test_flat

    cases = [
        (
            "Correct Predictions ✅",
            "08_correct_predictions.png",
            "green",
            np.where(correct_mask)[0],
            lambda idx: f"{CLASS_NAMES[y_test_flat[idx]]} ({np.max(y_pred[idx]) * 100:.0f}%)",
        ),
        (
            "Wrong Predictions ❌",
            "09_wrong_predictions.png",
            "red",
            np.where(~correct_mask)[0],
            lambda idx: (
                f"True: {CLASS_NAMES[y_test_flat[idx]]}\n"
                f"Pred: {CLASS_NAMES[y_pred_classes[idx]]} "
                f"({np.max(y_pred[idx]) * 100:.0f}%)"
            ),
        ),
    ]

    for suptitle, filename, color, indices, title_fn in cases:
        sample = np.random.choice(indices, size=10, replace=False)
        images = [X_test[i] for i in sample]
        titles = [title_fn(i) for i in sample]

        plot_image_grid(
            images,
            titles,
            filename,
            suptitle=suptitle,
            suptitle_color=color,
            title_color=color,
        )

    total = len(y_test_flat)
    correct = correct_mask.sum()

    print(
        f"\nTotal: {total}  Correct: {correct} ({correct / total:.2%})  "
        f"Wrong: {total - correct} ({(total - correct) / total:.2%})"
    )


# ============================================================
# 12. PER-CLASS ACCURACY
# ============================================================
def plot_per_class_accuracy(y_test_flat, y_pred_classes):
    """Plot accuracy for each class."""
    section("PER-CLASS ACCURACY")

    class_accuracies = [
        np.mean(y_pred_classes[y_test_flat == i] == i) for i in range(10)
    ]

    # Sort by accuracy
    sorted_indices = np.argsort(class_accuracies)
    sorted_names = [CLASS_NAMES[i] for i in sorted_indices]
    sorted_accs = [class_accuracies[i] for i in sorted_indices]

    print("\nAccuracy per class (sorted):")
    for name, acc in zip(sorted_names, sorted_accs):
        bar = "█" * int(acc * 30)
        print(f"  {name:12s}: {acc:.2%} {bar}")

    colors = ["green" if a > 0.85 else "orange" if a > 0.75 else "red"
              for a in sorted_accs]

    with save_plot("10_per_class_accuracy.png"):
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_names, sorted_accs, color=colors, alpha=0.8)
        plt.xlabel("Accuracy")
        plt.title("Per-Class Accuracy (Green > 85%, Orange > 75%, Red ≤ 75%)")
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3, axis="x")

        for i, acc in enumerate(sorted_accs):
            plt.text(acc + 0.01, i, f"{acc:.1%}", va="center", fontsize=10)

    print(f"\nBest class:  {CLASS_NAMES[np.argmax(class_accuracies)]} ({max(class_accuracies):.2%})")
    print(f"Worst class: {CLASS_NAMES[np.argmin(class_accuracies)]} ({min(class_accuracies):.2%})")

    return class_accuracies


# ============================================================
# 13. CONFIDENCE ANALYSIS
# ============================================================
def plot_confidence_analysis(y_test_flat, y_pred, y_pred_classes):
    """Analyze prediction confidence for correct vs wrong predictions."""
    section("CONFIDENCE ANALYSIS")

    max_confidences = np.max(y_pred, axis=1) * 100

    correct_mask = y_pred_classes == y_test_flat
    correct_conf = max_confidences[correct_mask]
    wrong_conf = max_confidences[~correct_mask]

    print(f"Correct predictions — Avg confidence: {correct_conf.mean():.1f}%")
    print(f"Wrong predictions   — Avg confidence: {wrong_conf.mean():.1f}%")

    buckets = [(0, 50), (50, 70), (70, 80), (80, 90), (90, 95), (95, 100)]
    bucket_data = []
    for low, high in buckets:
        mask = (max_confidences >= low) & (max_confidences < high)
        if mask.sum() > 0:
            acc = correct_mask[mask].mean()
            bucket_data.append((f"{low}-{high}%", acc, int(mask.sum())))
            print(f"  Confidence {low:3d}-{high:3d}%: Accuracy={acc:.2%} (n={mask.sum()})")

    with save_plot("11_confidence_analysis.png"):
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(correct_conf, bins=30, alpha=0.7, color="green", label="Correct", density=True)
        axes[0].hist(wrong_conf, bins=30, alpha=0.7, color="red", label="Wrong", density=True)
        axes[0].set(xlabel="Confidence (%)", ylabel="Density",
                    title="Confidence Distribution: Correct vs Wrong")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar([b[0] for b in bucket_data], [b[1] for b in bucket_data],
                    color="steelblue", alpha=0.8)
        axes[1].set(xlabel="Confidence Bucket", ylabel="Accuracy",
                    title="Accuracy by Confidence Level", ylim=(0, 1))
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].tick_params(axis="x", rotation=45)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================
def print_summary(model, test_accuracy, test_loss, class_accuracies, history):
    """Print final summary."""
    section("FINAL SUMMARY")

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
