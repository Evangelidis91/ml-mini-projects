import os
from contextlib import contextmanager

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt

os.makedirs("plots", exist_ok=True)   # ensures the folder exists for any importer

def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

@contextmanager
def save_plot(file_name):
    try:
        yield
        plt.tight_layout()
        plt.savefig(f"plots/{file_name}", dpi=150, bbox_inches="tight")
        print(f"Saved: plots/{file_name}")
    finally:
        plt.close()


def evaluate(model, X_test, y_test, label=""):
    """Return a dict of regression metrics + predictions."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if label:
        print(f"[{label}] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    return {
        "model": model,
        "y_pred": y_pred,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

def plot_image_grid(
    images,
    titles,
    filename,
    suptitle,
    rows=2,
    cols=5,
    figsize=(15, 6),
    title_color=None,
    suptitle_color=None,
):
    """Plot a grid of images with titles. Saves automatically."""
    with save_plot(filename):
        _, axes = plt.subplots(rows, cols, figsize=figsize)

        for ax, img, title in zip(axes.flatten(), images, titles):
            ax.imshow(img)
            kwargs = {"fontsize": 10}

            if title_color:
                kwargs["color"] = title_color

            ax.set_title(title, **kwargs)
            ax.axis("off")

        plt.suptitle(suptitle, fontsize=14, color=suptitle_color or "black")