"""
Customer Churn Prediction
=========================
Binary classification model to predict whether a telecom customer
will churn based on usage patterns and account information

Dataset: Telco Customer Churn (IBM)
Models: Logistic Regression, Random Forest, Gradient Boosting, GB (Weighted)
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from contextlib import contextmanager

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

TUNING_CONFIG = {
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        },
    ),
}
TUNING_CONFIG["GB (Weighted)"] = TUNING_CONFIG["Gradient Boosting"]

# =================================================
# 1. LOAD DATA
# =================================================
def load_data():
    """ Load the Telco Customer Churn dataset. """
    section("LOADING DATA")

    url = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
    df = pd.read_csv(url)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
    return df

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
def explore_data(df):
    """Perform exploratory data analysis and save plots."""
    section("EXPLORATORY DATA ANALYSIS")

    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['Churn'].value_counts(normalize=True)}")
    print(f"\nMissing values:\n{df.isnull().sum() > 0}")
    print(f"\nData types:\n{df.dtypes.value_counts()}")

    with save_plot("01_eda_overview.png"):
        # Plot 1: Churn distribution and key features
        _, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Churn distribution
        df["Churn"].value_counts().plot(
           kind="bar", ax=axes[0], color=["steelblue", "coral"]
        )

        axes[0].set_title("Churn Distribution")
        axes[0].set_ylabel("Count")
        axes[0].set_xticklabels(["No", "Yes"], rotation=0)

        # Churn by contract type
        pd.crosstab(df["Contract"], df["Churn"]).plot(kind="bar", ax=axes[1])
        axes[1].set_title("Churn by Contract Type")
        axes[1].tick_params(axis="x", rotation=45)

        # Monthly charges by churn
        df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[2])
        axes[2].set_title("Monthly Charges by Churn")
        plt.suptitle("")

# ============================================================
# 3. PREPROCESSING
# ============================================================
def preprocess_data(df):
    """Clean and preprocess the dataset."""
    section("PREPROCESSING")

    # Drop customer ID
    df = df.drop("customerID", axis=1)

    # Fix TotalCharges (has some spaces instead of numbers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    print(f"Final dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Churn'].value_counts()}")
    print(f"Encoded {len(categorical_cols)} categorical columns")

    return df

# ============================================================
# 4. CORRELATION ANALYSIS
# ============================================================
def plot_correlation(df):
    """Plot correlation matrix and identify top features."""
    section("CORRELATION ANALYSIS")

    correlation = df.corr()
    with save_plot("02_correlation_matrix.png"):
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=False, cmap="coolwarm", center=0, square=True)
        plt.title("Feature Correlation Matrix")

    # Top correlations with churn
    churn_corr = correlation["Churn"].sort_values(ascending=False)[1:11]
    print(f"\nTop features correlated with Churn:\n{churn_corr}")

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================
def split_and_scale(df):
    """Split data into train/test and scale features."""
    section("TRAIN-TEST SPLIT")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

# ============================================================
# 6. MODEL TRAINING & COMPARISON
# ============================================================
def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance."""
    section("MODEL TRAINING")

    sample_weights = compute_sample_weight("balanced", y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "GB (Weighted)": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"\n--- Training: {name} ---")
        fit_kwargs = {"sample_weight": sample_weights} if name == "GB (Weighted)" else {}
        model.fit(X_train, y_train, **fit_kwargs)
        results[name] = evaluate(model, X_test, y_test, name)

    return results

# ============================================================
# 7. ROC CURVES
# ============================================================
def plot_roc_curves(results, y_test):
    section("ROC CURVES")

    with save_plot("03_roc_curves.png"):
        plt.figure(figsize=(8, 6))

        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result["y_prob"])
            plt.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves — Model Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(results, y_test):
    """Plot confusion matrix for the best model."""
    section("CONFUSION MATRIX")
    """
        ROC-AUC:  "Which model SEPARATES classes best overall?"
        F1:       "Which model BALANCES precision and recall?"
        Recall:   "Which model CATCHES the most churners?
    """
    # best_name = max(results, key=lambda x: results[x]["roc_auc"])
    best_name = max(results, key=lambda x: results[x]["f1"])
    # best_name = max(results, key=lambda x: results[x]["recall"])

    print(f"Best model: {best_name}")

    with save_plot("04_confusion_matrix.png"):
        plt.figure(figsize=(6, 5))
        draw_confusion_matrix(
            y_test,
            results[best_name]["y_pred"],
            title=f"Confusion Matrix — {best_name}",
        )

    return best_name

# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================
def plot_feature_importance(results, best_name, feature_names):
    section("FEATURE IMPORTANCE")

    best_model = results[best_name]["model"]
    importance = (
        best_model.feature_importances_
        if hasattr(best_model, "feature_importances_")
        else np.abs(best_model.coef_[0])
    )

    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    print("\nTop 10 features:")
    print(feature_importance.head(10).to_string(index=False))

    with save_plot("05_feature_importance.png"):
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importance.head(10),
            x="importance",
            y="feature",
            hue="feature",
            palette="viridis",
            legend=False,  # silences seaborn warning
        )
        plt.title(f"Top 10 Features — {best_name}")


# ============================================================
# 10. HYPERPARAMETER TUNING
# ============================================================
def tune_best_model(best_name, X_train, X_test, y_train, y_test):
    section(f"HYPERPARAMETER TUNING — {best_name}")

    # GB (Weighted) shares the same hyperparameter space as Gradient Boosting
    config_key = "Gradient Boosting" if best_name == "GB (Weighted)" else best_name
    base_model, param_grid = TUNING_CONFIG[config_key]

    grid = GridSearchCV(
        base_model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

    metrics = evaluate(grid.best_estimator_, X_test, y_test, label="tuned")

    return metrics["accuracy"], metrics["roc_auc"]

# ============================================================
# 11. FINAL SUMMARY
# ============================================================
def print_summary(results, best_name, tuned_accuracy, tuned_roc_auc):
    """Print final results summary."""
    section("FINAL RESULTS SUMMARY")

    summary = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "Accuracy": [r["accuracy"] for r in results.values()],
            "ROC-AUC": [r["roc_auc"] for r in results.values()],
        }
    ).sort_values("ROC-AUC", ascending=False)

    print(f"\n{summary.to_string(index=False)}")
    print(f"\nBest model: {best_name}")
    print(f"After tuning — Accuracy: {tuned_accuracy:.4f} | ROC-AUC: {tuned_roc_auc:.4f}")
    print("\n" + "=" * 60)

def evaluate(model, X_test, y_test, label=""):
    """Return a dict of standard metrics + predictions."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    if label:
        print(f"[{label}] Acc={accuracy_score(y_test, y_pred):.4f} "
            f"AUC={roc_auc_score(y_test, y_prob):.4f} "
            f"Recall={report['1']['recall']:.4f}")
    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

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
        print("Saved: plots/" + file_name)
    finally:
        plt.close()

def draw_confusion_matrix(y_true, y_pred, title, ax=None):
    """Draw confusion matrix."""
    if ax is None:
        ax = plt.gca()

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# ============================================================
# MAIN
# ============================================================
def main():
    # 1. Load data
    df = load_data()

    # 2. Explore data
    explore_data(df)

    # 3. Preprocess
    df_processed = preprocess_data(df)

    # 4. Correlation analysis
    plot_correlation(df_processed)

    # 5. Split and scale
    X_train, X_test, y_train, y_test, feature_names = split_and_scale(df_processed)

    # 6. Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # 7. Plot ROC curves
    plot_roc_curves(results, y_test)

    # 8. Plot confusion matrix
    best_name = plot_confusion_matrix(results, y_test)

    # 9. Feature importance
    plot_feature_importance(results, best_name, feature_names)

    # 10. Hyperparameter tuning
    tuned_accuracy, tuned_roc_auc = tune_best_model(
        best_name, X_train, X_test, y_train, y_test
    )

    # 11. Summary (update number)
    print_summary(results, best_name, tuned_accuracy, tuned_roc_auc)

    print("\nAll plots saved in /plots folder")
    print("Done! ✅")


if __name__ == "__main__":
    main()
