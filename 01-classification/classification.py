"""
Customer Churn Prediction
=========================
Binary classification model to predict whether a telecom customer
will churn based on usage patterns and account information

Dataset: Telco Customer Churn (IBM)
Models: Logistic Regression, Random Forest, Gradient Boosting
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

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

# =================================================
# 1. LOAD DATA
# =================================================
def load_data():
    """ Load the Telco Customer Churn dataset. """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    url = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
    df = pd.read_csv(url)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
def explore_data(df):
    """Perform exploratory data analysis and save plots."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['Churn'].value_counts(normalize=True)}")
    print(f"\nMissing values:\n{df.isnull().sum() > 0}")
    print(f"\nData types:\n{df.dtypes.value_counts()}")

    # Plot 1: hurn distribution and key features
    fig, axes = plt.subplots(1,3,figsize=(18,5))

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
    plt.suptitle("")  # Remove auto title

    plt.tight_layout()
    plt.savefig("plots/01_eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/01_eda_overview.png")



# ============================================================
# 3. PREPROCESSING
# ============================================================

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # Drop customer ID
    df = df.drop("customerID", axis=1)

    # Fix TotalCharges (has some spaces instead of numbers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    print(f"Final dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Churn'].value_counts()}")
    print(f"Encoded {len(categorical_cols)} categorical columns")

    return df, label_encoders


# ============================================================
# 4. CORRELATION ANALYSIS
# ============================================================
def plot_correlation(df):
    """Plot correlation matrix and identify top features."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    plt.figure(figsize=(12, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=False, cmap="coolwarm", center=0, square=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("plots/02_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/02_correlation_matrix.png")

    # Top correlations with churn
    churn_corr = correlation["Churn"].sort_values(ascending=False)[1:11]
    print(f"\nTop features correlated with Churn:\n{churn_corr}")

    return correlation


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================
def split_and_scale(df):
    """Split data into train/test and scale features."""
    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

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

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler


# ============================================================
# 6. MODEL TRAINING & COMPARISON
# ============================================================
def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training: {name} ---")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC:  {roc_auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")

    return results


# ============================================================
# 7. ROC CURVES
# ============================================================
def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models."""
    print("\n" + "=" * 60)
    print("ROC CURVES")
    print("=" * 60)

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
    plt.tight_layout()
    plt.savefig("plots/03_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/03_roc_curves.png")


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(results, y_test):
    """Plot confusion matrix for the best model."""
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    best_name = max(results, key=lambda x: results[x]["roc_auc"])
    best_result = results[best_name]

    print(f"Best model: {best_name}")

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, best_result["y_pred"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig("plots/04_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/04_confusion_matrix.png")

    return best_name


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================
def plot_feature_importance(results, best_name, feature_names):
    """Plot feature importance for the best model."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    best_model = results[best_name]["model"]

    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
    else:
        importance = np.abs(best_model.coef_[0])

    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    print(f"\nTop 10 features:")
    print(feature_importance.head(10).to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance.head(10),
        x="importance",
        y="feature",
        palette="viridis",
    )
    plt.title(f"Top 10 Features — {best_name}")
    plt.tight_layout()
    plt.savefig("plots/05_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/05_feature_importance.png")


# ============================================================
# 10. HYPERPARAMETER TUNING
# ============================================================
def tune_best_model(results, best_name, X_train, X_test, y_train, y_test):
    """Hyperparameter tuning for the best model."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Tuning: {best_name}")

    if best_name == "Gradient Boosting":
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3,5,7],
            "learning_rate": [0.01,0.1,0.2],
        }
        base_model = GradientBoostingClassifier(random_state=42)

    elif best_name == "Random Forest":
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
        base_model = RandomForestClassifier(random_state=42)

    else:
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        }
        base_model = LogisticRegression(max_iter=1000, random_state=42)

    grid_search = GridSearchCV(
        base_model, param_grid=param_grid, cv=5,  scoring="roc_auc", n_jobs=-1, verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    # Final evaluation
    y_pred_tuned = grid_search.predict(X_test)
    y_prob_tuned = grid_search.predict_proba(X_test)[:, 1]

    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    tuned_roc_auc = roc_auc_score(y_test, y_prob_tuned)

    print(f"\nAfter tuning:")
    print(f"Test Accuracy: {tuned_accuracy:.4f}")
    print(f"Test ROC-AUC:  {tuned_roc_auc:.4f}")

    return grid_search, tuned_accuracy, tuned_roc_auc

# ============================================================
# 11. FINAL SUMMARY
# ============================================================
def print_summary(results, best_name, tuned_accuracy, tuned_roc_auc):
    """Print final results summary."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

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


# ============================================================
# MAIN
# ============================================================
def main():
    """Run the complete classification pipeline."""

    # 1. Load data
    df = load_data()

    # 2. Explore data
    explore_data(df)

    # 3. Preprocess
    df_processed, label_encoders = preprocess_data(df)

    # 4. Correlation analysis
    plot_correlation(df_processed)

    # 5. Split and scale
    X_train, X_test, y_train, y_test, feature_names, scaler = split_and_scale(
        df_processed
    )

    # 6. Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # 7. Plot ROC curves
    plot_roc_curves(results, y_test)

    # 8. Plot confusion matrix
    best_name = plot_confusion_matrix(results, y_test)

    # 9. Feature importance
    plot_feature_importance(results, best_name, feature_names)

    # 10. Hyperparameter tuning
    _, tuned_accuracy, tuned_roc_auc = tune_best_model(
        results, best_name, X_train, X_test, y_train, y_test
    )

    # 11. Summary
    print_summary(results, best_name, tuned_accuracy, tuned_roc_auc)

    print("\nAll plots saved in /plots folder")
    print("Done! ✅")



if __name__ == "__main__":
    main()
