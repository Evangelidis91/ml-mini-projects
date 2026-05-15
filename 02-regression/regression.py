"""
House Price Prediction
=====================
Regression models to predict house prices based on property
features using the California Housing dataset.

Dataset: California Housing (scikit-learn built-in)
Models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy import stats
import warnings

from helper_methods import section, save_plot, evaluate

warnings.filterwarnings("ignore")

TUNING_CONFIG = {
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        },
    ),
    "Random Forest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10]
        },
    ),
    "Ridge Regression": (
        Ridge(),
        {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    ),
    "Lasso Regression": (
        Lasso(max_iter=10000),
        {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    ),
    "Linear Regression": (
        Ridge(),
        {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    ),
}

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data():
    """Load the California Housing dataset."""
    section("LOADING DATA")

    housing = fetch_california_housing(as_frame=True)

    df = housing.frame

    print(f"Dataset shape: {df.shape}")
    print(f"\nFeatures:")
    for name in housing.feature_names:
        print(f"  - {name}")
    print(f"  - MedHouseVal (target)")
    print(f"\nTarget range: {df['MedHouseVal'].min():.2f} - {df['MedHouseVal'].max():.2f} ($100k)")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())

    return df


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
def explore_data(df):
    """Perform exploratory data analysis and save plots."""
    section("EXPLORATORY DATA ANALYSIS")

    print(f"\nDataset Statistics:")
    print(df.describe().to_string())

    # Check for missing values
    print(f"\nMissing values: {df.isnull().sum().sum()}")

    # Plot: Feature distributions
    with save_plot("01_feature_distributions.png"):
        _, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()

        for i, col in enumerate(df.columns[:-1]):
            axes[i].hist(df[col], bins=50, color="steelblue", alpha=0.7, edgecolor="white")
            axes[i].set_title(col, fontsize=12)
            axes[i].set_ylabel("Count")

        axes[-1].hist(df["MedHouseVal"], bins=50, color="coral", alpha=0.7, edgecolor="white")
        axes[-1].set_title("MedHouseVal (Target)", fontsize=12)
        axes[-1].set_ylabel("Count")

        plt.suptitle("Feature Distributions", fontsize=14, y=1.01)


# ============================================================
# 3. CORRELATION ANALYSIS
# ============================================================
def plot_correlation(df):
    """Plot correlation matrix."""
    section("CORRELATION ANALYSIS")

    correlation = df.corr()

    with save_plot("02_correlation_matrix.png"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0,
                    fmt=".2f", square=True, linewidths=0.5)
        plt.title("Feature Correlation Matrix")

    print("\nCorrelations with target (MedHouseVal):")
    target_corr = correlation["MedHouseVal"].sort_values(ascending=False)[1:]
    for feature, corr in target_corr.items():
        print(f"  {feature:15s}: {corr:+.4f}")


# ============================================================
# 4. GEOGRAPHIC VISUALIZATION
# ============================================================
def plot_geographic(df):
    """Plot house prices on a geographic map of California."""
    section("GEOGRAPHIC VISUALIZATION")

    with save_plot("03_geographic_maps.png"):
        _, axes = plt.subplots(1, 2, figsize=(20, 8))

        s1 = axes[0].scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"],
                             cmap="viridis", alpha=0.3, s=5)
        plt.colorbar(s1, ax=axes[0], label="Median House Value ($100k)")
        axes[0].set(xlabel="Longitude", ylabel="Latitude", title="House Prices by Location")

        s2 = axes[1].scatter(df["Longitude"], df["Latitude"], c=df["Population"],
                             cmap="magma", alpha=0.3, s=5)
        plt.colorbar(s2, ax=axes[1], label="Population")
        axes[1].set(xlabel="Longitude", ylabel="Latitude", title="Population Density by Location")


# ============================================================
# 5. FEATURE RELATIONSHIPS
# ============================================================
def plot_feature_relationships(df):
    """Plot key feature relationships with target."""
    section("FEATURE RELATIONSHIPS")

    features = ["MedInc", "AveRooms", "AveBedrms", "HouseAge", "AveOccup", "Population"]

    with save_plot("04_feature_relationships.png"):
        _, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, feature in zip(axes.flatten(), features):
            ax.scatter(df[feature], df["MedHouseVal"], alpha=0.1, s=5, color="steelblue")
            ax.set_xlabel(feature)
            ax.set_ylabel("MedHouseVal")
            ax.set_title(f"{feature} vs House Value")
            ax.grid(True, alpha=0.3)


# ============================================================
# 6. PREPROCESSING
# ============================================================
def preprocess_data(df):
    """Split and scale the data."""
    section("PREPROCESSING")

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    print(f"Features:  {X_train.shape[1]}")
    print(f"\nTarget statistics (train):")
    print(f"  Mean:   {y_train.mean():.4f}")
    print(f"  Median: {y_train.median():.4f}")
    print(f"  Std:    {y_train.std():.4f}")
    print(f"  Range:  {y_train.min():.4f} - {y_train.max():.4f}")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


# ============================================================
# 7. MODEL TRAINING & COMPARISON
# ============================================================
def train_models(X_train, X_test, y_train, y_test):
    """Train multiple regression models and compare."""
    section("MODEL TRAINING")

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, label=name)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()
        print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        results[name] = metrics

    return results


# ============================================================
# 8. MODEL COMPARISON CHART
# ============================================================
def plot_model_comparison(results):
    section("MODEL COMPARISON")

    summary = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "RMSE": [r["rmse"] for r in results.values()],
            "MAE": [r["mae"] for r in results.values()],
            "R²": [r["r2"] for r in results.values()],
            "CV R²": [r["cv_mean"] for r in results.values()],
        }
    ).sort_values("R²", ascending=False)

    print(f"\n{summary.to_string(index=False)}")

    chart_specs = [
        ("RMSE", "RMSE (lower is better)", "coral"),
        ("MAE", "MAE (lower is better)", "steelblue"),
        ("R²", "R² Score (higher is better)", "green"),
    ]

    with save_plot("05_model_comparison.png"):
        _, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (col, title, color) in zip(axes, chart_specs):
            ax.barh(summary["Model"], summary[col], color=color, alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel(col)
            ax.grid(True, alpha=0.3, axis="x")

    return summary


# ============================================================
# 9. ACTUAL VS PREDICTED
# ============================================================
def plot_actual_vs_predicted(results, y_test):
    """Plot actual vs predicted for the best model."""
    section("ACTUAL VS PREDICTED")

    best_name = max(results, key=lambda x: results[x]["r2"])
    best_pred = results[best_name]["y_pred"]

    print(f"Best model: {best_name}")
    print(f"  R²:   {results[best_name]['r2']:.4f}")
    print(f"  RMSE: {results[best_name]['rmse']:.4f}")
    print(f"  MAE:  {results[best_name]['mae']:.4f}")

    with save_plot("06_actual_vs_predicted.png"):
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, best_pred, alpha=0.3, s=10, color="steelblue")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 "r--", linewidth=2, label="Perfect prediction")
        plt.xlabel("Actual Price ($100k)")
        plt.ylabel("Predicted Price ($100k)")
        plt.title(f"Actual vs Predicted — {best_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

    return best_name


# ============================================================
# 10. RESIDUAL ANALYSIS
# ============================================================
def plot_residuals(results, best_name, y_test):
    section("RESIDUAL ANALYSIS")

    best_pred = results[best_name]["y_pred"]
    residuals = y_test.values - best_pred

    print("Residual statistics:")
    print(f"  Mean:    {residuals.mean():.4f}")
    print(f"  Median:  {np.median(residuals):.4f}")
    print(f"  Std:     {residuals.std():.4f}")
    print(f"  Min:     {residuals.min():.4f}")
    print(f"  Max:     {residuals.max():.4f}")

    with save_plot("07_residual_analysis.png"):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Residuals vs predicted
        axes[0].scatter(
            best_pred, residuals, alpha=0.3, s=10, color="steelblue"
        )
        axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[0].set(
            xlabel="Predicted Price ($100k)",
            ylabel="Residual",
            title="Residuals vs Predicted",
        )
        axes[0].grid(True, alpha=0.3)

        # Distribution
        axes[1].hist(
            residuals, bins=50, color="steelblue", alpha=0.7, edgecolor="white"
        )
        axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
        axes[1].set(
            xlabel="Residual", ylabel="Count", title="Residual Distribution"
        )

        # Real Q-Q plot via scipy
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot")
        axes[2].grid(True, alpha=0.3)

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================
def plot_feature_importance(results, best_name, feature_names):
    section("FEATURE IMPORTANCE")

    best_model = results[best_name]["model"]
    importance = (
        best_model.feature_importances_
        if hasattr(best_model, "feature_importances_")
        else np.abs(best_model.coef_)
    )

    feat_imp = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values("Importance", ascending=False)

    print(f"\nFeature importance ({best_name}):")
    for _, row in feat_imp.iterrows():
        print(f"  {row['Feature']:15s}: {row['Importance']:.4f}")

    with save_plot("08_feature_importance.png"):
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feat_imp,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="viridis",
            legend=False,
        )
        plt.title(f"Feature Importance — {best_name}")


# ============================================================
# 12. HYPERPARAMETER TUNING
# ============================================================
def tune_best_model(best_name, X_train, X_test, y_train, y_test):
    """Hyperparameter tuning for the best model."""
    section(f"HYPERPARAMETER TUNING — {best_name}")

    base_model, param_grid = TUNING_CONFIG[best_name]
    grid = GridSearchCV(base_model, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV R²: {grid.best_score_:.4f}")

    metrics = evaluate(grid.best_estimator_, X_test, y_test, label="tuned")
    return metrics["rmse"], metrics["mae"], metrics["r2"]


# ============================================================
# 13. PREDICTION ERROR ANALYSIS
# ============================================================
def analyze_prediction_errors(results, best_name, y_test):
    section("PREDICTION ERROR ANALYSIS")

    best_pred = results[best_name]["y_pred"]
    errors = np.abs(y_test.values - best_pred)

    price_ranges = [
        ("Budget (< $1.5)", y_test < 1.5),
        ("Mid ($1.5 - $3)", (y_test >= 1.5) & (y_test < 3.0)),
        ("High ($3 - $4)", (y_test >= 3.0) & (y_test < 4.0)),
        ("Premium (>= $4)", y_test >= 4.0),
    ]

    range_data = [
        (name, errors[mask].mean(), int(mask.sum()))
        for name, mask in price_ranges
        if mask.sum() > 0
    ]

    print("\nMean Absolute Error by price range ($100k):")
    for name, mae, n in range_data:
        print(f"  {name:20s}: MAE={mae:.4f} (n={n})")

    with save_plot("09_error_by_range.png"):
        plt.figure(figsize=(10, 5))
        plt.bar(
            [r[0] for r in range_data],
            [r[1] for r in range_data],
            color="coral",
            alpha=0.8,
        )
        plt.ylabel("Mean Absolute Error")
        plt.title("Prediction Error by Price Range")
        plt.grid(True, alpha=0.3, axis="y")


# ============================================================
# 14. FINAL SUMMARY
# ============================================================
def print_summary(results, best_name, tuned_rmse, tuned_mae, tuned_r2):
    """Print final results summary."""
    section("FINAL RESULTS SUMMARY")

    summary = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "RMSE": [r["rmse"] for r in results.values()],
            "MAE": [r["mae"] for r in results.values()],
            "R²": [r["r2"] for r in results.values()],
            "CV R² (mean)": [r["cv_mean"] for r in results.values()],
            "CV R² (std)": [r["cv_std"] for r in results.values()],
        }
    ).sort_values("R²", ascending=False)

    print(f"\nAll Models:")
    print(summary.to_string(index=False))

    print(f"\nBest model: {best_name}")
    print(f"  Before tuning — R²: {results[best_name]['r2']:.4f}")
    print(f"  After tuning  — R²: {tuned_r2:.4f}, RMSE: {tuned_rmse:.4f}, MAE: {tuned_mae:.4f}")
    print("\n" + "=" * 60)

# ============================================================
# MAIN
# ============================================================
def main():
    # 1. Load data
    df = load_data()

    # 2. Explore data
    explore_data(df)

    # 3. Correlation analysis
    plot_correlation(df)

    # 4. Geographic visualization
    plot_geographic(df)

    # 5. Feature relationships
    plot_feature_relationships(df)

    # 6. Preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    # 7. Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # 8. Model comparison
    plot_model_comparison(results)

    # 9. Actual vs predicted
    best_name = plot_actual_vs_predicted(results, y_test)

    # 10. Residual analysis
    plot_residuals(results, best_name, y_test)

    # 11. Feature importance
    plot_feature_importance(results, best_name, feature_names)

    # 12. Hyperparameter tuning
    tuned_rmse, tuned_mae, tuned_r2 = tune_best_model(
        best_name, X_train, X_test, y_train, y_test
    )

    # 13. Prediction error analysis
    analyze_prediction_errors(results, best_name, y_test)

    # 14. Summary
    print_summary(results, best_name, tuned_rmse, tuned_mae, tuned_r2)

    print("\nAll plots saved in /plots folder")
    print("Done! ✅")


if __name__ == "__main__":
    main()
