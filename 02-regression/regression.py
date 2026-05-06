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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings("ignore")

# Create output folder for plots
os.makedirs("plots", exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data():
    """Load the California Housing dataset."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

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

    return df, housing


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
def explore_data(df):
    """Perform exploratory data analysis and save plots."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nDataset Statistics:")
    print(df.describe().to_string())

    # Check for missing values
    missing = df.isnull().sum()
    print(f"\nMissing values: {missing.sum()}")

    # Plot: Feature distributions
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for i, col in enumerate(df.columns[:-1]):
        axes[i].hist(df[col], bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        axes[i].set_title(col, fontsize=12)
        axes[i].set_ylabel("Count")

    axes[-1].hist(df["MedHouseVal"], bins=50, color="coral", alpha=0.7, edgecolor="white")
    axes[-1].set_title("MedHouseVal (Target)", fontsize=12)
    axes[-1].set_ylabel("Count")

    plt.suptitle("Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("plots/01_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/01_feature_distributions.png")


# ============================================================
# 3. CORRELATION ANALYSIS
# ============================================================
def plot_correlation(df):
    """Plot correlation matrix."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    correlation = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.5,
    )

    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("plots/02_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/02_correlation_matrix.png")

    print("\nCorrelations with target (MedHouseVal):")
    target_corr = correlation["MedHouseVal"].sort_values(ascending=False)[1:]
    for feature, corr in target_corr.items():
        print(f"  {feature:15s}: {corr:+.4f}")

    return correlation


# ============================================================
# 4. GEOGRAPHIC VISUALIZATION
# ============================================================
def plot_geographic(df):
    """Plot house prices on a geographic map of California."""
    print("\n" + "=" * 60)
    print("GEOGRAPHIC VISUALIZATION")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Price map
    scatter1 = axes[0].scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["MedHouseVal"],
        cmap="viridis",
        alpha=0.3,
        s=5,
    )
    plt.colorbar(scatter1, ax=axes[0], label="Median House Value ($100k)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("House Prices by Location")

    # Population density map
    scatter2 = axes[1].scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["Population"],
        cmap="magma",
        alpha=0.3,
        s=5,
    )
    plt.colorbar(scatter2, ax=axes[1], label="Population")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title("Population Density by Location")

    plt.tight_layout()
    plt.savefig("plots/03_geographic_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/03_geographic_maps.png")


# ============================================================
# 5. FEATURE RELATIONSHIPS
# ============================================================
def plot_feature_relationships(df):
    """Plot key feature relationships with target."""
    print("\n" + "=" * 60)
    print("FEATURE RELATIONSHIPS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    features = ["MedInc", "AveRooms", "AveBedrms", "HouseAge", "AveOccup", "Population"]

    for i, feature in enumerate(features):
        row = i // 3
        col = i % 3
        axes[row][col].scatter(
            df[feature], df["MedHouseVal"], alpha=0.1, s=5, color="steelblue"
        )
        axes[row][col].set_xlabel(feature)
        axes[row][col].set_ylabel("MedHouseVal")
        axes[row][col].set_title(f"{feature} vs House Value")
        axes[row][col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/04_feature_relationships.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/04_feature_relationships.png")


# ============================================================
# 6. PREPROCESSING
# ============================================================
def preprocess_data(df):
    """Split and scale the data."""
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

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

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler


# ============================================================
# 7. MODEL TRAINING & COMPARISON
# ============================================================
def train_models(X_train, X_test, y_train, y_test):
    """Train multiple regression models and compare."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training: {name} ---")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
        )

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        print(f"  RMSE:     {rmse:.4f}")
        print(f"  MAE:      {mae:.4f}")
        print(f"  R²:       {r2:.4f}")
        print(f"  CV R²:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results


# ============================================================
# 8. MODEL COMPARISON CHART
# ============================================================
def plot_model_comparison(results):
    """Plot model comparison charts."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

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

    # Bar charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = summary["Model"].tolist()

    axes[0].barh(models, summary["RMSE"].tolist(), color="coral", alpha=0.8)
    axes[0].set_title("RMSE (lower is better)")
    axes[0].set_xlabel("RMSE")
    axes[0].grid(True, alpha=0.3, axis="x")

    axes[1].barh(models, summary["MAE"].tolist(), color="steelblue", alpha=0.8)
    axes[1].set_title("MAE (lower is better)")
    axes[1].set_xlabel("MAE")
    axes[1].grid(True, alpha=0.3, axis="x")

    axes[2].barh(models, summary["R²"].tolist(), color="green", alpha=0.8)
    axes[2].set_title("R² Score (higher is better)")
    axes[2].set_xlabel("R²")
    axes[2].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("plots/05_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/05_model_comparison.png")

    return summary


# ============================================================
# 9. ACTUAL VS PREDICTED
# ============================================================
def plot_actual_vs_predicted(results, y_test):
    """Plot actual vs predicted for the best model."""
    print("\n" + "=" * 60)
    print("ACTUAL VS PREDICTED")
    print("ACTUAL VS PREDICTED")
    print("=" * 60)

    best_name = max(results, key=lambda x: results[x]["r2"])
    best_pred = results[best_name]["y_pred"]

    print(f"Best model: {best_name}")
    print(f"  R²:   {results[best_name]['r2']:.4f}")
    print(f"  RMSE: {results[best_name]['rmse']:.4f}")
    print(f"  MAE:  {results[best_name]['mae']:.4f}")

    # Actual vs Predicted scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, best_pred, alpha=0.3, s=10, color="steelblue")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        linewidth=2,
        label="Perfect prediction",
    )
    plt.xlabel("Actual Price ($100k)")
    plt.ylabel("Predicted Price ($100k)")
    plt.title(f"Actual vs Predicted — {best_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/06_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/06_actual_vs_predicted.png")

    return best_name

# ============================================================
# 10. RESIDUAL ANALYSIS
# ============================================================
def plot_residuals(results, best_name, y_test):
    """Plot residual analysis for the best model."""
    print("\n" + "=" * 60)
    print("RESIDUAL ANALYSIS")
    print("=" * 60)

    best_pred = results[best_name]["y_pred"]
    residuals = y_test.values - best_pred

    print(f"Residual statistics:")
    print(f"  Mean:   {residuals.mean():.4f}")
    print(f"  Median: {np.median(residuals):.4f}")
    print(f"  Std:    {residuals.std():.4f}")
    print(f"  Min:    {residuals.min():.4f}")
    print(f"  Max:    {residuals.max():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Residual scatter plot
    axes[0].scatter(best_pred, residuals, alpha=0.3, s=10, color="steelblue")
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Predicted Price ($100k)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    # Residual distribution
    axes[1].hist(residuals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    # QQ-like plot (sorted residuals)
    sorted_residuals = np.sort(residuals)
    theoretical = np.random.normal(residuals.mean(), residuals.std(), len(residuals))
    theoretical.sort()
    axes[2].scatter(theoretical, sorted_residuals, alpha=0.3, s=10, color="steelblue")
    axes[2].plot(
        [theoretical.min(), theoretical.max()],
        [theoretical.min(), theoretical.max()],
        "r--",
        linewidth=2,
    )
    axes[2].set_xlabel("Theoretical Quantiles")
    axes[2].set_ylabel("Sample Quantiles")
    axes[2].set_title("Q-Q Plot")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/07_residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/07_residual_analysis.png")

# ============================================================
# 11. FEATURE IMPORTANCE
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
        importance = np.abs(best_model.coef_)

    feat_imp = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values("Importance", ascending=False)

    print(f"\nFeature importance ({best_name}):")
    for _, row in feat_imp.iterrows():
        print(f"  {row['Feature']:15s}: {row['Importance']:.4f}")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig("plots/08_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/08_feature_importance.png")


# ============================================================
# 12. HYPERPARAMETER TUNING
# ============================================================
def tune_best_model(results, best_name, X_train, X_test, y_train, y_test):
    """Hyperparameter tuning for the best model."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Tuning: {best_name}")

    if best_name == "Gradient Boosting":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
        }
        base_model = GradientBoostingRegressor(random_state=42)

    elif best_name == "Random Forest":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    else:
        param_grid = {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        }
        base_model = Ridge()

    print(f"Parameter grid: {param_grid}")

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

    # Final evaluation
    y_pred_tuned = grid_search.predict(X_test)
    tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    tuned_mae = mean_absolute_error(y_test, y_pred_tuned)
    tuned_r2 = r2_score(y_test, y_pred_tuned)

    print(f"\nAfter tuning:")
    print(f"  RMSE: {tuned_rmse:.4f}")
    print(f"  MAE:  {tuned_mae:.4f}")
    print(f"  R²:   {tuned_r2:.4f}")

    # Compare before and after
    original_r2 = results[best_name]["r2"]
    improvement = tuned_r2 - original_r2
    print(f"\nImprovement: R² {original_r2:.4f} → {tuned_r2:.4f} ({improvement:+.4f})")

    return grid_search, tuned_rmse, tuned_mae, tuned_r2

# ============================================================
# 13. PREDICTION ERROR ANALYSIS
# ============================================================
def analyze_prediction_errors(results, best_name, y_test):
    """Analyze where the model makes the biggest errors."""
    print("\n" + "=" * 60)
    print("PREDICTION ERROR ANALYSIS")
    print("=" * 60)

    best_pred = results[best_name]["y_pred"]
    errors = np.abs(y_test.values - best_pred)

    # Error by price range
    price_ranges = [
        ("Budget (< $1.5)", y_test < 1.5),
        ("Mid ($1.5 - $3)", (y_test >= 1.5) & (y_test < 3.0)),
        ("High ($3 - $4)", (y_test >= 3.0) & (y_test < 4.0)),
        ("Premium (>= $4)", y_test >= 4.0),
    ]

    print("\nMean Absolute Error by price range ($100k):")
    range_names = []
    range_maes = []

    for range_name, mask in price_ranges:
        if mask.sum() > 0:
            range_mae = errors[mask].mean()
            range_count = mask.sum()
            range_names.append(range_name)
            range_maes.append(range_mae)
            print(f"  {range_name:20s}: MAE={range_mae:.4f} (n={range_count})")

    plt.figure(figsize=(10, 5))
    plt.bar(range_names, range_maes, color="coral", alpha=0.8)
    plt.ylabel("Mean Absolute Error")
    plt.title("Prediction Error by Price Range")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("plots/09_error_by_range.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/09_error_by_range.png")

# ============================================================
# 14. FINAL SUMMARY
# ============================================================
def print_summary(results, best_name, tuned_rmse, tuned_mae, tuned_r2):
    """Print final results summary."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

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
    """Run the complete regression pipeline."""

    # 1. Load data
    df, housing = load_data()

    # 2. Explore data
    explore_data(df)

    # 3. Correlation analysis
    plot_correlation(df)

    # 4. Geographic visualization
    plot_geographic(df)

    # 5. Feature relationships
    plot_feature_relationships(df)

    # 6. Preprocess
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)

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
    _, tuned_rmse, tuned_mae, tuned_r2 = tune_best_model(
        results, best_name, X_train, X_test, y_train, y_test
    )

    # 13. Prediction error analysis
    analyze_prediction_errors(results, best_name, y_test)

    # 14. Summary
    print_summary(results, best_name, tuned_rmse, tuned_mae, tuned_r2)

    print("\nAll plots saved in /plots folder")
    print("Done! ✅")

if __name__ == "__main__":
    main()