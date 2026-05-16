"""
Customer Segmentation with Clustering
=====================================
Unsupervised learning to segment customers based on purchasing
behavior using K-Means and DBSCAN clustering.

Dataset: Mall Customers
Models: K-Means, DBSCAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import warnings

from helper_methods import section, save_plot

K_CLUSTERS = 5

warnings.filterwarnings("ignore")

# ============================================================
# 1. GENERATE DATASET
# ============================================================
def load_data():
    """Load Mall Customers dataset or generate realistic data."""
    section("LOADING DATA")

    # Try to load real dataset first
    try:
        df = pd.read_csv("Mall_Customers.csv")
        print("Loaded Mall_Customers.csv from file")
    except FileNotFoundError:
        print("Generating realistic customer dataset...")
        print("(For real data, download from Kaggle:)")
        print("  https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial")
        print()
        df = generate_dataset()

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nBasic statistics:")
    print(df.describe().to_string())

    return df

def generate_dataset():
    """Generate a realistic customer dataset with natural clusters."""
    np.random.seed(42)

    # Create 5 distinct customer segments
    segments = [
        # High income, high spending — Premium customers
        {"n": 40, "income_mean": 88, "income_std": 12, "spend_mean": 82, "spend_std": 10, "age_mean": 33, "age_std": 7},
        # High income, low spending — Careful/Saving
        {"n": 35, "income_mean": 86, "income_std": 14, "spend_mean": 18, "spend_std": 8, "age_mean": 45, "age_std": 10},
        # Average income, average spending — Standard
        {"n": 55, "income_mean": 55, "income_std": 10, "spend_mean": 50, "spend_std": 10, "age_mean": 40, "age_std": 12},
        # Low income, high spending — At risk
        {"n": 35, "income_mean": 24, "income_std": 7, "spend_mean": 78, "spend_std": 10, "age_mean": 27, "age_std": 5},
        # Low income, low spending — Budget conscious
        {"n": 35, "income_mean": 23, "income_std": 7, "spend_mean": 20, "spend_std": 9, "age_mean": 48, "age_std": 11},
    ]

    data = []
    customer_id = 1

    for params in segments:
        for _ in range(params["n"]):
            income = int(np.clip(
                np.random.normal(params["income_mean"], params["income_std"]), 15, 137
            ))
            spend = int(np.clip(
                np.random.normal(params["spend_mean"], params["spend_std"]), 1, 99
            ))
            age = int(np.clip(
                np.random.normal(params["age_mean"], params["age_std"]), 18, 70
            ))
            gender = np.random.choice(["Male", "Female"])

            data.append({
                "CustomerID": customer_id,
                "Gender": gender,
                "Age": age,
                "Annual Income (k$)": income,
                "Spending Score (1-100)": spend,
            })
            customer_id += 1

    df = pd.DataFrame(data)
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
def explore_data(df):
    """Perform exploratory data analysis."""
    section("EXPLORATORY DATA ANALYSIS")

    # Gender distribution
    if "Gender" in df.columns:
        gender_dist = df["Gender"].value_counts()
        print("\nGender distribution:")
        for gender, count in gender_dist.items():
            print(f"  {gender}: {count} ({count/len(df):.1%})")

    # Age statistics
    print(f"\nAge: mean={df['Age'].mean():.1f}, median={df['Age'].median():.1f}, range={df['Age'].min()}-{df['Age'].max()}")
    print(f"Income: mean={df['Annual Income (k$)'].mean():.1f}k, range={df['Annual Income (k$)'].min()}-{df['Annual Income (k$)'].max()}k")
    print(f"Spending: mean={df['Spending Score (1-100)'].mean():.1f}, range={df['Spending Score (1-100)'].min()}-{df['Spending Score (1-100)'].max()}")

    # ---- Plot 1: Distributions (3 histograms) ----
    hist_specs = [
        ("Age", "steelblue", "Age Distribution"),
        ("Annual Income (k$)", "coral", "Annual Income Distribution"),
        ("Spending Score (1-100)", "green", "Spending Score Distribution"),
    ]

    with save_plot("01_distributions.png"):
        _, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (col, color, title) in zip(axes, hist_specs):
            ax.hist(df[col], bins=20, color=color, alpha=0.7, edgecolor="white")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            mean = df[col].mean()
            ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.0f}")
            ax.legend()

    # ---- Plot 2: Scatter pairs ----
    scatter_specs = [
        ("Annual Income (k$)", "Spending Score (1-100)", "steelblue", "Income vs Spending"),
        ("Age", "Spending Score (1-100)", "coral", "Age vs Spending"),
        ("Age", "Annual Income (k$)", "green", "Age vs Income"),
    ]

    with save_plot("02_scatter_plots.png"):
        _, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (xcol, ycol, color, title) in zip(axes, scatter_specs):
            ax.scatter(df[xcol], df[ycol], alpha=0.6, s=50, color=color)
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    # ---- Plot 3: Gender-based analysis ----
    if "Gender" in df.columns:
        with save_plot("03_gender_analysis.png"):
            _, axes = plt.subplots(1, 2, figsize=(14, 5))

            for gender, color in [("Male", "steelblue"), ("Female", "coral")]:
                subset = df[df["Gender"] == gender]
                axes[0].scatter(
                    subset["Annual Income (k$)"],
                    subset["Spending Score (1-100)"],
                    alpha=0.6, s=50, color=color, label=gender,
                )

            axes[0].set_xlabel("Annual Income (k$)")
            axes[0].set_ylabel("Spending Score (1-100)")
            axes[0].set_title("Income vs Spending by Gender")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            df.boxplot(
                column=["Annual Income (k$)", "Spending Score (1-100)"],
                by="Gender",
                ax=axes[1],
            )
            axes[1].set_title("Income & Spending by Gender")
            plt.suptitle("")

# ============================================================
# 3. FEATURE PREPARATION
# ============================================================
def prepare_features(df):
    """Select and scale features."""
    section("FEATURE PREPARATION")

    # 2D features (Income + Spending) for main analysis
    X_2d = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

    scaler_2d = StandardScaler()
    X_2d_scaled = scaler_2d.fit_transform(X_2d)

    # 3D features (Age + Income + Spending) for additional analysis
    X_3d = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values

    scaler_3d = StandardScaler()
    X_3d_scaled = scaler_3d.fit_transform(X_3d)

    print("2D Features: Annual Income, Spending Score")
    print("3D Features: Age, Annual Income, Spending Score")
    print(f"Samples: {X_2d.shape[0]}")
    print(f"\n2D Scaled — Mean: {X_2d_scaled.mean(axis=0).round(4)}, Std: {X_2d_scaled.std(axis=0).round(4)}")
    print(f"3D Scaled — Mean: {X_3d_scaled.mean(axis=0).round(4)}, Std: {X_3d_scaled.std(axis=0).round(4)}")

    return X_2d, X_2d_scaled, X_3d, X_3d_scaled

# ============================================================
# 4. OPTIMAL K SELECTION
# ============================================================
def find_optimal_k(X_scaled):
    """Find optimal number of clusters."""
    section("OPTIMAL K SELECTION")

    K_range = range(2, 11)
    inertias, silhouette_scores = [], []

    print("\nEvaluating K=2 to K=10:")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"  K={k:2d}: Inertia={kmeans.inertia_:8.2f}, Silhouette={score:.4f}")

    best_k = list(K_range)[np.argmax(silhouette_scores)]

    with save_plot("04_optimal_k.png"):
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(list(K_range), inertias, "bo-", linewidth=2, markersize=8)
        axes[0].set(xlabel="Number of Clusters (K)",
                    ylabel="Inertia (Within-cluster sum of squares)",
                    title="Elbow Method")
        axes[0].axvline(x=K_CLUSTERS, color="red", linestyle="--", alpha=0.5,
                        label=f"K={K_CLUSTERS}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(list(K_range), silhouette_scores, "ro-", linewidth=2, markersize=8)
        axes[1].set(xlabel="Number of Clusters (K)",
                    ylabel="Silhouette Score",
                    title="Silhouette Analysis")
        axes[1].axvline(x=best_k, color="blue", linestyle="--", alpha=0.5,
                        label=f"Best K={best_k}")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    print(f"\nBest K by silhouette score: {best_k} (score={max(silhouette_scores):.4f})")

# ============================================================
# 5. K-MEANS CLUSTERING
# ============================================================
def run_kmeans(X_2d, X_2d_scaled, df, k=K_CLUSTERS):
    """Run K-Means clustering."""
    section(f"K-MEANS CLUSTERING (K={k})")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_2d_scaled)

    df_result = df.copy()
    df_result["Cluster"] = clusters

    score = silhouette_score(X_2d_scaled, clusters)
    print(f"Silhouette Score: {score:.4f}")
    print(f"Inertia: {kmeans.inertia_:.2f}")

    centers_original = X_2d.mean(axis=0) + kmeans.cluster_centers_ * X_2d.std(axis=0)

    with save_plot("05_kmeans_clusters.png"):
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, k))

        for i in range(k):
            mask = clusters == i
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[colors[i]], s=100, alpha=0.6,
                        edgecolors="white", linewidth=0.5,
                        label=f"Cluster {i} (n={mask.sum()})")

        plt.scatter(centers_original[:, 0], centers_original[:, 1],
                    c="red", marker="X", s=300, edgecolors="black",
                    linewidths=2, label="Centroids", zorder=5)

        plt.xlabel("Annual Income (k$)", fontsize=12)
        plt.ylabel("Spending Score (1-100)", fontsize=12)
        plt.title(f"K-Means Clustering (K={k}, Silhouette={score:.3f})", fontsize=14)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

    return df_result, clusters, score

# ============================================================
# 6. SILHOUETTE ANALYSIS (DETAILED)
# ============================================================
def plot_silhouette_detail(X_2d_scaled, clusters, k):
    """Plot detailed silhouette diagram."""
    section("SILHOUETTE ANALYSIS (DETAILED)")

    sample_silhouette_values = silhouette_samples(X_2d_scaled, clusters)
    avg_score = silhouette_score(X_2d_scaled, clusters)

    with save_plot("06_silhouette_detail.png"):
        plt.figure(figsize=(10, 7))
        y_lower = 10
        colors = plt.cm.viridis(np.linspace(0, 1, k))

        for i in range(k):
            cluster_values = sample_silhouette_values[clusters == i]
            cluster_values.sort()
            size = cluster_values.shape[0]
            y_upper = y_lower + size

            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values,
                              facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * size, f"Cluster {i}", fontsize=10)
            y_lower = y_upper + 10

        plt.axvline(x=avg_score, color="red", linestyle="--", linewidth=2,
                    label=f"Average ({avg_score:.3f})")
        plt.xlabel("Silhouette Coefficient", fontsize=12)
        plt.ylabel("Cluster", fontsize=12)
        plt.title("Silhouette Diagram for K-Means Clustering", fontsize=14)
        plt.legend(fontsize=11)

    # Print per-cluster stats
    print("\nPer-cluster silhouette scores:")
    for i in range(k):
        cluster_values = sample_silhouette_values[clusters == i]
        print(f"  Cluster {i}: mean={cluster_values.mean():.4f}, "
              f"min={cluster_values.min():.4f}, size={len(cluster_values)}")

# ============================================================
# 7. CLUSTER ANALYSIS
# ============================================================
def analyze_clusters(df_result, k=K_CLUSTERS):
    """Analyze and label each cluster."""
    section("CLUSTER ANALYSIS")

    segment_labels = {}

    for i in range(k):
        cluster_data = df_result[df_result["Cluster"] == i]
        avg_income = cluster_data["Annual Income (k$)"].mean()
        avg_spend = cluster_data["Spending Score (1-100)"].mean()
        avg_age = cluster_data["Age"].mean()
        count = len(cluster_data)

        # Auto-label based on income and spending
        if avg_income > 65 and avg_spend > 65:
            label = "Premium (High Income, High Spend)"
        elif avg_income > 65 and avg_spend < 40:
            label = "Careful (High Income, Low Spend)"
        elif avg_income < 40 and avg_spend > 65:
            label = "At Risk (Low Income, High Spend)"
        elif avg_income < 40 and avg_spend < 40:
            label = "Budget (Low Income, Low Spend)"
        else:
            label = "Standard (Average)"

        segment_labels[i] = label

        print(f"\n  Cluster {i}: {label}")
        print(f"    Customers:    {count}")
        print(f"    Avg Age:      {avg_age:.1f}")
        print(f"    Avg Income:   ${avg_income:.1f}k")
        print(f"    Avg Spending:  {avg_spend:.1f}")

        if "Gender" in cluster_data.columns:
            male = (cluster_data["Gender"] == "Male").sum()
            female = (cluster_data["Gender"] == "Female").sum()
            print(f"    Gender:       Male={male}, Female={female}")

    return segment_labels

# ============================================================
# 8. CLUSTER PROFILES VISUALIZATION
# ============================================================
def plot_cluster_profiles(df_result, segment_labels, k=K_CLUSTERS):
    """Visualize cluster profiles."""
    section("CLUSTER PROFILES")

    cluster_summary = (
        df_result.groupby("Cluster")
        .agg({
            "Age": "mean",
            "Annual Income (k$)": "mean",
            "Spending Score (1-100)": "mean",
        })
        .round(1)
    )

    labels = [f"C{i}\n{segment_labels[i].split('(')[0].strip()}" for i in range(k)]
    metric_specs = [
        ("Age", "Average Age by Cluster", "Age", "steelblue"),
        ("Annual Income (k$)", "Average Income by Cluster", "Income (k$)", "coral"),
        ("Spending Score (1-100)", "Average Spending Score by Cluster", "Spending Score", "green"),
    ]

    # Bar charts
    with save_plot("07_cluster_profiles_bar.png"):
        _, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (col, title, ylabel, color) in zip(axes, metric_specs):
            ax.bar(labels, cluster_summary[col], color=color, alpha=0.8)
            ax.set_title(title, fontsize=12)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis="y")

    # Box plots
    with save_plot("08_cluster_boxplots.png"):
        _, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (col, _, _, _) in zip(axes, metric_specs):
            df_result.boxplot(column=col, by="Cluster", ax=ax)
            ax.set_title(f"{col} by Cluster")
            ax.set_xlabel("Cluster")
        plt.suptitle("")

    print("\nCluster summary table:")
    print(cluster_summary.to_string())

# ============================================================
# 9. DBSCAN CLUSTERING
# ============================================================
def run_dbscan(X_2d, X_2d_scaled, clusters_kmeans):
    """Run DBSCAN and compare with K-Means."""
    section("DBSCAN CLUSTERING")

    # Test different eps values
    print("\nTesting different eps values (min_samples=5):")
    best_eps = 0.5
    best_score = -1

    for eps in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X_2d_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        score_str = ""
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 0 and len(set(labels[mask])) > 1:
                score = silhouette_score(X_2d_scaled[mask], labels[mask])
                score_str = f", Silhouette={score:.4f}"
                if score > best_score:
                    best_score, best_eps = score, eps

        print(f"  eps={eps:.1f}: {n_clusters} clusters, {n_noise} noise points{score_str}")

    # Run with best eps
    print(f"\nUsing eps={best_eps}:")
    dbscan = DBSCAN(eps=best_eps, min_samples=5)
    dbscan_clusters = dbscan.fit_predict(X_2d_scaled)

    n_clusters = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
    n_noise = list(dbscan_clusters).count(-1)

    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise}")

    with save_plot("09_kmeans_vs_dbscan.png"):
        _, axes = plt.subplots(1, 2, figsize=(16, 6))

        # K-Means
        axes[0].scatter(X_2d[:, 0], X_2d[:, 1],
                        c=clusters_kmeans, cmap="viridis", alpha=0.6, s=80,
                        edgecolors="white", linewidth=0.5)
        axes[0].set(title=f"K-Means Clustering (K={K_CLUSTERS})",
                    xlabel="Annual Income (k$)", ylabel="Spending Score")
        axes[0].grid(True, alpha=0.3)

        # DBSCAN
        mask = dbscan_clusters != -1
        if mask.any():
            axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                            c=dbscan_clusters[mask], cmap="viridis", alpha=0.6, s=80,
                            edgecolors="white", linewidth=0.5, label="Clustered")
        noise_mask = dbscan_clusters == -1
        if noise_mask.any():
            axes[1].scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                            c="grey", marker="x", s=60, alpha=0.8,
                            label=f"Noise ({n_noise})")

        axes[1].set(title=f"DBSCAN (eps={best_eps}, {n_clusters} clusters)",
                    xlabel="Annual Income (k$)", ylabel="Spending Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    return dbscan_clusters, best_eps

# ============================================================
# 10. 3D CLUSTERING WITH PCA
# ============================================================
def run_3d_clustering(X_3d, X_3d_scaled, k=K_CLUSTERS):
    """Run K-Means on 3D features and visualize with PCA."""
    section("3D CLUSTERING (Age + Income + Spending)")

    # K-Means on 3D
    kmeans_3d = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_3d = kmeans_3d.fit_predict(X_3d_scaled)

    score_3d = silhouette_score(X_3d_scaled, clusters_3d)
    print(f"3D K-Means Silhouette Score: {score_3d:.4f}")

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_3d_scaled)

    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
    print(f"Total explained: {sum(explained_variance):.2%}")

    # Plot
    with save_plot("10_3d_clustering.png"):
        _, axes = plt.subplots(1, 2, figsize=(16, 6))

        s1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                             c=clusters_3d, cmap="viridis", alpha=0.6, s=100,
                             edgecolors="white", linewidth=0.5)
        axes[0].set(xlabel=f"PC1 ({explained_variance[0]:.1%} variance)",
                    ylabel=f"PC2 ({explained_variance[1]:.1%} variance)",
                    title=f"3D K-Means Clusters (PCA projection)\nSilhouette={score_3d:.3f}")
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(s1, ax=axes[0], label="Cluster")

        s2 = axes[1].scatter(X_3d[:, 1], X_3d[:, 2],
                             c=clusters_3d, cmap="viridis", alpha=0.6, s=100,
                             edgecolors="white", linewidth=0.5)
        axes[1].set(xlabel="Annual Income (k$)", ylabel="Spending Score",
                    title="3D Clusters (Income vs Spending view)")
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(s2, ax=axes[1], label="Cluster")

    return score_3d

# ============================================================
# 11. CLUSTER STABILITY ANALYSIS
# ============================================================
def analyze_stability(X_2d_scaled, k=K_CLUSTERS):
    """Analyze cluster stability with different random seeds."""
    section("CLUSTER STABILITY ANALYSIS")

    scores = []
    for seed in range(20):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X_2d_scaled)
        scores.append(silhouette_score(X_2d_scaled, labels))

    mean = np.mean(scores)
    std = np.std(scores)
    score_min, score_max = np.min(scores), np.max(scores)

    print("Silhouette scores across 20 runs:")
    print(f"  Mean:  {mean:.4f}")
    print(f"  Std:   {std:.4f}")
    print(f"  Min:   {score_min:.4f}")
    print(f"  Max:   {score_max:.4f}")
    print(f"  Range: {score_max - score_min:.4f}")

    if std < 0.01:
        print("  ✅ Clusters are very stable (low variance)")
    elif std < 0.05:
        print("  ⚠️  Some instability in clustering")
    else:
        print("  ❌ Clusters are unstable")

    # Plot stability
    with save_plot("11_stability.png"):
        plt.figure(figsize=(10, 5))
        plt.bar(range(20), scores, color="steelblue", alpha=0.7)
        plt.axhline(y=mean, color="red", linestyle="--", label=f"Mean: {mean:.4f}")
        plt.xlabel("Random Seed")
        plt.ylabel("Silhouette Score")
        plt.title(f"Cluster Stability (K={k}, 20 runs)")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

# ============================================================
# 12. BUSINESS RECOMMENDATIONS
# ============================================================
def print_recommendations(segment_labels):
    """Print business recommendations for each segment."""
    section("BUSINESS RECOMMENDATIONS")

    recommendations = {
        "Premium (High Income, High Spend)": {
            "strategy": "Retain & Reward",
            "actions": [
                "VIP loyalty program with exclusive perks",
                "Early access to new products",
                "Personal shopping assistant",
                "Premium customer support",
            ],
        },
        "Careful (High Income, Low Spend)": {
            "strategy": "Convert & Upsell",
            "actions": [
                "Targeted campaigns highlighting value",
                "Personalized product recommendations",
                "Limited-time exclusive offers",
                "Focus on quality and ROI messaging",
            ],
        },
        "Standard (Average)": {
            "strategy": "Maintain & Grow",
            "actions": [
                "Regular promotions and discounts",
                "Loyalty points program",
                "Seasonal campaigns",
                "Cross-selling complementary products",
            ],
        },
        "At Risk (Low Income, High Spend)": {
            "strategy": "Monitor & Retain",
            "actions": [
                "Budget-friendly alternatives",
                "Installment payment options",
                "Value bundles",
                "Satisfaction surveys to prevent churn",
            ],
        },
        "Budget (Low Income, Low Spend)": {
            "strategy": "Engage Efficiently",
            "actions": [
                "Volume-based promotions",
                "Clearance and discount events",
                "Low-cost digital engagement",
                "Don't over-invest in acquisition",
            ],
        },
    }

    for cluster_id, label in segment_labels.items():
        rec = recommendations.get(label, {"strategy": "Custom", "actions": ["Analyze further"]})
        print(f"\n  Cluster {cluster_id}: {label}")
        print(f"  Strategy: {rec['strategy']}")
        print("  Actions:")
        for action in rec["actions"]:
            print(f"    • {action}")

# ============================================================
# 13. FINAL SUMMARY
# ============================================================
def print_summary(k, kmeans_score, segment_labels, score_3d):
    """Print final summary."""
    section("FINAL SUMMARY")

    print(f"""
Dataset: Mall Customers (200 samples)
Features used: Annual Income (k$), Spending Score (1-100)

K-Means Clustering (2D):
Optimal K: {k}
Silhouette Score: {kmeans_score:.4f}

K-Means Clustering (3D, with Age):
Silhouette Score: {score_3d:.4f}

Customer Segments Identified:""")

    for cluster_id, label in segment_labels.items():
        print(f"  Cluster {cluster_id}: {label}")

    print(f"""
Key Business Insight:
The 'Careful' segment (High Income, Low Spend) represents the 
biggest revenue opportunity. These customers have purchasing 
power but aren't spending — targeted marketing campaigns with 
personalized offers could unlock significant growth.
""")

# ============================================================
# MAIN
# ============================================================
def main():
    """Run the complete clustering pipeline."""

    # 1. Load data
    df = load_data()

    # 2. Explore data
    explore_data(df)

    # 3. Prepare features
    X_2d, X_2d_scaled, X_3d, X_3d_scaled = prepare_features(df)

    # 4. Find optimal K
    find_optimal_k(X_2d_scaled)

    # 5. K-Means clustering
    df_result, clusters, score = run_kmeans(X_2d, X_2d_scaled, df, k=K_CLUSTERS)

    # 6. Silhouette detail
    plot_silhouette_detail(X_2d_scaled, clusters, K_CLUSTERS)

    # 7. Analyze clusters
    segment_labels = analyze_clusters(df_result, k=K_CLUSTERS)

    # 8. Cluster profiles
    plot_cluster_profiles(df_result, segment_labels, k=K_CLUSTERS)

    # 9. DBSCAN comparison
    run_dbscan(X_2d, X_2d_scaled, clusters)

    # 10. 3D clustering
    score_3d = run_3d_clustering(X_3d, X_3d_scaled, k=K_CLUSTERS)

    # 11. Stability analysis
    analyze_stability(X_2d_scaled, k=K_CLUSTERS)

    # 12. Business recommendations
    print_recommendations(segment_labels)

    # 13. Summary
    print_summary(K_CLUSTERS, score, segment_labels, score_3d)

    print("\nAll plots saved in /plots folder")
    print("Done! ✅")

if __name__ == "__main__":
    main()