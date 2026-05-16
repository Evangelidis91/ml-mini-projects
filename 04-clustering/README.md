# Customer Segmentation with Clustering

Unsupervised learning to segment customers based on purchasing behavior using K-Means and DBSCAN clustering.

## Overview

This project performs customer segmentation on mall customers to identify distinct groups based on their annual income and spending patterns. The resulting segments enable targeted marketing strategies and resource allocation.

## Dataset

**Mall Customers** (generated with realistic cluster structure)
- **Samples:** 200 customers
- **Features:** CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
- **Type:** Unsupervised (no labels)
- **Gender split:** Male 52%, Female 48%
- **Source:** [Kaggle Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)

> If `Mall_Customers.csv` is not found in the working directory, the script automatically generates a realistic dataset with 5 natural clusters so the pipeline still runs end-to-end.

### Statistics
| Feature | Mean | Min | Max | Std |
|---------|------|-----|-----|-----|
| Age | 38.5 | 18 | 70 | 11.7 |
| Annual Income (k$) | 57.3 | 15 | 137 | 30.5 |
| Spending Score (1-100) | 49.3 | 1 | 99 | 27.6 |

## Approach

1. **Exploratory Data Analysis** — Distributions, scatter plots, gender analysis
2. **Feature Scaling** — StandardScaler for equal feature contribution
3. **Optimal K Selection** — Elbow method + Silhouette analysis (K=2 to K=10)
4. **K-Means Clustering** — Main segmentation (K=5)
5. **Silhouette Analysis** — Per-cluster quality assessment
6. **Cluster Profiling** — Demographics and business labeling
7. **DBSCAN Comparison** — Density-based alternative with multiple eps values
8. **3D Clustering** — Adding Age dimension with PCA visualization
9. **Stability Analysis** — Verifying robustness across 20 random seeds
10. **Business Recommendations** — Actionable strategies per segment

## Results

### Optimal K Selection
| K | Inertia | Silhouette Score |
|---|---------|------------------|
| 2 | 243.83 | 0.3654 |
| 3 | 149.49 | 0.4443 |
| 4 | 84.16 | 0.5130 |
| **5** | **44.27** | **0.5877** |
| 6 | 38.76 | 0.5539 |
| 7 | 33.84 | 0.4811 |

**Optimal K = 5** (highest silhouette score: 0.5877)

### K-Means Clustering (K=5)
| Cluster | Label | Customers | Avg Age | Avg Income | Avg Spending | Silhouette |
|---------|-------|-----------|---------|------------|--------------|------------|
| 0 | Standard (Average) | 49 | 42.1 | $57.2k | 50.4 | 0.538 |
| 1 | Careful (High Income, Low Spend) | 36 | 46.1 | $91.4k | 17.6 | 0.560 |
| 2 | Budget (Low Income, Low Spend) | 39 | 44.1 | $25.3k | 20.1 | 0.631 |
| 3 | Premium (High Income, High Spend) | 40 | 32.7 | $88.6k | 82.5 | 0.576 |
| 4 | At Risk (Low Income, High Spend) | 36 | 26.2 | $23.2k | 74.5 | 0.649 |

**Overall Silhouette Score: 0.5877**

> All 5 clusters have silhouette scores above 0.53, indicating well-defined and meaningfully separated segments.

### DBSCAN Comparison
| eps | Clusters | Noise Points | Silhouette |
|-----|----------|--------------|------------|
| 0.3 | 4 | 20 | 0.521 |
| 0.4 | 2 | 5 | 0.349 |
| 0.5 | 2 | 1 | 0.346 |
| 0.6+ | 1 | 0 | — |

> K-Means outperforms DBSCAN for this dataset — the clusters are spherical and well-separated, which is exactly what K-Means is designed for. DBSCAN's best result (eps=0.3) finds 4 clusters but treats 20 customers (10%) as noise.

### 3D Clustering (Age + Income + Spending)
| Metric | Score |
|--------|-------|
| Silhouette Score | 0.3977 |
| PCA PC1 variance | 51.06% |
| PCA PC2 variance | 34.27% |
| Total explained | 85.34% |

> Adding Age reduces cluster separation (0.59 → 0.40). Income and Spending are the dominant signals; Age adds noise without improving structure. The 2D model is the better choice here.

### Cluster Stability
| Metric | Value |
|--------|-------|
| Mean silhouette (20 runs) | 0.5877 |
| Std deviation | 0.0000 |
| Range | 0.0000 |
| Status | ✅ Perfectly stable |

> Zero variance across 20 random seeds — clusters are completely reproducible. Combined with `n_init=10`, K-Means converges to the same global optimum every time.

## Customer Segments & Business Recommendations

### 🟢 Cluster 3: Premium (High Income, High Spend)
- **Size:** 40 customers (20%)
- **Profile:** Young (avg 33), high income ($89k), heavy spenders (83/100)
- **Strategy:** Retain & Reward
- **Actions:** VIP loyalty, early access, personal shopping, premium support

### 🟡 Cluster 1: Careful (High Income, Low Spend) — BIGGEST OPPORTUNITY
- **Size:** 36 customers (18%)
- **Profile:** Older (avg 46), high income ($91k), low spending (18/100)
- **Strategy:** Convert & Upsell
- **Actions:** Targeted campaigns, personalized recommendations, limited-time offers

### 🔵 Cluster 0: Standard (Average)
- **Size:** 49 customers (24.5%) — largest segment
- **Profile:** Middle-aged (avg 42), average income ($57k), average spending (50/100)
- **Strategy:** Maintain & Grow
- **Actions:** Regular promotions, loyalty points, seasonal campaigns

### 🟠 Cluster 4: At Risk (Low Income, High Spend)
- **Size:** 36 customers (18%)
- **Profile:** Youngest (avg 26), low income ($23k), high spending (75/100)
- **Strategy:** Monitor & Retain
- **Actions:** Budget alternatives, installment plans, value bundles

### ⚪ Cluster 2: Budget (Low Income, Low Spend)
- **Size:** 39 customers (19.5%)
- **Profile:** Older (avg 44), low income ($25k), low spending (20/100)
- **Strategy:** Engage Efficiently
- **Actions:** Volume promotions, clearance events, don't over-invest

## Visualizations

| Plot | Description |
|------|-------------|
| `01_distributions.png` | Age, Income, and Spending Score distributions |
| `02_scatter_plots.png` | Income vs Spending, Age vs Spending, Age vs Income |
| `03_gender_analysis.png` | Income vs Spending by gender + box plots |
| `04_optimal_k.png` | Elbow method + Silhouette score for K=2-10 |
| `05_kmeans_clusters.png` | K-Means clusters with centroids |
| `06_silhouette_detail.png` | Per-point silhouette diagram |
| `07_cluster_profiles_bar.png` | Average Age/Income/Spending per cluster |
| `08_cluster_boxplots.png` | Distribution spread per cluster |
| `09_kmeans_vs_dbscan.png` | K-Means vs DBSCAN comparison |
| `10_3d_clustering.png` | 3D clusters with PCA projection |
| `11_stability.png` | Silhouette consistency across 20 seeds |

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — K-Means, DBSCAN, Silhouette, PCA
- **pandas** — Data manipulation
- **numpy** — Numerical operations
- **matplotlib** — Plotting

## Project Structure
04-clustering/ 
    ├── clustering.py # Main pipeline script 
    ├── requirements.txt # Python dependencies 
    ├── README.md # This file 
    └── plots/ # Generated visualizations 
        ├── 01_distributions.png 
        ├── 02_scatter_plots.png 
        ├── 03_gender_analysis.png 
        ├── 04_optimal_k.png 
        ├── 05_kmeans_clusters.png 
        ├── 06_silhouette_detail.png 
        ├── 07_cluster_profiles_bar.png 
        ├── 08_cluster_boxplots.png 
        ├── 09_kmeans_vs_dbscan.png 
        ├── 10_3d_clustering.png 
        └── 11_stability.png

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python clustering.py