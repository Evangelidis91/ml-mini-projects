# House Price Prediction

Regression models to predict California house prices based on property and location features.

## Overview

This project builds and compares multiple regression models to predict median house values in California. It demonstrates the significant performance gap between linear and tree-based models when dealing with non-linear real estate data.

## Dataset

**California Housing** (scikit-learn built-in)
- **Samples:** 20,640 housing blocks
- **Features:** 8 numerical features
- **Target:** Median house value (units of $100k)
- **Target range:** $14,999 — $500,001
- **Split:** 16,512 train / 4,128 test (80/20)
- **Missing values:** None

### Features
| Feature | Description | Correlation with target |
|---------|-------------|----------------------|
| MedInc | Median income in block | +0.688 |
| AveRooms | Average rooms per household | +0.152 |
| HouseAge | Median house age | +0.106 |
| AveOccup | Average household occupancy | -0.024 |
| Population | Block population | -0.025 |
| Longitude | Block longitude | -0.046 |
| AveBedrms | Average bedrooms per household | -0.047 |
| Latitude | Block latitude | -0.144 |

## Approach

1. **Exploratory Data Analysis** — Feature distributions, statistics
2. **Correlation Analysis** — Linear relationships with target
3. **Geographic Mapping** — Price and population density across California
4. **Feature Relationships** — Scatter plots of each feature vs house value
5. **Model Training** — Compare 5 regression models with cross-validation
6. **Hyperparameter Tuning** — GridSearchCV on best model
7. **Residual Analysis** — Verify assumptions and identify error patterns
8. **Prediction Error Analysis** — Performance across price ranges

## Results

### Model Comparison
| Model | RMSE | MAE | R² | CV R² |
|-------|------|-----|-----|-------|
| Linear Regression | 0.746 | 0.533 | 0.576 | 0.611 |
| Ridge Regression | 0.746 | 0.533 | 0.576 | 0.611 |
| Lasso Regression | 0.740 | 0.535 | 0.582 | 0.608 |
| Gradient Boosting | 0.542 | 0.372 | 0.776 | 0.787 |
| **Random Forest** | **0.506** | **0.328** | **0.805** | **0.805** |

### Hyperparameter Tuning (Random Forest)
| Metric | Before tuning | After tuning |
|--------|--------------|--------------|
| R² | 0.8050 | 0.8066 |
| RMSE | 0.5055 | 0.5035 |
| MAE | 0.3276 | 0.3265 |

**Best parameters:** `max_depth=None, min_samples_split=2, n_estimators=300`

### Prediction Error by Price Range
| Range | MAE | Samples |
|-------|-----|---------|
| Budget (< $150k) | 0.233 ($23,300) | 1,500 |
| Mid ($150k - $300k) | 0.292 ($29,200) | 1,870 |
| High ($300k - $400k) | 0.498 ($49,800) | 415 |
| Premium (≥ $400k) | 0.729 ($72,900) | 343 |

### Residual Analysis
| Statistic | Value |
|-----------|-------|
| Mean | -0.012 (no systematic bias) |
| Median | -0.050 |
| Std | 0.505 |
| Range | -3.038 to +3.089 |

### Feature Importance (Random Forest)
| Feature | Importance |
|---------|------------|
| MedInc (Median Income) | 0.525 |
| AveOccup (Avg Occupancy) | 0.139 |
| Latitude | 0.089 |
| Longitude | 0.089 |
| HouseAge | 0.055 |
| AveRooms | 0.044 |
| Population | 0.031 |
| AveBedrms | 0.030 |

## Visualizations

| Plot | Description |
|------|-------------|
| `01_feature_distributions.png` | Distribution of all 8 features + target |
| `02_correlation_matrix.png` | Feature correlation heatmap |
| `03_geographic_maps.png` | California price map + population density map |
| `04_feature_relationships.png` | Scatter plots of each feature vs house value |
| `05_model_comparison.png` | RMSE, MAE, R² comparison across all models |
| `06_actual_vs_predicted.png` | Actual vs predicted scatter for Random Forest |
| `07_residual_analysis.png` | Residual plots + distribution + Q-Q plot |
| `08_feature_importance.png` | Feature importance bar chart |
| `09_error_by_range.png` | Prediction error by price range |

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — Model training, evaluation, hyperparameter tuning
- **pandas** — Data manipulation
- **numpy** — Numerical operations
- **matplotlib** — Plotting
- **seaborn** — Statistical visualizations

## Project Structure
02-regression/ 
    ├── regression.py # Main pipeline script 
    ├── requirements.txt # Python dependencies 
    ├── README.md # This file 
    └── plots/ # Generated visualizations 
        ├── 01_feature_distributions.png 
        ├── 02_correlation_matrix.png 
        ├── 03_geographic_maps.png 
        ├── 04_feature_relationships.png 
        ├── 05_model_comparison.png 
        ├── 06_actual_vs_predicted.png 
        ├── 07_residual_analysis.png 
        ├── 08_feature_importance.png 
        └── 09_error_by_range.png

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python regression.py