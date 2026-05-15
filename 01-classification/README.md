# Customer Churn Prediction

Binary classification model to predict whether a telecom customer will churn based on usage patterns and account information.

## Overview

Customer churn is a critical business problem in the telecom industry. This project builds and compares multiple machine learning models to identify customers likely to leave, enabling proactive retention strategies.

## Dataset

**Telco Customer Churn** (IBM)
- **Source:** [IBM Sample Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Samples:** 7,043 customers
- **Features:** 20 (after dropping customerID)
- **Target:** Churn (binary — 0 or 1)
- **Class balance:** 73.5% No Churn (5,174) / 26.5% Churn (1,869)
- **Split:** 5,634 train / 1,409 test (80/20, stratified)

### Key Features
| Feature | Description |
|---------|-------------|
| Contract | Month-to-month, One year, Two year |
| tenure | Months as a customer |
| MonthlyCharges | Monthly payment amount |
| TotalCharges | Total amount charged |
| OnlineSecurity | Has online security add-on |
| TechSupport | Has tech support add-on |
| InternetService | DSL, Fiber optic, None |
| PaymentMethod | Electronic check, Mailed check, etc. |

## Approach

1. **Exploratory Data Analysis** — Distribution analysis, churn patterns by contract type and charges
2. **Preprocessing** — Label encoding of 15 categorical columns, handling missing values in TotalCharges, feature scaling with StandardScaler
3. **Correlation Analysis** — Identified top features correlated with churn
4. **Model Training** — Compared Logistic Regression, Random Forest, Gradient Boosting, and a class-weighted Gradient Boosting variant to handle the imbalanced target
5. **Hyperparameter Tuning** — GridSearchCV with 5-fold cross-validation on best model
6. **Evaluation** — Accuracy, ROC-AUC, recall, confusion matrix, feature importance

## Results

| Model | Accuracy | ROC-AUC | Churn Recall |
|-------|----------|---------|--------------|
| Logistic Regression | 74.0% | 0.840 | 79.7% |
| Random Forest | 79.2% | 0.821 | 49.5% |
| **Gradient Boosting** | **80.1%** | **0.845** | 50.5% |
| GB (Weighted) | 74.6% | 0.844 | **79.1%** |

> **Note on model choice:** Gradient Boosting wins on ROC-AUC and accuracy, but only catches ~50% of actual churners. The class-weighted variant (`GB (Weighted)`) reaches a similar ROC-AUC while catching ~80% of churners — usually the more useful tradeoff when missing a churner costs more than a false alarm.

### After Hyperparameter Tuning (Gradient Boosting)
| Metric | Score |
|--------|-------|
| Test Accuracy | 80.3% |
| Test ROC-AUC | 0.845 |
| Best CV ROC-AUC | 0.846 |

**Best parameters:** `learning_rate=0.05, max_depth=3, n_estimators=100`

### Top 5 Most Important Features
| Feature | Importance |
|---------|------------|
| Contract | 0.404 |
| tenure | 0.143 |
| MonthlyCharges | 0.135 |
| OnlineSecurity | 0.081 |
| TotalCharges | 0.078 |

## Visualizations

The script generates the following plots in the `plots/` folder:

| Plot | Description |
|------|-------------|
| `01_eda_overview.png` | Churn distribution, churn by contract type, monthly charges |
| `02_correlation_matrix.png` | Feature correlation heatmap |
| `03_roc_curves.png` | ROC curves comparing all four models |
| `04_confusion_matrix.png` | Confusion matrix for the best model |
| `05_feature_importance.png` | Top 10 most predictive features |

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — Model training, evaluation, hyperparameter tuning
- **pandas** — Data manipulation
- **numpy** — Numerical operations
- **matplotlib** — Plotting
- **seaborn** — Statistical visualizations

## Project Structure
01-classification/ 
    ├── classification.py # Main pipeline script    
    ├── requirements.txt # Python dependencies  
    ├── README.md # This file 
    └── plots/ # Generated visualizations 
        ├── 01_eda_overview.png 
        ├── 02_correlation_matrix.png 
        ├── 03_roc_curves.png 
        ├── 04_confusion_matrix.png 
        └── 05_feature_importance.png

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python classification.py