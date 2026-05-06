My thought process
ml-mini-projects/README.md
markdown
# Machine Learning Mini Projects

A collection of machine learning projects covering classification, regression, deep learning (CNN), and unsupervised learning (clustering). Built during MSc in Data Science and Machine Learning.

## Projects

| # | Project | Technique | Algorithm | Dataset |
|---|---------|-----------|-----------|---------|
| 1 | [Customer Churn Prediction](./01-classification/) | Classification | Logistic Regression, Random Forest, Gradient Boosting | Telco Customer Churn (IBM) |
| 2 | [House Price Prediction](./02-regression/) | Regression | Linear, Ridge, Lasso, Random Forest, Gradient Boosting | California Housing |
| 3 | [Image Classification](./03-cnn-image-classification/) | Deep Learning (CNN) | Convolutional Neural Network | CIFAR-10 |
| 4 | [Customer Segmentation](./04-clustering/) | Clustering | K-Means, DBSCAN | Mall Customers |

## Results Summary

| Project | Best Model | Key Metric | Score |
|---------|-----------|------------|-------|
| Classification | Gradient Boosting | ROC-AUC | 0.845 |
| Regression | Random Forest | R² | 0.807 |
| CNN | 3-Block CNN + BatchNorm | Accuracy | 78% |
| Clustering | K-Means (K=5) | Silhouette | ~0.55 |

## Skills Demonstrated

### Machine Learning
- Supervised learning (classification & regression)
- Unsupervised learning (clustering & segmentation)
- Deep learning (convolutional neural networks)
- Model selection and comparison
- Hyperparameter tuning (GridSearchCV)
- Cross-validation
- Feature importance analysis

### Data Science
- Exploratory data analysis
- Data preprocessing and feature scaling
- Handling missing values and class imbalance
- Statistical evaluation metrics
- Data visualization
- Business insights from ML results

### Technical
- Python, scikit-learn, TensorFlow/Keras
- pandas, numpy, matplotlib, seaborn
- Clean code structure with modular functions
- Automated plot generation
- Reproducible pipelines

## Tech Stack

| Library | Usage |
|---------|-------|
| scikit-learn | ML models, preprocessing, evaluation |
| TensorFlow/Keras | CNN architecture and training |
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib | Visualizations |
| seaborn | Statistical plots |

