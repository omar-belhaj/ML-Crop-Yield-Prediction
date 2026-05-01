# 🌾 ML Crop Yield Prediction

Academic project : M1 AI, Data, Agentics - Dauphine Tunis  
Course: Fouille de données 

**All analyses and interpretations are written in French.**

## Context

This notebook implements a complete machine learning pipeline to predict agricultural yield (`yield_value`) from a set of structured features.

**The dataset is artificially generated.** It was provided as part of the course material to simulate a realistic agricultural dataset. This is why the features have generic names such as `feature_score_1`, `feature_score_2`, `exposure_index`, `special_flag`, etc. They represent abstracted agronomic variables without domain-specific labels.

The notebook is inspired by a guide provided by the instructor (`DM_Insurance_Guide`) and structured around a series of analytical questions.

## Objective

Predict crop yield using supervised and unsupervised machine learning techniques, and explain model behavior using interpretability tools.

## Pipeline Overview

| Part | Description |
|------|-------------|
| **EDA** | Initial inspection, descriptive statistics, missing values, outliers, distributions |
| **Correlation Analysis** | Pearson, Spearman, Kendall, Cramér's V, ANOVA, Mutual Information |
| **Statistical Tests** | Shapiro-Wilk, Levene, Mann-Whitney |
| **Regression** | Benchmark of 7 regressors, cross-validation shortlist, GridSearchCV on Ridge |
| **Diagnostics** | Train/test gap, residual analysis, overfitting/underfitting assessment |
| **PCA** | Dimensionality reduction, impact comparison with and without PCA |
| **Multiclass Classification** | Discretization of target into 3 classes, benchmark of 5 classifiers |
| **Clustering** | KMeans, silhouette score, impact of cluster label as feature |
| **Binary Classification + SMOTE** | High-yield detection (top 20%), class imbalance handling with SMOTE |
| **Interpretability** | Global explanations with SHAP, local explanations with LIME |
| **Interpretation Questions** | Synthesis and critical analysis of results |

## Key Results

| Task | Best Model | Score |
|------|-----------|-------|
| Regression | Ridge (α=10) | R² = **0.774**, RMSE = 1033 |
| Multiclass Classification | Logistic Regression | Accuracy = **70.9%**, F1 = 0.711 |
| Binary Classification (baseline) | Logistic Regression | ROC-AUC = **0.942** |
| Binary Classification (SMOTE) | Logistic Regression | Recall class 1 = **0.898** |

**Notable finding:** Linear models (Ridge, Lasso, LinearRegression) outperform ensemble methods (GradientBoosting, RandomForest), consistent with the additive and linear structure of the artificially generated dataset.

## Tech Stack

- **Language:** Python
- **Core ML:** `scikit-learn`, `imbalanced-learn` (SMOTE)
- **Interpretability:** `shap`, `lime`
- **Data:** `pandas`, `numpy`, `scipy`
- **Visualization:** `matplotlib`
- **Environment:** Jupyter Notebook
