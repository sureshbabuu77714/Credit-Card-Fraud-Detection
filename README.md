# Credit-Card-Fraud-Detection
This project was developed as an end-to-end machine learning project to demonstrate practical skills in Python, imbalanced classification, XGBoost, model evaluation, hyperparameter tuning, threshold optimization, and model explainability.

# Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Since fraudulent transactions represent a very small proportion of the total transactions, the dataset is highly imbalanced. The project addresses this challenge using **SMOTE (Synthetic Minority Over-sampling Technique)** and builds an optimized **XGBoost classification model**.

The project follows an end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, imbalance handling, model training, cross-validation, hyperparameter tuning, threshold optimization, model evaluation, and explainability using SHAP.

---

## 🎯 Objectives

- Detect fraudulent credit card transactions accurately.
- Handle severe class imbalance using SMOTE.
- Build and evaluate an XGBoost classification model.
- Compare tuned and untuned XGBoost performance.
- Optimize the classification threshold instead of relying only on the default 0.5 threshold.
- Evaluate the model using fraud-focused metrics such as Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
- Interpret model predictions using SHAP explainability.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- SHAP
- Jupyter Notebook

---

## 🔄 Project Workflow

### 1. Data Preprocessing
- Loaded and inspected the transaction dataset.
- Checked missing values and data types.
- Performed exploratory data analysis.
- Analyzed the distribution of fraudulent and legitimate transactions.

### 2. Handling Class Imbalance

The dataset contains significantly fewer fraudulent transactions than legitimate transactions.

To address this imbalance, **SMOTE** was applied within an imbalanced-learn pipeline to generate synthetic minority-class samples.

This prevents the model from becoming biased toward the majority class.

### 3. XGBoost Model

An **XGBoost Classifier** was selected because it performs well on tabular datasets and can capture complex nonlinear relationships between features.

The model was initially trained using an SMOTE + XGBoost pipeline.

### 4. Stratified K-Fold Cross-Validation

A **5-fold Stratified Cross-Validation** strategy was used to maintain the class distribution across folds.

The model was evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC

### 5. Hyperparameter Tuning

`RandomizedSearchCV` was used to search for better XGBoost hyperparameters.

The search included parameters such as:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`

The best-performing configuration was selected as the tuned XGBoost model.

### 6. Tuned vs Untuned Model Comparison

The tuned XGBoost model was compared with the initial XGBoost pipeline using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

This helped determine whether hyperparameter tuning improved the model's performance.

### 7. Threshold Optimization

Instead of automatically using the default classification threshold of `0.5`, multiple thresholds were evaluated.

The threshold was varied from `0.05` to `0.95` and Precision, Recall, and F1-score were calculated for each threshold.

A final threshold of **0.85** was selected based on the desired precision-recall trade-off.

### 8. Final Model Evaluation

At the final threshold of **0.85**, the tuned XGBoost model achieved approximately:

| Metric | Score |
|---|---:|
| Precision | **92.5%** |
| Recall | **77.9%** |
| F1-score | **84.6%** |
| ROC-AUC | **96.9%** |
| PR-AUC | **81.2%** |

### Confusion Matrix

```text
[[56645     6]
 [   21    74]]
