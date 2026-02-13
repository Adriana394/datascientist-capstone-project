# Insurance Fraud Detection – Capstone Project (DataScientest)

This repository contains an **end-to-end machine learning project** for predicting fraudulent insurance claims (`fraud_reported`).  
The focus is on **tabular data**, **class imbalance**, **threshold optimization**, and **interpretability** (including SHAP).

## Objective & Metrics

- **Objective**: Binary classification of "fraud" vs. "no fraud" for individual claims.
- **Challenges**: Rare target class, costly false negatives, leakage risk, many categorical features.
- **Evaluation**: Primarily **PR-AUC** and **Precision/Recall/F1** for the fraud class; additionally **threshold tuning** (operating point).

## Project Structure

```text
datascientist_project_remake/
├── data/                         # Raw/cleaned/preprocessed data (CSV)
├── notebooks/                    # EDA + modeling (01–04)
├── models/                       # Saved models per approach
├── reports/
│   ├── artifacts/                # Notebook artifacts (e.g., tuning, metrics)
│   └── figures/                  # Exported plots
├── src/                          # (Template) Python package structure
│   └── streamlit/app.py          # Currently empty (no dashboard yet)
├── requirements.txt
└── README.md
```

## Notebooks (Workflow)

### 1. Data Exploration (`01_data_exploration.ipynb`)

Comprehensive exploratory data analysis including:

- **Data Quality**: Missing values inspection, data type analysis, basic cleaning
- **Outlier Detection**: IQR-based outlier analysis (notably in `umbrella_limit`)
- **Distribution Analysis**: 
  - Numerical feature distributions and skewness
  - Categorical feature distributions
  - Class imbalance assessment (~24.7% fraud)
- **Correlation Analysis**: Pearson correlation heatmap revealing multicollinearity among claim amount variables (`total_claim_amount`, `vehicle_claim`, `property_claim`, `injury_claim`)
- **Statistical Testing**:
  - **Mann-Whitney U test** for numerical features (with Cliff's Delta effect size)
  - **Chi-Square test** for categorical features (with Cramér's V)
  - Results show `incident_severity` as the strongest categorical predictor (V = 0.514)
  - Claim amount variables show moderate effect sizes for fraud discrimination

**Key Insights**: High incident severity and larger claim amounts correlate with fraud. Missing categorical data encoded as "Unknown" to preserve uncertainty. Feature engineering: `has_umbrella_policy` created as binary indicator.

---

### 2. Baseline & Random Forest Models (`02_model_implementation.ipynb`)

Implements baseline and Random Forest models with multiple variants:

**Baseline Model:**
- **Logistic Regression** with `class_weight='balanced'` and standardized features
- Serves as interpretable linear baseline
- Uses One-Hot Encoding for categorical features

**Random Forest Models:**
1. **Base RF**: `RandomForestClassifier` with `class_weight='balanced_subsample'` (300 trees)
2. **RF + BorderlineSMOTE**: Adds oversampling via BorderlineSMOTE to handle class imbalance
3. **RF + BorderlineSMOTE + Feature Selection**: Adds `SelectFromModel` feature selection based on RF importance

**Preprocessing:**
- One-Hot Encoding for categorical features
- StandardScaler for Logistic Regression
- Train/validation/test split (stratified)
- Stratified K-Fold cross-validation

**Evaluation & Interpretability:**
- PR-AUC and fraud-class Precision/Recall/F1 metrics
- Threshold optimization (F1-maximizing threshold)
- **SHAP Summary plots** for global feature importance
- **Permutation Importance** as secondary validation
- t-SNE visualization for class separation intuition

**Results**: RF + SMOTE + Feature Selection achieves best performance (PR-AUC ≈ 0.65, F1 ≈ 0.69 after threshold tuning).

---

### 3. XGBoost Model (`03_model_implementation_xgb.ipynb`)

Gradient boosting model with native categorical feature handling:

**Model Configuration:**
- `XGBClassifier` with `enable_categorical=True` (native categorical support, no OHE needed)
- `scale_pos_weight` for class imbalance handling
- `eval_metric='aucpr'` (PR-AUC)
- `tree_method='hist'` for efficiency

**Hyperparameter Tuning:**
- **GridSearchCV** with 5-fold StratifiedKFold
- Parameter grid: `n_estimators` [300, 600, 1000], `learning_rate` [0.03, 0.05, 0.1], `max_depth` [2, 3, 5], `min_child_weight` [5, 10], `subsample` [0.7, 0.9, 1.0], `colsample_bytree` [0.7, 0.9, 1.0]
- Scoring: `average_precision` (PR-AUC)

**Evaluation:**
- Threshold optimization on validation set
- SHAP analysis (summary plots, dependence plots)
- Permutation importance
- Test set evaluation with optimized threshold

**Results**: Strong balance between recall and precision (Recall ≈ 0.84, Precision ≈ 0.55-0.64 depending on threshold, F1 ≈ 0.67-0.72).

---

### 4. CatBoost Model (`04_model_implementation_catboost.ipynb`)

CatBoost implementation leveraging native categorical feature handling:

**Model Configuration:**
- `CatBoostClassifier` with explicit categorical feature specification via `Pool` objects
- `loss_function='Logloss'`, `eval_metric='PRAUC'`
- `scale_pos_weight` for class imbalance
- `bootstrap_type='Bernoulli'`
- **Early stopping** with validation set (50 rounds patience)

**Hyperparameter Tuning:**
- **RandomizedSearchCV** with 5-fold StratifiedKFold
- Parameter distributions: `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `border_count`, `bagging_temperature`
- Scoring: `average_precision` (PR-AUC)

**Categorical Feature Handling:**
- Categorical columns identified and converted to pandas `category` dtype
- Explicitly passed to CatBoost via `Pool` objects (avoids one-hot encoding overhead)
- Reduces overfitting risk on small datasets

**Evaluation:**
- Threshold optimization
- SHAP analysis (summary plots, dependence plots)
- Permutation importance
- Test set evaluation

**Results**: Stable performance with good interpretability (Precision ≈ 0.61, Recall ≈ 0.80, F1 ≈ 0.69 after threshold optimization). `incident_severity` remains the strongest predictor.

---

## Common Methodologies Across Notebooks

### Threshold Optimization

All models (except baseline Logistic Regression) undergo threshold optimization:

- **Precision-Recall Curve Analysis**: Evaluates trade-offs across threshold values
- **F1-Maximizing Threshold**: Finds optimal threshold that maximizes F1 score on validation set
- **Optional Recall Constraints**: Can enforce minimum recall requirements for business needs
- **Test Set Application**: Selected threshold applied to final test set evaluation

This ensures models operate at business-aligned decision points rather than default 0.5 threshold.

### Model Interpretability

Both tree-based models (RF, XGBoost, CatBoost) include interpretability analysis:

- **SHAP (SHapley Additive exPlanations)**:
  - Summary plots for global feature importance
  - Dependence plots showing feature interactions
  - Local explanations for individual predictions
  - Helps identify monotonicity and feature interactions

- **Permutation Importance**:
  - Used as secondary validation method
  - Helps rule out data leakage
  - Validates SHAP findings

**Consistent Finding**: `incident_severity` consistently emerges as the strongest predictor across all models, followed by claim amount variables (`total_claim_amount`, `vehicle_claim`, etc.).

---

## Models

Trained models and related outputs are located in:

- **`models/`**: Saved model states (e.g., `xgboost/`, `catboost/`, `random_forest_final/`, `logistic_regression/`)
- **`reports/artifacts/`** and **`reports/figures/`**: Artifacts and figures per notebook/experiment

## Installation & Reproducibility

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Then run the notebooks in order **01 → 04**.  
Results (models/plots/artifacts) will be saved to `models/` and `reports/` respectively.

## Streamlit (Status)

The repository contains `src/streamlit/app.py`, but this file is currently **empty**.  
If a dashboard is planned, it can be added there (dependencies: `streamlit` is already in `requirements.txt`).

## License & Template

See `LICENSE` for license information. Project structure is based on the Cookiecutter Data Science template.
