# Insurance Claim Fraud Prediction - Capstone Project
---
*End-to-End Machine Learning Workflow  · Streamlit Dashboard*

This repository contains the code and artifacts for my **Capstone Project** in the **DataScientest Data Scientist Program**. 
The goal is to build an end-to-end fraud detection pipeline for insurance claims, including:

- **EDA & Statistical Testing**
- **Feature Engineering**
- multiple **ML Models**
- **Threshold Optimization**
- **SHAP & Permutation Importance**
- a **Streamlit App** for interactive exploration of results

## Project Structure
---

├── data/<br>                             
├── notebooks/<br>
│   ├── 01_data_exploration.ipynb          
│   ├── 02_model_implementation.ipynb     
│   ├── 03_model_implementation_xgb.ipynb  
│   └── 04_model_implementation_catboost.ipynb<br>    
│
├── models/<br>                          
│
├── fraud_app/                     
│    ├── home.py                    
│    ├── pages/                      
│    └── artifacts/<br>                   
│
|── reports/                     
│   ├── figures                    
│   ├── report.md<br>                       
│   
|
├── requirements.txt<br>  
└── README.md


## 🧠 Problem Description

Insurance fraud is relatively rare but financially damaging.  
The task is to classify each claim as **fraudulent** or **non-fraudulent** using structured customer, incident, and claim information.

### Key Challenges

- **Strong class imbalance** (fraud ≈ 24,7%)  
- **High cost of false negatives**  
- **Need for interpretability** for business adoption  
- Avoiding **data leakage**  
- Handling **mixed data types**, **missing values** & **inconsistent labels**  

---

## 🔍 Exploratory Data Analysis (EDA)  
Notebook: **`01_data_exploration.ipynb`**

The EDA examined:

- Dataset overview & data types  
- Missing values inspection  
- Outlier analysis using IQR  
- Numerical distributions & skewness  
- Categorical feature distributions  
- Correlation heatmaps  
- Statistical tests:  
  - **Mann–Whitney U** for numerical features  
  - **Chi-Square** for categorical features  

### Key Insights

- Fraud is correlated with **high incident severity** and **higher claim amounts**  
- Claim subcomponents (`injury_claim`, `property_claim`, `vehicle_claim`) show **strong multicollinearity** with `total_claim_amount`  
- Outliers occur mainly in `umbrella_limit` — left untouched because tree-based models handle them well  
- Missing categorical data was encoded as **"Unknown"** to preserve uncertainty  
- A new feature **`has_umbrella_policy`** was engineered to capture policy coverage in a clean binary form  

---

## 🤖 Machine Learning Models

All model development is documented in:

- `02_model_implementation.ipynb` — Logistic Regression & Random Forest  
- `03_model_implementation_xgb.ipynb` — XGBoost  
- `04_model_implementation_catboost.ipynb` — CatBoost  

Shared preprocessing pipeline:

- Train/validation/test split  
- **Encoding**
  - **LogReg / Random Forest:** One-Hot Encoding  
  - **CatBoost:** native categorical handling (no OHE)  
  - **XGBoost:** consistent handling as implemented in the notebook (OHE or native categorical)  
- Standardization for LR  
- Stratified K-Fold cross-validation  
- Threshold optimization based on **F1** and optional recall constraint  

---

## 1️⃣ Logistic Regression (Baseline)

Logistic Regression was used as a **transparent linear baseline**.

### Interpretability

- Good interpretability  
- Underfits complex non-linear patterns  
- Relatively low recall, which misses too many fraud cases  
- Useful as a baseline, not suitable for production  

---

## 2️⃣ Random Forest (Base, SMOTE, Feature Selection)  
Notebook: **`02_model_implementation.ipynb`**

### Models Trained

1. **Base Random Forest**  
2. **RF + BorderlineSMOTE**  
3. **RF + BorderlineSMOTE + Feature Selection**

### Results — default Threshold (0.50)

| Model | Precision | Recall | F1 | Accuracy |
|------|----------:|-------:|---:|---------:|
| Base RF | 0.529 | 0.720 | 0.610 | 0.770 |
| RF + SMOTE | 0.600 | 0.360 | 0.450 | 0.780 |
| RF + SMOTE + FS | 0.606 | 0.800 | 0.690 | 0.820 |

### Results after Threshold Optimization

| Model | τ* | Precision | Recall | F1 | Accuracy |
|------|---:|----------:|-------:|---:| ----------:|
| Base RF | 0.480 | 0.528 | 0.760 | 0.623 | 0.77
| RF + SMOTE | 0.380 | 0.571 | 0.8 | 0.667 | 0.80
| RF + SMOTE + FS | 0.578 | 0.593 | 0.640 | 0.615 | 0.80

### Interpretability

- **Permutation Importance** → `incident_severity` dominates  
- **SHAP Summary** → High severity, missing damage info and larger claim amounts ↑ fraud likelihood  

Random Forest delivers strong performance but is surpassed by boosting models.

---

## 3️⃣ XGBoost  
Notebook: **`03_model_implementation_xgb.ipynb`**

XGBoost uses gradient boosting with strong regularization and learns complex non-linear boundaries.

### Results — default Threshold (0.50)

| Precision | Recall | F1 | Accuracy |
|----------:|-------:|---:|---------:|
| 0.636 | 0.840 | 0.724 | 0.84 |

### Results after Threshold Optimization

| τ* | Precision | Recall | F1 | Accuracy |
|--:|----------:|-------:|---:|---------:|
| 0.496 | 0.553 | 0.84 | 0.667 | 0.79 |

### Interpretability

- Excellent balance between recall and precision  
- Outperforms both LR and RF  
- SHAP shows incident severity and claim amounts as dominant drivers  
- A reduced **Top 3- model** retains most performance (F1 ≈ 0.69)  

XGBoost is a high-performing and interpretable model.

---

## 4️⃣ CatBoost
Notebook: **`04_model_implementation_catboost.ipynb`**

CatBoost is designed for **categorical handling**, **ordered boosting**, and **strong generalization** on tabular data.

### Model Setup

- Categorical features passed directly (no OHE)  
- Loss function: `Logloss`  
- Eval metric: `PRAUC`  
- Class imbalance handled via `scale_pos_weight`  
- Randomized Search for hyperparameter tuning  
- Early stopping  

### Results — default Threshold (0.50)

| Precision | Recall | F1 | Accuracy |
|----------:|-------:|---:|---------:|
| **0.625** | **0.80** | **0.702** | **0.83** |

### Results after Threshold Optimization

| τ* | Precision | Recall | F1 | Accuracy |
|---:|----------:|-------:|---:|---------:|
| 0.464 | 0.606 | 0.80 | 0.690 | 0.82 |

### Interpretability

- **SHAP Summary & Dependence Plots**  
- **Permutation Importance**  
- `incident_severity` remains the strongest predictor  
- Policy-related variables and “Unknown” categories influence risk  
- CatBoost provides **stable** and **interpretable** predictions  

---

## 🎯 Threshold Optimization

Because fraud data is imbalanced, default thresholds are suboptimal.  
For each model:

- Precision–Recall analysis  
- F1-maximizing threshold  
- Optionally enforce minimum recall  
- Selected threshold applied to test set  

This ensures realistic, business-driven operating points.

---

## 🧮 Interpretability

### SHAP  
Used for:

- Global feature importance  
- Local explanations  
- Dependence plots  
- Detecting monotonicity and interactions  

### Permutation Importance  
Used as a secondary check to validate SHAP and rule out leakage.

---

## 🖥 Streamlit Dashboard

Available in `fraud_app/`.

### Pages include:

1. **Home + Introduction**
2. **EDA & Data Quality**
3. **Logistic Regression & Random Forest**
4. **XGBoost**
5. **CatBoost**
6. **Feature Reduction**
7. **Final Project Summary**

Includes:

- KPI cards  
- Feature distribution plots  
- Threshold optimization charts  
- SHAP visualizations  
- Model comparison  
- Recommended thresholds  

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
cd fraud_app
streamlit run Home.py
```
---

## Reproducibility

- All modeling steps are documented in Jupyter Notebooks
- Models saved to models/
- Streamlit dynamically loads saved artifacts and plots

---

## Conclusion
This project delivers a complete real-world fraud detection solution:

- EDA & statistical testing

- Advanced ML modeling

- threshold optimization for business alignment

- model explainability

- a polished, interactive Streamlit app

- clear evaluation & comparison

The final model – XGBoost – achieves the strongest balance between recall, precision, stability, and explainability, making it suitable for deployment in real insurance environments.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
