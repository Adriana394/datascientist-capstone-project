## Insurance Fraud Detection – End-to-End ML Project & Streamlit App

This repository contains an **end-to-end machine learning project** for predicting fraudulent insurance claims (`fraud_reported`) – including **EDA**, **feature engineering**, **model comparison**, **threshold optimization**, **interpretability (SHAP, permutation importance)** and a **portfolio-focused Streamlit app**.

The project is designed both as a realistic fraud detection case study and as a polished portfolio piece.

---

## 🎯 Project Goals & Business Context

- **Business goal**: Support fraud analysts in the **early identification of suspicious claims**.
- **ML task**: Binary classification (“Fraud” vs. “No Fraud”) with:
  - a strongly **imbalanced target** (fraud is the minority class),
  - many **categorical features**,
  - potential **data leakage risks**, and
  - high cost of **false negatives** (missed fraud).

**Why this matters**  
Insurance fraud is costly and operationally disruptive, while fraudulent claims represent only a small fraction of all cases. A well‑designed model can help:

- highlight suspicious claims earlier,
- make case screening more consistent,
- and let analysts focus on the highest‑risk cases.

However, this cannot be reduced to a single metric. A useful fraud detection workflow must balance:

- **detection performance**,  
- **false positives**, and  
- **interpretability**  

in a way that remains practical for real decision‑making.

---

## 🧱 Project Structure

```text
datascientist_project_remake/
├── data/                         # Raw and cleaned data (CSV)
├── notebooks/                    # EDA & modeling notebooks (01–04)
├── models/                       # Saved models (per approach)
├── reports/
│   ├── artifacts/                # Metrics, tables, tuning results
│   └── figures/                  # Exported plots for EDA & models
├── src/
│   └── streamlit/
│       ├── Welcome.py            # Streamlit app start page
│       ├── utils.py              # Styling, helpers, path logic, code snippets
│       └── pages/
│           ├── 01_Project_Introduction.py
│           ├── 02_EDA_Overview.py
│           ├── 03_EDA_Deep_Dive.py
│           ├── 04_Baseline_and_Random_Forest.py
│           ├── 05_XGBoost.py
│           ├── 06_CatBoost.py
│           └── 07_Summary.py
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset & Problem Setup

- **Domain**: Property & casualty insurance claims.
- **Target variable**: `fraud_reported` (binary; highly imbalanced).
- **Features**:
  - Numerical claim and customer features,
  - Multiple categorical variables (policy, incident, customer attributes),
  - Engineered features (e.g. claim composition ratios, `has_umbrella_policy`).

**Key challenges**

- Fraud cases are much rarer than non‑fraud cases.
- Plain **accuracy is misleading** under class imbalance.
- The default probability threshold of **0.50** is often not the best business operating point.
- Strong performance is not enough if the model cannot be **explained and justified**.

The project focuses on fraud‑relevant metrics:

- **PR‑AUC** (area under precision–recall curve),
- **precision, recall and F1 for the fraud class**,
- **threshold‑aware evaluation**, not just fixed 0.50.

---

## 🛠️ Tech Stack

**Data & Modeling**

- Python, Pandas, NumPy  
- Scikit‑learn (baseline and Random Forest)  
- XGBoost  
- CatBoost  
- Imbalanced‑learn (SMOTE and class‑imbalance handling)

**Interpretability & Visualization**

- SHAP  
- Permutation importance  
- Matplotlib, Seaborn

**Application Layer**

- Streamlit (multi‑page portfolio app)

---

## 🚀 How to Run the Streamlit App

### 1. Clone the repository

```bash
git clone <[your-repo-url]>
cd datascientist_project
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.\.venv\Scripts\activate    # on Windows PowerShell
# source .venv/bin/activate # on macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare data and artifacts

Ensure that:

- the main dataset (e.g. `insurance_claims.csv`) is available in `data/`, and  
- the analysis notebooks have been run so that exported **figures** and **artifacts** (CSV metrics/tables) exist under `reports/figures/` and `reports/artifacts/`.

The Streamlit pages read these artifacts to display plots, tables and metrics consistent with the notebooks.

### 5. Start the app

From the project root:

```bash
streamlit run src/streamlit/Welcome.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## 🧭 App Navigation

The Streamlit app is intentionally structured like a guided portfolio walkthrough. Use the **sidebar** to navigate:

- **Welcome**  
  High‑level overview of the project, app, and author context.

- **Project Introduction**  
  Business framing, dataset scope, technical overview, workflow, challenges and benefits.

- **EDA Overview**  
  - Dataset structure, types and quality checks  
  - Target distribution and imbalance  
  - First correlation patterns among numerical variables  

- **EDA Deep Dive**  
  - Numerical and categorical distributions  
  - Feature engineering (e.g. claim ratios, `has_umbrella_policy`)  
  - Statistical tests (Mann–Whitney U, chi‑square, Cliff’s delta)  
  - Outlier analysis based on IQR

- **Baseline & Random Forest**  
  - Logistic Regression baseline  
  - Multiple Random Forest variants:
    - base model,
    - SMOTE‑enhanced,
    - feature‑selection variant,
    - final refit  
  - Threshold optimization and permutation importance

- **XGBoost**  
  - Boosted tree model with native categorical handling  
  - Hyperparameter tuning and validation design  
  - Default vs. validation‑tuned threshold  
  - PR curves, classification reports, and SHAP summary

- **CatBoost**  
  - Final boosted‑tree model with native categorical support  
  - Compact preprocessing pipeline  
  - Validation‑based threshold checks  
  - Permutation importance & SHAP interpretation

- **Summary**  
  - Side‑by‑side comparison of the strongest operating points for each model family  
  - Highlight of the **best fraud F1 / accuracy combination** (CatBoost)  
  - Discussion of **threshold takeaways** and **consistent high‑impact features** (e.g. `incident_severity`)

---

## 🧠 Modeling Overview

The project moves deliberately from simple to more complex models, always under class imbalance:

- **Baseline: Logistic Regression**
  - Transparent reference model.
  - Very high fraud recall but low precision and limited practical usability.
  - Serves as a benchmark to evaluate improvements.

- **Random Forest (multiple variants)**
  - Base model at threshold 0.50.
  - SMOTE‑based variant to improve learning on the minority fraud class.
  - Feature‑selection variant to test a more compact feature space.
  - **Threshold‑tuned** versions to explicitly trade off fraud recall vs precision.
  - Final refit trained with the selected configuration.
  - Interpretation via **permutation importance**.

- **XGBoost**
  - Boosted trees with **native categorical handling**.
  - Hyperparameter tuning with PR‑AUC target and `scale_pos_weight`.
  - Default threshold 0.50 provides a very strong operating point.
  - Validation‑based threshold tuning is tested but does **not** improve fraud F1 on the test set.

- **CatBoost**
  - Boosted tree model designed for categorical/tabular data.
  - Clean pipeline (no one‑hot encoding or scaling required).
  - Achieves the **strongest fraud‑class F1 and highest accuracy** among full model operating points in this project.
  - Threshold tuning offers little extra benefit here; the default 0.50 is already near‑optimal.
  - Interpretation again confirms the importance of incident severity and claim‑related monetary features.

---

## 📊 Evaluation & Threshold Tuning

Instead of relying on accuracy and a fixed 0.50 threshold, the project emphasizes:

- **PR‑AUC** as a ranking metric,
- **fraud‑class precision, recall, and F1**,
- explicit **threshold optimization logic** that:
  - enforces a **minimum precision** constraint,
  - then maximizes recall over candidate thresholds.

Key findings:

- **Random Forest** benefits the most from threshold tuning (especially the RF + SMOTE variant).
- **XGBoost** is strong at the default threshold; the tuned threshold does not improve test F1.
- **CatBoost** already operates near a strong default point; validation‑tuned thresholds do not provide a meaningful test‑set improvement.

---

## 🔍 Interpretability

Interpretability is a central part of this project:

- **Permutation importance** is used across Random Forest, XGBoost and CatBoost to compare feature importance.
- **SHAP** summary plots are used for XGBoost and CatBoost to understand how individual features influence predictions.

A key, consistent signal across models is:

- **`incident_severity`** – repeatedly among the top features, which is operationally plausible and strengthens trust in the models.

---

## 🧩 Notebooks vs. App

The **notebooks** under `notebooks/` contain the original, exploratory and modeling workflows:

- Detailed EDA,
- step‑by‑step feature engineering,
- model training and tuning,
- threshold search and interpretation.

The **Streamlit app** in `src/streamlit/` turns these results into:

- a **curated, portfolio‑ready narrative**,
- interactive metric tables and plots loaded from `reports/artifacts/` and `reports/figures/`,
- code snippets demonstrating key pipeline components (e.g., preprocessing, Random Forest + SMOTE pipeline, threshold tuning function, XGBoost and CatBoost setups).

Both are aligned so that the app displays values, figures and tables that match the notebook outputs.

---

## ⚖️ Limitations & Possible Extensions

Current limitations:

- Dataset size and scope are constrained; results should not be treated as production‑ready without further validation.
- Only a fixed set of models and hyperparameter ranges is explored.
- Cost‑sensitive optimization (e.g. expected monetary cost of errors) is modeled implicitly via thresholds, not via an explicit cost function.

Potential extensions:

- Add **cost‑based** or **utility‑based** evaluation.
- Integrate **calibration checks** and calibration plots.
- Explore **additional model families** (e.g. LightGBM, simple neural nets) for comparison.
- Deploy a lightweight **API** and connect the model to a simple case‑triage UI.

---

## 🤝 About the Project

This project and app were built as part of a **Data Science capstone portfolio**.  
The goal is **not** to simulate a full production fraud‑detection system, but to show a **structured, interpretable and business‑aware ML workflow** from:

- data exploration and feature engineering,  
- through class‑imbalance‑aware modeling and threshold tuning,  
- to comparative evaluation and interpretability.

If you are reviewing this repository as part of an application or portfolio review, the **Streamlit app** is the recommended starting point to explore the project.
