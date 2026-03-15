import io
import pandas as pd
import streamlit as st

from utils import set_page, hero, load_csv, show_figure, art_path, data_path

set_page("EDA Overview", "📊")

hero(
    "📊 Exploratory Data Analysis",
    "This section introduces the dataset, its quality, and the first fraud-related patterns before moving into deeper feature exploration."
)

# -------------------------------------------------------------------
# Load data from your exported notebook artifacts
# -------------------------------------------------------------------
original_data = load_csv(data_path("insurance_claims.csv"))
cleaned_data = load_csv(art_path("eda", "claim_data_cleaned.csv"))
missing_df = load_csv(art_path("eda", "missing_values.csv"))

tabs = st.tabs(["Overview", "Data", "Target Distribution"])

# -------------------------------------------------------------------
# Tab 1 - Overview
# -------------------------------------------------------------------
with tabs[0]:
    st.subheader("EDA Overview")

    st.markdown(
        """
Exploratory Data Analysis — short **EDA** — is the first and most critical step in any data science workflow.  
It helps to **understand the dataset** before moving on to modeling or prediction.

During EDA, the goal is to:

- explore **data quality**, **structure**, and **relationships** between variables  
- detect **missing values**, **outliers**, and **inconsistencies**  
- evaluate the **distribution** of numerical features and the **balance** of categorical levels  
- identify potential **biases** and **redundancies**  
- gain first insights into which features may influence the target variable `fraud_reported`
"""
    )

    st.markdown(
        """
### 🔍 What is done in this EDA

1. **Data Quality & Cleaning** – Checking for missing values, invalid entries and duplicate issues  
2. **Univariate Analysis** – Inspecting numerical and categorical feature distributions  
3. **Target-wise Exploration** – Comparing fraudulent and non-fraudulent claims  
4. **Correlation Analysis** – Identifying redundant information and relationship structures  
5. **Feature Engineering** – Creating and refining variables that may improve fraud detection
"""
    )

    st.markdown(
        """
### ⚠️ Challenges

- **Imbalanced target:** Fraud cases are relatively rare  
- **Data quality issues:** Some features contain inconsistencies or missing values  
- **Redundant information:** Several claim-related variables are strongly related  
- **Non-normal distributions:** Many numerical variables are skewed or contain outliers
"""
    )

    st.markdown(
        """
### 💡 Why this matters

A solid EDA improves both the **quality of preprocessing decisions** and the **interpretability of later model results**.  
It helps ensure that later modeling is based on informed choices rather than trial and error.
"""
    )

# -------------------------------------------------------------------
# Tab 2 - Data
# -------------------------------------------------------------------
with tabs[1]:
    st.subheader("📁 Data Overview & Quality")

    if not original_data.empty:
        n_rows, n_cols = original_data.shape
        fraud_rate = (
            (original_data["fraud_reported"] == "Y").mean()
            if "fraud_reported" in original_data.columns
            else None
        )
        missing_total = original_data.isna().sum().sum()
        missing_pct = missing_total / (n_rows * n_cols) if n_rows and n_cols else 0

        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.metric("Rows", f"{n_rows}")
        with kpi_cols[1]:
            st.metric("Columns", f"{n_cols}")
        with kpi_cols[2]:
            if fraud_rate is not None:
                st.metric("Fraud Rate (Y)", f"{fraud_rate * 100:.1f}%")
            else:
                st.metric("Fraud Rate (Y)", "n/a")
        with kpi_cols[3]:
            st.metric("Overall Missing", f"{missing_pct * 100:.1f}%")
    else:
        st.warning("Could not load `insurance_claims.csv` from reports/artifacts/eda/.")

    st.markdown("## Original Dataset")
    if not original_data.empty:
        st.dataframe(original_data.head(10), use_container_width=True)
        st.caption(
            f"The original dataset contains {original_data.shape[0]} rows and {original_data.shape[1]} columns."
        )
    else:
        st.info("Original dataset preview is not available.")

    st.markdown("## Data Info")
    if not original_data.empty:
        buf = io.StringIO()
        original_data.info(buf=buf)
        st.code(buf.getvalue(), language="bash")
    else:
        st.info("Data info could not be generated because the dataset is missing.")

    st.markdown("## Missing Values")
    if not missing_df.empty:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.info("No exported missing-values table found in `reports/artifacts/eda/`.")

    st.markdown(
        """
The dataset contains a mix of **categorical** and **numerical** variables.  
A few fields contain missing values or inconsistent entries, which makes this EDA step important before any modeling begins.
"""
    )

# -------------------------------------------------------------------
# Tab 3 - Target Distribution
# -------------------------------------------------------------------
with tabs[2]:
    st.subheader("🎯 Target Distribution")

    if not original_data.empty and "fraud_reported" in original_data.columns:
        target_counts = original_data["fraud_reported"].value_counts().sort_index()
        target_ratio = original_data["fraud_reported"].value_counts(normalize=True).sort_index()

        cols_t = st.columns(3)
        with cols_t[0]:
            st.metric("Total Claims", int(target_counts.sum()))
        with cols_t[1]:
            st.metric(
                "Non-fraud (N)",
                f"{target_counts.get('N', 0)} ({target_ratio.get('N', 0) * 100:.1f}%)"
            )
        with cols_t[2]:
            st.metric(
                "Fraud (Y)",
                f"{target_counts.get('Y', 0)} ({target_ratio.get('Y', 0) * 100:.1f}%)"
            )

        st.markdown(
            """
The target variable `fraud_reported` is **clearly imbalanced**:  
most claims are labeled as non-fraudulent (`N`), while only a minority are labeled as fraudulent (`Y`).

This is typical for real-world fraud detection and one of the key reasons why later model evaluation focuses on **fraud-class metrics**, **PR-AUC**, and **threshold tuning** rather than accuracy alone.
"""
        )

        st.bar_chart(target_counts.rename(index={"N": "Non-fraud", "Y": "Fraud"}))


    else:
        st.info("Target column `fraud_reported` is not available in the loaded data.")