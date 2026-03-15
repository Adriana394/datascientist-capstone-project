import io
import streamlit as st

from utils import (
    set_page,
    hero,
    load_csv,
    show_figure,
    art_path,
    data_path,
)

set_page("EDA Overview", "📊")

hero(
    "📊 Exploratory Data Analysis",
    "This section introduces the dataset, its quality, and the first fraud-related patterns before moving into deeper analysis."
)

original_data = load_csv(data_path("insurance_claims.csv"))
missing_df = load_csv(art_path("eda", "missing_values.csv"))
stats_df = load_csv(art_path("eda", "stats_data.csv"))

tabs = st.tabs(["Overview", "Data", "Target Distribution", "Correlation"])

# =========================================================
# Tab 1 - Overview
# =========================================================
with tabs[0]:
    st.subheader("EDA Overview")

    st.markdown(
        """
Exploratory Data Analysis (EDA) is the first and most important step before modeling.  
It helps uncover the overall structure of the dataset, identify data quality issues, and reveal first patterns related to fraudulent and non-fraudulent claims.

In this section, the focus is on:

- understanding the dataset structure  
- checking data quality and missing values  
- exploring the target distribution  
- identifying first relationships between variables
"""
    )

    st.markdown(
        """
### What is covered here

1. **Data Overview** – dataset dimensions, data types, and quality checks  
2. **Target Distribution** – fraud vs non-fraud balance  
3. **Correlation Analysis** – first structural relationships between numerical variables  

This first EDA page is intended as a high-level introduction before moving into deeper variable-level analysis.
"""
    )

    st.markdown(
        """
### Why this page matters

This page sets the foundation for the rest of the project.  
It shows early on that this is **not** a standard balanced classification problem and that later modeling choices need to account for:

- minority-class fraud detection  
- uneven feature distributions  
- possible redundancy between claim-related variables
"""
    )

    st.info(
        "This page focuses on the overall dataset structure and the first descriptive patterns. More detailed variable analysis is shown on the next EDA page."
    )

# =========================================================
# Tab 2 - Data
# =========================================================
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

        num_cols = original_data.select_dtypes(include=["number"]).shape[1]
        cat_cols = original_data.select_dtypes(exclude=["number"]).shape[1]

        kpi_cols = st.columns(6)
        with kpi_cols[0]:
            st.metric("Rows", f"{n_rows}")
        with kpi_cols[1]:
            st.metric("Columns", f"{n_cols}")
        with kpi_cols[2]:
            st.metric("Numerical", f"{num_cols}")
        with kpi_cols[3]:
            st.metric("Categorical", f"{cat_cols}")
        with kpi_cols[4]:
            if fraud_rate is not None:
                st.metric("Fraud Rate (Y)", f"{fraud_rate * 100:.1f}%")
            else:
                st.metric("Fraud Rate (Y)", "n/a")
        with kpi_cols[5]:
            st.metric("Overall Missing", f"{missing_pct * 100:.1f}%")
    else:
        st.warning("Could not load `insurance_claims.csv` from `data/`.")

    st.markdown("### Original Dataset")
    if not original_data.empty:
        st.dataframe(original_data.head(10), use_container_width=True)
        st.caption(
            f"The original dataset contains {original_data.shape[0]} rows and {original_data.shape[1]} columns."
        )
    else:
        st.info("Original dataset preview is not available.")

    st.markdown("### Dataset Info")
    if not original_data.empty:
        buf = io.StringIO()
        original_data.info(buf=buf)
        st.code(buf.getvalue(), language="bash")
    else:
        st.info("Data info could not be generated because the dataset is missing.")

    st.markdown("### Missing Values")
    if not missing_df.empty:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.info("No exported missing-values artifact found in `reports/artifacts/eda/`.")

    st.markdown("### Summary Statistics")
    if not stats_df.empty:
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("No exported `stats_data.csv` artifact found in `reports/artifacts/eda/`.")

# =========================================================
# Tab 3 - Target Distribution
# =========================================================
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
The target variable `fraud_reported` is clearly imbalanced:  
most claims are labeled as non-fraudulent, while only a minority are labeled as fraudulent.

This matters because it makes **accuracy alone misleading**.  
It also explains why later model evaluation focuses more strongly on **PR-AUC**, **fraud-class precision/recall/F1**, and **threshold tuning**.
"""
        )

        st.bar_chart(target_counts.rename(index={"N": "Non-fraud", "Y": "Fraud"}))

        st.markdown("### Saved Plot from the Notebook")
        show_figure(
            "01_data_exploration",
            "target_distribution.png",
            caption="Target Distribution - Fraud Reported",
        )
    else:
        st.info("Target column `fraud_reported` is not available in the loaded data.")

# =========================================================
# Tab 4 - Correlation
# =========================================================
with tabs[3]:
    st.subheader("📌 Correlation Analysis")

    st.markdown(
        """
The heatmap provides an initial overview of linear relationships between numerical features.  
It is useful for spotting broad structure, likely redundancy, and clusters of variables that move together.

This is especially relevant in this dataset because several claim-related variables are naturally linked to one another.
"""
    )

    show_figure(
        "01_data_exploration",
        "heatmap.png",
        caption="Correlation Heatmap",
    )

    st.markdown(
        """
### Interpretation

The heatmap is best understood as a **high-level orientation tool**.

In this dataset, it is especially useful for identifying:

- likely overlap between total claim amount and its claim components  
- broader financial structure in claim-related features  
- customer-related relationships such as age and duration as a customer  

These patterns matter because they help flag potential redundancy and motivate the later feature-engineering choices shown on the next EDA page.
"""
    )