import streamlit as st
import pandas as pd

from utils import (
    set_page,
    hero,
    nice_df,
    CODE_SNIPPETS,
    show_figure,
    load_csv,
    art_path,
)

set_page("XGBoost", "🚀")

hero(
    "🚀 XGBoost",
    "This section covers the boosted-tree XGBoost model, validation-based threshold tuning, and model interpretation."
)

# -------------------------------------------------------------------
# Notebook-aligned result snapshot
# -------------------------------------------------------------------
results_df = pd.DataFrame(
    [
        {
            "Model": "XGBoost",
            "Variant": "Default @ 0.50",
            "PR-AUC": 0.6043,
            "Fraud Precision": 0.5882,
            "Fraud Recall": 0.8000,
            "Fraud F1": 0.6780,
            "Accuracy": 0.8100,
            "Threshold": 0.50,
        },
        {
            "Model": "XGBoost",
            "Variant": "Validation-tuned",
            "PR-AUC": 0.6043,
            "Fraud Precision": 0.5405,
            "Fraud Recall": 0.8000,
            "Fraud F1": 0.6452,
            "Accuracy": 0.7800,
            "Threshold": 0.3909,
        },
    ]
)

best_params_df = pd.DataFrame(
    [
        {
            "colsample_bytree": 1.0,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_child_weight": 10,
            "n_estimators": 300,
            "subsample": 1.0,
        }
    ]
)

xgb_report_default = load_csv(
    art_path("03_model_implementation_xgb", "classification_report_xgb_threshold_0_5.csv")
)
xgb_report_tuned = load_csv(
    art_path("03_model_implementation_xgb", "classification_report_xgb_tuned.csv")
)
xgb_threshold_comparison = load_csv(
    art_path("03_model_implementation_xgb", "xgb_threshold_comparison.csv")
)
xgb_perm_imp = load_csv(
    art_path("03_model_implementation_xgb", "perm_imp.csv")
)

tabs = st.tabs([
    "Model Overview",
    "Preparation Steps",
    "XGBoost Classifier",
    "Threshold Optimization",
    "Model Interpretation",
])

# =========================================================
# Tab 1 - Model Overview
# =========================================================
with tabs[0]:
    st.subheader("Model Overview")

    st.markdown(
        """
After the baseline and Random Forest experiments, this notebook evaluates **XGBoost** as a stronger boosted-tree model for the fraud detection task.

The goal is to test whether a more flexible boosting approach can improve fraud detection performance while keeping the workflow interpretable and operationally useful.
"""
    )

    st.markdown(
        """
### Why XGBoost?

XGBoost is a strong choice for tabular fraud data because it can:

- capture **non-linear relationships** and interactions  
- handle **class imbalance** through `scale_pos_weight`  
- work efficiently on relatively **small structured datasets**  
- support **native categorical handling** in this setup
"""
    )

    st.markdown("### Notebook-Aligned Result Snapshot")
    st.dataframe(nice_df(results_df), use_container_width=True, hide_index=True)

    st.info(
        "XGBoost delivers one of the cleanest overall operating points in the project, but threshold tuning does not improve the fraud-class F1 on the test set."
    )

# =========================================================
# Tab 2 - Preparation Steps
# =========================================================
with tabs[1]:
    st.subheader("Preparation Steps")

    st.markdown(
        """
The XGBoost notebook builds on the cleaned and reduced feature set from the earlier workflow.

Compared with the baseline and Random Forest pages, the preprocessing here is simpler because categorical variables are handled natively.  
That reduces manual encoding overhead and keeps the feature representation closer to the original data structure.
"""
    )

    st.markdown("### Native Categorical Handling")
    st.code(CODE_SNIPPETS["xgb_native_cat"], language="python")

    st.markdown("### Best Hyperparameters")
    st.dataframe(best_params_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
### Training Design

- **5-fold stratified cross-validation**
- optimization target: **average precision / PR-AUC**
- class imbalance handled with **`scale_pos_weight`**
- best validation result: **CV PR-AUC = 0.5528**
"""
    )

    with st.expander("Overfitting Check"):
        st.markdown(
            """
The notebook compares training and test performance at the same threshold of 0.50.

- **Train PR-AUC = 0.7206**
- **Test PR-AUC = 0.6043**

This indicates a **moderate generalization gap**, but not a catastrophic one.  
The selected configuration is regularized enough to justify continuing with threshold tuning rather than discarding the model.
"""
        )

# =========================================================
# Tab 3 - XGBoost Classifier
# =========================================================
with tabs[2]:
    st.subheader("XGBoost Classifier")

    st.markdown(
        """
This section reports the full-feature XGBoost model at the **default threshold of 0.50**.

The default threshold gives the cleanest operating point in the notebook:
it combines strong minority-class recall with better precision than the earlier baseline models.
"""
    )

    if not xgb_report_default.empty:
        st.markdown("### Classification Report")
        st.dataframe(xgb_report_default, use_container_width=True)
    else:
        st.info("`classification_report_xgb_threshold_0_5.csv` not found.")

    show_figure(
        "03_model_implementation_xgb",
        "pr_curve_xgb_threshold_0_5.png",
        caption="Precision–Recall Curve (XGBoost at threshold 0.50)",
    )

    st.markdown(
        """
### Interpretation

At the default threshold of **0.50**, XGBoost reaches:

- **PR-AUC = 0.6043**
- **Fraud precision = 0.5882**
- **Fraud recall = 0.8000**
- **Fraud F1 = 0.6780**
- **Accuracy = 0.8100**

That makes it one of the strongest overall trade-offs in the project.
"""
    )

# =========================================================
# Tab 4 - Threshold Optimization
# =========================================================
with tabs[3]:
    st.subheader("Threshold Optimization")

    st.markdown(
        """
As in the Random Forest notebook, the XGBoost page checks whether moving away from the default threshold of **0.50** improves the fraud-detection operating point.

The threshold is selected on a **validation split** under a minimum precision constraint.
"""
    )

    st.markdown("### Threshold Optimization Logic")
    st.code(CODE_SNIPPETS["threshold_tuning"], language="python")

    if not xgb_threshold_comparison.empty:
        st.markdown("### Default vs Validation-Tuned Threshold")
        st.dataframe(xgb_threshold_comparison, use_container_width=True, hide_index=True)
    else:
        st.dataframe(nice_df(results_df), use_container_width=True, hide_index=True)

    show_figure(
        "03_model_implementation_xgb",
        "threshold_tuning_xgb.png",
        caption="Threshold Tuning – XGBoost",
    )

    if not xgb_report_tuned.empty:
        st.markdown("### Classification Report at the Tuned Threshold")
        st.dataframe(xgb_report_tuned, use_container_width=True)

    st.markdown(
        """
### Interpretation

Validation tuning selects a threshold of approximately **0.3909**.

However, on the **test set** this change does **not** improve the fraud-class F1:

- fraud recall stays at **0.80**
- fraud precision drops from **0.5882** to **0.5405**
- fraud F1 drops from **0.6780** to **0.6452**

That is an important and honest result: threshold tuning is useful to test, but it is **not automatically beneficial**.
For this notebook, the **default threshold of 0.50 remains the stronger practical choice**.
"""
    )

# =========================================================
# Tab 5 - Model Interpretation
# =========================================================
with tabs[4]:
    st.subheader("Model Interpretation")

    st.markdown(
        """
To understand why XGBoost flags certain claims as suspicious, the notebook uses:

- **Permutation Importance**
- **SHAP Summary Plot**

Together, these help verify that the model relies on meaningful fraud-related signals rather than arbitrary noise.
"""
    )

    show_figure(
        "03_model_implementation_xgb",
        "permutation_importance_xgb.png",
        caption="Permutation Importance – XGBoost",
    )

    if not xgb_perm_imp.empty:
        st.markdown("### Permutation Importance Table")
        st.dataframe(xgb_perm_imp, use_container_width=True, hide_index=True)

    show_figure(
        "03_model_implementation_xgb",
        "SHAP_summary.png",
        caption="SHAP Summary Plot – XGBoost",
    )

    st.markdown(
        """
### Interpretation

The interpretation outputs are consistent with the rest of the project:

- **`incident_severity`** remains one of the strongest drivers
- claim-size-related variables continue to matter
- the learned signal aligns with the earlier EDA and feature-engineering steps

That coherence is important because it shows the model is not only scoring well, but also learning from features that are operationally plausible in a fraud context.
"""
    )