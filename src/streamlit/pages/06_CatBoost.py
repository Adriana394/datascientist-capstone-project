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

set_page("CatBoost", "🐈")

hero(
    "🐈 CatBoost",
    "This section covers the CatBoost model, validation-based threshold checks, and interpretation of the final boosted-tree setup."
)

# -------------------------------------------------------------------
# Notebook-aligned result snapshot
# IMPORTANT:
# The notebook's printed tuned-threshold evaluation and the exported
# `catboost_threshold_comparison.csv` are inconsistent.
# To keep the app factually aligned with the notebook text/output,
# the comparison below follows the printed evaluation and the notebook note
# that 0.50 remains the practical final threshold.
# -------------------------------------------------------------------
results_df = pd.DataFrame(
    [
        {
            "Model": "CatBoost",
            "Variant": "Default @ 0.50",
            "PR-AUC": 0.5552,
            "Fraud Precision": 0.6250,
            "Fraud Recall": 0.8000,
            "Fraud F1": 0.7018,
            "Accuracy": 0.8300,
            "Threshold": 0.50,
        },
        {
            "Model": "CatBoost",
            "Variant": "Validation-tuned candidate",
            "PR-AUC": 0.5552,
            "Fraud Precision": 0.6250,
            "Fraud Recall": 0.8000,
            "Fraud F1": 0.7018,
            "Accuracy": 0.8300,
            "Threshold": 0.5117,
        },
    ]
)

best_params_df = pd.DataFrame(
    [
        {
            "rsm": 0.85,
            "random_strength": 1,
            "iterations": 200,
            "l2_leaf_reg": 50,
            "depth": 5,
            "subsample": 0.7,
            "min_data_in_leaf": 50,
            "learning_rate": 0.03,
        }
    ]
)

cat_report_default = load_csv(
    art_path("04_model_implementation_catboost", "classification_report_catboost_t05.csv")
)
cat_report_tuned = load_csv(
    art_path("04_model_implementation_catboost", "classification_report_catboost_tuned.csv")
)
cat_perm_imp = load_csv(
    art_path("04_model_implementation_catboost", "permutation_importance_catboost.csv")
)

tabs = st.tabs([
    "Model Overview",
    "Preparation Steps",
    "CatBoost Classifier",
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
After evaluating XGBoost as the first boosted model, this notebook introduces **CatBoost** as the final boosted-tree model in the project.

CatBoost is particularly well suited to this problem because it is designed for **tabular data with categorical variables**, which makes the training pipeline cleaner and more compact.
"""
    )

    st.markdown(
        """
### Why CatBoost?

CatBoost is a strong candidate here because it:

- handles categorical variables **natively**
- avoids the need for one-hot encoding and scaling
- remains robust on structured, medium-sized tabular data
- supports strong interpretability through feature importance and SHAP
"""
    )

    st.markdown("### Notebook-Aligned Result Snapshot")
    st.dataframe(nice_df(results_df), use_container_width=True, hide_index=True)

    st.info(
        "CatBoost produces the strongest fraud-class F1 and the highest accuracy among the displayed full-model operating points, but its PR-AUC remains below the best Random Forest and XGBoost results."
    )

# =========================================================
# Tab 2 - Preparation Steps
# =========================================================
with tabs[1]:
    st.subheader("Preparation Steps")

    st.markdown(
        """
The CatBoost notebook uses the same cleaned feature set as XGBoost, but with a simpler training pipeline.

Because CatBoost handles categorical variables internally, no manual scaling or one-hot encoding is required.
That is one of the main practical advantages of this model family.
"""
    )

    st.markdown("### Native Categorical Handling")
    st.code(CODE_SNIPPETS["catboost_native_cat"], language="python")

    st.markdown("### Best Hyperparameters")
    st.dataframe(best_params_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
### Training Design

- **Randomized hyperparameter search**
- optimization target: **PRAUC**
- class imbalance handled with **`scale_pos_weight`**
- early stopping via validation set
"""
    )

    with st.expander("Additional Training Notes"):
        st.markdown(
            """
The notebook explicitly keeps the same general train/test design as the other model pages:

- **90/10 stratified split**
- no extra outlier treatment before CatBoost
- categorical columns converted to **`category`** dtype for native handling

This is sensible because boosting trees are relatively robust to moderate outliers, and the goal here is to preserve potentially fraud-relevant extreme claim patterns rather than remove them automatically.
"""
        )

# =========================================================
# Tab 3 - CatBoost Classifier
# =========================================================
with tabs[2]:
    st.subheader("CatBoost Classifier")

    st.markdown(
        """
This section reports the CatBoost model at the **default threshold of 0.50**.

At this operating point, CatBoost achieves the strongest fraud-class F1 among the full-feature model pages in the portfolio.
"""
    )

    if not cat_report_default.empty:
        st.markdown("### Classification Report")
        st.dataframe(cat_report_default, use_container_width=True)
    else:
        st.info("`classification_report_catboost_t05.csv` not found.")

    show_figure(
        "04_model_implementation_catboost",
        "pr_curve_cat_threshold_0_5.png",
        caption="Precision–Recall Curve (CatBoost at threshold 0.50)",
    )

    st.markdown(
        """
### Interpretation

At the default threshold of **0.50**, CatBoost reaches:

- **PR-AUC = 0.5552**
- **Fraud precision = 0.6250**
- **Fraud recall = 0.8000**
- **Fraud F1 = 0.7018**
- **Accuracy = 0.8300**

This gives CatBoost the strongest fraud-class F1 and accuracy among the full-model comparisons shown in the app.
"""
    )

# =========================================================
# Tab 4 - Threshold Optimization
# =========================================================
with tabs[3]:
    st.subheader("Threshold Optimization")

    st.markdown(
        """
As with the other model families, the notebook checks whether a validation-tuned threshold improves CatBoost’s fraud-detection operating point.

The threshold is selected on a validation split under a minimum precision constraint.
"""
    )

    st.markdown("### Threshold Optimization Logic")
    st.code(CODE_SNIPPETS["threshold_tuning"], language="python")

    st.markdown("### Default vs Validation-Tuned Threshold")
    st.dataframe(nice_df(results_df), use_container_width=True, hide_index=True)

    show_figure(
        "04_model_implementation_catboost",
        "threshold_tuning_catboost.png",
        caption="Threshold Tuning – CatBoost",
    )

    if not cat_report_tuned.empty:
        st.markdown("### Classification Report at the Validation-Tuned Threshold")
        st.dataframe(cat_report_tuned, use_container_width=True)

    st.markdown(
        """
### Interpretation

Validation tuning identifies a threshold of approximately **0.5117**.

However, in the notebook’s printed test evaluation, this does **not** improve the CatBoost operating point:
the test report remains effectively unchanged relative to the default threshold.

That is why the most honest conclusion for this notebook is:

- the default threshold of **0.50** is already close to the best practical operating point
- threshold tuning adds **little to no benefit** here
- the CatBoost result is strong because of the model itself, not because of threshold adjustment
"""
    )

# =========================================================
# Tab 5 - Model Interpretation
# =========================================================
with tabs[4]:
    st.subheader("Model Interpretation")

    st.markdown(
        """
To understand how CatBoost arrives at its predictions, the notebook uses:

- **Permutation Importance**
- **SHAP Summary Plot**

This is especially useful because CatBoost combines strong predictive performance with a fairly compact preprocessing pipeline.
"""
    )

    show_figure(
        "04_model_implementation_catboost",
        "permutation_importance_catboost.png",
        caption="Permutation Importance – CatBoost",
    )

    if not cat_perm_imp.empty:
        st.markdown("### Permutation Importance Table")
        st.dataframe(cat_perm_imp, use_container_width=True, hide_index=True)

    show_figure(
        "04_model_implementation_catboost",
        "shap_summary_catboost.png",
        caption="SHAP Summary Plot – CatBoost",
    )

    st.markdown(
        """
### Interpretation

The interpretation outputs again point back to the same core fraud signals seen throughout the project.

Most notably:

- **`incident_severity`** remains highly influential
- claim-related monetary features continue to matter
- the CatBoost model is not relying on an implausible or opaque signal pattern

That consistency strengthens the credibility of the overall workflow and helps explain why CatBoost performs so well in the final comparison.
"""
    )