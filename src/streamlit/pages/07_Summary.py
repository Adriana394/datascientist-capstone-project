import streamlit as st
import pandas as pd

from utils import set_page, hero, section_title, metric_with_bar, nice_df

set_page("Summary", "✅")

hero(
    "✅ Project Summary",
    "A compact final view of the strongest models, their trade-offs, and the main lessons from the project."
)

# -------------------------------------------------------------------
# Notebook-aligned summary values
# -------------------------------------------------------------------
family_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest",
            "Variant": "SMOTE tuned",
            "PR-AUC": 0.6190,
            "Fraud Precision": 0.5250,
            "Fraud Recall": 0.8400,
            "Fraud F1": 0.6462,
            "Accuracy": 0.7700,
            "Threshold": 0.3091,
        },
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
            "Model": "CatBoost",
            "Variant": "Default @ 0.50",
            "PR-AUC": 0.5552,
            "Fraud Precision": 0.6250,
            "Fraud Recall": 0.8000,
            "Fraud F1": 0.7018,
            "Accuracy": 0.8300,
            "Threshold": 0.50,
        },
    ]
)

# Full comparison table across all notebook-relevant variants shown in the app
comparison_df = pd.DataFrame(
    [
        {
            "Model": "Logistic Regression",
            "Variant": "Baseline @ 0.50",
            "PR-AUC": 0.3749,
            "Fraud Precision": 0.2875,
            "Fraud Recall": 0.9200,
            "Fraud F1": 0.4381,
            "Accuracy": 0.4100,
            "Threshold": 0.50,
        },
        {
            "Model": "Random Forest",
            "Variant": "Base @ 0.50",
            "PR-AUC": 0.5588,
            "Fraud Precision": 0.5000,
            "Fraud Recall": 0.1600,
            "Fraud F1": 0.2424,
            "Accuracy": 0.7500,
            "Threshold": 0.50,
        },
        {
            "Model": "Random Forest",
            "Variant": "Base tuned",
            "PR-AUC": 0.5430,
            "Fraud Precision": 0.4651,
            "Fraud Recall": 0.8000,
            "Fraud F1": 0.5882,
            "Accuracy": 0.7200,
            "Threshold": 0.2500,
        },
        {
            "Model": "Random Forest",
            "Variant": "SMOTE tuned",
            "PR-AUC": 0.6190,
            "Fraud Precision": 0.5250,
            "Fraud Recall": 0.8400,
            "Fraud F1": 0.6462,
            "Accuracy": 0.7700,
            "Threshold": 0.3091,
        },
        {
            "Model": "Random Forest",
            "Variant": "SMOTE + feature selection",
            "PR-AUC": 0.6491,
            "Fraud Precision": 0.4545,
            "Fraud Recall": 0.8000,
            "Fraud F1": 0.5797,
            "Accuracy": 0.7100,
            "Threshold": 0.2638,
        },
        {
            "Model": "Random Forest",
            "Variant": "Final refit",
            "PR-AUC": 0.6353,
            "Fraud Precision": 0.4878,
            "Fraud Recall": 0.8000,
            "Fraud F1": 0.6061,
            "Accuracy": 0.7400,
            "Threshold": 0.3091,
        },
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

# Best family-level operating point for the final headline card
best = family_df.sort_values(["Fraud F1", "Accuracy"], ascending=False).iloc[0]

section_title("🏆 Best performing Model in Fraud Detection")
st.markdown(f"### {best['Model']} — {best['Variant']}")
cols_best = st.columns(4)
with cols_best[0]:
    metric_with_bar("F1 (fraud)", float(best["Fraud F1"]))
with cols_best[1]:
    metric_with_bar("Recall (fraud)", float(best["Fraud Recall"]))
with cols_best[2]:
    metric_with_bar("Precision (fraud)", float(best["Fraud Precision"]))
with cols_best[3]:
    metric_with_bar("Accuracy", float(best["Accuracy"]))
st.caption(f"Threshold shown on this page: {float(best['Threshold']):.2f}")

section_title("📊 Quick comparison across model families")
st.markdown(
    """
These cards show the strongest representative operating point for each model family in the portfolio app.

For Random Forest, the family is represented by the **SMOTE-tuned** variant because it gives the strongest fraud-detection balance among the RF operating points, even though another RF variant reaches a higher PR-AUC.
"""
)
cols = st.columns(3)
for col, (_, row) in zip(cols, family_df.iterrows()):
    with col:
        st.markdown(f"#### {row['Model']}")
        st.caption(row['Variant'])
        metric_with_bar("F1 (fraud)", float(row["Fraud F1"]))
        metric_with_bar("Recall (fraud)", float(row["Fraud Recall"]))
        metric_with_bar("Precision (fraud)", float(row["Fraud Precision"]))
        metric_with_bar("Accuracy", float(row["Accuracy"]))
        st.caption(f"Threshold: {float(row['Threshold']):.2f}")

section_title("📋 Full comparison table")
st.dataframe(
    nice_df(comparison_df.sort_values(["Fraud F1", "PR-AUC"], ascending=False)),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Note: the CatBoost validation-tuned candidate is shown for completeness. In the notebook’s printed test evaluation, it does not improve the practical test outcome relative to the default threshold."
)

section_title("🧠 Most consistent high-impact signal")
st.code("incident_severity", language="text")
st.markdown(
    """
Across the interpretation outputs, `incident_severity` stands out as the most consistently dominant signal.

That is a good sign because it is operationally plausible: higher incident severity is repeatedly associated with stronger fraud risk, and this pattern remains visible across Random Forest, XGBoost, and CatBoost interpretation views.
"""
)

section_title("⚙️ Threshold takeaways")
st.markdown(
    """
- **Random Forest** benefited the most from threshold tuning.  
- **XGBoost** stayed strong, but threshold tuning did not improve the fraud-class F1 on the test set.  
- **CatBoost** already operated near a strong default point, so threshold adjustment added little to no practical value.
"""
)

section_title("✅ Final takeaways")
st.markdown(
    """
- There is **no single universally best model** across every metric.  
- **CatBoost** gives the strongest fraud-class F1 and the highest accuracy in this portfolio comparison.  
- **Random Forest + feature selection** reaches the strongest observed PR-AUC in the notebook-derived results.  
- **Random Forest + SMOTE (tuned)** is the strongest operational Random Forest variant in the family-level comparison.  
- **XGBoost** offers a very competitive middle ground with strong default-threshold performance and a clean, explainable setup.  
- The strongest part of the project is not one score alone: it is the combination of **EDA, feature engineering, threshold logic, and interpretability**.
"""
)