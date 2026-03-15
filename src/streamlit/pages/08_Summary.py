import streamlit as st
from utils import set_page, hero, section_title, metric_with_bar, BEST_FAMILY_DF, nice_df, COMPARISON_DF

set_page("Summary", "✅")

hero(
    "✅ Project Summary",
    "A compact final view of the strongest models, their trade-offs and the main lessons from the project."
)

best = BEST_FAMILY_DF.sort_values(["Fraud F1", "Accuracy"], ascending=False).iloc[0]

section_title("🏆 Best overall model in this portfolio view")
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
cols = st.columns(3)
for col, (_, row) in zip(cols, BEST_FAMILY_DF.iterrows()):
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
    nice_df(COMPARISON_DF.sort_values(["Fraud F1", "PR-AUC"], ascending=False)),
    use_container_width=True,
    hide_index=True,
)

section_title("🧠 Most influential signal")
st.code("incident_severity", language="text")
st.markdown(
    """
Across the interpretation outputs, `incident_severity` stands out as the most consistent high-impact feature.  
That is a good sign because it is operationally plausible: severe incidents are more strongly associated with suspicious claim behaviour in this dataset.
"""
)

section_title("⚙️ Threshold takeaways")
st.markdown(
    """
- **Random Forest** benefited the most from threshold tuning.  
- **XGBoost** stayed strong, but threshold tuning did not improve the fraud-class F1 in the same way.  
- **CatBoost** already operated near a strong default point, so threshold adjustment added little value.
"""
)

section_title("✅ Final takeaways")
st.markdown(
    """
- There is **no single universally best model** across every metric.  
- **CatBoost** gives the strongest fraud-class F1 and accuracy in this public comparison.  
- **Random Forest** reaches the strongest observed PR-AUC in the notebook-derived results.  
- **XGBoost** offers a very competitive middle ground with a clean, easy-to-explain setup.  
- The strongest part of the project is not one score: it is the combination of **EDA, feature engineering, threshold logic and interpretability**.
"""
)