import streamlit as st
from utils import set_page, hero, metric_row, CODE_SNIPPETS, img

set_page("CatBoost", "🐈")

hero(
    "CatBoost",
    "CatBoost is evaluated as another tree-based model with strong native categorical support and a compact training pipeline."
)

metric_row([
    {"label": "Best search config", "value": "depth=5"},
    {"label": "Test PR-AUC", "value": f"{0.5552:.4f}"},
    {"label": "Fraud F1", "value": f"{0.7018:.4f}"},
    {"label": "Accuracy", "value": f"{0.8300:.4f}"},
])

st.markdown("""
### What stands out

CatBoost produces the **highest fraud-class F1** and the **highest accuracy** among the displayed operating points.  
At the same time, its **PR-AUC is weaker** than the stronger Random Forest and XGBoost results.

That makes CatBoost a perfect example of why this project should not be summarized with a single metric.
""")

c1, c2 = st.columns(2)
with c1:
    st.image(img("04_model_implementation_catboost_cell23_out0.png"), caption="CatBoost evaluation output", use_container_width=True)
with c2:
    st.image(img("04_model_implementation_catboost_cell30_out0.png"), caption="CatBoost threshold tuning", use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.image(img("04_model_implementation_catboost_cell41_out0.png"), caption="CatBoost permutation importance", use_container_width=True)
with c4:
    st.image(img("04_model_implementation_catboost_cell44_out0.png"), caption="CatBoost SHAP summary", use_container_width=True)

st.markdown("""
### Honest interpretation

The threshold-tuning step barely changes the CatBoost outcome here.  
That is not a weakness in the notebook; it is useful information. It suggests the default operating point was already close to the selected validation preference.

For a portfolio, this is a strong page because it shows you understand metric trade-offs instead of just chasing the biggest single number.
""")

st.markdown("### Native categorical handling in CatBoost")
st.code(CODE_SNIPPETS["catboost_native_cat"], language="python")

st.markdown("### Manual permutation importance for CatBoost")
st.code(CODE_SNIPPETS["manual_permutation_catboost"], language="python")