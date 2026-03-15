import streamlit as st
from utils import set_page, hero, metric_row, CODE_SNIPPETS, img

set_page("XGBoost", "🚀")

hero(
    "XGBoost",
    "The XGBoost notebook explores a stronger boosted-tree model with native categorical support."
)

metric_row([
    {"label": "Best CV PR-AUC", "value": f"{0.5528:.4f}"},
    {"label": "Test PR-AUC", "value": f"{0.6043:.4f}"},
    {"label": "Fraud F1 @ 0.50", "value": f"{0.6780:.4f}"},
    {"label": "Tuned threshold", "value": f"{0.3909:.4f}"},
])

st.markdown("""
### What stands out

XGBoost delivers one of the cleanest operating points in the project:

- strong fraud-class precision relative to the baseline  
- recall remains high at **0.80**  
- accuracy is solid at **0.81**  
- native categorical handling keeps the pipeline cleaner than one-hot-heavy alternatives
""")

c1, c2 = st.columns(2)
with c1:
    st.image(img("03_model_implementation_xgb_cell26_out0.png"), caption="XGBoost evaluation output", use_container_width=True)
with c2:
    st.image(img("03_model_implementation_xgb_cell35_out1.png"), caption="Threshold search for XGBoost", use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.image(img("03_model_implementation_xgb_cell48_out0.png"), caption="XGBoost permutation importance", use_container_width=True)
with c4:
    st.image(img("03_model_implementation_xgb_cell51_out0.png"), caption="XGBoost SHAP summary", use_container_width=True)

st.markdown("""
### Important nuance

Threshold tuning did **not** improve the fraud-class F1 here.  
That is exactly the kind of result worth showing publicly, because it proves the project is not forcing a success narrative where one does not exist.

XGBoost is still a strong candidate, but the notebook correctly shows that threshold optimization is model-specific and not automatically beneficial.
""")

st.markdown("### Native categorical handling")
st.code(CODE_SNIPPETS["xgb_native_cat"], language="python")