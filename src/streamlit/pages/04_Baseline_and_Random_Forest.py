import streamlit as st
from utils import set_page, hero, metric_row, COMPARISON_DF, nice_df, CODE_SNIPPETS, img

set_page("Baseline & Random Forest", "🌲")

hero(
    "Baseline & Random Forest Modeling",
    "The second notebook moves from a baseline Logistic Regression to several Random Forest variants."
)

st.markdown("""
### Why this page matters

This notebook shows a real modeling progression rather than a one-shot experiment:

- simple baseline with Logistic Regression  
- base Random Forest  
- Random Forest with SMOTE  
- Random Forest with feature selection  
- threshold tuning and final refit
""")

subset = COMPARISON_DF[COMPARISON_DF["Model"].isin(["Logistic Regression", "Random Forest"])].copy()
st.dataframe(nice_df(subset), use_container_width=True, hide_index=True)

metric_row([
    {"label": "Baseline PR-AUC", "value": f"{0.3749:.4f}"},
    {"label": "RF base PR-AUC", "value": f"{0.5588:.4f}"},
    {"label": "RF SMOTE tuned PR-AUC", "value": f"{0.6190:.4f}"},
    {"label": "RF best observed PR-AUC", "value": f"{0.6491:.4f}"},
])

st.markdown("### Visual checkpoints")
c1, c2 = st.columns(2)
with c1:
    st.image(img("02_model_implementation_cell18_out0.png"), caption="t-SNE projection used as exploratory class-separation check", use_container_width=True)
with c2:
    st.image(img("02_model_implementation_cell26_out0.png"), caption="Precision–Recall curve for the baseline approach", use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.image(img("02_model_implementation_cell35_out0.png"), caption="Random Forest evaluation curve", use_container_width=True)
with c4:
    st.image(img("02_model_implementation_cell52_out0.png"), caption="Threshold tuning for the stronger RF setup", use_container_width=True)

st.markdown("""
### What improved

The baseline Logistic Regression achieves very high fraud recall, but the precision is weak and the overall operating point is too noisy.  
The Random Forest variants substantially improve ranking quality and produce a much more usable fraud-class F1 once threshold tuning is introduced.

That is one of the clearest lessons in the whole project:
**threshold selection changes the story almost as much as the algorithm choice.**
""")

st.markdown("### Core model-building snippet")
st.code(CODE_SNIPPETS["rf_pipeline"], language="python")

st.markdown("### Threshold logic snippet")
st.code(CODE_SNIPPETS["threshold_tuning"], language="python")