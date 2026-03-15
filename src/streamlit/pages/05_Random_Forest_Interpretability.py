import streamlit as st
from utils import set_page, hero, img, CODE_SNIPPETS

set_page("Random Forest Interpretability", "🧠")

hero(
    "Random Forest Interpretability",
    "The notebook does not stop at metrics; it also checks what the model is actually using."
)

st.markdown("""
### Why interpretability is important here

Fraud models can look strong for the wrong reasons.  
That is why SHAP and permutation importance are not optional decoration in this project.

They help answer two questions:

- Which features drive the model most strongly?  
- Do those drivers look operationally plausible?
""")

c1, c2 = st.columns(2)
with c1:
    st.image(img("02_model_implementation_cell87_out0.png"), caption="Random Forest SHAP summary", use_container_width=True)
with c2:
    st.image(img("02_model_implementation_cell92_out1.png"), caption="Random Forest permutation importance", use_container_width=True)

st.markdown("""
### Reading these plots

The strongest features concentrate around claim severity, claim amount structure, and the engineered ratios created earlier in the workflow.  
That is a good sign because it means the EDA and feature-engineering decisions are visible again at interpretation time.

The feature-selection variant also supports the idea that a smaller, cleaner feature set can improve ranking quality even if it does not win every point metric.
""")

st.markdown("### SHAP snippet")
st.code(CODE_SNIPPETS["shap"], language="python")

st.markdown("""
### Portfolio value

This page makes the project look more mature.  
Without it, the modeling story would be just “I trained a forest and got a score.”  
With it, the story becomes “I trained it, tuned it, and checked whether the signal makes sense.”
""")