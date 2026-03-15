import streamlit as st
from utils import set_page, hero, img, CODE_SNIPPETS

set_page("EDA • Feature Engineering", "🧪")

hero(
    "EDA — Feature Engineering & Data Refinement",
    "Later sections of the EDA notebook move from observation to actionable dataset design."
)

st.markdown("""
### Feature-engineering focus

A big strength of the project is that the EDA notebook does not stop at descriptive plots.  
It transitions into actual dataset refinement:

- date-based transformations  
- ratio-style financial features  
- feature pruning after weak statistical evidence  
- outlier profiling for numeric variables
""")

c1, c2 = st.columns(2)
with c1:
    st.image(img("01_data_exploration_cell149_out0.png"), caption="Engineered ratio feature inspection", use_container_width=True)
with c2:
    st.image(img("01_data_exploration_cell163_out0.png"), caption="IQR-based outlier summary in the final dataset", use_container_width=True)

st.markdown("### Additional claim amount views")
c3, c4, c5 = st.columns(3)
with c3:
    st.image(img("01_data_exploration_cell116_out0.png"), caption="Monetary sub-claim distribution", use_container_width=True)
with c4:
    st.image(img("01_data_exploration_cell116_out1.png"), caption="Monetary sub-claim distribution", use_container_width=True)
with c5:
    st.image(img("01_data_exploration_cell116_out2.png"), caption="Monetary sub-claim distribution", use_container_width=True)

st.markdown("""
### Why this matters

This is the point where many beginner projects get weak.  
They either keep almost everything or drop columns without a defensible story.

Here, the notebook tries to do the harder thing:
build a **leaner modeling table** based on a mix of domain reasoning, EDA findings, and practical model needs.
""")

st.markdown("### Representative preprocessing snippet")
st.code(CODE_SNIPPETS["preprocessing"], language="python")

st.markdown("""
### Honest read

The strongest portfolio message here is **not** “look how many plots I made”.  
It is: *the EDA notebook changes the dataset in a way that clearly affects downstream model quality and interpretability.*
""")