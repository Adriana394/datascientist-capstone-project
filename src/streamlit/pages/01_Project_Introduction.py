import streamlit as st
import pandas as pd
from utils import set_page, hero, section_title, feature_cards

set_page("Project Introduction", "📘")

hero(
    "📘 Project Introduction",
    "This page introduces the business problem, dataset scope, technical setup, and the key goals of the project."
)

section_title("🎯 Project Goals")
st.markdown(
    """
- identify potentially suspicious insurance claims  
- compare multiple model families under class imbalance  
- evaluate fraud-relevant metrics beyond accuracy  
- assess threshold tuning as a business-oriented decision tool  
- present the full workflow in a clear portfolio format
"""
)

section_title("🏢 Business Context")
st.markdown(
    """
Insurance fraud is costly, operationally disruptive, and often difficult to detect at an early stage.  
Although fraudulent claims represent only a minority of all cases, they can generate disproportionate financial losses and increase the workload for claims and investigation teams.

This makes fraud detection a valuable machine learning use case.  
A well-designed model can support analysts by highlighting suspicious claims earlier, improving consistency in case screening, and helping teams focus their attention where it is most needed.

At the same time, this is not a problem that can be reduced to a single metric.  
A useful fraud detection workflow has to balance detection performance, false positives, and interpretability in a way that remains practical for real decision-making.
"""
)

section_title("🧪 Technical Overview")
feature_cards([
    {
        "title": "Dataset",
        "body": "Insurance claims data with mixed numerical and categorical features."
    },
    {
        "title": "Target",
        "body": "Binary fraud label with clear class imbalance."
    },
    {
        "title": "Models",
        "body": "Logistic Regression, Random Forest, XGBoost and CatBoost."
    },
])

section_title("🧠 Workflow")
workflow_df = pd.DataFrame(
    {
        "Step": [
            "Preprocessing",
            "Balancing",
            "Validation",
            "Optimization",
            "Interpretability",
            "Communication",
        ],
        "Description": [
            "Imputation, encoding and model-specific data preparation.",
            "SMOTE and class weighting for the minority fraud class.",
            "Cross-validation and consistent evaluation logic.",
            "Hyperparameter tuning and threshold selection.",
            "SHAP and permutation importance for model interpretation.",
            "Streamlit app for public-facing project presentation.",
        ],
    }
)
st.dataframe(workflow_df, use_container_width=True, hide_index=True)

section_title("🛠️ Tech Stack")
tech_stack_df = pd.DataFrame(
    {
        "Category": [
            "Data Processing",
            "Modeling",
            "Imbalance Handling",
            "Interpretability",
            "Visualization",
            "Application",
        ],
        "Tools": [
            "Pandas, NumPy",
            "Scikit-learn, XGBoost, CatBoost",
            "Imbalanced-learn",
            "SHAP, Permutation Importance",
            "Matplotlib, Seaborn",
            "Streamlit",
        ],
    }
)
st.dataframe(tech_stack_df, use_container_width=True, hide_index=True)

section_title("⚖️ Challenges and Benefits")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
<div style="
    padding: 1rem 1.1rem;
    border-radius: 1rem;
    background: #1e293b;
    border: 1px solid rgba(148,163,184,0.22);
    min-height: 220px;
">
    <h4 style="margin-top:0; color:#f8fafc;">Challenges</h4>
    <ul style="color:#e2e8f0; padding-left: 1.2rem; margin-bottom:0;">
        <li>Fraud cases are much rarer than non-fraud cases.</li>
        <li>Accuracy alone can therefore be misleading.</li>
        <li>The default threshold of 0.50 is not always the most useful operating point.</li>
        <li>Strong performance is not enough if the model cannot be interpreted convincingly.</li>
        <li>The dataset combines numerical and categorical features, which requires different preprocessing strategies across models.</li> 
    </ul>
</div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
<div style="
    padding: 1rem 1.1rem;
    border-radius: 1rem;
    background: #1e293b;
    border: 1px solid rgba(148,163,184,0.22);
    min-height: 220px;
">
    <h4 style="margin-top:0; color:#f8fafc;">Benefits</h4>
    <ul style="color:#e2e8f0; padding-left: 1.2rem; margin-bottom:0;">
        <li>Several model families are compared instead of relying on one single algorithm.</li>
        <li>PR-AUC, fraud precision, recall and F1 are better aligned with the business problem.</li>
        <li>Threshold tuning adds a practical decision layer beyond standard classification output.</li>
        <li>SHAP and permutation importance improve transparency and communication.</li>
        <li>Earlier identification of suspicious claims can support more forcused manual review.</li>
    </ul>
</div>
        """,
        unsafe_allow_html=True,
    )