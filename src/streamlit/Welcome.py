import streamlit as st
from utils import set_page, hero, section_title, feature_cards, timeline_cards

set_page("Insurance Fraud Detection", "🛡️")

hero(
    "🛡️ Insurance Claim Fraud Detection",
    "A portfolio-focused Streamlit app presenting an end-to-end machine learning workflow for insurance fraud detection."
)

section_title("👩‍💻 About the Author")
st.markdown(
    """
Hi, I’m Adriana, and this app is part of my Data Science capstone project portfolio.  
It was built to present the project in a clearer, more professional way than a collection of notebooks alone.

The goal is not to simulate a production fraud platform, but to show a structured machine learning workflow from **exploration and preprocessing** to **model comparison, threshold tuning, and interpretation**.
"""
)

section_title("💡 Project Overview")
st.markdown(
    """
Insurance fraud is expensive — but detecting it early is challenging.  
This app demonstrates how a professional end-to-end ML workflow can assist fraud analysts by:

- highlighting suspicious claims  
- comparing multiple model families  
- adjusting decision thresholds to business needs  
- offering interpretable insights via SHAP and permutation importance
"""
)

section_title("🧩 What this app highlights")
feature_cards([
    {
        "title": "📊 Exploratory Analysis",
        "body": "Important fraud patterns, target imbalance, data cleaning, preparation, and engineered features steps are presented visually and clearly."
    },
    {
        "title": "🤖 Model Comparison",
        "body": "Logistic Regression, Random Forest, XGBoost and CatBoost are compared using fraud-relevant metrics instead of relying on accuracy only."
    },
    {
        "title": "🧠 Explainability",
        "body": "SHAP and permutation importance are used to identify which features contribute most strongly to suspicious claim predictions."
    },
])

section_title("📈 Project Workflow")
timeline_cards([
    {"title": "1️⃣ EDA", "body": "Explore structure, missing values, class imbalance and first fraud-related patterns."},
    {"title": "2️⃣ Preprocessing", "body": "Prepare mixed numerical and categorical features for different model families."},
    {"title": "3️⃣ Class Imbalance", "body": "Use class weights and SMOTE-based strategies to support minority-class learning."},
    {"title": "4️⃣ Modeling", "body": "Train baseline and tree-based models to compare different decision behaviors."},
    {"title": "5️⃣ Threshold Tuning", "body": "Evaluate alternative thresholds beyond the default 0.5."},
    {"title": "6️⃣ Interpretation", "body": "Use explainability tools to connect model behavior back to meaningful claim features."},
])

section_title("🧭 Explore the App")
st.markdown(
    """
Use the sidebar to navigate through the project:

- **Introduction** – business framing, workflow and technical setup  
- **EDA pages** – data quality, fraud patterns and feature engineering  
- **Baseline & Random Forest** – first benchmark and threshold effects  
- **XGBoost** – boosted tree model with native categorical handling  
- **CatBoost** – strong fraud-class F1 and compact preprocessing pipeline  
- **Summary** – final comparison with visual metric bars and key takeaways
"""
)