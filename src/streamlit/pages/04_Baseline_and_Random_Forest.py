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

set_page("Baseline & Random Forest", "🌲")

hero(
    "🌲 Baseline & Random Forest",
    "This section covers the baseline Logistic Regression model, multiple Random Forest variants, threshold optimization, and model interpretation."
)

# -------------------------------------------------------------------
# Notebook-aligned result snapshot
# -------------------------------------------------------------------
results_df = pd.DataFrame(
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
    ]
)

rf_default_comparison = load_csv(art_path("02_model_implementation", "rf_default_comparison.csv"))
topt_results = load_csv(art_path("02_model_implementation", "topt_results.csv"))
lr_report = load_csv(art_path("02_model_implementation", "classification_report_lr_base.csv"))
rf_base_report = load_csv(art_path("02_model_implementation", "classification_report_rf_base.csv"))
rf_base_tuned_report = load_csv(art_path("02_model_implementation", "classification_report_rf_base_tuned.csv"))
rf_smote_tuned_report = load_csv(art_path("02_model_implementation", "classification_report_rf_smote_tuned.csv"))
rf_fs_tuned_report = load_csv(art_path("02_model_implementation", "classification_report_rf_fs_tuned.csv"))
rf_final_report = load_csv(art_path("02_model_implementation", "classification_report_rf_final.csv"))
perm_results = load_csv(art_path("02_model_implementation", "perm_results.csv"))

tabs = st.tabs([
    "Model Overview",
    "Preparation Steps",
    "Logistic Regression",
    "Random Forest",
    "Optimized Threshold",
    "Model Interpretation",
])

# =========================================================
# Tab 1 - Model Overview
# =========================================================
with tabs[0]:
    st.subheader("Model Overview")

    st.markdown(
        """
In this notebook, the workflow moves from a simple baseline model toward more flexible tree-based approaches.

The goal is not just to train models, but to compare how different modeling choices behave under class imbalance.
That includes:

- a **Logistic Regression baseline**
- a **base Random Forest**
- a **SMOTE-enhanced Random Forest**
- a **feature-selection variant**
- **threshold optimization**
- **permutation-based interpretation**
"""
    )

    st.markdown(
        """
### Why start with Logistic Regression?

Logistic Regression provides a transparent baseline and makes it easier to judge whether more complex models are actually improving the fraud-detection workflow.

### Why Random Forest?

Random Forest is a strong next step because it can capture non-linear relationships and interactions between features that a linear baseline may miss.  
In a fraud setting, that matters more than raw accuracy alone.

To better address the class imbalance, the notebook evaluates several Random Forest variants rather than relying on a single model fit.
"""
    )

    st.markdown("### Notebook-Aligned Result Snapshot")
    st.dataframe(nice_df(results_df), use_container_width=True, hide_index=True)

    st.info(
        "This page shows how the modeling workflow evolves from a simple linear baseline toward stronger tree-based models with threshold-aware evaluation."
    )

# =========================================================
# Tab 2 - Preparation Steps
# =========================================================
with tabs[1]:
    st.subheader("Preparation Steps")

    st.markdown(
        """
This notebook prepares the data differently depending on the model family.

The linear baseline and the tree-based models do **not** need exactly the same preprocessing,
so the workflow separates preparation steps in a way that stays aligned with each model’s strengths.
"""
    )

    st.info(
        "The goal of this stage is not just to clean the data, but to prepare it in a way that supports fair model comparison under class imbalance."
    )

    st.divider()

    # ---------------------------------------------------------
    # Section 1
    # ---------------------------------------------------------
    st.markdown("### 1. Model-Specific Preprocessing")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
**For Logistic Regression**

- missing values are imputed  
- categorical variables are encoded  
- numerical variables are scaled  

This is necessary because linear models are more sensitive to feature scale and representation.
"""
        )
    with col2:
        st.markdown(
            """
**For Random Forest**

- missing values are imputed  
- categorical features are prepared through the tree pipeline  
- no scaling is required  

Tree-based models are less sensitive to scale, so the preprocessing can stay more compact.
"""
        )

    st.markdown("### Preprocessing Design")
    st.code(CODE_SNIPPETS["preprocessing"], language="python")

    st.divider()

    # ---------------------------------------------------------
    # Section 2
    # ---------------------------------------------------------
    st.markdown("### 2. Exploratory Class Structure")

    show_figure(
        "02_model_implementation",
        "t_sne_plot_target.png",
        caption="t-SNE projection of the preprocessed training data",
    )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            """
**What this shows**

The t-SNE projection gives an exploratory two-dimensional view of the preprocessed feature space.

It is used here as a visual checkpoint to see whether some class structure is already visible before the final model comparison.
"""
        )
    with col4:
        st.markdown(
            """
**Why it matters**

This is **not** a validation metric and should not be overinterpreted.

But it helps illustrate whether fraud and non-fraud observations show at least partial separation after preprocessing, which can make the later model results easier to interpret.
"""
        )

    st.divider()

    # ---------------------------------------------------------
    # Section 3
    # ---------------------------------------------------------
    st.markdown("### 3. Random Forest Pipeline")

    st.code(CODE_SNIPPETS["rf_pipeline"], language="python")

    with st.expander("Why no aggressive outlier removal here?"):
        st.markdown(
            """
The notebook explicitly avoids automatic removal of extreme claim-related values at this stage.

That is a sensible choice in fraud detection because very large or unusual monetary values may themselves carry fraud signal rather than simply representing noise.
"""
        )

    st.success(
        "Key takeaway: this preparation stage is designed to support fair comparison between a simple linear baseline and stronger tree-based models, while preserving potentially meaningful fraud-related signal."
    )

# =========================================================
# Tab 3 - Logistic Regression
# =========================================================
with tabs[2]:
    st.subheader("Logistic Regression Baseline")

    lr_df = results_df[results_df["Model"] == "Logistic Regression"].copy()

    st.markdown(
        """
Logistic Regression serves as the baseline model in this workflow.

Its role is important because it gives a simple, interpretable reference point before moving to more flexible ensemble methods.
That makes later improvements easier to evaluate honestly.
"""
    )

    st.dataframe(nice_df(lr_df), use_container_width=True, hide_index=True)

    if not lr_report.empty:
        st.markdown("### Classification Report")
        st.dataframe(lr_report, use_container_width=True)

    show_figure(
        "02_model_implementation",
        "pr_curve_logreg.png",
        caption="Precision–Recall Curve (Logistic Regression)",
    )

    st.markdown(
        """
### Interpretation

The baseline achieves very strong fraud recall, but precision is weak and the overall operating point is not very practical.

This is a common pattern in imbalanced classification:
a simple baseline can catch many positives, but often at the cost of too many false alarms.
"""
    )

# =========================================================
# Tab 4 - Random Forest
# =========================================================
with tabs[3]:
    st.subheader("Random Forest Variants")

    st.markdown(
        """
The Random Forest section goes beyond a single model fit and compares several variants of the same model family.

This is important because performance under fraud imbalance is influenced not only by the algorithm itself, but also by:

- how the minority class is handled  
- whether feature selection is applied  
- which threshold is used at prediction time
"""
    )

    rf_view = st.selectbox(
        "Select a Random Forest view",
        [
            "Default-threshold comparison",
            "Basic Random Forest",
            "Random Forest Base (Tuned)",
            "Random Forest + SMOTE (Tuned)",
            "Random Forest + Feature Selection (Tuned)",
            "Final Refit",
        ],
        key="rf_view_select",
    )

    if rf_view == "Default-threshold comparison":
        st.markdown(
            """
### What this view shows

This comparison summarizes the main Random Forest variants at their default operating point.

It provides a quick way to see how much the model family changes once balancing strategies and later threshold optimization are introduced.
"""
        )

        st.markdown(
            """
### Why it matters

This view makes one thing very clear:
model performance is not determined by the classifier alone.

Even within the same model family, different preparation and threshold choices can produce very different fraud-detection behavior.
"""
        )

        if not rf_default_comparison.empty:
            st.markdown("### Comparison at Default Threshold (0.50)")
            st.dataframe(rf_default_comparison, use_container_width=True, hide_index=True)
        else:
            st.dataframe(
                nice_df(results_df[results_df["Model"] == "Random Forest"]),
                use_container_width=True,
                hide_index=True,
            )

        show_figure(
            "02_model_implementation",
            "pr_curve_rf_base.png",
            caption="Precision–Recall Curve (Random Forest Base)",
        )

    elif rf_view == "Basic Random Forest":
        st.markdown(
            """
### What this view shows

This is the untuned Random Forest at the default threshold of **0.50**.

It acts as the first tree-based benchmark after Logistic Regression and shows what the model can do before threshold optimization or class-balancing improvements are introduced.
"""
        )

        st.markdown(
            """
### Why it matters

The base model already improves ranking quality compared with the baseline,
but its fraud recall at the default threshold remains too low to be the best practical operating point.
"""
        )

        if not rf_base_report.empty:
            st.markdown("### Classification Report")
            st.dataframe(rf_base_report, use_container_width=True)

        show_figure(
            "02_model_implementation",
            "pr_curve_rf_base.png",
            caption="Precision–Recall Curve (Random Forest Base)",
        )

    elif rf_view == "Random Forest Base (Tuned)":
        st.markdown(
            """
### What this view shows

This version keeps the base Random Forest model but adjusts the threshold away from the standard **0.50** cutoff.

The goal is to improve fraud recall while maintaining a minimum acceptable precision.
"""
        )

        st.markdown(
            """
### Why it matters

This is the clearest demonstration that threshold choice can materially change model usefulness.

The underlying model stays the same, but the operating point becomes far more practical for fraud detection.
"""
        )

        if not rf_base_tuned_report.empty:
            st.markdown("### Classification Report")
            st.dataframe(rf_base_tuned_report, use_container_width=True)

    elif rf_view == "Random Forest + SMOTE (Tuned)":
        st.markdown(
            """
### What this view shows

This variant combines Random Forest with **SMOTE** and then applies threshold tuning.

SMOTE is used to help the model learn the minority fraud class more effectively during training.
"""
        )

        st.markdown(
            """
### Why it matters

Among the Random Forest operating points shown in the app, this is the strongest **practical** fraud-detection setup.

It balances fraud recall, fraud precision, and generalization better than the simpler RF variants.
"""
        )

        if not rf_smote_tuned_report.empty:
            st.markdown("### Classification Report")
            st.dataframe(rf_smote_tuned_report, use_container_width=True)

    elif rf_view == "Random Forest + Feature Selection (Tuned)":
        st.markdown(
            """
### What this view shows

This version applies feature selection before the tuned Random Forest evaluation.

The idea is to test whether a smaller and cleaner feature space can improve ranking quality and reduce noise.
"""
        )

        st.markdown(
            """
### Why it matters

This variant reaches the **highest PR-AUC** among the Random Forest notebook results,
which makes it especially interesting from a ranking perspective even though it is not the final operational choice.
"""
        )

        if not rf_fs_tuned_report.empty:
            st.markdown("### Classification Report")
            st.dataframe(rf_fs_tuned_report, use_container_width=True)

        show_figure(
            "02_model_implementation",
            "selected_features_rf.png",
            caption="Selected Features from the Random Forest Feature Selection Step",
        )

    else:
        st.markdown(
            """
### What this view shows

The final refit reuses the strongest operational Random Forest setup and trains it on the full training design chosen after tuning.

It represents the final Random Forest result that is carried forward for interpretation.
"""
        )

        st.markdown(
            """
### Why it matters

This is the most deployment-like Random Forest result in the notebook:
not necessarily the best on every single metric, but the most reasonable final RF configuration after the earlier comparisons.
"""
        )

        if not rf_final_report.empty:
            st.markdown("### Classification Report")
            st.dataframe(rf_final_report, use_container_width=True)

        show_figure(
            "02_model_implementation",
            "selected_features_rf.png",
            caption="Selected Features from the Random Forest Feature Selection Step",
        )

    st.divider()

    st.markdown(
        """
### Overall interpretation

Compared with Logistic Regression, the Random Forest variants provide a much stronger ranking of suspicious claims.

The notebook shows very clearly that better fraud detection comes from a combination of:

- stronger model structure  
- better minority-class handling  
- smarter threshold choice  
- more deliberate feature-space design
"""
    )

# =========================================================
# Tab 5 - Optimized Threshold
# =========================================================
with tabs[4]:
    st.subheader("Optimized Threshold")

    st.markdown(
        """
A key lesson from this notebook is that the default classification threshold of **0.50** is not always the most useful operating point.

This is especially important in fraud detection, where the practical balance between:

- fraud recall  
- fraud precision  
- false alarms  

often matters more than the default model cutoff.
"""
    )

    st.markdown("### Threshold Optimization Logic")
    st.code(CODE_SNIPPETS["threshold_tuning"], language="python")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Before Optimization")
        if not rf_default_comparison.empty:
            st.dataframe(rf_default_comparison, use_container_width=True, hide_index=True)
        else:
            st.info("`rf_default_comparison.csv` not found.")

    with col_right:
        st.markdown("### After Optimization")
        if not topt_results.empty:
            st.dataframe(topt_results, use_container_width=True, hide_index=True)
        else:
            st.info("`topt_results.csv` not found.")

    st.markdown("### Threshold Tuning Plots")
    show_figure(
        "02_model_implementation",
        "threshold_tuning_rf_base.png",
        caption="Threshold Tuning – RF Base",
    )

    show_figure(
        "02_model_implementation",
        "threshold_tuning_rf_smote.png",
        caption="Threshold Tuning – RF + SMOTE",
    )

    rf_fs_threshold_plot = art_path("02_model_implementation", "threshold_tuning_rf_fs.png")
    if rf_fs_threshold_plot.exists():
        st.image(
            str(rf_fs_threshold_plot),
            caption="Threshold Tuning – RF + Feature Selection",
            use_container_width=True,
        )

    st.markdown(
        """
### Interpretation

Threshold tuning is one of the most practically relevant parts of this notebook.

The comparison between the default-threshold table and the tuned-threshold table makes the effect much easier to understand:
the decision rule changes, and with it the balance between fraud recall and fraud precision.

In this notebook, the **RF + SMOTE** variant generalizes best after threshold tuning and is therefore selected as the basis for the final refit.
"""
    )

# =========================================================
# Tab 6 - Model Interpretation
# =========================================================
with tabs[5]:
    st.subheader("Model Interpretation")

    st.markdown(
        """
Interpretability matters in fraud detection because good performance alone is not enough.
A useful model should also provide insight into which features are driving suspicious predictions.

This notebook uses permutation importance to better understand the final Random Forest model.
"""
    )

    show_figure(
        "02_model_implementation",
        "permutation_importance_rf_final.png",
        caption="Permutation Importance for the Final Random Forest Model",
    )

    if not perm_results.empty:
        st.markdown("### Permutation Importance Table")
        st.dataframe(perm_results, use_container_width=True, hide_index=True)

    st.markdown(
        """
### Interpretation

The most important features are concentrated around claim severity, claim-amount structure, and the engineered relationships created earlier during EDA.

That is a strong sign that the workflow is internally coherent:
the signals identified during exploratory analysis also appear again in the final model interpretation.

Without this step, the page would only show scores.
With it, the project becomes much more credible and portfolio-ready.
"""
    )