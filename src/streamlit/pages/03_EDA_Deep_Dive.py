import pandas as pd
import streamlit as st

from utils import (
    set_page,
    hero,
    show_figure,
    show_artifact_table,
    load_csv,
    art_path,
)

set_page("EDA Deep Dive", "🔬")

hero(
    "🔬 EDA Deep Dive",
    "This section explores variable distributions, engineered features, statistical tests, and outlier patterns in more detail."
)

tabs = st.tabs([
    "Numerical Distribution",
    "Categorical Distribution",
    "Feature Engineering",
    "Statistical Tests",
    "Outlier Analysis",
])


def _pval_style(value):
    try:
        value = float(value)
        if value < 0.05:
            return "background-color: rgba(20,184,166,0.25); color: white;"
        return "background-color: rgba(244,63,94,0.18); color: white;"
    except Exception:
        return ""


def style_pvalue_columns(df: pd.DataFrame):
    if df.empty:
        return df

    p_cols = [col for col in df.columns if "p" in col.lower()]
    if not p_cols:
        return df

    styler = df.style
    for col in p_cols:
        styler = styler.map(_pval_style, subset=[col])
    return styler


# =========================================================
# Tab 1 - Numerical Distribution
# =========================================================
with tabs[0]:
    st.subheader("Numerical Distribution")

    st.markdown(
        """
This tab focuses on the distribution of numerical variables.  
The goal is to identify skewness, spread, potential outliers, and first differences in claim-related numeric features.
"""
    )

    show_figure(
        "01_data_exploration",
        "num_hist_continuous.png",
        caption="Continuous Numerical Feature Distributions",
    )

    show_figure(
        "01_data_exploration",
        "num_hist_discrete.png",
        caption="Discrete Numerical Feature Distributions",
    )

    show_figure(
        "01_data_exploration",
        "boxplots_continuous_by_target.png",
        caption="Continuous Variables by Fraud Status",
    )

    show_figure(
        "01_data_exploration",
        "discrete_num_by_target.png",
        caption="Discrete Numerical Variables by Fraud Status",
    )

    st.markdown(
        """
### Interpretation

Many numerical variables in insurance claims data are not normally distributed.  
This matters because skewed distributions and extreme values can affect:

- interpretability  
- threshold behavior  
- model sensitivity  
- feature engineering decisions

The target-wise plots are especially helpful because they show whether fraud and non-fraud cases differ only in level or also in shape and spread.
"""
    )

# =========================================================
# Tab 2 - Categorical Distribution
# =========================================================
with tabs[1]:
    st.subheader("Categorical Distribution")

    st.markdown(
        """
Categorical variables often contain some of the most useful fraud-related signals.  
This tab highlights how claim categories, policy characteristics, and incident attributes are distributed.
"""
    )

    show_figure(
        "01_data_exploration",
        "categorical_features_fraud.png",
        caption="Categorical Feature Distribution by Fraud Status",
    )

    focus_plot = st.selectbox(
        "Select a focused categorical view",
        [
            "Number of Vehicles Involved",
            "Witnesses",
            "Incident Hour",
            "Fraud Rate by Weekday",
            "Fraud Rate: Weekend vs Weekday",
        ],
        key="eda_cat_focus_plot",
    )

    plot_map = {
        "Number of Vehicles Involved": (
            "vehicles_fraud_percent.png",
            "Fraud Percentage by Number of Vehicles Involved",
        ),
        "Witnesses": (
            "witnesses_fraud.png",
            "Number of Witnesses by Fraud Status",
        ),
        "Incident Hour": (
            "incident_hour_fraud_percent.png",
            "Incident Hour Distribution by Fraud Status",
        ),
        "Fraud Rate by Weekday": (
            "fraud_rate_by_weekday.png",
            "Fraud Rate by Day of Week",
        ),
        "Fraud Rate: Weekend vs Weekday": (
            "fraud_rate_weekend.png",
            "Fraud Rate: Weekend vs Weekday",
        ),
    }

    filename, caption = plot_map[focus_plot]
    show_figure("01_data_exploration", filename, caption=caption)

    st.markdown(
        """
### Interpretation

Categorical EDA is especially valuable here because many fraud-relevant patterns are tied to:

- incident type  
- claim characteristics  
- policyholder attributes  
- categorical group imbalances

The focused views help move from a broad categorical overview to more specific fraud-related behavioral patterns.
"""
    )

# =========================================================
# Tab 3 - Feature Engineering
# =========================================================
with tabs[2]:
    st.subheader("Feature Engineering")

    st.markdown(
        """
This section moves from descriptive EDA toward **feature refinement for modeling**.

The goal is to understand which transformations or engineered views of the data provide
clearer fraud-related signal than the raw variables alone.
"""
    )

    st.info(
        "Most of the plots below are used to evaluate whether transformed or engineered feature views reveal clearer fraud patterns. One engineered feature that is explicitly retained later is `has_umbrella_policy`."
    )

    st.divider()

    # ---------------------------------------------------------
    # Section 1
    # ---------------------------------------------------------
    st.markdown("### 1. Claim Composition")

    show_figure(
        "01_data_exploration",
        "claim_composition_shares_boxplot.png",
        caption="Claim Composition Shares by Fraud Status",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
**What this shows**

This plot breaks the total claim into its component shares and compares how these relative claim compositions differ by fraud status.

Instead of looking only at absolute claim size, it highlights how the **structure of the claim** changes between fraud and non-fraud cases.
"""
        )
    with col2:
        st.markdown(
            """
**Why it matters**

Fraud-related signal may appear not only in how large a claim is, but in **how the claim is composed**.

This helps evaluate whether relative composition features may be more informative than raw totals alone.
"""
        )

    st.divider()

    # ---------------------------------------------------------
    # Section 2
    # ---------------------------------------------------------
    st.markdown("### 2. Engineered Ratio Features")

    show_figure(
        "01_data_exploration",
        "claim_ratio_features_boxplot.png",
        caption="Engineered Claim Ratio Features",
    )

    show_figure(
        "01_data_exploration",
        "claim_ratio_fraud.png",
        caption="Claim Ratio Distribution by Fraud Status",
    )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            """
**What this shows**

These plots summarize ratio-style features derived from the claim components.

They are useful for checking whether fraud and non-fraud cases differ more clearly in their **relative financial structure** than in the original raw claim variables.
"""
        )
    with col4:
        st.markdown(
            """
**Why it matters**

Ratio-based features are often easier to interpret in fraud settings because they capture **relationships between claim components**, not just magnitude.

Even when not every engineered ratio is carried forward into every final model, these views help justify later feature selection and refinement decisions.
"""
        )

    st.divider()

    # ---------------------------------------------------------
    # Section 3
    # ---------------------------------------------------------
    st.markdown("### 3. Umbrella Policy Feature")

    umbrella_view = st.selectbox(
        "Select an umbrella-policy view",
        [
            "Umbrella Policy Presence",
            "Umbrella Limit Tiers",
        ],
        key="eda_umbrella_view",
    )

    if umbrella_view == "Umbrella Policy Presence":
        show_figure(
            "01_data_exploration",
            "umbrella_policy_dist.png",
            caption="Umbrella Policy Presence by Fraud Status",
        )
    else:
        show_figure(
            "01_data_exploration",
            "umbrella_limit_tiers_percent.png",
            caption="Umbrella Limit Tiers by Fraud Status",
        )

    col5, col6 = st.columns(2)
    with col5:
        st.markdown(
            """
**What this shows**

The original `umbrella_limit` variable does not provide especially clean separation on its own.

To make the information more usable, the notebook derives the binary feature:

- `has_umbrella_policy = 1` if `umbrella_limit > 0`
- `has_umbrella_policy = 0` otherwise
"""
        )
    with col6:
        st.markdown(
            """
**Why it matters**

This is the clearest example of an engineered feature that is later retained in the modeling workflow.

It simplifies interpretation and creates a cleaner representation of whether a customer has umbrella coverage at all, which proved more practical than using the raw limit alone.
"""
        )

    st.divider()

    # ---------------------------------------------------------
    # Section 4
    # ---------------------------------------------------------
    st.markdown("### 4. Refinement Decisions")

    st.markdown(
        """
Feature refinement also includes deciding **what not to keep**.

One important example is `incident_month`, which was removed because the available data only covers the **first quarter of the year**.  
Keeping it could introduce temporal bias instead of robust fraud signal.

This is an important reminder that feature engineering is not only about creating new variables — it is also about removing variables that may hurt generalization.
"""
    )

    st.success(
        "Key takeaway: this tab supports later modeling by showing how transformed feature views improve interpretability and by highlighting `has_umbrella_policy` as a feature that is explicitly carried forward."
    )

# =========================================================
# Tab 4 - Statistical Tests
# =========================================================
with tabs[3]:
    st.subheader("Statistical Tests")

    st.markdown(
        """
This tab summarizes the statistical tests used during EDA to compare fraudulent and non-fraudulent claims more systematically.

These tests help answer questions such as:

- which variables differ meaningfully between groups  
- which patterns are likely to be non-random  
- which variables may deserve deeper attention in modeling or feature selection
"""
    )

    mw_df = load_csv(art_path("eda", "mannwhitney_cliffs.csv"))
    chi2_df = load_csv(art_path("eda", "chi2_results.csv"))

    st.markdown("### Mann–Whitney U Test & Cliff's Delta")
    if not mw_df.empty:
        st.dataframe(style_pvalue_columns(mw_df), use_container_width=True, hide_index=True)
    else:
        show_artifact_table("eda", "mannwhitney_cliffs.csv")

    st.markdown("### Chi-Square Test Results")
    if not chi2_df.empty:
        st.dataframe(style_pvalue_columns(chi2_df), use_container_width=True, hide_index=True)
    else:
        show_artifact_table("eda", "chi2_results.csv")

    st.markdown(
        """
### Interpretation

Statistical tests do not replace modeling, but they strengthen the analytical foundation of the project.

They help identify variables that show meaningful group differences between fraud and non-fraud cases and support later decisions around:

- variable relevance  
- feature engineering  
- feature retention or removal  
- interpretation of fraud-related patterns
"""
    )

    st.info(
        "Green-highlighted p-values indicate stronger evidence of a meaningful group difference. Red-highlighted p-values suggest weaker statistical support."
    )

# =========================================================
# Tab 5 - Outlier Analysis
# =========================================================
with tabs[4]:
    st.subheader("Outlier Analysis")

    st.markdown(
        """
Outliers are common in financial and claim-related data.  
This section summarizes the IQR-based outlier analysis used during EDA.

The goal is to identify variables with unusually extreme values that may influence:

- summary statistics  
- visual interpretation  
- feature engineering decisions  
- model behavior
"""
    )

    st.markdown("### IQR Summary")
    show_artifact_table("eda", "IQR_summary.csv")

    st.markdown(
        """
### Interpretation

Outlier analysis is especially relevant in this project because claim-related variables often show skewed distributions and extreme observations.

These patterns can reflect:

- natural variation in claim amounts  
- rare but important fraud-related cases  
- variables that may require special treatment or closer interpretation
"""
    )