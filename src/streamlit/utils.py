from pathlib import Path
import pandas as pd
import streamlit as st

# =========================================================
# Paths
# =========================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = REPORTS_DIR / "artifacts"

EDA_FIG_DIR = FIGURES_DIR / "01_data_exploration"
EDA_ART_DIR = ARTIFACTS_DIR / "eda"

# =========================================================
# Model Results
# =========================================================
MODEL_RESULTS = [
    {"Model": "Logistic Regression", "Variant": "Baseline @ 0.50", "PR-AUC": 0.3749, "Fraud Precision": 0.2875, "Fraud Recall": 0.9200, "Fraud F1": 0.4381, "Accuracy": 0.4100, "Threshold": 0.50},
    {"Model": "Random Forest", "Variant": "Base @ 0.50", "PR-AUC": 0.5588, "Fraud Precision": 0.5000, "Fraud Recall": 0.1600, "Fraud F1": 0.2424, "Accuracy": 0.7500, "Threshold": 0.50},
    {"Model": "Random Forest", "Variant": "Base tuned", "PR-AUC": None, "Fraud Precision": 0.4651, "Fraud Recall": 0.8000, "Fraud F1": 0.5882, "Accuracy": 0.7200, "Threshold": 0.2500},
    {"Model": "Random Forest", "Variant": "SMOTE tuned", "PR-AUC": 0.6190, "Fraud Precision": 0.5250, "Fraud Recall": 0.8400, "Fraud F1": 0.6462, "Accuracy": 0.7700, "Threshold": 0.3091},
    {"Model": "Random Forest", "Variant": "SMOTE + feature selection", "PR-AUC": 0.6491, "Fraud Precision": 0.4545, "Fraud Recall": 0.8000, "Fraud F1": 0.5797, "Accuracy": 0.7100, "Threshold": 0.2638},
    {"Model": "Random Forest", "Variant": "Final refit", "PR-AUC": 0.6353, "Fraud Precision": 0.4878, "Fraud Recall": 0.8000, "Fraud F1": 0.6061, "Accuracy": 0.7400, "Threshold": 0.3091},
    {"Model": "XGBoost", "Variant": "Default @ 0.50", "PR-AUC": 0.6043, "Fraud Precision": 0.5882, "Fraud Recall": 0.8000, "Fraud F1": 0.6780, "Accuracy": 0.8100, "Threshold": 0.50},
    {"Model": "XGBoost", "Variant": "Tuned threshold", "PR-AUC": 0.6043, "Fraud Precision": 0.5405, "Fraud Recall": 0.8000, "Fraud F1": 0.6452, "Accuracy": 0.7800, "Threshold": 0.3909},
    {"Model": "CatBoost", "Variant": "Default @ 0.50", "PR-AUC": 0.5552, "Fraud Precision": 0.6250, "Fraud Recall": 0.8000, "Fraud F1": 0.7018, "Accuracy": 0.8300, "Threshold": 0.50},
    {"Model": "CatBoost", "Variant": "Tuned threshold", "PR-AUC": 0.5552, "Fraud Precision": 0.6250, "Fraud Recall": 0.8000, "Fraud F1": 0.7018, "Accuracy": 0.8300, "Threshold": 0.5117},
]

COMPARISON_DF = pd.DataFrame(MODEL_RESULTS)

BEST_FAMILY = [
    {"Model": "Random Forest", "Variant": "SMOTE tuned", "PR-AUC": 0.6190, "Fraud Precision": 0.5250, "Fraud Recall": 0.8400, "Fraud F1": 0.6462, "Accuracy": 0.7700, "Threshold": 0.3091},
    {"Model": "XGBoost", "Variant": "Default @ 0.50", "PR-AUC": 0.6043, "Fraud Precision": 0.5882, "Fraud Recall": 0.8000, "Fraud F1": 0.6780, "Accuracy": 0.8100, "Threshold": 0.50},
    {"Model": "CatBoost", "Variant": "Default @ 0.50", "PR-AUC": 0.5552, "Fraud Precision": 0.6250, "Fraud Recall": 0.8000, "Fraud F1": 0.7018, "Accuracy": 0.8300, "Threshold": 0.50},
]
BEST_FAMILY_DF = pd.DataFrame(BEST_FAMILY)

CODE_SNIPPETS = {
    "preprocessing": """
ord_cols = ['incident_severity']
ord_order = [['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss']]

ord_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=ord_order))
])

nom_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
""",
    "threshold_tuning": """
def threshold_opt_recall_with_precision(y_true, y_proba, min_precision=0.35):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]

    valid = precision >= min_precision
    if not valid.any():
        return 0.5, precision[0], recall[0]

    idx = np.argmax(recall[valid])
    best_threshold = thresholds[valid][idx]
    return best_threshold, precision[valid][idx], recall[valid][idx]
""",
    "rf_pipeline": """
rf_pipe_smote = ImbPipeline([
    ('preprocessor', preprocessor_tree),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid_rf = {
    'clf__n_estimators': [200, 400, 600],
    'clf__max_depth': [6, 10, None],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2],
    'clf__max_features': ['sqrt', 'log2']
}
""",
    "xgb_native_cat": """
cat_features = X.select_dtypes('object').columns
X_cat = X.copy()
X_cat[cat_features] = X_cat[cat_features].astype('category')

xgb_clf = XGBClassifier(
    enable_categorical=True,
    scale_pos_weight=scale_weight,
    eval_metric='aucpr',
    random_state=42
)
""",
    "catboost_native_cat": """
cat_cols = X_train.select_dtypes(['object', 'category']).columns.tolist()

best_cb = CatBoostClassifier(
    scale_pos_weight=scale_weight,
    loss_function='Logloss',
    eval_metric='PRAUC',
    verbose=False,
    random_seed=42,
    **best_params_cb
)

train_pool = Pool(Xtr, y_train, cat_features=cat_cols)
valid_pool = Pool(Xval, y_val, cat_features=cat_cols)
best_cb.fit(train_pool, eval_set=valid_pool, use_best_model=True)
""",
    "shap": """
X_shap_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap_sample)

shap.summary_plot(
    shap_values,
    X_shap_sample,
    plot_type='dot',
    show=False
)
""",
    "manual_permutation_catboost": """
baseline = average_precision_score(y_test, best_cb.predict_proba(X_test)[:, 1])
results = []

for col in X_test.columns:
    scores = []
    for _ in range(20):
        X_perm = X_test.copy()
        X_perm[col] = np.random.permutation(X_perm[col].values)
        pr_auc = average_precision_score(y_test, best_cb.predict_proba(X_perm)[:, 1])
        scores.append(baseline - pr_auc)

    results.append((col, np.mean(scores), np.std(scores)))
"""
}

# =========================================================
# Page / Styling Helpers
# =========================================================
def set_page(title: str, icon: str = "📊"):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    inject_css()


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2.2rem;
        }

        .hero {
            padding: 1.6rem 1.7rem;
            border-radius: 1.2rem;
            background: linear-gradient(135deg, rgba(79,70,229,0.20), rgba(20,184,166,0.16));
            border: 1px solid rgba(100,116,139,0.22);
            margin-bottom: 1.2rem;
        }

        .hero h1 {
            margin: 0 0 0.2rem 0;
            font-size: 2.5rem;
            color: #f8fafc;
        }

        .hero p {
            margin: 0;
            color: #e2e8f0;
            font-size: 1.05rem;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin: 1.4rem 0 0.8rem 0;
            color: #f8fafc;
        }

        .feature-box, .timeline-step, .kpi-card {
            padding: 1rem 1.05rem;
            border-radius: 1rem;
            background: #1e293b;
            border: 1px solid rgba(148,163,184,0.22);
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.18);
            height: 100%;
            color: #f8fafc;
        }

        .feature-box h3,
        .timeline-step h4,
        .kpi-label,
        .kpi-value,
        .feature-box p,
        .timeline-step p {
            color: #f8fafc !important;
        }

        .feature-box h3,
        .timeline-step h4 {
            margin-top: 0;
            margin-bottom: 0.35rem;
        }

        .kpi-label {
            font-size: 0.85rem;
            margin-bottom: 0.2rem;
            color: #cbd5e1 !important;
        }

        .kpi-value {
            font-size: 1.45rem;
            font-weight: 750;
        }

        .timeline-row {
            margin-bottom: 1.3rem;
        }

        .metric-block {
            margin-bottom: 0.9rem;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #cbd5e1;
            margin-bottom: 3px;
        }

        .metric-value {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 4px;
            color: #f8fafc;
        }

        .metric-bar {
            width: 100%;
            height: 8px;
            border-radius: 999px;
            background: rgba(148,163,184,0.22);
            overflow: hidden;
        }

        .metric-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #14b8a6);
        }

        code {
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def feature_cards(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="feature-box">
                    <h3>{item['title']}</h3>
                    <p>{item['body']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def timeline_cards(items, n_cols: int = 3):
    for i in range(0, len(items), n_cols):
        st.markdown('<div class="timeline-row">', unsafe_allow_html=True)
        cols = st.columns(n_cols)
        for col, item in zip(cols, items[i:i + n_cols]):
            with col:
                st.markdown(
                    f"""
                    <div class="timeline-step">
                        <h4>{item['title']}</h4>
                        <p>{item['body']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


def kpi_cards(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{item['label']}</div>
                    <div class="kpi-value">{item['value']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def metric_row(metrics):
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            st.metric(metric["label"], metric["value"])


def metric_with_bar(label: str, value: float | None):
    if value is None:
        pct = 0
        value_txt = "—"
    else:
        pct = max(0, min(100, int(round(value * 100))))
        value_txt = f"{value:.3f}"

    st.markdown(
        f"""
        <div class="metric-block">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value_txt}</div>
            <div class="metric-bar">
                <div class="metric-bar-fill" style="width:{pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# Path Helpers
# =========================================================
def data_path(filename: str) -> Path:
    return DATA_DIR / filename


def fig_path(subfolder: str, filename: str) -> Path:
    return FIGURES_DIR / subfolder / filename


def art_path(subfolder: str, filename: str) -> Path:
    return ARTIFACTS_DIR / subfolder / filename

# =========================================================
# Load Helpers
# =========================================================
def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Could not read CSV: {path.name} ({e})")
            return pd.DataFrame()
    return pd.DataFrame()


def load_excel(path: Path, sheet_name=0) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except Exception as e:
            st.warning(f"Could not read Excel file: {path.name} ({e})")
            return pd.DataFrame()
    return pd.DataFrame()

# =========================================================
# Display Helpers
# =========================================================
def show_figure(
    subfolder: str,
    filename: str,
    caption: str | None = None,
    use_container_width: bool = True,
):
    path = fig_path(subfolder, filename)
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=use_container_width)
    else:
        st.warning(f"Missing figure: reports/figures/{subfolder}/{filename}")


def show_artifact_table(subfolder: str, filename: str):
    path = art_path(subfolder, filename)

    if not path.exists():
        st.warning(f"Missing artifact: reports/artifacts/{subfolder}/{filename}")
        return

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(f"Artifact found, but preview is not implemented for: {filename}")
    except Exception as e:
        st.error(f"Could not open artifact {filename}: {e}")

# =========================================================
# DataFrame Formatting
# =========================================================
def nice_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["PR-AUC", "Fraud Precision", "Fraud Recall", "Fraud F1", "Accuracy", "Threshold"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "—")
    return out