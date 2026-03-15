# LinkedIn Post: Insurance Fraud Detection Project

🎯 Just completed my Capstone Project: Building an ML system to detect fraudulent insurance claims!

This end-to-end project tackles a real-world problem with imbalanced data (~24.7% fraud) where missing fraud cases is costly. Here's what I built:

---

## 🔧 Technical Approach

**4 Models Compared:**
- Logistic Regression (baseline)
- Random Forest (with SMOTE & feature selection)
- XGBoost ⭐ (best performer)
- CatBoost

**Key Challenge:** Class imbalance + need for interpretability

---

## 💻 Code Highlights

### 1️⃣ XGBoost with Native Categorical Features

Instead of one-hot encoding, I leveraged XGBoost's native categorical handling:

```python
# Handle class imbalance
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

# XGBoost with native categorical support
xgb_clf = XGBClassifier(
    enable_categorical=True,  # No OHE needed!
    scale_pos_weight=scale_weight,
    tree_method='hist',
    eval_metric='aucpr',
    random_state=42
)

# Hyperparameter tuning with GridSearchCV
grid_xgb = GridSearchCV(
    xgb_clf,
    param_grid={
        'n_estimators': [300, 600, 1000],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [2, 3, 5],
        'min_child_weight': [5, 10],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0]
    },
    cv=StratifiedKFold(n_splits=5, shuffle=True),
    scoring='average_precision',
    n_jobs=-1
)
```

---

### 2️⃣ Threshold Optimization for Business Alignment

Default 0.5 threshold? Not for imbalanced data! I optimized thresholds to maximize recall while maintaining minimum precision:

```python
def threshold_opt_recall_with_precision(y_true, y_proba, 
                                        precision_min=0.35):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find threshold maximizing recall with precision constraint
    mask = precision[1:] >= precision_min
    if mask.any():
        best_idx = np.argmax(recall[1:][mask])
        best_threshold = thresholds[mask][best_idx]
    else:
        best_idx = np.argmax(recall[1:])
        best_threshold = thresholds[best_idx]
    
    return best_threshold, precision[1:][best_idx], recall[1:][best_idx]
```

This ensures models operate at business-aligned decision points, not arbitrary defaults!

**[IMAGE: Precision-Recall Curve with optimal threshold marked]**
*Visualizing the precision-recall trade-off helps identify the best operating point for business needs*

---

### 3️⃣ Model Interpretability with SHAP

Understanding *why* a model predicts fraud is crucial for business adoption:

```python
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test_sample)

# Visualize feature importance
shap.summary_plot(
    shap_values[1],  # Fraud class
    X_test_sample,
    feature_names=X_test.columns,
    show=False
)
plt.title("SHAP Summary - Fraud Class")
plt.savefig('shap_summary.png')
```

**Key Finding:** `incident_severity` consistently emerged as the strongest predictor across all models! 📊

**[IMAGE: SHAP Summary Plot]**
*SHAP values reveal that incident_severity dominates fraud predictions, with claim amounts as secondary signals*

---

## 📈 Results

**[IMAGE: Model Comparison Table]**
*Comparison of all 4 models showing XGBoost's superior performance*

**Best Model (XGBoost):**
- **Recall:** 84% (catches most fraud cases)
- **Precision:** 55-64% (depending on threshold)
- **F1 Score:** 0.67-0.72
- **PR-AUC:** 0.55+

**Why XGBoost won:**
✅ Best precision-recall balance
✅ Native categorical handling (no encoding overhead)
✅ Strong regularization prevents overfitting
✅ Excellent interpretability via SHAP

---

## ✨ Best Practices Demonstrated

✅ **Statistical Rigor:** Mann-Whitney U & Chi-Square tests with effect sizes
✅ **Proper Validation:** Stratified K-Fold cross-validation
✅ **Business Alignment:** Threshold optimization for real-world deployment
✅ **Interpretability:** SHAP + Permutation Importance
✅ **Feature Engineering:** Created `has_umbrella_policy` binary feature
✅ **Reproducibility:** Full pipeline documented in Jupyter notebooks

---

## 🎓 Key Learnings

1. **Threshold optimization** is critical for imbalanced problems - default 0.5 rarely works!
2. **Native categorical handling** (XGBoost/CatBoost) beats one-hot encoding for efficiency
3. **SHAP** bridges the gap between model performance and business trust
4. **Statistical testing** validates feature importance beyond correlation

---

🔗 Full project available on GitHub: [Link to your repo]

Would love to hear your thoughts on handling imbalanced classification problems! What strategies have worked best for you?

---

#MachineLearning #DataScience #FraudDetection #XGBoost #SHAP #Python #ImbalancedLearning #MLOps #DataScientest #CapstoneProject
