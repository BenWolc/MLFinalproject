"""
Best Classification Model for Layoff Prediction
CS 4774 Final Project

Key insight from data analysis:
- 77% of all company-years have exactly ~3% layoffs (very narrow band)
- Only ~22% exceed 3%, making "above normal" a meaningful signal
- Post-2020 has 41% high-layoff years vs 15% pre-2020 — big distribution shift
- Test set only has 4% high-layoff years — extremely imbalanced

Strategy:
- Use threshold of >3% (above the "normal" band) for better class balance
- Use cross-validation on training data rather than relying on tiny test set
- Try multiple classifiers and pick best by ROC-AUC (robust to imbalance)
- Use SMOTE-style class weighting to handle imbalance
- Report both CV results AND test results honestly
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score, precision_score, recall_score,
                             accuracy_score, f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────────
X_train    = pd.read_csv("X_train.csv")
X_validate = pd.read_csv("X_validate.csv")
X_test     = pd.read_csv("X_test.csv")
y_train    = pd.read_csv("y_train.csv").squeeze()
y_validate = pd.read_csv("y_validate.csv").squeeze()
y_test     = pd.read_csv("y_test.csv").squeeze()

# ── Drop leaky columns ─────────────────────────────────────────────────────────
LEAKY = ["employees_end", "net_change", "attrition_rate_pct",
         "workforce_growth_pct", "hire_to_layoff_ratio", "layoff_pct_log"]
def drop_leaky(df):
    return df.drop(columns=[c for c in LEAKY if c in df.columns])

# X_train    = drop_leaky(X_train)
# X_validate = drop_leaky(X_validate)
# X_test     = drop_leaky(X_test)

# Combine train + validate for cross-validation
# (gives more data for CV while still keeping test set untouched)
X_trainval = pd.concat([X_train, X_validate], ignore_index=True)
y_trainval = pd.concat([y_train, y_validate], ignore_index=True)

# ── Define classification threshold ───────────────────────────────────────────
# >3% = "elevated layoffs" (above the normal steady-state band)
# This gives ~22% positive class — much better balance than 5% threshold
THRESHOLD = np.log1p(3.0)

y_train_clf    = (y_train    > THRESHOLD).astype(int)
y_validate_clf = (y_validate > THRESHOLD).astype(int)
y_test_clf     = (y_test     > THRESHOLD).astype(int)
y_trainval_clf = (y_trainval > THRESHOLD).astype(int)

print("=" * 60)
print("  CLASSIFICATION SETUP")
print("=" * 60)
print(f"Threshold: layoff % > 3% of workforce next year")
print(f"Train+Val positive rate: {y_trainval_clf.mean():.1%}")
print(f"Test positive rate:      {y_test_clf.mean():.1%}")
print(f"Features: {X_train.shape[1]}")

# ── Cross validation setup ─────────────────────────────────────────────────────
# Stratified 5-fold — preserves class ratio in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Models to compare ─────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                   max_iter=1000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=4, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        scale_pos_weight=(y_trainval_clf==0).sum()/(y_trainval_clf==1).sum(),
        random_state=42, verbosity=0, eval_metric="logloss"
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=1.0, class_weight="balanced", probability=True,
                    random_state=42))
    ]),
}

# ── Cross-validation results ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  5-FOLD CROSS-VALIDATION RESULTS (Train+Validate)")
print("=" * 60)
print(f"{'Model':<25} {'ROC-AUC':>10} {'F1':>10} {'Accuracy':>10}")
print("─" * 60)

cv_results = []
for name, model in models.items():
    auc_scores = cross_val_score(model, X_trainval, y_trainval_clf,
                                  cv=cv, scoring="roc_auc")
    
    recall_scores = cross_val_score(model, X_trainval, y_trainval_clf,
                                  cv=cv, scoring="recall")
    precision_scores = cross_val_score(model, X_trainval, y_trainval_clf,
                                  cv=cv, scoring="precision")
    f1_scores  = cross_val_score(model, X_trainval, y_trainval_clf,
                                  cv=cv, scoring="f1")
    acc_scores = cross_val_score(model, X_trainval, y_trainval_clf,
                                  cv=cv, scoring="accuracy")
    print(f"{name:<25} {auc_scores.mean():.4f}±{auc_scores.std():.3f}"
          f" {f1_scores.mean():.4f}±{f1_scores.std():.3f}"
          f" {acc_scores.mean():.4f}±{acc_scores.std():.3f}")
    cv_results.append({
        "Model": name,
        "CV_ROC_AUC": round(auc_scores.mean(), 4),
        "CV_F1": round(f1_scores.mean(), 4),
        "CV_Accuracy": round(acc_scores.mean(), 4),
        "CV_Recall": round(recall_scores.mean(), 4),
        "CV_Precision": round(precision_scores.mean(), 4),
    })

# ── Pick best model by ROC-AUC and evaluate on test set ───────────────────────
best_name = max(cv_results, key=lambda x: x["CV_ROC_AUC"])["Model"]
print(f"\nBest model by CV ROC-AUC: {best_name}")

best_model = models[best_name]
best_model.fit(X_trainval, y_trainval_clf)
test_preds = best_model.predict(X_test)
test_probs = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print(f"  BEST MODEL ({best_name}) — TEST SET RESULTS")
print("=" * 60)
print(f"  Accuracy : {accuracy_score(y_test_clf, test_preds):.4f}")
print(f"  F1 Score : {f1_score(y_test_clf, test_preds, zero_division=0):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test_clf, test_probs):.4f}")
print()
print(classification_report(y_test_clf, test_preds,
      target_names=["Normal (<3%)", "Elevated (>3%)"], zero_division=0))

cm = confusion_matrix(y_test_clf, test_preds)
print("Confusion Matrix:")
print(f"  True Negative  (correctly said normal):   {cm[0,0]}")
print(f"  False Positive (wrongly flagged elevated): {cm[0,1]}")
print(f"  False Negative (missed elevated):          {cm[1,0]}")
print(f"  True Positive  (correctly flagged):        {cm[1,1]}")

# ── Also evaluate ALL models on test set ──────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL MODELS — TEST SET SUMMARY")
print("=" * 60)
print(f"{'Model':<25} {'ROC-AUC':>10} {'F1':>8} {'Accuracy':>10}")
print("─" * 60)

test_results = []
for name, model in models.items():
    model.fit(X_trainval, y_trainval_clf)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_clf, probs)
    f1  = f1_score(y_test_clf, preds, zero_division=0)
    acc = accuracy_score(y_test_clf, preds)
    precision = precision_score(y_test_clf, preds)
    recall = recall_score(y_test_clf, preds)
    print(f"{name:<25} {auc:.4f}     {f1:.4f}   {acc:.4f}")
    test_results.append({
        "Model": name,
        "Test_ROC_AUC": round(auc,4),
        "Test_F1": round(f1,4),
        "Test_Accuracy": round(acc,4),
        "Test_precision": round(precision,4),
        "Test_recall": round(recall,4)
    })

# ── Feature importance from Random Forest ─────────────────────────────────────
rf = models["Random Forest"]
rf.fit(X_trainval, y_trainval_clf)
feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\n  Top 10 Features (Random Forest importance):")
print(feat_imp.head(10).to_string(index=False))

# ── Save results ───────────────────────────────────────────────────────────────
cv_df   = pd.DataFrame(cv_results)
test_df = pd.DataFrame(test_results)
merged  = cv_df.merge(test_df, on="Model")
merged.to_csv("classification_results.csv", index=False)
feat_imp.to_csv("classification_feature_importance.csv", index=False)

print("\n✅ Saved: classification_results.csv")
print("✅ Saved: classification_feature_importance.csv")
print("\n" + "=" * 60)
print("  FULL RESULTS TABLE")
print("=" * 60)
print(merged.to_string(index=False))