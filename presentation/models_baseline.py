"""
Baseline Models + Ridge Linear Regression
CS 4774 Final Project

Models:
  1. Predict Last Year's Value  (naive baseline)
  2. Predict Historical Average (naive baseline)
  3. Ridge Linear Regression    (first real model)

Metrics reported: MAE, RMSE, R²
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
X_validate = pd.read_csv("X_validate.csv")
y_train = pd.read_csv("y_train.csv").squeeze()  # squeeze to Series
y_test  = pd.read_csv("y_test.csv").squeeze()
y_validate = pd.read_csv("y_validate.csv").squeeze()

X_train_orig = X_train.copy()
X_test_orig = X_test.copy()
X_validate_orig = X_validate.copy()

# Feature scaling
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
onehot_cols = [c for c in numeric_cols if c.startswith("co_")]
scale_cols  = [c for c in numeric_cols if c not in onehot_cols]

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_validate[scale_cols] = scaler.fit_transform(X_validate[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

# Load processed data for model features
df = pd.read_csv("processed_data.csv")

# Also load the raw data to get the original company labels for naive baselines
# (company was one-hot encoded in processed_data, so we reconstruct from raw)
raw = pd.read_csv("tech_employment_2000_2025.csv")
raw["layoff_pct"] = raw["layoffs"] / raw["employees_start"] * 100

SPLIT_YEAR = 2020

# ── Helper: print metrics ──────────────────────────────────────────────────────
def report(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  MAE  : {mae:.4f}%")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  R²   : {r2:.4f}")
    return {"Model": name, "MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,4)}

results = []

# BASELINE 1: Predict Last Year's Value
df_sorted = raw.sort_values(["company", "year"]).copy()
df_sorted["layoff_pct_next"] = df_sorted.groupby("company")["layoff_pct"].shift(-1)

test_df = df_sorted[df_sorted["year"] >= SPLIT_YEAR].dropna(subset=["layoff_pct", "layoff_pct_next"])
results.append(report(
    "Baseline 1: Last Year's Value",
    np.log1p(test_df["layoff_pct_next"]),
    np.log1p(test_df["layoff_pct"])
))

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 2: Predict Historical Average
# For each company, compute the mean layoff_pct over all TRAINING years,
# then use that single number to predict every test year for that company.
# ══════════════════════════════════════════════════════════════════════════════
train_df = df_sorted[df_sorted["year"] < SPLIT_YEAR]
company_avg = train_df.groupby("company")["layoff_pct"].mean().rename("hist_avg_pred")

test_df2 = df_sorted[df_sorted["year"] >= SPLIT_YEAR].join(company_avg, on="company")
test_df2 = test_df2.dropna(subset=["hist_avg_pred", "layoff_pct_next"])

results.append(report(
    "Baseline 2: Historical Average",
    np.log1p(test_df2["layoff_pct_next"]),
    np.log1p(test_df2["hist_avg_pred"])
))

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 3: Global Mean
# Predict the overall training mean for every test example.
# The simplest possible model — everything else should beat this.
# ══════════════════════════════════════════════════════════════════════════════
global_mean = y_train.mean()
global_mean_preds = np.full(len(y_test), global_mean)

results.append(report(
    "Baseline 3: Global Mean",
    y_test,
    global_mean_preds
))

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: Ridge Linear Regression
# Uses RidgeCV to automatically pick the best regularization strength (alpha)
# via cross-validation on the training set.
# ══════════════════════════════════════════════════════════════════════════════

alphas = [0.1, 1, 10, 100, 1000]

best_alpha, best_score = None, -np.inf

for alpha in alphas:
    model = Ridge(alpha)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_validate)
    score = r2_score(y_validate, val_preds)
    print(f"alpha={alpha:.1f}  val R²={score:.3f}")
    if score > best_score:
        best_score, best_alpha = score, alpha


final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train, y_train)
test_preds = final_model.predict(X_test)
print(f"\n  Best Ridge alpha chosen by CV: {best_alpha}")
results.append(report(
    "Model 1: Ridge Linear Regression",
    y_test,
    test_preds
))

# Top features by coefficient magnitude
feature_names = X_train.columns.tolist()
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": final_model.coef_
}).reindex(pd.Series(final_model.coef_).abs().sort_values(ascending=False).index)

print("\n  Top 10 most important features (by |coefficient|):")
print(coef_df.head(10).to_string(index=False))

print("Residuals for Ridge Regularized Linear Regression: ")
residuals = pd.DataFrame({
    "year":      X_test_orig["year"].values,
    "actual":    y_test.values,
    "predicted": test_preds,
    "error":     test_preds - y_test.values,
    "abs_error": np.abs(test_preds - y_test.values),
})
print(residuals.sort_values("abs_error", ascending=False).head(10))

# print("Residuals for Global Mean Prediction: ")
# residuals = pd.DataFrame({
#     "year":      X_test["year"].values,
#     "actual":    y_test.values,
#     "predicted": test_preds,
#     "error":     test_preds - y_test.values,
#     "abs_error": np.abs(test_preds - y_test.values),
# })
# print(residuals.sort_values("abs_error", ascending=False).head(10))

# ══════════════════════════════════════════════════════════════════════════════
# Summary Table
# ══════════════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 55)
print("  RESULTS SUMMARY")
print("=" * 55)
summary = pd.DataFrame(results)
print(summary.to_string(index=False))