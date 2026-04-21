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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# ── Load data ──────────────────────────────────────────────────────────────────
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()  # squeeze to Series
y_test  = pd.read_csv("y_test.csv").squeeze()

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

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 1: Predict Last Year's Value
# For each (company, year) in the test set, predict using that company's
# layoff_pct from the previous year.
# This is exactly what the lag1 feature captures — but we evaluate it standalone.
# ══════════════════════════════════════════════════════════════════════════════
df_sorted = raw.sort_values(["company", "year"]).copy()
df_sorted["prev_year_pred"] = df_sorted.groupby("company")["layoff_pct"].shift(1)

test_df = df_sorted[df_sorted["year"] >= SPLIT_YEAR].dropna(subset=["prev_year_pred", "layoff_pct"])
results.append(report(
    "Baseline 1: Last Year's Value",
    test_df["layoff_pct"],
    test_df["prev_year_pred"]
))

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 2: Predict Historical Average
# For each company, compute the mean layoff_pct over all TRAINING years,
# then use that single number to predict every test year for that company.
# ══════════════════════════════════════════════════════════════════════════════
train_df = df_sorted[df_sorted["year"] < SPLIT_YEAR]
company_avg = train_df.groupby("company")["layoff_pct"].mean().rename("hist_avg_pred")

test_df2 = df_sorted[df_sorted["year"] >= SPLIT_YEAR].join(company_avg, on="company")
test_df2 = test_df2.dropna(subset=["hist_avg_pred", "layoff_pct"])

results.append(report(
    "Baseline 2: Historical Average",
    test_df2["layoff_pct"],
    test_df2["hist_avg_pred"]
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
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train, y_train)

ridge_preds = ridge.predict(X_test)

# Clip negatives — layoff % can't be negative
ridge_preds = np.clip(ridge_preds, 0, None)

print(f"\n  Best Ridge alpha chosen by CV: {ridge.alpha_}")
results.append(report(
    "Model 1: Ridge Linear Regression",
    y_test,
    ridge_preds
))

# Top features by coefficient magnitude
feature_names = X_train.columns.tolist()
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": ridge.coef_
}).reindex(pd.Series(ridge.coef_).abs().sort_values(ascending=False).index)

print("\n  Top 10 most important features (by |coefficient|):")
print(coef_df.head(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Summary Table
# ══════════════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 55)
print("  RESULTS SUMMARY")
print("=" * 55)
summary = pd.DataFrame(results)
print(summary.to_string(index=False))
summary.to_csv("baseline_results.csv", index=False)
print("\n Saved: baseline_results.csv")