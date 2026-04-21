"""
Preprocessing script for Tech Hiring and Layoffs Dataset (2000-2025)
CS 4774 Final Project

Outputs:
  - processed_data.csv        : fully preprocessed dataset (all features)
  - X_train.csv, X_test.csv   : feature matrices (time-aware split)
  - y_train.csv, y_test.csv   : target vectors (layoff_pct)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv("tech_employment_2000_2025.csv")
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# ── 2. Target variable ─────────────────────────────────────────────────────────
# We predict PERCENTAGE of workforce laid off (as recommended by professor feedback)
# This accounts for differences in company size (e.g. 10k layoffs at Amazon vs a startup)
df["layoff_pct"] = df["layoffs"] / df["employees_start"] * 100

# Sanity check: flag any extreme outliers
outliers = df[df["layoff_pct"] > 50]
if not outliers.empty:
    print(f"\n⚠  {len(outliers)} row(s) with layoff_pct > 50%:")
    print(outliers[["company", "year", "layoffs", "employees_start", "layoff_pct"]])

# ── 3. Derived / engineered features ──────────────────────────────────────────
# These capture dynamics the raw columns don't directly express

# Workforce growth rate year-over-year (how fast was the company expanding?)
df["workforce_growth_pct"] = (
    (df["employees_end"] - df["employees_start"]) / df["employees_start"] * 100
)

# Hire-to-layoff ratio: high ratio = aggressive expansion; low = contraction mode
# Add 1 to avoid division by zero when layoffs == 0
df["hire_to_layoff_ratio"] = df["new_hires"] / (df["layoffs"] + 1)

# Revenue per employee (efficiency metric)
df["revenue_per_employee"] = (
    df["revenue_billions_usd"] * 1e9 / df["employees_start"]
)

# ── 4. Lag features (previous year's data for the same company) ────────────────
# These are critical for time series-aware models and the baseline strategies.
# We sort first so shift() works correctly within each company group.
df = df.sort_values(["company", "year"]).reset_index(drop=True)

for col in ["layoff_pct", "employees_start", "revenue_billions_usd",
            "stock_price_change_pct", "new_hires"]:
    df[f"{col}_lag1"] = df.groupby("company")[col].shift(1)

print(f"\nRows with NaN lag features (first year per company): "
      f"{df['layoff_pct_lag1'].isna().sum()}")

# ── 5. Encode categorical columns ──────────────────────────────────────────────

# 5a. One-hot encode company (25 companies → 25 binary columns)
df = pd.get_dummies(df, columns=["company"], prefix="co")

# 5b. Ordinal encode confidence_level  (Medium=0, High=1)
df["confidence_level"] = df["confidence_level"].map({"Medium": 0, "High": 1})

# 5c. is_estimated is already bool → cast to int (False=0, True=1)
df["is_estimated"] = df["is_estimated"].astype(int)

# ── 6. Drop columns we won't use as model features ────────────────────────────
# - layoffs (raw count) is replaced by layoff_pct
# - net_change / employees_end are downstream of layoffs (data leakage risk)
# - attrition_rate_pct is derived from layoffs (also leakage)
# - data_quality_score has essentially no variance — won't help
df = df.drop(columns=["layoffs", "net_change", "employees_end",
                       "attrition_rate_pct", "data_quality_score"])

print(f"\nFinal columns ({len(df.columns)}):")
print(df.columns.tolist())

# ── 7. Time-aware train/test split ────────────────────────────────────────────
# IMPORTANT: never train on future years — that would be data leakage.
# Train on 2001–2019, test on 2020–2025.
# This also naturally captures the COVID + post-pandemic layoff wave in the test set.
SPLIT_YEAR = 2020

target = "layoff_pct"
feature_cols = [c for c in df.columns if c not in [target, "year"]]

train = df[df["year"] < SPLIT_YEAR].dropna(subset=feature_cols + [target])
test  = df[df["year"] >= SPLIT_YEAR].dropna(subset=feature_cols + [target])

X_train = train[feature_cols]
y_train = train[target]
X_test  = test[feature_cols]
y_test  = test[target]

print(f"\nTrain: {len(train)} rows ({train['year'].min()}–{train['year'].max()})")
print(f"Test:  {len(test)} rows  ({test['year'].min()}–{test['year'].max()})")

# ── 8. Feature scaling (L2 / StandardScaler) ──────────────────────────────────
# Scale numeric columns only (leave one-hot columns as-is)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Identify one-hot company columns (already 0/1, no need to scale)
onehot_cols = [c for c in numeric_cols if c.startswith("co_")]
scale_cols  = [c for c in numeric_cols if c not in onehot_cols]

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])   # use train stats!

print(f"\nScaled {len(scale_cols)} numeric features.")
print(f"Left {len(onehot_cols)} one-hot company features unscaled.")

# ── 9. Save outputs ────────────────────────────────────────────────────────────
df.to_csv("processed_data.csv", index=False)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv",  index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv",  index=False)

print("\n✅ Saved: processed_data.csv, X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print(f"\nFeature matrix shape — Train: {X_train.shape}, Test: {X_test.shape}")