"""
Preprocessing script for Tech Hiring and Layoffs Dataset (2000-2025)
CS 4774 Final Project

Target: predict NEXT year's layoff percentage for a given company,
        using the current year's features as input.

Outputs:
  - processed_data.csv        : fully preprocessed dataset (all features)
  - X_train.csv, X_test.csv   : feature matrices (time-aware split)
  - y_train.csv, y_test.csv   : target vectors (layoff_pct_next)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv("tech_employment_2000_2025.csv")
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
df["layoff_pct_log"] = np.log1p(df["layoffs"] / df["employees_start"] * 100)

# Creating layoff pct next feature, which will be used as target feature for our data to prevent data leakage.
df = df.sort_values(["company", "year"]).reset_index(drop=True)
# df["layoff_pct_next"] = df.groupby("company")["layoff_pct"].shift(-1)
df["layoff_pct_next_log"] = df.groupby("company")["layoff_pct_log"].shift(-1)

# Feature Engineering
df["workforce_growth_pct"] = (
    (df["employees_end"] - df["employees_start"]) / df["employees_start"] * 100
)
df["hire_to_layoff_ratio"] = df["new_hires"] / (df["layoffs"] + 1)
df["revenue_per_employee"] = (
    df["revenue_billions_usd"] * 1e9 / df["employees_start"]
)

df = df.dropna()

df = pd.get_dummies(df, columns=["company"], prefix="co")
df = df.drop(columns=["confidence_level", "is_estimated", "layoffs", "data_quality_score"])


# Adding macroeconomic factors as data:
macro_features = {
    2000: {"fed_rate": 6.5,  "sp500_return": -9.1,  "recession": 0},
    2001: {"fed_rate": 1.75, "sp500_return": -11.9, "recession": 1},
    2002: {"fed_rate": 1.25, "sp500_return": -22.1, "recession": 0},
    2003: {"fed_rate": 1.0,  "sp500_return": 28.7,  "recession": 0},
    2004: {"fed_rate": 2.25, "sp500_return": 10.9,  "recession": 0},
    2005: {"fed_rate": 4.25, "sp500_return": 4.9,   "recession": 0},
    2006: {"fed_rate": 5.25, "sp500_return": 15.8,  "recession": 0},
    2007: {"fed_rate": 4.25, "sp500_return": 5.5,   "recession": 0},
    2008: {"fed_rate": 0.25, "sp500_return": -37.0, "recession": 1},
    2009: {"fed_rate": 0.25, "sp500_return": 26.5,  "recession": 1},
    2010: {"fed_rate": 0.25, "sp500_return": 15.1,  "recession": 0},
    2011: {"fed_rate": 0.25, "sp500_return": 2.1,   "recession": 0},
    2012: {"fed_rate": 0.25, "sp500_return": 16.0,  "recession": 0},
    2013: {"fed_rate": 0.25, "sp500_return": 32.4,  "recession": 0},
    2014: {"fed_rate": 0.25, "sp500_return": 13.7,  "recession": 0},
    2015: {"fed_rate": 0.5,  "sp500_return": 1.4,   "recession": 0},
    2016: {"fed_rate": 0.75, "sp500_return": 12.0,  "recession": 0},
    2017: {"fed_rate": 1.5,  "sp500_return": 21.8,  "recession": 0},
    2018: {"fed_rate": 2.5,  "sp500_return": -4.4,  "recession": 0},
    2019: {"fed_rate": 1.75, "sp500_return": 31.5,  "recession": 0},
    2020: {"fed_rate": 0.25, "sp500_return": 18.4,  "recession": 1},
    2021: {"fed_rate": 0.25, "sp500_return": 28.7,  "recession": 0},
    2022: {"fed_rate": 4.5,  "sp500_return": -18.1, "recession": 0},
    2023: {"fed_rate": 5.5,  "sp500_return": 26.3,  "recession": 0},
    2024: {"fed_rate": 4.5,  "sp500_return": 25.0,  "recession": 0},
}

macro_df = pd.DataFrame(macro_features).T.reset_index().rename(columns={"index": "year"})
macro_df["year"] = macro_df["year"].astype(int)

df = df.merge(macro_df, on="year", how="left")

print(f"\nFinal columns ({len(df.columns)}):")
print(df.columns.tolist())
SPLIT_YEAR1 = 2020   # rows with year < SPLIT_YEAR go to train
SPLIT_YEAR2 = 2023

target = "layoff_pct_next_log"
feature_cols = [c for c in df.columns if c != target]

train = df[df["year"] < SPLIT_YEAR1].dropna(subset=feature_cols + [target])
rest = df[df["year"] >= SPLIT_YEAR1].dropna(subset=feature_cols + [target])

validate = rest[rest["year"] < SPLIT_YEAR2].dropna(subset=feature_cols + [target])
test  = rest[rest["year"] >= SPLIT_YEAR2].dropna(subset=feature_cols + [target])

X_train = train[feature_cols]
y_train = train[target]
X_validate = validate[feature_cols]
y_validate = validate[target]
X_test  = test[feature_cols]
y_test  = test[target]

print(f"\nTrain: {len(train)} rows  (input years {train['year'].min()}–{train['year'].max()}"
      f" → predicts {train['year'].min()+1}–{train['year'].max()+1})")
print(f"\nValidate: {len(validate)} rows  (input years {validate['year'].min()}–{validate['year'].max()}"
      f" → predicts {validate['year'].min()+1}–{validate['year'].max()+1})")
print(f"Test:  {len(test)} rows   (input years {test['year'].min()}–{test['year'].max()}"
      f" → predicts {test['year'].min()+1}–{test['year'].max()+1})")

# Feature scaling
# numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
# onehot_cols = [c for c in numeric_cols if c.startswith("co_")]
# scale_cols  = [c for c in numeric_cols if c not in onehot_cols]

# scaler = StandardScaler()
# X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
# X_validate[scale_cols] = scaler.fit_transform(X_validate[scale_cols])
# X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

df.to_csv("processed_data.csv", index=False)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv",  index=False)
X_validate.to_csv("X_validate.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv",  index=False)
y_validate.to_csv("y_validate.csv", index=False)

print(f"\nFeature matrix shape — Train: {X_train.shape}, Validate: {X_validate.shape}, Test: {X_test.shape}")