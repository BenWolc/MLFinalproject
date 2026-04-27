import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, DLinear
from neuralforecast.losses.pytorch import MAE
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics import r2_score

# ── 1. Rebuild long-format time series from processed_data.csv ────────────────
# NeuralForecast requires: unique_id, ds (datetime), y, + any covariates.
# The one-hot company columns let us reconstruct the company identifier.
df = pd.read_csv("processed_data.csv")

# Recover company name from one-hot columns
co_cols = [c for c in df.columns if c.startswith("co_")]
df["unique_id"] = (
    df[co_cols]
    .idxmax(axis=1)          # name of the 1-valued column, e.g. "co_Apple"
    .str.replace("co_", "", regex=False)
)

# NeuralForecast needs ds as a proper datetime
df["ds"] = pd.to_datetime(df["year"], format="%Y")

# Target is next year's layoff percentage
df = df.rename(columns={"layoff_pct_next_log": "y"})
print(df.columns)

# ── 2. Define exogenous covariates ────────────────────────────────────────────
# NeuralForecast distinguishes between:
#   hist_exog_list  — covariates known historically but NOT at forecast time
#   futr_exog_list  — covariates known at forecast time too (e.g. year)
#
# Lag features are historical only; year is known in the future.
hist_exog = [
    "employees_start","employees_end","new_hires","net_change","hiring_rate_pct",
    "attrition_rate_pct", "revenue_billions_usd", "stock_price_change_pct",
    "gdp_growth_us_pct", "unemployment_rate_us_pct", "layoff_pct_log",
    "workforce_growth_pct", "hire_to_layoff_ratio", "revenue_per_employee",
    "fed_rate", "sp500_return"
]
futr_exog = ["year"]

# Drop rows with NaN in y or any covariate
keep_cols = ["unique_id", "ds", "y"] + hist_exog + futr_exog
nf_df = df[keep_cols].dropna().reset_index(drop=True)

# ── 3. Time-aware train/test split ────────────────────────────────────────────
# Same boundary as the preprocessing script: input year < 2019 → train,
# input year >= 2019 → test. NeuralForecast takes the full df at fit time
# and you pass the test portion separately to predict().
SPLIT_YEAR = 2019
train_df = nf_df[nf_df["year"] < SPLIT_YEAR]
test_df  = nf_df[nf_df["year"] >= SPLIT_YEAR]

# ── 4. Build and fit the model ────────────────────────────────────────────────
# horizon=1 because we predict one year ahead.
# input_size is how many past timesteps the model sees — 5 years of history
# is reasonable given the dataset starts in 2000.
HORIZON = 1

model = NeuralForecast(
    models=[
        NHITS(
            h=HORIZON,
            input_size=3,
            hist_exog_list=hist_exog,
            futr_exog_list=futr_exog,
            loss=MAE(),
            max_steps=2000,
            learning_rate=1e-3,
            scaler_type='robust',
            early_stop_patience_steps=50,
            val_check_steps=50,
        ),
        DLinear(
            h=HORIZON,
            input_size=3,
            loss=MAE(),
            max_steps=2000,
            learning_rate=1e-3,
            scaler_type='robust',
            early_stop_patience_steps=50,
            val_check_steps=50,
        ),
    ],
    freq="YS",
)

model.fit(train_df, val_size=2)

# ── 5. Predict ────────────────────────────────────────────────────────────────
# With h=1, predict() returns one forecast per unique_id (the year immediately
# after the last training timestep). futr_df must have exactly 1 row per series.
# We take only the first test year per company to match this constraint.
first_test_year = (
    test_df.groupby("unique_id")["ds"].min().reset_index()
)
futr_df = first_test_year.merge(
    test_df[["unique_id", "ds"] + futr_exog],
    on=["unique_id", "ds"],
    how="left"
)

preds = model.predict(futr_df=futr_df)
preds = preds.merge(
    test_df[["unique_id", "ds", "y"]],
    on=["unique_id", "ds"],
    how="left"
)

model_cols = ["NHITS", "DLinear"]

for col in model_cols:
    if col not in preds.columns:
        continue
    subset = preds.dropna(subset=["y", col])
    mae  = mean_absolute_error(subset["y"], subset[col])
    rmse = root_mean_squared_error(subset["y"], subset[col])
    r2   = r2_score(subset["y"], subset[col])
    print(f"\n── {col} ──")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²:   {r2:.3f}")

print("\n── Side-by-side predictions ──")
print(preds[["unique_id", "ds", "y"] + model_cols].head(15))