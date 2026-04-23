import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

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
df = df.rename(columns={"layoff_pct_next": "y"})

# ── 2. Define exogenous covariates ────────────────────────────────────────────
# NeuralForecast distinguishes between:
#   hist_exog_list  — covariates known historically but NOT at forecast time
#   futr_exog_list  — covariates known at forecast time too (e.g. year)
#
# Lag features are historical only; year is known in the future.
hist_exog = [
    "layoff_pct", "workforce_growth_pct", "hire_to_layoff_ratio",
    "revenue_per_employee", "layoff_pct_lag1", "employees_start_lag1",
    "revenue_billions_usd_lag1", "stock_price_change_pct_lag1", "new_hires_lag1",
    "confidence_level", "is_estimated",
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
            input_size=5,
            hist_exog_list=hist_exog,
            futr_exog_list=futr_exog,
            loss=MAE(),
            max_steps=500,
            learning_rate=1e-3,
        )
    ],
    freq="YS",   # yearly, start of year
)

model.fit(train_df)

# ── 5. Predict ────────────────────────────────────────────────────────────────
# predict() needs the future exogenous values for the forecast horizon.
# Here that's just the year column for each company.
futr_df = test_df[["unique_id", "ds"] + futr_exog].drop_duplicates()

preds = model.predict(futr_df=futr_df)
preds = preds.merge(
    test_df[["unique_id", "ds", "y"]],
    on=["unique_id", "ds"],
    how="left"
)

print(f"MAE:  {mean_absolute_error(preds['y'], preds['NHITS']):.3f} percentage points")
print(f"RMSE: {root_mean_squared_error(preds['y'], preds['NHITS']):.3f} percentage points")
print(preds[["unique_id", "ds", "y", "NHITS"]].head(15))