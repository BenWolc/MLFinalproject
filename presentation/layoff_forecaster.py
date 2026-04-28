"""
Improved Tech Layoff Forecaster

Run from inside the presentation folder:

    python improved_layoff_forecaster.py

Expected files in same folder:
    tech_employment_2000_2025.csv
    layoffs.csv

Outputs:
    improved_predictions.csv
    improved_walk_forward_cv.csv
    improved_walk_forward_summary.csv
"""

import os
import re
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

PRIMARY_DATA = "tech_employment_2000_2025.csv"
EVENT_DATA = "layoffs.csv"

RANDOM_STATE = 42

TRAIN_END_YEAR = 2020
VALID_END_YEAR = 2023
TEST_START_YEAR = 2023

ELEVATED_THRESHOLD = 6.0


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def print_table(title, df, max_rows=25):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    if df.empty:
        print("No rows.")
        return

    show = df.head(max_rows).copy()
    for col in show.select_dtypes(include=[np.number]).columns:
        show[col] = show[col].round(4)

    print(show.to_string(index=False))


def normalize_company_name(name):
    if pd.isna(name):
        return ""

    raw = str(name).strip()
    low = raw.lower()
    clean = re.sub(r"[^a-z0-9]+", " ", low).strip()

    aliases = {
        "amd": "AMD",
        "adobe": "Adobe",
        "airbnb": "Airbnb",
        "alphabet": "Alphabet",
        "google": "Alphabet",
        "amazon": "Amazon",
        "apple": "Apple",
        "block": "Block",
        "square": "Block",
        "intel": "Intel",
        "linkedin": "LinkedIn",
        "linked in": "LinkedIn",
        "lyft": "Lyft",
        "meta": "Meta",
        "facebook": "Meta",
        "microsoft": "Microsoft",
        "nvidia": "NVIDIA",
        "netflix": "Netflix",
        "oracle": "Oracle",
        "paypal": "PayPal",
        "pay pal": "PayPal",
        "pinterest": "Pinterest",
        "sap": "SAP",
        "salesforce": "Salesforce",
        "shopify": "Shopify",
        "snap": "Snap",
        "snapchat": "Snap",
        "stripe": "Stripe",
        "tesla": "Tesla",
        "uber": "Uber",
        "twitter": "X (Twitter)",
        "x twitter": "X (Twitter)",
        "x": "X (Twitter)",
    }

    if clean in aliases:
        return aliases[clean]

    for key, value in aliases.items():
        if re.search(rf"\b{re.escape(key)}\b", clean):
            return value

    return raw


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

def add_macro_features(df):
    macro_features = {
        2000: {"fed_rate": 6.50, "sp500_return": -9.1, "nasdaq_return": -39.3, "recession": 0},
        2001: {"fed_rate": 1.75, "sp500_return": -11.9, "nasdaq_return": -21.1, "recession": 1},
        2002: {"fed_rate": 1.25, "sp500_return": -22.1, "nasdaq_return": -31.5, "recession": 0},
        2003: {"fed_rate": 1.00, "sp500_return": 28.7, "nasdaq_return": 50.0, "recession": 0},
        2004: {"fed_rate": 2.25, "sp500_return": 10.9, "nasdaq_return": 8.6, "recession": 0},
        2005: {"fed_rate": 4.25, "sp500_return": 4.9, "nasdaq_return": 1.4, "recession": 0},
        2006: {"fed_rate": 5.25, "sp500_return": 15.8, "nasdaq_return": 9.5, "recession": 0},
        2007: {"fed_rate": 4.25, "sp500_return": 5.5, "nasdaq_return": 9.8, "recession": 0},
        2008: {"fed_rate": 0.25, "sp500_return": -37.0, "nasdaq_return": -40.5, "recession": 1},
        2009: {"fed_rate": 0.25, "sp500_return": 26.5, "nasdaq_return": 43.9, "recession": 1},
        2010: {"fed_rate": 0.25, "sp500_return": 15.1, "nasdaq_return": 18.2, "recession": 0},
        2011: {"fed_rate": 0.25, "sp500_return": 2.1, "nasdaq_return": -1.8, "recession": 0},
        2012: {"fed_rate": 0.25, "sp500_return": 16.0, "nasdaq_return": 15.9, "recession": 0},
        2013: {"fed_rate": 0.25, "sp500_return": 32.4, "nasdaq_return": 40.1, "recession": 0},
        2014: {"fed_rate": 0.25, "sp500_return": 13.7, "nasdaq_return": 14.8, "recession": 0},
        2015: {"fed_rate": 0.50, "sp500_return": 1.4, "nasdaq_return": 7.0, "recession": 0},
        2016: {"fed_rate": 0.75, "sp500_return": 12.0, "nasdaq_return": 8.9, "recession": 0},
        2017: {"fed_rate": 1.50, "sp500_return": 21.8, "nasdaq_return": 29.6, "recession": 0},
        2018: {"fed_rate": 2.50, "sp500_return": -4.4, "nasdaq_return": -2.8, "recession": 0},
        2019: {"fed_rate": 1.75, "sp500_return": 31.5, "nasdaq_return": 35.2, "recession": 0},
        2020: {"fed_rate": 0.25, "sp500_return": 18.4, "nasdaq_return": 43.6, "recession": 1},
        2021: {"fed_rate": 0.25, "sp500_return": 28.7, "nasdaq_return": 21.4, "recession": 0},
        2022: {"fed_rate": 4.50, "sp500_return": -18.1, "nasdaq_return": -33.1, "recession": 0},
        2023: {"fed_rate": 5.50, "sp500_return": 26.3, "nasdaq_return": 43.4, "recession": 0},
        2024: {"fed_rate": 4.50, "sp500_return": 25.0, "nasdaq_return": 29.6, "recession": 0},
        2025: {"fed_rate": 4.50, "sp500_return": 0.0, "nasdaq_return": 0.0, "recession": 0},
    }

    macro = pd.DataFrame(macro_features).T.reset_index()
    macro = macro.rename(columns={"index": "year"})
    macro["year"] = macro["year"].astype(int)

    return df.merge(macro, on="year", how="left")


def add_event_features(base, event_path=EVENT_DATA):
    """
    Adds features from layoffs.csv.

    For a row with year Y, this only uses layoff events up to Dec 31 of year Y.
    Since the target is year Y+1, this avoids future leakage.
    """

    event_cols = [
        "events_last_1y",
        "events_last_2y",
        "events_last_3y",
        "event_laid_off_last_1y",
        "event_laid_off_last_2y",
        "event_laid_off_last_3y",
        "event_max_pct_last_3y",
        "months_since_last_event",
        "had_event_last_1y",
        "had_event_last_2y",
        "event_laid_off_last_1y_per_employee",
        "event_laid_off_last_2y_per_employee",
        "event_laid_off_last_3y_per_employee",
    ]

    out = base.copy()

    for c in event_cols:
        out[c] = 0.0

    out["months_since_last_event"] = 999.0

    if not os.path.exists(event_path):
        print(f"{event_path} not found. Continuing without event-level features.")
        return out

    events = pd.read_csv(event_path)

    if "company" not in events.columns or "date" not in events.columns:
        print(f"{event_path} missing company/date columns. Continuing without event-level features.")
        return out

    events["company"] = events["company"].map(normalize_company_name)
    events["date"] = pd.to_datetime(events["date"], errors="coerce")

    events = events.dropna(subset=["date"])
    events = events[events["company"].isin(set(out["company"].unique()))].copy()

    if "total_laid_off" not in events.columns:
        events["total_laid_off"] = 0.0

    if "percentage_laid_off" not in events.columns:
        events["percentage_laid_off"] = 0.0

    events["total_laid_off"] = pd.to_numeric(events["total_laid_off"], errors="coerce").fillna(0.0)
    events["percentage_laid_off"] = pd.to_numeric(events["percentage_laid_off"], errors="coerce").fillna(0.0)

    rows = []

    for _, row in out.iterrows():
        company = row["company"]
        year = int(row["year"])

        end = pd.Timestamp(year=year, month=12, day=31)
        start_1 = end - pd.DateOffset(years=1)
        start_2 = end - pd.DateOffset(years=2)
        start_3 = end - pd.DateOffset(years=3)

        ev = events[(events["company"] == company) & (events["date"] <= end)]

        ev_1 = ev[ev["date"] >= start_1]
        ev_2 = ev[ev["date"] >= start_2]
        ev_3 = ev[ev["date"] >= start_3]

        if ev.empty:
            months_since = 999.0
        else:
            months_since = max(0.0, (end - ev["date"].max()).days / 30.44)

        rows.append({
            "events_last_1y": len(ev_1),
            "events_last_2y": len(ev_2),
            "events_last_3y": len(ev_3),
            "event_laid_off_last_1y": float(ev_1["total_laid_off"].sum()),
            "event_laid_off_last_2y": float(ev_2["total_laid_off"].sum()),
            "event_laid_off_last_3y": float(ev_3["total_laid_off"].sum()),
            "event_max_pct_last_3y": float(ev_3["percentage_laid_off"].max()) if len(ev_3) else 0.0,
            "months_since_last_event": float(months_since),
            "had_event_last_1y": int(len(ev_1) > 0),
            "had_event_last_2y": int(len(ev_2) > 0),
        })

    feat = pd.DataFrame(rows, index=out.index)

    for c in feat.columns:
        out[c] = feat[c]

    out["event_laid_off_last_1y_per_employee"] = out["event_laid_off_last_1y"] / (out["employees_start"] + 1.0)
    out["event_laid_off_last_2y_per_employee"] = out["event_laid_off_last_2y"] / (out["employees_start"] + 1.0)
    out["event_laid_off_last_3y_per_employee"] = out["event_laid_off_last_3y"] / (out["employees_start"] + 1.0)

    print(f"Added event-level features from {event_path}: {len(events)} matching events.")

    return out


def build_forecasting_frame(primary_path=PRIMARY_DATA, event_path=EVENT_DATA):
    df = pd.read_csv(primary_path)
    df = df.sort_values(["company", "year"]).reset_index(drop=True)

    df["company"] = df["company"].map(normalize_company_name)

    if "is_estimated" in df.columns:
        df["is_estimated"] = df["is_estimated"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    else:
        df["is_estimated"] = 0

    df["layoff_pct"] = df["layoffs"] / (df["employees_start"] + 1e-9) * 100.0
    df["log_layoff_pct"] = np.log1p(df["layoff_pct"])
    df["workforce_growth_pct"] = (df["employees_end"] - df["employees_start"]) / (df["employees_start"] + 1e-9) * 100.0
    df["revenue_per_employee"] = df["revenue_billions_usd"] * 1e9 / (df["employees_start"] + 1.0)
    df["hire_to_layoff_ratio"] = df["new_hires"] / (df["layoffs"] + 1.0)
    df["layoff_to_hire_ratio"] = df["layoffs"] / (df["new_hires"] + 1.0)
    df["layoff_spike_current"] = (df["layoff_pct"] >= ELEVATED_THRESHOLD).astype(int)

    g = df.groupby("company", group_keys=False)

    df["target_year"] = g["year"].shift(-1)
    df["target_layoff_pct_next"] = g["layoff_pct"].shift(-1)
    df["target_layoff_count_next"] = g["layoffs"].shift(-1)
    df["target_employees_start_next"] = g["employees_start"].shift(-1)
    df["target_is_estimated"] = g["is_estimated"].shift(-1).fillna(0).astype(int)

    lag_cols = [
        "layoff_pct",
        "log_layoff_pct",
        "employees_start",
        "employees_end",
        "new_hires",
        "net_change",
        "hiring_rate_pct",
        "attrition_rate_pct",
        "revenue_billions_usd",
        "stock_price_change_pct",
        "gdp_growth_us_pct",
        "unemployment_rate_us_pct",
        "workforce_growth_pct",
        "revenue_per_employee",
    ]

    for col in lag_cols:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f"{col}_lag{lag}"] = g[col].shift(lag)

    df["layoff_pct_roll3_mean"] = g["layoff_pct"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["layoff_pct_roll3_max"] = g["layoff_pct"].transform(lambda s: s.rolling(3, min_periods=1).max())
    df["layoff_pct_roll5_mean"] = g["layoff_pct"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["layoff_pct_expanding_mean"] = g["layoff_pct"].transform(lambda s: s.expanding(min_periods=1).mean())

    df["layoff_pct_diff1"] = df["layoff_pct"] - df["layoff_pct_lag1"]
    df["revenue_growth_pct"] = g["revenue_billions_usd"].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
    df["employee_start_growth_pct"] = g["employees_start"].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
    df["hiring_acceleration"] = df["hiring_rate_pct"] - df["hiring_rate_pct_lag1"]
    df["stock_change_lag_mean3"] = g["stock_price_change_pct"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    df = add_macro_features(df)

    df["bad_macro_score"] = (
        -0.05 * df["sp500_return"].fillna(0)
        -0.04 * df["nasdaq_return"].fillna(0)
        -0.25 * df["gdp_growth_us_pct"].fillna(0)
        + 0.35 * df["unemployment_rate_us_pct"].fillna(0)
        + 1.50 * df["recession"].fillna(0)
    )

    df["layoff_pct_times_bad_macro"] = df["layoff_pct"] * df["bad_macro_score"]
    df["spike_times_bad_macro"] = df["layoff_spike_current"] * df["bad_macro_score"]
    df["stock_underperformance"] = df["stock_price_change_pct"] - df["sp500_return"]
    df["tech_underperformance"] = df["stock_price_change_pct"] - df["nasdaq_return"]

    df = add_event_features(df, event_path)

    df = df.dropna(subset=["target_layoff_pct_next", "target_year"]).reset_index(drop=True)

    return df


def get_feature_columns(df):
    leakage = {
        "target_year",
        "target_layoff_pct_next",
        "target_layoff_count_next",
        "target_employees_start_next",
        "target_is_estimated",
    }

    return [c for c in df.columns if c not in leakage]


def split_time_aware(df):
    train = df[df["year"] < TRAIN_END_YEAR].copy()
    validate = df[(df["year"] >= TRAIN_END_YEAR) & (df["year"] < VALID_END_YEAR)].copy()
    test = df[df["year"] >= TEST_START_YEAR].copy()
    train_plus_val = df[df["year"] < VALID_END_YEAR].copy()

    return train, validate, test, train_plus_val


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_predictions(y_true, pred, actual_count=None, pred_count=None):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 100.0)

    if len(y_true) == 0:
        return {
            "n": 0,
            "MAE_pct_points": np.nan,
            "RMSE_pct_points": np.nan,
            "MedianAE_pct_points": np.nan,
            "R2_pct": np.nan,
            "F1_elevated": np.nan,
            "Precision_elevated": np.nan,
            "Recall_elevated": np.nan,
        }

    y_spike = (y_true >= ELEVATED_THRESHOLD).astype(int)
    p_spike = (pred >= ELEVATED_THRESHOLD).astype(int)

    out = {
        "n": len(y_true),
        "MAE_pct_points": mean_absolute_error(y_true, pred),
        "RMSE_pct_points": rmse(y_true, pred),
        "MedianAE_pct_points": median_absolute_error(y_true, pred),
        "R2_pct": r2_score(y_true, pred) if len(np.unique(y_true)) > 1 else np.nan,
        "F1_elevated": f1_score(y_spike, p_spike, zero_division=0),
        "Precision_elevated": precision_score(y_spike, p_spike, zero_division=0),
        "Recall_elevated": recall_score(y_spike, p_spike, zero_division=0),
    }

    if actual_count is not None and pred_count is not None:
        actual_count = np.asarray(actual_count, dtype=float)
        pred_count = np.asarray(pred_count, dtype=float)

        out["MAE_layoff_count"] = mean_absolute_error(actual_count, pred_count)
        out["RMSE_layoff_count"] = rmse(actual_count, pred_count)

    return out


# ------------------------------------------------------------
# Baselines and models
# ------------------------------------------------------------

class CurrentYearPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["layoff_pct"].to_numpy(dtype=float)


class GlobalMedianPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.global_median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.global_median_, dtype=float)


class CompanyMedianPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        temp = X[["company"]].copy()
        temp["target"] = np.asarray(y, dtype=float)

        self.company_median_ = temp.groupby("company")["target"].median()
        self.global_median_ = float(np.median(y))

        return self

    def predict(self, X):
        return X["company"].map(self.company_median_).fillna(self.global_median_).to_numpy(dtype=float)


class EmpiricalBayesCompanyPredictor(BaseEstimator, RegressorMixin):
    """
    Regularized company-history model.

    Each company gets a historical layoff baseline, but that baseline is shrunk
    toward the global median to avoid overfitting.
    """

    def __init__(self, strength=8.0, current_adjustment=True):
        self.strength = strength
        self.current_adjustment = current_adjustment

    def fit(self, X, y):
        temp = X[["company"]].copy()
        temp["target"] = np.asarray(y, dtype=float)

        stats = temp.groupby("company")["target"].agg(["median", "count"])

        self.global_median_ = float(np.median(y))

        weight = stats["count"] / (stats["count"] + self.strength)
        self.company_prior_ = weight * stats["median"] + (1.0 - weight) * self.global_median_

        prior = X["company"].map(self.company_prior_).fillna(self.global_median_).to_numpy(dtype=float)

        if self.current_adjustment:
            residual = np.asarray(y, dtype=float) - prior

            z1 = X["layoff_pct"].to_numpy(dtype=float) - prior
            z2 = X.get("spike_times_bad_macro", pd.Series(0, index=X.index)).fillna(0).to_numpy(dtype=float)
            z3 = X.get("event_laid_off_last_1y_per_employee", pd.Series(0, index=X.index)).fillna(0).to_numpy(dtype=float)

            Z = np.vstack([z1, z2, z3]).T

            lam = 25.0
            coef = np.linalg.pinv(Z.T @ Z + lam * np.eye(Z.shape[1])) @ Z.T @ residual

            self.beta_ = np.clip(coef, -0.35, 0.35)
        else:
            self.beta_ = np.zeros(3)

        return self

    def predict(self, X):
        prior = X["company"].map(self.company_prior_).fillna(self.global_median_).to_numpy(dtype=float)

        z1 = X["layoff_pct"].to_numpy(dtype=float) - prior
        z2 = X.get("spike_times_bad_macro", pd.Series(0, index=X.index)).fillna(0).to_numpy(dtype=float)
        z3 = X.get("event_laid_off_last_1y_per_employee", pd.Series(0, index=X.index)).fillna(0).to_numpy(dtype=float)

        Z = np.vstack([z1, z2, z3]).T

        pred = prior + Z @ self.beta_

        return np.clip(pred, 0.0, 100.0)


class ResidualRegressor(BaseEstimator, RegressorMixin):
    """
    Predicts deviation from a company-history baseline instead of raw layoff %.

    Final prediction = company baseline + model-predicted residual.
    """

    def __init__(self, base_model=None, strength=8.0):
        self.base_model = base_model
        self.strength = strength

    def fit(self, X, y):
        temp = X[["company"]].copy()
        temp["target"] = np.asarray(y, dtype=float)

        stats = temp.groupby("company")["target"].agg(["median", "count"])
        self.global_median_ = float(np.median(y))

        weight = stats["count"] / (stats["count"] + self.strength)
        self.company_prior_ = weight * stats["median"] + (1.0 - weight) * self.global_median_

        prior = X["company"].map(self.company_prior_).fillna(self.global_median_).to_numpy(dtype=float)
        residual = np.asarray(y, dtype=float) - prior

        self.model_ = clone(self.base_model)
        self.model_.fit(X, residual)

        return self

    def predict(self, X):
        prior = X["company"].map(self.company_prior_).fillna(self.global_median_).to_numpy(dtype=float)
        residual_pred = self.model_.predict(X)

        return np.clip(prior + residual_pred, 0.0, 100.0)


def build_preprocessor(feature_cols, scale_numeric=False):
    categorical_cols = [c for c in ["company", "confidence_level"] if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    if scale_numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipe = SimpleImputer(strategy="median")

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def make_model_zoo(feature_cols):
    linear_pre = build_preprocessor(feature_cols, scale_numeric=True)
    tree_pre = build_preprocessor(feature_cols, scale_numeric=False)

    ridge_pipeline = Pipeline([
        ("pre", linear_pre),
        ("model", Ridge(alpha=25.0)),
    ])

    huber_pipeline = Pipeline([
        ("pre", linear_pre),
        ("model", HuberRegressor(alpha=0.10, epsilon=1.35, max_iter=1000)),
    ])

    rf_pipeline = Pipeline([
        ("pre", tree_pre),
        ("model", RandomForestRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    gbr_pipeline = Pipeline([
        ("pre", tree_pre),
        ("model", GradientBoostingRegressor(
            loss="huber",
            n_estimators=180,
            learning_rate=0.025,
            max_depth=2,
            min_samples_leaf=18,
            subsample=0.80,
            random_state=RANDOM_STATE,
        )),
    ])

    hist_pipeline = Pipeline([
        ("pre", tree_pre),
        ("model", HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.035,
            max_iter=160,
            max_leaf_nodes=8,
            min_samples_leaf=18,
            l2_regularization=0.75,
            random_state=RANDOM_STATE,
        )),
    ])

    return {
        "Baseline: current-year layoff pct": CurrentYearPredictor(),
        "Baseline: global median": GlobalMedianPredictor(),
        "Baseline: company median": CompanyMedianPredictor(),

        "Empirical Bayes company-history model": EmpiricalBayesCompanyPredictor(),

        "Residual Ridge": ResidualRegressor(ridge_pipeline),
        "Residual Huber": ResidualRegressor(huber_pipeline),
        "Residual Random Forest": ResidualRegressor(rf_pipeline),
        "Residual Gradient Boosting": ResidualRegressor(gbr_pipeline),
        "Residual Hist Gradient Boosting": ResidualRegressor(hist_pipeline),

        "Direct Ridge": ridge_pipeline,
        "Direct Huber": huber_pipeline,
        "Direct Random Forest": rf_pipeline,
        "Direct Gradient Boosting": gbr_pipeline,
        "Direct Hist Gradient Boosting": hist_pipeline,
    }


# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------

def fit_predict(model, X_train, y_train, X_eval):
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    pred = fitted.predict(X_eval)
    return np.clip(np.asarray(pred, dtype=float), 0.0, 100.0)


def evaluate_model_table(models, train, eval_df, feature_cols):
    rows = []

    X_train = train[feature_cols]
    y_train = train["target_layoff_pct_next"]

    X_eval = eval_df[feature_cols]
    y_eval = eval_df["target_layoff_pct_next"]

    for name, model in models.items():
        try:
            pred_pct = fit_predict(model, X_train, y_train, X_eval)
            pred_count = pred_pct / 100.0 * eval_df["employees_end"].to_numpy(dtype=float)

            metrics = evaluate_predictions(
                y_eval,
                pred_pct,
                eval_df["target_layoff_count_next"],
                pred_count,
            )

            rows.append({"model": name, **metrics})

        except Exception as e:
            rows.append({"model": name, "error": str(e)})

    out = pd.DataFrame(rows)

    if "MAE_pct_points" in out.columns:
        out = out.sort_values("MAE_pct_points", ascending=True)

    return out


def walk_forward_cv(df, feature_cols, min_train_year=2006, max_eval_year=2022):
    models = make_model_zoo(feature_cols)
    rows = []

    for eval_year in range(min_train_year, max_eval_year + 1):
        train = df[df["year"] < eval_year].copy()
        valid = df[df["year"] == eval_year].copy()

        if len(train) < 50 or len(valid) == 0:
            continue

        X_train = train[feature_cols]
        y_train = train["target_layoff_pct_next"]

        X_valid = valid[feature_cols]
        y_valid = valid["target_layoff_pct_next"]

        for name, model in models.items():
            try:
                pred = fit_predict(model, X_train, y_train, X_valid)
                metrics = evaluate_predictions(y_valid, pred)
                rows.append({
                    "eval_input_year": eval_year,
                    "model": name,
                    **metrics,
                })

            except Exception as e:
                rows.append({
                    "eval_input_year": eval_year,
                    "model": name,
                    "error": str(e),
                })

    return pd.DataFrame(rows)


def summarize_cv(cv):
    if "error" in cv.columns:
        good = cv[cv["error"].isna()].copy()
    else:
        good = cv.copy()

    summary = (
        good.groupby("model")
        .agg(
            CV_MAE=("MAE_pct_points", "mean"),
            CV_RMSE=("RMSE_pct_points", "mean"),
            CV_MedianAE=("MedianAE_pct_points", "mean"),
            CV_F1=("F1_elevated", "mean"),
            CV_Precision=("Precision_elevated", "mean"),
            CV_Recall=("Recall_elevated", "mean"),
            folds=("eval_input_year", "nunique"),
        )
        .sort_values(["CV_MAE", "CV_RMSE"], ascending=True)
        .reset_index()
    )

    return summary


def choose_final_model(cv_summary, models):
    """
    Prefer a real model over a pure baseline, unless all real models are awful.
    """

    baseline_names = {
        "Baseline: current-year layoff pct",
        "Baseline: global median",
        "Baseline: company median",
    }

    non_baseline = cv_summary[~cv_summary["model"].isin(baseline_names)].copy()

    if len(non_baseline) == 0:
        best_name = cv_summary.iloc[0]["model"]
    else:
        best_name = non_baseline.iloc[0]["model"]

    return best_name, models[best_name]


def save_predictions(model_name, model, train, test, feature_cols):
    fitted = clone(model)
    fitted.fit(train[feature_cols], train["target_layoff_pct_next"])

    pred_pct = np.clip(fitted.predict(test[feature_cols]), 0.0, 100.0)
    pred_count = pred_pct / 100.0 * test["employees_end"].to_numpy(dtype=float)

    out = test[[
        "company",
        "year",
        "target_year",
        "is_estimated",
        "target_is_estimated",
        "employees_start",
        "employees_end",
        "layoff_pct",
        "target_layoff_pct_next",
        "target_layoff_count_next",
    ]].copy()

    out.insert(0, "selected_model", model_name)
    out["pred_layoff_pct_next"] = pred_pct
    out["pred_layoff_count_next"] = pred_count
    out["abs_error_pct_points"] = (out["pred_layoff_pct_next"] - out["target_layoff_pct_next"]).abs()

    out.to_csv("improved_predictions.csv", index=False)

    return out


def print_real_vs_estimated_metrics(predictions):
    all_metrics = evaluate_predictions(
        predictions["target_layoff_pct_next"],
        predictions["pred_layoff_pct_next"],
        predictions["target_layoff_count_next"],
        predictions["pred_layoff_count_next"],
    )

    print_table("SELECTED MODEL TEST METRICS: all target rows", pd.DataFrame([all_metrics]))

    real = predictions[predictions["target_is_estimated"] == 0]

    if len(real) > 0:
        real_metrics = evaluate_predictions(
            real["target_layoff_pct_next"],
            real["pred_layoff_pct_next"],
            real["target_layoff_count_next"],
            real["pred_layoff_count_next"],
        )

        print_table("SELECTED MODEL TEST METRICS: real target rows only", pd.DataFrame([real_metrics]))
    else:
        print("\nNo real target rows in the final test set. All test targets are estimated.")

    estimated = predictions[predictions["target_is_estimated"] == 1]

    if len(estimated) > 0:
        est_metrics = evaluate_predictions(
            estimated["target_layoff_pct_next"],
            estimated["pred_layoff_pct_next"],
            estimated["target_layoff_count_next"],
            estimated["pred_layoff_count_next"],
        )

        print_table("SELECTED MODEL TEST METRICS: estimated target rows only", pd.DataFrame([est_metrics]))


def try_save_feature_importances(model_name, model, train, feature_cols):
    """
    Saves feature importances if the selected model exposes them.
    Works for tree pipelines and residual tree pipelines.
    """

    try:
        fitted = clone(model)
        fitted.fit(train[feature_cols], train["target_layoff_pct_next"])

        inner = fitted

        if isinstance(fitted, ResidualRegressor):
            inner = fitted.model_

        if not isinstance(inner, Pipeline):
            return

        estimator = inner.named_steps.get("model")
        pre = inner.named_steps.get("pre")

        if not hasattr(estimator, "feature_importances_"):
            return

        try:
            names = pre.get_feature_names_out()
        except Exception:
            names = [f"feature_{i}" for i in range(len(estimator.feature_importances_))]

        imp = pd.DataFrame({
            "feature": names,
            "importance": estimator.feature_importances_,
        }).sort_values("importance", ascending=False)

        imp.to_csv("improved_feature_importances.csv", index=False)

        print_table("TOP FEATURE IMPORTANCES", imp, max_rows=20)
        print("Saved improved_feature_importances.csv")

    except Exception:
        return


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    if not os.path.exists(PRIMARY_DATA):
        raise FileNotFoundError(
            f"Could not find {PRIMARY_DATA}. Put this script in the same folder as your CSV files."
        )

    df = build_forecasting_frame(PRIMARY_DATA, EVENT_DATA)
    feature_cols = get_feature_columns(df)

    train, validate, test, train_plus_val = split_time_aware(df)

    print(f"Rows after feature engineering: {len(df)}")
    print(f"Companies: {df['company'].nunique()}")
    print(
        f"Train rows: {len(train)} "
        f"({train['year'].min()}-{train['year'].max()} input years) | "
        f"Validate rows: {len(validate)} "
        f"({validate['year'].min()}-{validate['year'].max()} input years) | "
        f"Test rows: {len(test)} "
        f"({test['year'].min()}-{test['year'].max()} input years)"
    )

    print("Target = next year's layoff percentage.")
    print(f"Estimated target rows included: {int(df['target_is_estimated'].sum())}")

    models = make_model_zoo(feature_cols)

    cv = walk_forward_cv(
        df[df["year"] < VALID_END_YEAR].copy(),
        feature_cols,
        min_train_year=2006,
        max_eval_year=2022,
    )

    cv_summary = summarize_cv(cv)

    cv.to_csv("improved_walk_forward_cv.csv", index=False)
    cv_summary.to_csv("improved_walk_forward_summary.csv", index=False)

    print_table("WALK-FORWARD CV SUMMARY — lower MAE is better", cv_summary)

    selected_name, selected_model = choose_final_model(cv_summary, models)

    print(f"\nSelected final non-baseline model: {selected_name}")

    val_table = evaluate_model_table(models, train, validate, feature_cols)
    test_train_only_table = evaluate_model_table(models, train, test, feature_cols)
    test_final_table = evaluate_model_table(models, train_plus_val, test, feature_cols)

    print_table("VALIDATION RESULTS: train <= 2019, validate 2020-2022", val_table)
    print_table("HELD-OUT TEST RESULTS: train <= 2019 only, test 2023-2024", test_train_only_table)
    print_table("FINAL HELD-OUT TEST RESULTS: train <= 2022, test 2023-2024", test_final_table)

    predictions = save_predictions(
        selected_name,
        selected_model,
        train_plus_val,
        test,
        feature_cols,
    )

    print("\nSaved improved_predictions.csv")
    print("Saved improved_walk_forward_cv.csv")
    print("Saved improved_walk_forward_summary.csv")

    selected_row = test_final_table[test_final_table["model"] == selected_name]

    if len(selected_row) > 0:
        print_table("SELECTED MODEL FINAL TEST ROW", selected_row)

    print_real_vs_estimated_metrics(predictions)

    print_table(
        "WORST SELECTED-MODEL TEST ERRORS",
        predictions.sort_values("abs_error_pct_points", ascending=False)[[
            "company",
            "year",
            "target_year",
            "target_is_estimated",
            "target_layoff_pct_next",
            "pred_layoff_pct_next",
            "abs_error_pct_points",
        ]],
        max_rows=15,
    )

    try_save_feature_importances(selected_name, selected_model, train_plus_val, feature_cols)

    print("\nInterpretation:")
    print(
        "If a baseline still wins on the final test set, that is likely because the "
        "2023-2024 target rows are estimated and close to a constant value. "
        "Use walk-forward CV as your main model-selection evidence, and clearly report "
        "real vs estimated target performance."
    )


if __name__ == "__main__":
    main()