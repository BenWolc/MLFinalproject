"""
Compare all layoff prediction models on the same dataset, target, and splits.

Put this file in the same folder as your current layoff_forecaster.py and CSV files.
Then run:

    python compare_all_models.py

This script imports your improved preprocessing/model code from layoff_forecaster.py
(or improved_layoff_forecaster.py if that is the filename), then adds the remaining
models from your project: Poisson, XGBoost, LightGBM, MLP, etc.

Outputs:
    all_models_walk_forward_cv.csv
    all_models_walk_forward_summary.csv
    all_models_validation_results.csv
    all_models_test_train_only_results.csv
    all_models_final_test_results.csv
    all_models_best_predictions.csv
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, PoissonRegressor, Ridge, TweedieRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ELEVATED_THRESHOLD = 6.0
TRAIN_END_YEAR = 2020
VALID_END_YEAR = 2023
TEST_START_YEAR = 2023

PRIMARY_DATA = "tech_employment_2000_2025.csv"
EVENT_DATA = "layoffs.csv"


def import_base_module():
    """Import your improved layoff_forecaster code."""
    for module_name in ["layoff_forecaster", "improved_layoff_forecaster"]:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            pass
    raise ImportError(
        "Could not import layoff_forecaster.py or improved_layoff_forecaster.py. "
        "Put compare_all_models.py in the same folder as your improved forecaster script."
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(feature_cols: Iterable[str], scale_numeric: bool = False) -> ColumnTransformer:
    feature_cols = list(feature_cols)
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


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(y_true, pred, actual_count=None, pred_count=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 100.0)

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
        out["MAE_layoff_count"] = mean_absolute_error(actual_count, pred_count)
        out["RMSE_layoff_count"] = rmse(actual_count, pred_count)

    return out


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
    """Regularized company-history model with small current/event adjustment."""

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
            z4 = X.get("events_last_1y", pd.Series(0, index=X.index)).fillna(0).to_numpy(dtype=float)
            Z = np.vstack([z1, z2, z3, z4]).T
            lam = 30.0
            coef = np.linalg.pinv(Z.T @ Z + lam * np.eye(Z.shape[1])) @ Z.T @ residual
            self.beta_ = np.clip(coef, -0.30, 0.30)
        else:
            self.beta_ = np.zeros(4)
        return self

    def predict(self, X):
        prior = X["company"].map(self.company_prior_).fillna(self.global_median_).to_numpy(dtype=float)
        z1 = X["layoff_pct"].to_numpy(dtype=float) - prior
        z2 = X.get("spike_times_bad_macro", pd.Series(0, index=X.index)).fillna(0).to_numpy(dtype=float)
        z3 = X.get("event_laid_off_last_1y_per_employee", pd.Series(0, index=X.index)).fillna(0).to_numpy(dtype=float)
        z4 = X.get("events_last_1y", pd.Series(0, index=X.index)).fillna(0).to_numpy(dtype=float)
        Z = np.vstack([z1, z2, z3, z4]).T
        return np.clip(prior + Z @ self.beta_, 0.0, 100.0)


class ResidualRegressor(BaseEstimator, RegressorMixin):
    """Predict residual from company baseline, then add it back."""

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
        residual = self.model_.predict(X)
        return np.clip(prior + residual, 0.0, 100.0)


def optional_xgb_model(feature_cols):
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None

    return Pipeline([
        ("pre", build_preprocessor(feature_cols, scale_numeric=False)),
        ("model", XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
        )),
    ])


def optional_lgbm_model(feature_cols):
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return None

    return Pipeline([
        ("pre", build_preprocessor(feature_cols, scale_numeric=False)),
        ("model", LGBMRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=5.0,
            min_child_samples=15,
            random_state=RANDOM_STATE,
            verbose=-1,
        )),
    ])


def make_model_zoo(feature_cols) -> Dict[str, object]:
    linear_pre = build_preprocessor(feature_cols, scale_numeric=True)
    tree_pre = build_preprocessor(feature_cols, scale_numeric=False)

    ridge = Pipeline([
        ("pre", linear_pre),
        ("model", Ridge(alpha=25.0)),
    ])

    elastic = Pipeline([
        ("pre", linear_pre),
        ("model", ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=10000, random_state=RANDOM_STATE)),
    ])

    huber = Pipeline([
        ("pre", linear_pre),
        ("model", HuberRegressor(alpha=0.10, epsilon=1.35, max_iter=1000)),
    ])

    poisson = Pipeline([
        ("pre", linear_pre),
        ("model", PoissonRegressor(alpha=5.0, max_iter=10000)),
    ])

    # Tweedie with power 1.5 is a practical sklearn substitute for overdispersed nonnegative targets.
    # It is not literally negative binomial, but it serves the same project idea better than forcing a fragile NB fit.
    tweedie = Pipeline([
        ("pre", linear_pre),
        ("model", TweedieRegressor(power=1.5, alpha=2.0, link="log", max_iter=10000)),
    ])

    rf = Pipeline([
        ("pre", tree_pre),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    gbr = Pipeline([
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

    hist = Pipeline([
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

    mlp = Pipeline([
        ("pre", linear_pre),
        ("model", MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=20.0,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.2,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])

    models = {
        "Baseline: current-year layoff pct": CurrentYearPredictor(),
        "Baseline: global median": GlobalMedianPredictor(),
        "Baseline: company median": CompanyMedianPredictor(),
        "Empirical Bayes company-history model": EmpiricalBayesCompanyPredictor(),
        "Direct Ridge": ridge,
        "Direct ElasticNet": elastic,
        "Direct Huber": huber,
        "Direct Poisson Regression": poisson,
        "Direct Tweedie / NB-style Regression": tweedie,
        "Direct Random Forest": rf,
        "Direct Gradient Boosting": gbr,
        "Direct Hist Gradient Boosting": hist,
        "Direct MLP Regressor": mlp,
        "Residual Ridge": ResidualRegressor(ridge),
        "Residual ElasticNet": ResidualRegressor(elastic),
        "Residual Huber": ResidualRegressor(huber),
        "Residual Random Forest": ResidualRegressor(rf),
        "Residual Gradient Boosting": ResidualRegressor(gbr),
        "Residual Hist Gradient Boosting": ResidualRegressor(hist),
    }

    xgb = optional_xgb_model(feature_cols)
    if xgb is not None:
        models["Direct XGBoost"] = xgb
        models["Residual XGBoost"] = ResidualRegressor(xgb)
    else:
        print("xgboost is not installed; skipping XGBoost.")

    lgbm = optional_lgbm_model(feature_cols)
    if lgbm is not None:
        models["Direct LightGBM"] = lgbm
        models["Residual LightGBM"] = ResidualRegressor(lgbm)
    else:
        print("lightgbm is not installed; skipping LightGBM.")

    return models


def split_time_aware(df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] < TRAIN_END_YEAR].copy()
    validate = df[(df["year"] >= TRAIN_END_YEAR) & (df["year"] < VALID_END_YEAR)].copy()
    test = df[df["year"] >= TEST_START_YEAR].copy()
    train_plus_val = df[df["year"] < VALID_END_YEAR].copy()
    return train, validate, test, train_plus_val


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
        except Exception as exc:
            rows.append({"model": name, "error": str(exc)})

    out = pd.DataFrame(rows)
    if "MAE_pct_points" in out.columns:
        out = out.sort_values("MAE_pct_points", ascending=True)
    return out


def walk_forward_cv(df, feature_cols, models, min_train_year=2006, max_eval_year=2022):
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
                rows.append({"eval_input_year": eval_year, "model": name, **metrics})
            except Exception as exc:
                rows.append({"eval_input_year": eval_year, "model": name, "error": str(exc)})

    return pd.DataFrame(rows)


def summarize_cv(cv):
    good = cv.copy()
    if "error" in good.columns:
        good = good[good["error"].isna()].copy()

    return (
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


def print_table(title, df, max_rows=30):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    if df.empty:
        print("No rows.")
        return
    show = df.head(max_rows).copy()
    for col in show.select_dtypes(include=[np.number]).columns:
        show[col] = show[col].round(4)
    print(show.to_string(index=False))


def save_best_predictions(best_name, best_model, train_plus_val, test, feature_cols):
    fitted = clone(best_model)
    fitted.fit(train_plus_val[feature_cols], train_plus_val["target_layoff_pct_next"])
    pred_pct = np.clip(fitted.predict(test[feature_cols]), 0.0, 100.0)
    pred_count = pred_pct / 100.0 * test["employees_end"].to_numpy(dtype=float)

    cols = [
        "company", "year", "target_year", "target_is_estimated",
        "employees_start", "employees_end", "layoff_pct",
        "target_layoff_pct_next", "target_layoff_count_next",
    ]
    out = test[cols].copy()
    out.insert(0, "selected_model", best_name)
    out["pred_layoff_pct_next"] = pred_pct
    out["pred_layoff_count_next"] = pred_count
    out["abs_error_pct_points"] = (out["pred_layoff_pct_next"] - out["target_layoff_pct_next"]).abs()
    out.to_csv("all_models_best_predictions.csv", index=False)
    return out


def main():
    if not os.path.exists(PRIMARY_DATA):
        raise FileNotFoundError(f"Could not find {PRIMARY_DATA}. Run this from the presentation folder.")

    base = import_base_module()

    # Use your improved data builder if available.
    try:
        df = base.build_forecasting_frame(PRIMARY_DATA, EVENT_DATA)
    except TypeError:
        df = base.build_forecasting_frame(PRIMARY_DATA)

    try:
        feature_cols = base.get_feature_columns(df)
    except Exception:
        leakage = {
            "target_year", "target_layoff_pct_next", "target_layoff_count_next",
            "target_employees_start_next", "target_is_estimated",
        }
        feature_cols = [c for c in df.columns if c not in leakage]

    train, validate, test, train_plus_val = split_time_aware(df)

    print(f"Rows after feature engineering: {len(df)}")
    print(f"Companies: {df['company'].nunique()}")
    print(f"Train rows: {len(train)} | Validate rows: {len(validate)} | Test rows: {len(test)}")
    if "target_is_estimated" in df.columns:
        print(f"Estimated target rows included: {int(df['target_is_estimated'].sum())}")
        print(f"Final test real targets: {int((test['target_is_estimated'] == 0).sum())}")
        print(f"Final test estimated targets: {int((test['target_is_estimated'] == 1).sum())}")
    print("Target units: percentage points of workforce laid off.")

    models = make_model_zoo(feature_cols)
    print(f"\nModels being compared: {len(models)}")
    for name in models:
        print(f" - {name}")

    cv = walk_forward_cv(df[df["year"] < VALID_END_YEAR].copy(), feature_cols, models, min_train_year=2006, max_eval_year=2022)
    cv_summary = summarize_cv(cv)

    val_table = evaluate_model_table(models, train, validate, feature_cols)
    test_train_only_table = evaluate_model_table(models, train, test, feature_cols)
    final_test_table = evaluate_model_table(models, train_plus_val, test, feature_cols)

    cv.to_csv("all_models_walk_forward_cv.csv", index=False)
    cv_summary.to_csv("all_models_walk_forward_summary.csv", index=False)
    val_table.to_csv("all_models_validation_results.csv", index=False)
    test_train_only_table.to_csv("all_models_test_train_only_results.csv", index=False)
    final_test_table.to_csv("all_models_final_test_results.csv", index=False)

    print_table("WALK-FORWARD CV SUMMARY — main model-selection table", cv_summary, max_rows=50)
    print_table("VALIDATION RESULTS — train <= 2019, validate 2020-2022", val_table, max_rows=50)
    print_table("HELD-OUT TEST RESULTS — train <= 2019 only, test 2023-2024", test_train_only_table, max_rows=50)
    print_table("FINAL HELD-OUT TEST RESULTS — train <= 2022, test 2023-2024", final_test_table, max_rows=50)

    best_cv_name = cv_summary.iloc[0]["model"]
    best_cv_model = models[best_cv_name]
    print(f"\nBest model by walk-forward CV MAE: {best_cv_name}")
    print(
        "This is the fairest answer for 'best model' because it averages performance "
        "across many historical forecasting years instead of relying only on the estimated final test rows."
    )

    preds = save_best_predictions(best_cv_name, best_cv_model, train_plus_val, test, feature_cols)
    print("\nSaved:")
    print(" - all_models_walk_forward_cv.csv")
    print(" - all_models_walk_forward_summary.csv")
    print(" - all_models_validation_results.csv")
    print(" - all_models_test_train_only_results.csv")
    print(" - all_models_final_test_results.csv")
    print(" - all_models_best_predictions.csv")

    if "target_is_estimated" in preds.columns:
        real = preds[preds["target_is_estimated"] == 0]
        estimated = preds[preds["target_is_estimated"] == 1]
        if len(real) == 0:
            print("\nImportant: there are no real target rows in the final test set; all final test targets are estimated.")
        else:
            print_table("Best-CV model metrics on real final-test target rows", pd.DataFrame([
                evaluate_predictions(real["target_layoff_pct_next"], real["pred_layoff_pct_next"], real["target_layoff_count_next"], real["pred_layoff_count_next"])
            ]))
        if len(estimated) > 0:
            print_table("Best-CV model metrics on estimated final-test target rows", pd.DataFrame([
                evaluate_predictions(estimated["target_layoff_pct_next"], estimated["pred_layoff_pct_next"], estimated["target_layoff_count_next"], estimated["pred_layoff_count_next"])
            ]))

    print_table("Worst errors for best-CV model on final test", preds.sort_values("abs_error_pct_points", ascending=False), max_rows=15)


if __name__ == "__main__":
    main()
