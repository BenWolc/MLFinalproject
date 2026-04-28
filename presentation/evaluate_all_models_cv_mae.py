"""
evaluate_all_models_cv_mae.py

Unified walk-forward cross-validation evaluator for the tech layoffs project.

Goal
----
Evaluate your repo models AND the generalized/recommended models with the same
walk-forward CV method and the same human-readable target:

    target = next-year layoff percentage points

This is the fairest way to choose the best forecasting model because every fold
trains only on past years and predicts a future year.

Expected file in the same folder:
    processed_data.csv

Optional files:
    tech_employment_2000_2025.csv   # used only to mark estimated target rows

Outputs:
    cv_mae_all_models_fold_results.csv
    cv_mae_all_models_summary.csv
    cv_mae_all_model_predictions.csv
    cv_classification_fold_results.csv
    cv_classification_summary.csv
    cv_mae_audit_summary.txt

Run:
    python evaluate_all_models_cv_mae.py

Useful options:
    python evaluate_all_models_cv_mae.py --min-eval-year 2006 --max-eval-year 2022
    python evaluate_all_models_cv_mae.py --exclude-estimated-targets
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, PoissonRegressor, Ridge, TweedieRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
ELEVATED_THRESHOLD_PCT = 3.0
DEFAULT_MIN_EVAL_YEAR = 2006
DEFAULT_MAX_EVAL_YEAR = 2022


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def print_table(title: str, df: pd.DataFrame, max_rows: int = 80) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    if df.empty:
        print("No rows.")
        return
    show = df.head(max_rows).copy()
    for c in show.select_dtypes(include=[np.number]).columns:
        show[c] = show[c].round(4)
    print(show.to_string(index=False))


def get_company_columns(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c.startswith("co_")]


def get_company_label_from_onehot(X: pd.DataFrame) -> pd.Series:
    co_cols = get_company_columns(X)
    if not co_cols:
        return pd.Series(["GLOBAL"] * len(X), index=X.index)

    vals = X[co_cols].astype(float).to_numpy()
    idx = vals.argmax(axis=1)
    row_sums = vals.sum(axis=1)
    labels = []
    for i, j in enumerate(idx):
        if row_sums[i] <= 0:
            labels.append("GLOBAL")
        else:
            labels.append(co_cols[j].replace("co_", ""))
    return pd.Series(labels, index=X.index)


def current_layoff_pct_from_X(X: pd.DataFrame) -> np.ndarray:
    if "layoff_pct_log" in X.columns:
        return np.expm1(X["layoff_pct_log"].to_numpy(dtype=float))
    if "layoff_pct" in X.columns:
        return X["layoff_pct"].to_numpy(dtype=float)
    return np.zeros(len(X), dtype=float)


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    # The repo processed_data is already numeric/boolean. Keep one-hot company
    # columns unscaled so company identity remains clean.
    X_num = X.copy()
    for c in X_num.columns:
        if X_num[c].dtype == bool:
            X_num[c] = X_num[c].astype(int)

    numeric_cols = X_num.select_dtypes(include=[np.number, bool]).columns.tolist()
    onehot_cols = [c for c in numeric_cols if c.startswith("co_")]
    scale_cols = [c for c in numeric_cols if c not in onehot_cols]

    if scale_numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_pipe = SimpleImputer(strategy="median")

    transformers = []
    if scale_cols:
        transformers.append(("num", num_pipe, scale_cols))
    if onehot_cols:
        transformers.append(("onehot", SimpleImputer(strategy="most_frequent"), onehot_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 100.0)
    y_true_bin = (y_true > ELEVATED_THRESHOLD_PCT).astype(int)
    y_pred_bin = (y_pred > ELEVATED_THRESHOLD_PCT).astype(int)
    return {
        "MAE_pct_points": mean_absolute_error(y_true, y_pred),
        "RMSE_pct_points": rmse(y_true, y_pred),
        "MedianAE_pct_points": median_absolute_error(y_true, y_pred),
        "R2_pct_space": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        "F1_elevated_3pct": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "Precision_elevated_3pct": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "Recall_elevated_3pct": recall_score(y_true_bin, y_pred_bin, zero_division=0),
    }


def safe_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, score)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_processed_frame() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if not os.path.exists("processed_data.csv"):
        raise FileNotFoundError("processed_data.csv not found. Run this script from your presentation folder.")

    df = pd.read_csv("processed_data.csv")
    if "layoff_pct_next_log" not in df.columns:
        raise ValueError("processed_data.csv must contain layoff_pct_next_log.")
    if "year" not in df.columns:
        raise ValueError("processed_data.csv must contain year for walk-forward CV.")

    # Convert booleans / bool-like strings to numeric where possible.
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
        elif df[c].dtype == object:
            lowered = df[c].astype(str).str.lower()
            if set(lowered.dropna().unique()).issubset({"true", "false"}):
                df[c] = lowered.eq("true").astype(int)

    y_pct = np.expm1(df["layoff_pct_next_log"].to_numpy(dtype=float))
    X = df.drop(columns=["layoff_pct_next_log"]).copy()

    meta = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "year": df["year"].astype(int).to_numpy(),
        "target_year": df["year"].astype(int).to_numpy() + 1,
        "company": get_company_label_from_onehot(X),
        "target_layoff_pct_next": y_pct,
    })

    # Optional: mark whether the target year row was estimated in the raw file.
    meta["target_is_estimated"] = np.nan
    if os.path.exists("tech_employment_2000_2025.csv"):
        raw = pd.read_csv("tech_employment_2000_2025.csv")
        if {"company", "year", "is_estimated"}.issubset(raw.columns):
            raw = raw.copy()
            raw["is_estimated"] = raw["is_estimated"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
            est_map = raw.set_index(["company", "year"])["is_estimated"].to_dict()
            meta["target_is_estimated"] = [
                est_map.get((company, int(target_year)), np.nan)
                for company, target_year in zip(meta["company"], meta["target_year"])
            ]

    return X, y_pct, meta


# -----------------------------------------------------------------------------
# Baseline and recommended models
# -----------------------------------------------------------------------------

class CurrentYearLayoffPctPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return current_layoff_pct_from_X(X)


class GlobalMeanPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype=float)


class GlobalMedianPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.median_, dtype=float)


class CompanyMeanPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        labels = get_company_label_from_onehot(X)
        temp = pd.DataFrame({"company": labels, "target": np.asarray(y, dtype=float)})
        self.company_mean_ = temp.groupby("company")["target"].mean()
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        labels = get_company_label_from_onehot(X)
        return labels.map(self.company_mean_).fillna(self.global_mean_).to_numpy(dtype=float)


class CompanyMedianPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        labels = get_company_label_from_onehot(X)
        temp = pd.DataFrame({"company": labels, "target": np.asarray(y, dtype=float)})
        self.company_median_ = temp.groupby("company")["target"].median()
        self.global_median_ = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        labels = get_company_label_from_onehot(X)
        return labels.map(self.company_median_).fillna(self.global_median_).to_numpy(dtype=float)


class EmpiricalBayesCompanyPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, strength: float = 8.0, current_adjustment: bool = True):
        self.strength = strength
        self.current_adjustment = current_adjustment

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        labels = get_company_label_from_onehot(X)
        y = np.asarray(y, dtype=float)
        temp = pd.DataFrame({"company": labels, "target": y})
        stats = temp.groupby("company")["target"].agg(["median", "count"])
        self.global_median_ = float(np.median(y))
        w = stats["count"] / (stats["count"] + self.strength)
        self.company_prior_ = w * stats["median"] + (1.0 - w) * self.global_median_

        self.beta_current_ = 0.0
        if self.current_adjustment:
            prior = labels.map(self.company_prior_).fillna(self.global_median_).to_numpy(dtype=float)
            current = current_layoff_pct_from_X(X)
            z = (current - prior).reshape(-1, 1)
            residual = y - prior
            lam = 25.0
            coef = np.linalg.pinv(z.T @ z + lam * np.eye(1)) @ z.T @ residual
            self.beta_current_ = float(np.clip(coef[0], -0.35, 0.35))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        labels = get_company_label_from_onehot(X)
        prior = labels.map(self.company_prior_).fillna(self.global_median_).to_numpy(dtype=float)
        pred = prior.copy()
        if self.current_adjustment:
            pred += self.beta_current_ * (current_layoff_pct_from_X(X) - prior)
        return np.clip(pred, 0.0, 100.0)


class ResidualRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_model: Any, strength: float = 8.0):
        self.base_model = base_model
        self.strength = strength

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.prior_model_ = EmpiricalBayesCompanyPredictor(strength=self.strength, current_adjustment=False)
        self.prior_model_.fit(X, y)
        prior = self.prior_model_.predict(X)
        residual = np.asarray(y, dtype=float) - prior
        self.model_ = clone(self.base_model)
        self.model_.fit(X, residual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.prior_model_.predict(X) + self.model_.predict(X), 0.0, 100.0)


# -----------------------------------------------------------------------------
# Model specs
# -----------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    group: str
    source: str
    factory: Callable[[pd.DataFrame], Any]
    tune_param: Optional[str] = None
    tune_values: Optional[List[Any]] = None
    notes: str = ""


def maybe_xgb_regressor(**kwargs):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(**kwargs)
    except Exception as exc:
        raise ImportError(f"xgboost unavailable: {exc}")


def maybe_lgbm_regressor(**kwargs):
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**kwargs)
    except Exception as exc:
        raise ImportError(f"lightgbm unavailable: {exc}")


def maybe_xgb_classifier(y_train_binary: np.ndarray):
    try:
        from xgboost import XGBClassifier
        neg = max(1, int((y_train_binary == 0).sum()))
        pos = max(1, int((y_train_binary == 1).sum()))
        return XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            scale_pos_weight=neg / pos,
            random_state=RANDOM_STATE,
            verbosity=0,
            eval_metric="logloss",
        )
    except Exception as exc:
        raise ImportError(f"xgboost unavailable: {exc}")


def make_pipeline(X: pd.DataFrame, estimator: Any, scale_numeric: bool = False) -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(X, scale_numeric=scale_numeric)),
        ("model", estimator),
    ])


def regression_model_specs(X: pd.DataFrame) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    specs += [
        ModelSpec("Repo/Baseline: Last Year's Value", "Repo baseline", "models_baseline.py", lambda X: CurrentYearLayoffPctPredictor()),
        ModelSpec("Repo/Baseline: Historical Average", "Repo baseline", "models_baseline.py", lambda X: CompanyMeanPredictor()),
        ModelSpec("Repo/Baseline: Global Mean", "Repo baseline", "models_baseline.py", lambda X: GlobalMeanPredictor()),
        ModelSpec("Baseline: global median", "Baseline", "unified script", lambda X: GlobalMedianPredictor()),
        ModelSpec("Baseline: company median", "Baseline", "unified script", lambda X: CompanyMedianPredictor()),
        ModelSpec("Recommended: Empirical Bayes company-history", "Recommended", "custom recommended", lambda X: EmpiricalBayesCompanyPredictor(strength=8.0, current_adjustment=True)),
    ]

    # Repo-style regression models, but trained/evaluated on pct-point target.
    specs += [
        ModelSpec(
            "Repo Ridge alpha tuned", "Repo regression", "models_baseline.py",
            lambda X: make_pipeline(X, Ridge(), scale_numeric=True),
            tune_param="model__alpha", tune_values=[0.1, 1, 10, 100, 1000],
            notes="Same alpha grid as repo; tuned inside each CV fold."
        ),
        ModelSpec(
            "Repo Poisson alpha tuned", "Repo regression", "poisson.py",
            lambda X: make_pipeline(X, PoissonRegressor(max_iter=2000), scale_numeric=True),
            tune_param="model__alpha", tune_values=[0.1, 1, 10, 100, 1000],
            notes="Same alpha grid as repo; tuned inside each CV fold."
        ),
        ModelSpec(
            "Repo XGBoost reg_lambda tuned", "Repo regression", "xgb.py",
            lambda X: maybe_xgb_regressor(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, random_state=RANDOM_STATE, objective="reg:squarederror"),
            tune_param="reg_lambda", tune_values=[0.1, 1, 5, 10, 100],
            notes="Same regularization grid as repo; tuned inside each CV fold."
        ),
        ModelSpec(
            "Repo LightGBM reg_lambda tuned", "Repo regression", "lgbm.py",
            lambda X: make_pipeline(X, maybe_lgbm_regressor(n_estimators=250, max_depth=5, learning_rate=0.05,
                                                            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                                                            verbosity=-1), scale_numeric=True),
            tune_param="model__reg_lambda", tune_values=[0.1, 1, 5, 10, 100],
            notes="Same regularization grid as repo; tuned inside each CV fold."
        ),
        ModelSpec(
            "Repo MLP alpha tuned", "Repo regression", "mlp.py",
            lambda X: make_pipeline(X, MLPRegressor(hidden_layer_sizes=(200, 200), max_iter=1200, random_state=RANDOM_STATE), scale_numeric=True),
            tune_param="model__alpha", tune_values=[0.1, 1, 5, 10, 100],
            notes="Same alpha grid as repo; tuned inside each CV fold."
        ),
        ModelSpec(
            "Repo MLP bug-faithful alpha=100", "Repo regression", "mlp.py",
            lambda X: make_pipeline(X, MLPRegressor(hidden_layer_sizes=(100, 100), alpha=100, max_iter=1200, random_state=RANDOM_STATE), scale_numeric=True),
            notes="Replicates the old high-alpha behavior."
        ),
    ]

    # Broad / generalized model zoo.
    specs += [
        ModelSpec("Broad Direct Ridge", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, Ridge(alpha=25.0), scale_numeric=True)),
        ModelSpec("Broad Direct ElasticNet", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=5000, random_state=RANDOM_STATE), scale_numeric=True)),
        ModelSpec("Broad Direct Huber", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, HuberRegressor(alpha=0.10, epsilon=1.35, max_iter=1000), scale_numeric=True)),
        ModelSpec("Broad Direct Poisson", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, PoissonRegressor(alpha=1.0, max_iter=2000), scale_numeric=True)),
        ModelSpec("Broad Direct Tweedie / NB-style", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, TweedieRegressor(power=1.5, alpha=0.5, link="log", max_iter=2000), scale_numeric=True)),
        ModelSpec("Broad Direct Random Forest", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, RandomForestRegressor(n_estimators=350, max_depth=4, min_samples_leaf=12, random_state=RANDOM_STATE, n_jobs=-1), scale_numeric=False)),
        ModelSpec("Broad Direct Gradient Boosting", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, GradientBoostingRegressor(loss="huber", n_estimators=160, learning_rate=0.025, max_depth=2, min_samples_leaf=18, subsample=0.8, random_state=RANDOM_STATE), scale_numeric=False)),
        ModelSpec("Broad Direct Hist Gradient Boosting", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.035, max_iter=140, max_leaf_nodes=8, min_samples_leaf=18, l2_regularization=0.75, random_state=RANDOM_STATE), scale_numeric=False)),
        ModelSpec("Broad Direct MLP", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, MLPRegressor(hidden_layer_sizes=(64, 32), alpha=2.0, early_stopping=True, max_iter=1200, random_state=RANDOM_STATE), scale_numeric=True)),
        ModelSpec("Broad Direct XGBoost", "Broad model zoo", "recommended comparison", lambda X: maybe_xgb_regressor(n_estimators=120, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=15.0, reg_alpha=1.0, random_state=RANDOM_STATE, objective="reg:squarederror")),
        ModelSpec("Broad Direct LightGBM", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, maybe_lgbm_regressor(n_estimators=120, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=15.0, reg_alpha=1.0, random_state=RANDOM_STATE, verbosity=-1), scale_numeric=False)),
    ]

    residual_base_specs = [
        ("Residual Ridge", lambda X: make_pipeline(X, Ridge(alpha=25.0), scale_numeric=True)),
        ("Residual ElasticNet", lambda X: make_pipeline(X, ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=5000, random_state=RANDOM_STATE), scale_numeric=True)),
        ("Residual Random Forest", lambda X: make_pipeline(X, RandomForestRegressor(n_estimators=350, max_depth=4, min_samples_leaf=12, random_state=RANDOM_STATE, n_jobs=-1), scale_numeric=False)),
        ("Residual Gradient Boosting", lambda X: make_pipeline(X, GradientBoostingRegressor(loss="huber", n_estimators=160, learning_rate=0.025, max_depth=2, min_samples_leaf=18, subsample=0.8, random_state=RANDOM_STATE), scale_numeric=False)),
        ("Residual Hist Gradient Boosting", lambda X: make_pipeline(X, HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.035, max_iter=140, max_leaf_nodes=8, min_samples_leaf=18, l2_regularization=0.75, random_state=RANDOM_STATE), scale_numeric=False)),
        ("Residual XGBoost", lambda X: maybe_xgb_regressor(n_estimators=120, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=15.0, reg_alpha=1.0, random_state=RANDOM_STATE, objective="reg:squarederror")),
        ("Residual LightGBM", lambda X: make_pipeline(X, maybe_lgbm_regressor(n_estimators=120, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=15.0, reg_alpha=1.0, random_state=RANDOM_STATE, verbosity=-1), scale_numeric=False)),
    ]
    for name, fac in residual_base_specs:
        specs.append(ModelSpec(f"Broad {name}", "Broad residual model", "recommended comparison", lambda X, fac=fac: ResidualRegressor(fac(X))))

    return specs


# -----------------------------------------------------------------------------
# Tuning and CV
# -----------------------------------------------------------------------------

def set_nested_param(model: Any, param_name: str, value: Any) -> Any:
    model.set_params(**{param_name: value})
    return model


def choose_inner_split(meta_train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    years = sorted(meta_train["year"].unique())
    if len(years) < 3:
        return meta_train.index.to_numpy(), np.array([], dtype=int)
    inner_val_year = years[-1]
    inner_train_idx = meta_train.index[meta_train["year"] < inner_val_year].to_numpy()
    inner_val_idx = meta_train.index[meta_train["year"] == inner_val_year].to_numpy()
    return inner_train_idx, inner_val_idx


def fit_model_for_fold(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    meta_train: pd.DataFrame,
) -> Tuple[Any, Any, Optional[str]]:
    """Tune inside the training window, then fit on the full training window."""
    train_idx = meta_train.index.to_numpy()

    try:
        best_value = None
        if spec.tune_param and spec.tune_values:
            inner_train_idx, inner_val_idx = choose_inner_split(meta_train)
            if len(inner_val_idx) == 0:
                best_value = spec.tune_values[0]
            else:
                best_mae = np.inf
                for value in spec.tune_values:
                    model = spec.factory(X.iloc[inner_train_idx])
                    model = set_nested_param(model, spec.tune_param, value)
                    model.fit(X.iloc[inner_train_idx], y[inner_train_idx])
                    pred = np.clip(model.predict(X.iloc[inner_val_idx]), 0.0, 100.0)
                    mae = mean_absolute_error(y[inner_val_idx], pred)
                    if mae < best_mae:
                        best_mae = mae
                        best_value = value

        final_model = spec.factory(X.iloc[train_idx])
        if spec.tune_param and best_value is not None:
            final_model = set_nested_param(final_model, spec.tune_param, best_value)
        final_model.fit(X.iloc[train_idx], y[train_idx])
        return final_model, best_value, None
    except Exception as exc:
        return None, None, str(exc)


def walk_forward_regression_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    min_eval_year: int,
    max_eval_year: int,
    exclude_estimated_targets: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = regression_model_specs(X)
    fold_rows = []
    pred_rows = []

    candidate_years = [yr for yr in sorted(meta["year"].unique()) if min_eval_year <= yr <= max_eval_year]
    print(f"Regression models to evaluate: {len(specs)}")
    print(f"Walk-forward input-year folds: {candidate_years}")

    for fold_no, eval_year in enumerate(candidate_years, 1):
        train_mask = meta["year"] < eval_year
        test_mask = meta["year"] == eval_year
        if exclude_estimated_targets and meta["target_is_estimated"].notna().any():
            test_mask = test_mask & (meta["target_is_estimated"] == 0)

        meta_train = meta[train_mask].copy()
        meta_test = meta[test_mask].copy()
        if len(meta_train) < 50 or len(meta_test) == 0:
            print(f"Skipping fold {eval_year}: train={len(meta_train)}, test={len(meta_test)}")
            continue

        print(f"\nFold {fold_no}/{len(candidate_years)}: train years < {eval_year}, eval year = {eval_year}, n_test={len(meta_test)}")
        for i, spec in enumerate(specs, 1):
            print(f"  [{i:02d}/{len(specs)}] {spec.name}")
            model, selected_value, error = fit_model_for_fold(spec, X, y, meta_train)
            if error:
                fold_rows.append({
                    "eval_year": eval_year,
                    "model": spec.name,
                    "group": spec.group,
                    "source": spec.source,
                    "status": "failed",
                    "error": error,
                })
                continue

            test_idx = meta_test.index.to_numpy()
            pred = np.clip(model.predict(X.iloc[test_idx]), 0.0, 100.0)
            metrics = evaluate_regression(y[test_idx], pred)
            fold_rows.append({
                "eval_year": eval_year,
                "n_test": len(test_idx),
                "model": spec.name,
                "group": spec.group,
                "source": spec.source,
                "status": "ok",
                "selected_param": spec.tune_param or "",
                "selected_value": selected_value if selected_value is not None else "",
                "notes": spec.notes,
                **metrics,
            })

            preds = meta_test[["row_id", "year", "target_year", "company", "target_is_estimated"]].copy()
            preds["model"] = spec.name
            preds["actual_layoff_pct_next"] = y[test_idx]
            preds["pred_layoff_pct_next"] = pred
            preds["abs_error_pct_points"] = np.abs(pred - y[test_idx])
            pred_rows.append(preds)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    if fold_df.empty:
        return fold_df, pd.DataFrame(), pred_df

    ok = fold_df[fold_df["status"] == "ok"].copy()
    summary = (
        ok.groupby(["model", "group", "source"], dropna=False)
        .agg(
            CV_MAE=("MAE_pct_points", "mean"),
            CV_RMSE=("RMSE_pct_points", "mean"),
            CV_MedianAE=("MedianAE_pct_points", "mean"),
            CV_R2=("R2_pct_space", "mean"),
            CV_F1_3pct=("F1_elevated_3pct", "mean"),
            CV_precision_3pct=("Precision_elevated_3pct", "mean"),
            CV_recall_3pct=("Recall_elevated_3pct", "mean"),
            folds=("eval_year", "nunique"),
            total_test_rows=("n_test", "sum"),
        )
        .sort_values(["CV_MAE", "CV_RMSE"], ascending=True)
        .reset_index()
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return fold_df, summary, pred_df


# -----------------------------------------------------------------------------
# Classification CV, separate from CV MAE
# -----------------------------------------------------------------------------

@dataclass
class ClassifierSpec:
    name: str
    group: str
    source: str
    factory: Callable[[pd.DataFrame, np.ndarray], Any]


def classifier_specs() -> List[ClassifierSpec]:
    return [
        ClassifierSpec("Repo Logistic Regression", "Repo classification", "next_year.py", lambda X, y: Pipeline([("pre", make_preprocessor(X, scale_numeric=True)), ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))])),
        ClassifierSpec("Repo Random Forest Classifier", "Repo classification", "next_year.py", lambda X, y: Pipeline([("pre", make_preprocessor(X, scale_numeric=False)), ("clf", RandomForestClassifier(n_estimators=250, max_depth=4, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))])),
        ClassifierSpec("Repo Gradient Boosting Classifier", "Repo classification", "next_year.py", lambda X, y: Pipeline([("pre", make_preprocessor(X, scale_numeric=False)), ("clf", GradientBoostingClassifier(n_estimators=160, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE))])),
        ClassifierSpec("Repo XGBoost Classifier", "Repo classification", "next_year.py", lambda X, y: maybe_xgb_classifier(y)),
        ClassifierSpec("Repo SVM", "Repo classification", "next_year.py", lambda X, y: Pipeline([("pre", make_preprocessor(X, scale_numeric=True)), ("clf", SVC(C=1.0, class_weight="balanced", probability=True, random_state=RANDOM_STATE))])),
        ClassifierSpec("Broad Logistic Regression balanced", "Broad classification", "recommended comparison", lambda X, y: Pipeline([("pre", make_preprocessor(X, scale_numeric=True)), ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))])),
    ]


def classifier_predict_score(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X)


def walk_forward_classification_cv(
    X: pd.DataFrame,
    y_pct: np.ndarray,
    meta: pd.DataFrame,
    min_eval_year: int,
    max_eval_year: int,
    exclude_estimated_targets: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    specs = classifier_specs()
    y_bin = (y_pct > ELEVATED_THRESHOLD_PCT).astype(int)
    rows = []
    candidate_years = [yr for yr in sorted(meta["year"].unique()) if min_eval_year <= yr <= max_eval_year]

    print(f"\nClassification models to evaluate separately: {len(specs)}")
    for eval_year in candidate_years:
        train_mask = meta["year"] < eval_year
        test_mask = meta["year"] == eval_year
        if exclude_estimated_targets and meta["target_is_estimated"].notna().any():
            test_mask = test_mask & (meta["target_is_estimated"] == 0)

        train_idx = meta.index[train_mask].to_numpy()
        test_idx = meta.index[test_mask].to_numpy()
        if len(train_idx) < 50 or len(test_idx) == 0 or len(np.unique(y_bin[train_idx])) < 2:
            continue

        for spec in specs:
            try:
                model = spec.factory(X.iloc[train_idx], y_bin[train_idx])
                model.fit(X.iloc[train_idx], y_bin[train_idx])
                pred = model.predict(X.iloc[test_idx])
                score = classifier_predict_score(model, X.iloc[test_idx])
                rows.append({
                    "eval_year": eval_year,
                    "n_test": len(test_idx),
                    "model": spec.name,
                    "group": spec.group,
                    "source": spec.source,
                    "status": "ok",
                    "ROC_AUC": safe_roc_auc(y_bin[test_idx], score),
                    "F1": f1_score(y_bin[test_idx], pred, zero_division=0),
                    "precision": precision_score(y_bin[test_idx], pred, zero_division=0),
                    "recall": recall_score(y_bin[test_idx], pred, zero_division=0),
                    "accuracy": accuracy_score(y_bin[test_idx], pred),
                    "positive_rate": float(y_bin[test_idx].mean()),
                })
            except Exception as exc:
                rows.append({
                    "eval_year": eval_year,
                    "model": spec.name,
                    "group": spec.group,
                    "source": spec.source,
                    "status": "failed",
                    "error": str(exc),
                })

    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        return fold_df, pd.DataFrame()

    ok = fold_df[fold_df["status"] == "ok"].copy()
    summary = (
        ok.groupby(["model", "group", "source"], dropna=False)
        .agg(
            CV_ROC_AUC=("ROC_AUC", "mean"),
            CV_F1=("F1", "mean"),
            CV_precision=("precision", "mean"),
            CV_recall=("recall", "mean"),
            CV_accuracy=("accuracy", "mean"),
            folds=("eval_year", "nunique"),
            total_test_rows=("n_test", "sum"),
        )
        .sort_values(["CV_ROC_AUC", "CV_F1"], ascending=False)
        .reset_index()
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return fold_df, summary


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

def write_summary(reg_summary: pd.DataFrame, clf_summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = []
    lines.append("CV MAE MODEL AUDIT SUMMARY")
    lines.append("=" * 80)
    lines.append("All regression models were evaluated with the same walk-forward CV method.")
    lines.append("Target unit: next-year layoff percentage points.")
    lines.append(f"Evaluation years: {args.min_eval_year} to {args.max_eval_year}.")
    lines.append(f"Exclude estimated target rows: {args.exclude_estimated_targets}.")
    lines.append("")

    if not reg_summary.empty:
        best = reg_summary.iloc[0]
        lines.append("Best regression model by CV MAE:")
        lines.append(f"  {best['model']} ({best['group']})")
        lines.append(f"  CV MAE = {best['CV_MAE']:.4f} percentage points")
        lines.append(f"  CV RMSE = {best['CV_RMSE']:.4f} percentage points")
        lines.append(f"  Folds = {int(best['folds'])}")
        lines.append("")

    if not clf_summary.empty:
        best_auc = clf_summary.sort_values(["CV_ROC_AUC", "CV_F1"], ascending=False).iloc[0]
        best_f1 = clf_summary.sort_values(["CV_F1", "CV_ROC_AUC"], ascending=False).iloc[0]
        lines.append("Classification models are not ranked by MAE because they predict classes, not percentages.")
        lines.append(f"Best classifier by CV ROC AUC: {best_auc['model']} | AUC = {best_auc['CV_ROC_AUC']:.4f} | F1 = {best_auc['CV_F1']:.4f}")
        lines.append(f"Best classifier by CV F1: {best_f1['model']} | F1 = {best_f1['CV_F1']:.4f} | AUC = {best_f1['CV_ROC_AUC']:.4f}")
        lines.append("")

    lines.append("Output files:")
    lines.append("  cv_mae_all_models_fold_results.csv")
    lines.append("  cv_mae_all_models_summary.csv")
    lines.append("  cv_mae_all_model_predictions.csv")
    lines.append("  cv_classification_fold_results.csv")
    lines.append("  cv_classification_summary.csv")
    lines.append("  cv_mae_audit_summary.txt")

    text = "\n".join(lines)
    with open("cv_mae_audit_summary.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all repo and recommended models with walk-forward CV MAE.")
    parser.add_argument("--min-eval-year", type=int, default=DEFAULT_MIN_EVAL_YEAR)
    parser.add_argument("--max-eval-year", type=int, default=DEFAULT_MAX_EVAL_YEAR)
    parser.add_argument("--exclude-estimated-targets", action="store_true", help="Exclude evaluation rows whose target year is marked estimated in tech_employment_2000_2025.csv.")
    parser.add_argument("--skip-classification", action="store_true", help="Only run regression CV MAE; skip classifier CV metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("All-model walk-forward CV MAE evaluation started.")
    print(f"Working folder: {os.getcwd()}")
    print("Regression target: next-year layoff percentage points.")
    print("Each fold trains on years before the eval year and tests on that eval year.")

    X, y, meta = load_processed_frame()
    print(f"Loaded processed_data.csv: X={X.shape}, rows={len(y)}")
    print(f"Companies detected: {meta['company'].nunique()}")
    print(f"Years available: {int(meta['year'].min())}-{int(meta['year'].max())}")
    if meta["target_is_estimated"].notna().any():
        print(f"Estimated target rows marked from raw file: {int(meta['target_is_estimated'].fillna(0).sum())}")

    reg_fold_df, reg_summary, reg_pred_df = walk_forward_regression_cv(
        X=X,
        y=y,
        meta=meta,
        min_eval_year=args.min_eval_year,
        max_eval_year=args.max_eval_year,
        exclude_estimated_targets=args.exclude_estimated_targets,
    )

    reg_fold_df.to_csv("cv_mae_all_models_fold_results.csv", index=False)
    reg_summary.to_csv("cv_mae_all_models_summary.csv", index=False)
    reg_pred_df.to_csv("cv_mae_all_model_predictions.csv", index=False)

    clf_fold_df = pd.DataFrame()
    clf_summary = pd.DataFrame()
    if not args.skip_classification:
        clf_fold_df, clf_summary = walk_forward_classification_cv(
            X=X,
            y_pct=y,
            meta=meta,
            min_eval_year=args.min_eval_year,
            max_eval_year=args.max_eval_year,
            exclude_estimated_targets=args.exclude_estimated_targets,
        )
        clf_fold_df.to_csv("cv_classification_fold_results.csv", index=False)
        clf_summary.to_csv("cv_classification_summary.csv", index=False)

    print_table("WALK-FORWARD CV MAE SUMMARY — all regression models, same folds", reg_summary, max_rows=80)
    if not args.skip_classification:
        print_table("WALK-FORWARD CLASSIFICATION CV SUMMARY — not ranked by MAE", clf_summary, max_rows=40)

    write_summary(reg_summary, clf_summary, args)


if __name__ == "__main__":
    main()
