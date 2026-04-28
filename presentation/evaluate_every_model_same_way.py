"""
evaluate_every_model_same_way.py

Unified evaluator for the tech layoffs project.

Goal
----
Evaluate EVERY model in one consistent way:
  1. Same X_train / X_validate / X_test files
  2. Same target scale: layoff percentage points, not log units
  3. Same validation-based hyperparameter selection
  4. Same final train+validate -> test evaluation
  5. Same metrics for every model

Expected files in the same folder:
  X_train.csv
  X_validate.csv
  X_test.csv
  y_train.csv
  y_validate.csv
  y_test.csv

Optional files:
  processed_data.csv
  tech_employment_2000_2025.csv

Outputs:
  unified_regression_validation_results.csv
  unified_regression_test_results.csv
  unified_regression_predictions.csv
  unified_classification_results.csv
  unified_model_audit_summary.txt

Run:
  python evaluate_every_model_same_way.py
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
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
ELEVATED_THRESHOLD_PCT = 3.0  # matches repo next_year.py threshold

# -----------------------------
# Utilities
# -----------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_expm1(y: pd.Series | np.ndarray) -> np.ndarray:
    """Converts repo log1p(layoff_pct) target back to percentage points."""
    arr = np.asarray(y, dtype=float)
    return np.expm1(arr)


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


def load_repo_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    required = ["X_train.csv", "X_validate.csv", "X_test.csv", "y_train.csv", "y_validate.csv", "y_test.csv"]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}. Run this script from the presentation folder.")

    X_train = pd.read_csv("X_train.csv")
    X_val = pd.read_csv("X_validate.csv")
    X_test = pd.read_csv("X_test.csv")

    y_train_log = pd.read_csv("y_train.csv").squeeze()
    y_val_log = pd.read_csv("y_validate.csv").squeeze()
    y_test_log = pd.read_csv("y_test.csv").squeeze()

    # All models are evaluated in human-readable percentage points.
    y_train = safe_expm1(y_train_log)
    y_val = safe_expm1(y_val_log)
    y_test = safe_expm1(y_test_log)

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_company_columns(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c.startswith("co_")]


def get_company_label_from_onehot(X: pd.DataFrame) -> pd.Series:
    co_cols = get_company_columns(X)
    if not co_cols:
        return pd.Series(["GLOBAL"] * len(X), index=X.index)
    vals = X[co_cols].to_numpy()
    idx = vals.argmax(axis=1)
    labels = [co_cols[i].replace("co_", "") for i in idx]
    # If a row has no active one-hot, use GLOBAL.
    row_sums = vals.sum(axis=1)
    labels = [lab if row_sums[i] > 0 else "GLOBAL" for i, lab in enumerate(labels)]
    return pd.Series(labels, index=X.index)


def current_layoff_pct_from_X(X: pd.DataFrame) -> np.ndarray:
    if "layoff_pct_log" in X.columns:
        return np.expm1(X["layoff_pct_log"].to_numpy(dtype=float))
    if "layoff_pct" in X.columns:
        return X["layoff_pct"].to_numpy(dtype=float)
    # Fallback: global zeros if unavailable.
    return np.zeros(len(X), dtype=float)


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    onehot_cols = [c for c in numeric_cols if c.startswith("co_")]
    scale_cols = [c for c in numeric_cols if c not in onehot_cols]

    if scale_numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_pipe = SimpleImputer(strategy="median")

    # Keep one-hot company columns unscaled.
    transformers = []
    if scale_cols:
        transformers.append(("num", num_pipe, scale_cols))
    if onehot_cols:
        transformers.append(("onehot", SimpleImputer(strategy="most_frequent"), onehot_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 100.0)
    return {
        "MAE_pct_points": mean_absolute_error(y_true, y_pred),
        "RMSE_pct_points": rmse(y_true, y_pred),
        "MedianAE_pct_points": median_absolute_error(y_true, y_pred),
        "R2_pct_space": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        "F1_elevated_3pct": f1_score((y_true > ELEVATED_THRESHOLD_PCT).astype(int), (y_pred > ELEVATED_THRESHOLD_PCT).astype(int), zero_division=0),
        "Precision_elevated_3pct": precision_score((y_true > ELEVATED_THRESHOLD_PCT).astype(int), (y_pred > ELEVATED_THRESHOLD_PCT).astype(int), zero_division=0),
        "Recall_elevated_3pct": recall_score((y_true > ELEVATED_THRESHOLD_PCT).astype(int), (y_pred > ELEVATED_THRESHOLD_PCT).astype(int), zero_division=0),
    }


# -----------------------------
# Custom baseline and recommended models
# -----------------------------

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
    """Recommended model: company-history baseline shrunk toward the global median."""

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
    """Fits a base model on residuals around Empirical Bayes company history."""

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


# -----------------------------
# Model factories
# -----------------------------

@dataclass
class ModelSpec:
    name: str
    group: str
    source: str
    factory: Callable[[pd.DataFrame], Any]
    tune_param: Optional[str] = None
    tune_values: Optional[List[Any]] = None
    scale: bool = False
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


def make_pipeline(X: pd.DataFrame, estimator: Any, scale_numeric: bool = False) -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(X, scale_numeric=scale_numeric)),
        ("model", estimator),
    ])


def model_specs(X: pd.DataFrame) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    # Baselines and recommended historical models.
    specs += [
        ModelSpec("Baseline: current-year layoff pct", "Baseline", "unified script", lambda X: CurrentYearLayoffPctPredictor()),
        ModelSpec("Baseline: global mean", "Baseline", "unified script", lambda X: GlobalMeanPredictor()),
        ModelSpec("Baseline: global median", "Baseline", "unified script", lambda X: GlobalMedianPredictor()),
        ModelSpec("Baseline: company mean", "Baseline", "unified script", lambda X: CompanyMeanPredictor()),
        ModelSpec("Baseline: company median", "Baseline", "unified script", lambda X: CompanyMedianPredictor()),
        ModelSpec("Recommended: Empirical Bayes company-history", "Recommended", "custom recommended", lambda X: EmpiricalBayesCompanyPredictor(strength=8.0, current_adjustment=True)),
    ]

    # Repo models, evaluated on pct target with consistent preprocessing and metrics.
    specs += [
        ModelSpec(
            "Repo Ridge alpha tuned", "Repo regression", "models_baseline.py",
            lambda X: make_pipeline(X, Ridge(), scale_numeric=True),
            tune_param="model__alpha", tune_values=[0.1, 1, 10, 100, 1000],
            notes="Same alpha grid as repo; target standardized to pct points."
        ),
        ModelSpec(
            "Repo Poisson alpha tuned", "Repo regression", "poisson.py",
            lambda X: make_pipeline(X, PoissonRegressor(max_iter=2000), scale_numeric=True),
            tune_param="model__alpha", tune_values=[0.1, 1, 10, 100, 1000],
            notes="Same alpha grid as repo; target standardized to pct points."
        ),
        ModelSpec(
            "Repo XGBoost reg_lambda tuned", "Repo regression", "xgb.py",
            lambda X: maybe_xgb_regressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, random_state=RANDOM_STATE, objective="reg:squarederror"),
            tune_param="reg_lambda", tune_values=[0.1, 1, 5, 10, 100],
            notes="Same hyperparameter grid as repo."
        ),
        ModelSpec(
            "Repo LightGBM reg_lambda tuned", "Repo regression", "lgbm.py",
            lambda X: make_pipeline(X, maybe_lgbm_regressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                                                            verbosity=-1), scale_numeric=True),
            tune_param="model__reg_lambda", tune_values=[0.1, 1, 5, 10, 100],
            notes="Same hyperparameter grid as repo."
        ),
        ModelSpec(
            "Repo MLP alpha tuned", "Repo regression", "mlp.py",
            lambda X: make_pipeline(X, MLPRegressor(hidden_layer_sizes=(200, 200), max_iter=2000, random_state=RANDOM_STATE), scale_numeric=True),
            tune_param="model__alpha", tune_values=[0.1, 1, 5, 10, 100],
            notes="Same alpha grid as repo, but uses stable random_state and pct target."
        ),
        ModelSpec(
            "Repo MLP bug-faithful alpha=100", "Repo regression", "mlp.py",
            lambda X: make_pipeline(X, MLPRegressor(hidden_layer_sizes=(100, 100), alpha=100, max_iter=2000, random_state=RANDOM_STATE), scale_numeric=True),
            notes="Replicates the old final-alpha behavior more closely."
        ),
    ]

    # Broad / optimal model zoo, using same split and pct target.
    specs += [
        ModelSpec("Broad Direct Ridge", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, Ridge(alpha=25.0), scale_numeric=True)),
        ModelSpec("Broad Direct ElasticNet", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=5000, random_state=RANDOM_STATE), scale_numeric=True)),
        ModelSpec("Broad Direct Huber", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, HuberRegressor(alpha=0.10, epsilon=1.35, max_iter=1000), scale_numeric=True)),
        ModelSpec("Broad Direct Poisson", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, PoissonRegressor(alpha=1.0, max_iter=2000), scale_numeric=True)),
        ModelSpec("Broad Direct Tweedie / NB-style", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, TweedieRegressor(power=1.5, alpha=0.5, link="log", max_iter=2000), scale_numeric=True)),
        ModelSpec("Broad Direct Random Forest", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, RandomForestRegressor(n_estimators=400, max_depth=4, min_samples_leaf=12, random_state=RANDOM_STATE, n_jobs=-1), scale_numeric=False)),
        ModelSpec("Broad Direct Gradient Boosting", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, GradientBoostingRegressor(loss="huber", n_estimators=180, learning_rate=0.025, max_depth=2, min_samples_leaf=18, subsample=0.8, random_state=RANDOM_STATE), scale_numeric=False)),
        ModelSpec("Broad Direct Hist Gradient Boosting", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.035, max_iter=160, max_leaf_nodes=8, min_samples_leaf=18, l2_regularization=0.75, random_state=RANDOM_STATE), scale_numeric=False)),
        ModelSpec("Broad Direct MLP", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, MLPRegressor(hidden_layer_sizes=(64, 32), alpha=2.0, early_stopping=True, max_iter=2000, random_state=RANDOM_STATE), scale_numeric=True)),
        ModelSpec("Broad Direct XGBoost", "Broad model zoo", "recommended comparison", lambda X: maybe_xgb_regressor(n_estimators=120, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=15.0, reg_alpha=1.0, random_state=RANDOM_STATE, objective="reg:squarederror")),
        ModelSpec("Broad Direct LightGBM", "Broad model zoo", "recommended comparison", lambda X: make_pipeline(X, maybe_lgbm_regressor(n_estimators=120, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=15.0, reg_alpha=1.0, random_state=RANDOM_STATE, verbosity=-1), scale_numeric=False)),
    ]

    # Residual versions of broad models.
    residual_base_specs = [
        ("Residual Ridge", lambda X: make_pipeline(X, Ridge(alpha=25.0), scale_numeric=True)),
        ("Residual ElasticNet", lambda X: make_pipeline(X, ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=5000, random_state=RANDOM_STATE), scale_numeric=True)),
        ("Residual Random Forest", lambda X: make_pipeline(X, RandomForestRegressor(n_estimators=400, max_depth=4, min_samples_leaf=12, random_state=RANDOM_STATE, n_jobs=-1), scale_numeric=False)),
        ("Residual Gradient Boosting", lambda X: make_pipeline(X, GradientBoostingRegressor(loss="huber", n_estimators=180, learning_rate=0.025, max_depth=2, min_samples_leaf=18, subsample=0.8, random_state=RANDOM_STATE), scale_numeric=False)),
        ("Residual Hist Gradient Boosting", lambda X: make_pipeline(X, HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.035, max_iter=160, max_leaf_nodes=8, min_samples_leaf=18, l2_regularization=0.75, random_state=RANDOM_STATE), scale_numeric=False)),
        ("Residual XGBoost", lambda X: maybe_xgb_regressor(n_estimators=120, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=15.0, reg_alpha=1.0, random_state=RANDOM_STATE, objective="reg:squarederror")),
        ("Residual LightGBM", lambda X: make_pipeline(X, maybe_lgbm_regressor(n_estimators=120, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=15.0, reg_alpha=1.0, random_state=RANDOM_STATE, verbosity=-1), scale_numeric=False)),
    ]
    for name, fac in residual_base_specs:
        specs.append(ModelSpec(f"Broad {name}", "Broad residual model", "recommended comparison", lambda X, fac=fac: ResidualRegressor(fac(X))))

    return specs


def set_nested_param(model: Any, param_name: str, value: Any) -> Any:
    model.set_params(**{param_name: value})
    return model


def tune_and_fit(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[Any, Optional[Any], Optional[float], Optional[str]]:
    """Tune on validation MAE, then return a fitted model on train only."""
    try:
        if spec.tune_param and spec.tune_values:
            best_value = None
            best_mae = np.inf
            best_model = None
            for value in spec.tune_values:
                model = spec.factory(X_train)
                model = set_nested_param(model, spec.tune_param, value)
                model.fit(X_train, y_train)
                val_pred = np.clip(model.predict(X_val), 0.0, 100.0)
                mae = mean_absolute_error(y_val, val_pred)
                if mae < best_mae:
                    best_mae = mae
                    best_value = value
                    best_model = model
            return best_model, best_value, best_mae, None
        else:
            model = spec.factory(X_train)
            model.fit(X_train, y_train)
            val_pred = np.clip(model.predict(X_val), 0.0, 100.0)
            return model, None, mean_absolute_error(y_val, val_pred), None
    except Exception as exc:
        return None, None, None, str(exc)


def evaluate_all_regression() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train, X_val, X_test, y_train, y_val, y_test = load_repo_split()
    specs = model_specs(X_train)

    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = np.concatenate([y_train, y_val])

    val_rows = []
    test_rows = []
    prediction_frames = []

    print(f"Loaded repo split: X_train={X_train.shape}, X_validate={X_val.shape}, X_test={X_test.shape}")
    print("All regression targets converted from log1p(layoff %) to percentage points before training/evaluation.")
    print(f"Models to evaluate: {len(specs)}")

    for i, spec in enumerate(specs, 1):
        print(f"[{i:02d}/{len(specs)}] {spec.name}")
        fitted, best_value, best_val_mae, error = tune_and_fit(spec, X_train, y_train, X_val, y_val)
        if error:
            val_rows.append({"model": spec.name, "group": spec.group, "source": spec.source, "status": "failed", "error": error})
            test_rows.append({"model": spec.name, "group": spec.group, "source": spec.source, "status": "failed", "error": error})
            continue

        val_pred = np.clip(fitted.predict(X_val), 0.0, 100.0)
        val_metrics = evaluate_regression(y_val, val_pred)
        val_rows.append({
            "model": spec.name,
            "group": spec.group,
            "source": spec.source,
            "status": "ok",
            "selected_param": spec.tune_param or "",
            "selected_value": best_value if best_value is not None else "",
            "notes": spec.notes,
            **val_metrics,
        })

        # Refit final model on train+validate using selected hyperparameter, then evaluate on test.
        try:
            final_model = spec.factory(X_trainval)
            if spec.tune_param and best_value is not None:
                final_model = set_nested_param(final_model, spec.tune_param, best_value)
            final_model.fit(X_trainval, y_trainval)
            test_pred = np.clip(final_model.predict(X_test), 0.0, 100.0)
            test_metrics = evaluate_regression(y_test, test_pred)
            test_rows.append({
                "model": spec.name,
                "group": spec.group,
                "source": spec.source,
                "status": "ok",
                "selected_param": spec.tune_param or "",
                "selected_value": best_value if best_value is not None else "",
                "notes": spec.notes,
                **test_metrics,
            })

            pred_df = pd.DataFrame({
                "model": spec.name,
                "actual_layoff_pct_next": y_test,
                "pred_layoff_pct_next": test_pred,
                "abs_error_pct_points": np.abs(test_pred - y_test),
            })
            if "year" in X_test.columns:
                pred_df.insert(1, "year", X_test["year"].to_numpy())
            prediction_frames.append(pred_df)
        except Exception as exc:
            test_rows.append({"model": spec.name, "group": spec.group, "source": spec.source, "status": "failed", "error": str(exc)})

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    preds_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    if "MAE_pct_points" in val_df.columns:
        val_df = val_df.sort_values(["status", "MAE_pct_points"], ascending=[False, True])
    if "MAE_pct_points" in test_df.columns:
        test_df = test_df.sort_values(["status", "MAE_pct_points"], ascending=[False, True])

    val_df.to_csv("unified_regression_validation_results.csv", index=False)
    test_df.to_csv("unified_regression_test_results.csv", index=False)
    preds_df.to_csv("unified_regression_predictions.csv", index=False)

    return val_df, test_df, preds_df


# -----------------------------
# Classification models, same split and same target threshold
# -----------------------------

@dataclass
class ClassifierSpec:
    name: str
    group: str
    source: str
    factory: Callable[[pd.DataFrame, np.ndarray], Any]


def maybe_xgb_classifier(y_train_binary: np.ndarray):
    try:
        from xgboost import XGBClassifier
        neg = max(1, int((y_train_binary == 0).sum()))
        pos = max(1, int((y_train_binary == 1).sum()))
        return XGBClassifier(
            n_estimators=300,
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


def classifier_specs() -> List[ClassifierSpec]:
    return [
        ClassifierSpec("Repo Logistic Regression", "Repo classification", "next_year.py", lambda X, y: Pipeline([("pre", make_preprocessor(X, scale_numeric=True)), ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))])),
        ClassifierSpec("Repo Random Forest Classifier", "Repo classification", "next_year.py", lambda X, y: Pipeline([("pre", make_preprocessor(X, scale_numeric=False)), ("clf", RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))])),
        ClassifierSpec("Repo Gradient Boosting Classifier", "Repo classification", "next_year.py", lambda X, y: Pipeline([("pre", make_preprocessor(X, scale_numeric=False)), ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE))])),
        ClassifierSpec("Repo XGBoost Classifier", "Repo classification", "next_year.py", lambda X, y: maybe_xgb_classifier(y)),
        ClassifierSpec("Repo SVM", "Repo classification", "next_year.py", lambda X, y: Pipeline([("pre", make_preprocessor(X, scale_numeric=True)), ("clf", SVC(C=1.0, class_weight="balanced", probability=True, random_state=RANDOM_STATE))])),
        ClassifierSpec("Broad Logistic Regression balanced", "Broad classification", "recommended comparison", lambda X, y: Pipeline([("pre", make_preprocessor(X, scale_numeric=True)), ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))])),
    ]


def classifier_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X)


def evaluate_classification() -> pd.DataFrame:
    X_train, X_val, X_test, y_train_pct, y_val_pct, y_test_pct = load_repo_split()
    y_train = (y_train_pct > ELEVATED_THRESHOLD_PCT).astype(int)
    y_val = (y_val_pct > ELEVATED_THRESHOLD_PCT).astype(int)
    y_test = (y_test_pct > ELEVATED_THRESHOLD_PCT).astype(int)

    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = np.concatenate([y_train, y_val])

    rows = []
    for spec in classifier_specs():
        print(f"[classifier] {spec.name}")
        try:
            # Validation fit on train only.
            model_val = spec.factory(X_train, y_train)
            model_val.fit(X_train, y_train)
            val_pred = model_val.predict(X_val)
            val_prob = classifier_predict_proba(model_val, X_val)

            # Final fit on train+validate, test on test.
            model_test = spec.factory(X_trainval, y_trainval)
            model_test.fit(X_trainval, y_trainval)
            test_pred = model_test.predict(X_test)
            test_prob = classifier_predict_proba(model_test, X_test)

            val_auc = roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else np.nan
            test_auc = roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else np.nan

            rows.append({
                "model": spec.name,
                "group": spec.group,
                "source": spec.source,
                "threshold_pct": ELEVATED_THRESHOLD_PCT,
                "val_ROC_AUC": val_auc,
                "val_F1": f1_score(y_val, val_pred, zero_division=0),
                "val_precision": precision_score(y_val, val_pred, zero_division=0),
                "val_recall": recall_score(y_val, val_pred, zero_division=0),
                "val_accuracy": accuracy_score(y_val, val_pred),
                "test_ROC_AUC": test_auc,
                "test_F1": f1_score(y_test, test_pred, zero_division=0),
                "test_precision": precision_score(y_test, test_pred, zero_division=0),
                "test_recall": recall_score(y_test, test_pred, zero_division=0),
                "test_accuracy": accuracy_score(y_test, test_pred),
                "test_positive_rate": float(y_test.mean()),
                "status": "ok",
            })
        except Exception as exc:
            rows.append({"model": spec.name, "group": spec.group, "source": spec.source, "status": "failed", "error": str(exc)})

    out = pd.DataFrame(rows)
    if "test_ROC_AUC" in out.columns:
        out = out.sort_values(["status", "test_ROC_AUC", "test_F1"], ascending=[False, False, False])
    out.to_csv("unified_classification_results.csv", index=False)
    return out


def write_summary(val_df: pd.DataFrame, test_df: pd.DataFrame, clf_df: pd.DataFrame) -> None:
    ok_test = test_df[test_df.get("status", "") == "ok"].copy()
    ok_val = val_df[val_df.get("status", "") == "ok"].copy()
    ok_clf = clf_df[clf_df.get("status", "") == "ok"].copy()

    lines = []
    lines.append("UNIFIED MODEL EVALUATION SUMMARY")
    lines.append("=" * 80)
    lines.append("All regression models were evaluated using the same repo split and the same target scale:")
    lines.append("target = next-year layoff percentage points, converted from log1p target CSVs.")
    lines.append("")

    if not ok_test.empty:
        best_test = ok_test.sort_values("MAE_pct_points").iloc[0]
        lines.append("Best final test regression model:")
        lines.append(f"  {best_test['model']} ({best_test['group']})")
        lines.append(f"  MAE = {best_test['MAE_pct_points']:.4f} percentage points")
        lines.append(f"  RMSE = {best_test['RMSE_pct_points']:.4f} percentage points")
        lines.append("")
    if not ok_val.empty:
        best_val = ok_val.sort_values("MAE_pct_points").iloc[0]
        lines.append("Best validation regression model:")
        lines.append(f"  {best_val['model']} ({best_val['group']})")
        lines.append(f"  MAE = {best_val['MAE_pct_points']:.4f} percentage points")
        lines.append(f"  RMSE = {best_val['RMSE_pct_points']:.4f} percentage points")
        lines.append("")
    if not ok_clf.empty and "test_ROC_AUC" in ok_clf.columns:
        best_auc = ok_clf.sort_values("test_ROC_AUC", ascending=False).iloc[0]
        best_f1 = ok_clf.sort_values("test_F1", ascending=False).iloc[0]
        lines.append("Best classification model by test ROC AUC:")
        lines.append(f"  {best_auc['model']} | ROC AUC = {best_auc['test_ROC_AUC']:.4f} | F1 = {best_auc['test_F1']:.4f}")
        lines.append("Best classification model by test F1:")
        lines.append(f"  {best_f1['model']} | F1 = {best_f1['test_F1']:.4f} | ROC AUC = {best_f1['test_ROC_AUC']:.4f}")
        lines.append("")

    lines.append("Output files:")
    lines.append("  unified_regression_validation_results.csv")
    lines.append("  unified_regression_test_results.csv")
    lines.append("  unified_regression_predictions.csv")
    lines.append("  unified_classification_results.csv")
    lines.append("  unified_model_audit_summary.txt")

    text = "\n".join(lines)
    with open("unified_model_audit_summary.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


def main() -> None:
    print("Unified model evaluation started.")
    print(f"Working folder: {os.getcwd()}")
    print("Evaluation method: same repo train/validate/test split for every model.")
    print("Regression target: next-year layoff percentage points.\n")

    val_df, test_df, preds_df = evaluate_all_regression()
    clf_df = evaluate_classification()

    print_table("UNIFIED REGRESSION VALIDATION RESULTS — all models same split/target", val_df)
    print_table("UNIFIED REGRESSION FINAL TEST RESULTS — all models same split/target", test_df)
    print_table("UNIFIED CLASSIFICATION RESULTS — all classifiers same split/threshold", clf_df)

    write_summary(val_df, test_df, clf_df)


if __name__ == "__main__":
    main()
