"""
Compare the actual model families present in the project zip.

Put this file inside the same folder as the project files, e.g. the folder containing:
    preprocess.py
    models_baseline.py
    poisson.py
    xgb.py
    lgbm.py
    mlp.py
    logreg.py
    next_year.py
    nf.py
    tech_employment_2000_2025.csv

Then run:
    python compare_zip_project_models.py

What this script does:
1. Runs preprocess.py if the expected X/y CSVs are missing.
2. Recreates the model logic from the repo model files under one consistent evaluator.
3. Evaluates regression models on the original repo split and log target, but reports errors
   in BOTH log units and real layoff percentage points.
4. Evaluates classification models from logreg.py / next_year.py style using >3% layoffs.
5. Optionally tries to run nf.py as a separate subprocess, because neuralforecast is often
   not installed and can be slow.
6. Saves comparison CSVs.

Important:
- The repo y_train/y_validate/y_test files are log1p(layoff percentage). This script converts
  predictions back with expm1() before reporting MAE_pct_points.
- It also flags scripts that exist but cannot be run because a dependency is missing.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PoissonRegressor, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LOG_TARGET = True
ELEVATED_THRESHOLD_PCT = 3.0
ELEVATED_THRESHOLD_LOG = float(np.log1p(ELEVATED_THRESHOLD_PCT))

EXPECTED_FILES = [
    "X_train.csv", "X_validate.csv", "X_test.csv",
    "y_train.csv", "y_validate.csv", "y_test.csv",
]

MODEL_FILES = [
    "models_baseline.py",
    "poisson.py",
    "xgb.py",
    "lgbm.py",
    "mlp.py",
    "logreg.py",
    "next_year.py",
    "nf.py",
]

LEAKY_CLASSIFICATION_COLUMNS = [
    "employees_end", "net_change", "attrition_rate_pct",
    "workforce_growth_pct", "hire_to_layoff_ratio", "layoff_pct_log",
]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def print_table(title: str, df: pd.DataFrame, max_rows: int = 40) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    if df is None or df.empty:
        print("No rows.")
        return
    show = df.head(max_rows).copy()
    for col in show.select_dtypes(include=[np.number]).columns:
        show[col] = show[col].round(4)
    print(show.to_string(index=False))


def package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def ensure_preprocessed_files() -> None:
    missing = [f for f in EXPECTED_FILES if not Path(f).exists()]
    if not missing:
        return
    if not Path("preprocess.py").exists():
        raise FileNotFoundError(
            f"Missing {missing}, and preprocess.py was not found. Run from the project presentation folder."
        )
    print(f"Missing {missing}; running preprocess.py first...")
    result = subprocess.run(
        [sys.executable, "preprocess.py"],
        text=True,
        capture_output=True,
        timeout=120,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("preprocess.py failed. Fix preprocessing before model comparison.")


def load_repo_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    ensure_preprocessed_files()
    X_train = pd.read_csv("X_train.csv")
    X_validate = pd.read_csv("X_validate.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").squeeze("columns")
    y_validate = pd.read_csv("y_validate.csv").squeeze("columns")
    y_test = pd.read_csv("y_test.csv").squeeze("columns")
    return X_train, X_validate, X_test, y_train, y_validate, y_test


def get_scale_columns(X: pd.DataFrame) -> List[str]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    onehot_cols = [c for c in numeric_cols if c.startswith("co_")]
    return [c for c in numeric_cols if c not in onehot_cols]


def scale_like_repo(X_train, X_validate, X_test, fix_validation_scaling: bool = True):
    """
    Repo scripts scale train with fit_transform, test with transform.
    Some scripts accidentally use fit_transform on validation.
    The default fixes that bug so validation tuning is meaningful.
    """
    Xtr = X_train.copy()
    Xva = X_validate.copy()
    Xte = X_test.copy()
    scale_cols = get_scale_columns(Xtr)
    if not scale_cols:
        return Xtr, Xva, Xte
    scaler = StandardScaler()
    Xtr[scale_cols] = scaler.fit_transform(Xtr[scale_cols])
    if fix_validation_scaling:
        Xva[scale_cols] = scaler.transform(Xva[scale_cols])
    else:
        # This reproduces the bug in several original repo scripts.
        Xva[scale_cols] = scaler.fit_transform(Xva[scale_cols])
    Xte[scale_cols] = scaler.transform(Xte[scale_cols])
    return Xtr, Xva, Xte


def log_to_pct(x) -> np.ndarray:
    return np.expm1(np.asarray(x, dtype=float))


def evaluate_regression(name: str, y_true_log, pred_log, selected_by: str, source_file: str) -> dict:
    y_true_log = np.asarray(y_true_log, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)
    y_true_pct = log_to_pct(y_true_log)
    pred_pct = np.clip(log_to_pct(pred_log), 0.0, 100.0)
    return {
        "model": name,
        "source_file": source_file,
        "selected_by": selected_by,
        "MAE_log_units": mean_absolute_error(y_true_log, pred_log),
        "RMSE_log_units": rmse(y_true_log, pred_log),
        "R2_log_units": r2_score(y_true_log, pred_log) if len(np.unique(y_true_log)) > 1 else np.nan,
        "MAE_pct_points": mean_absolute_error(y_true_pct, pred_pct),
        "RMSE_pct_points": rmse(y_true_pct, pred_pct),
        "R2_pct_space": r2_score(y_true_pct, pred_pct) if len(np.unique(y_true_pct)) > 1 else np.nan,
    }


def tune_by_validation_r2(model_factory, alphas, X_train, y_train, X_val, y_val):
    best_alpha = None
    best_score = -np.inf
    best_pred = None
    for alpha in alphas:
        model = model_factory(alpha)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = r2_score(y_val, pred) if len(np.unique(y_val)) > 1 else -np.inf
        if score > best_score:
            best_alpha = alpha
            best_score = score
            best_pred = pred
    return best_alpha, best_score, best_pred


def reconstruct_company_from_onehot(X: pd.DataFrame) -> pd.Series:
    co_cols = [c for c in X.columns if c.startswith("co_")]
    if not co_cols:
        return pd.Series(["UNKNOWN"] * len(X), index=X.index)
    return X[co_cols].idxmax(axis=1).str.replace("co_", "", regex=False)


def baseline_last_year_from_processed() -> Optional[dict]:
    if not Path("tech_employment_2000_2025.csv").exists():
        return None
    raw = pd.read_csv("tech_employment_2000_2025.csv")
    raw = raw.sort_values(["company", "year"]).copy()
    raw["layoff_pct"] = raw["layoffs"] / raw["employees_start"] * 100.0
    raw["layoff_pct_next"] = raw.groupby("company")["layoff_pct"].shift(-1)
    # Match repo baseline script: SPLIT_YEAR = 2020 and test includes year >= 2020.
    test_df = raw[raw["year"] >= 2020].dropna(subset=["layoff_pct", "layoff_pct_next"])
    return evaluate_regression(
        "Repo Baseline 1: Last Year's Value",
        np.log1p(test_df["layoff_pct_next"]),
        np.log1p(test_df["layoff_pct"]),
        "direct baseline",
        "models_baseline.py",
    )


def baseline_historical_average_from_processed() -> Optional[dict]:
    if not Path("tech_employment_2000_2025.csv").exists():
        return None
    raw = pd.read_csv("tech_employment_2000_2025.csv")
    raw = raw.sort_values(["company", "year"]).copy()
    raw["layoff_pct"] = raw["layoffs"] / raw["employees_start"] * 100.0
    raw["layoff_pct_next"] = raw.groupby("company")["layoff_pct"].shift(-1)
    train_df = raw[raw["year"] < 2020]
    company_avg = train_df.groupby("company")["layoff_pct"].mean().rename("hist_avg_pred")
    test_df = raw[raw["year"] >= 2020].join(company_avg, on="company")
    test_df = test_df.dropna(subset=["hist_avg_pred", "layoff_pct_next"])
    return evaluate_regression(
        "Repo Baseline 2: Historical Average",
        np.log1p(test_df["layoff_pct_next"]),
        np.log1p(test_df["hist_avg_pred"]),
        "direct baseline",
        "models_baseline.py",
    )


def run_repo_regression_models() -> pd.DataFrame:
    X_train, X_val, X_test, y_train, y_val, y_test = load_repo_split()
    rows: List[dict] = []

    # Baselines from models_baseline.py
    for fn in [baseline_last_year_from_processed, baseline_historical_average_from_processed]:
        res = fn()
        if res:
            rows.append(res)

    global_mean = float(y_train.mean())
    rows.append(evaluate_regression(
        "Repo Baseline 3: Global Mean",
        y_test,
        np.full(len(y_test), global_mean),
        "train mean",
        "models_baseline.py",
    ))

    Xtr_s, Xva_s, Xte_s = scale_like_repo(X_train, X_val, X_test, fix_validation_scaling=True)

    # Ridge from models_baseline.py
    ridge_alphas = [0.1, 1, 10, 100, 1000]
    best_alpha, best_score, _ = tune_by_validation_r2(
        lambda a: Ridge(alpha=a), ridge_alphas, Xtr_s, y_train, Xva_s, y_val
    )
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(Xtr_s, y_train)
    rows.append(evaluate_regression(
        f"Repo Model 1: Ridge Linear Regression alpha={best_alpha}",
        y_test,
        ridge.predict(Xte_s),
        f"validation R2={best_score:.4f}",
        "models_baseline.py",
    ))

    # Poisson from poisson.py. It trains on log target in the repo.
    poisson_alphas = [0.1, 1, 10, 100, 1000]
    best_alpha, best_score, _ = tune_by_validation_r2(
        lambda a: PoissonRegressor(max_iter=1000, alpha=a), poisson_alphas, Xtr_s, y_train, Xva_s, y_val
    )
    poisson = PoissonRegressor(max_iter=1000, alpha=best_alpha)
    poisson.fit(Xtr_s, y_train)
    rows.append(evaluate_regression(
        f"Repo Model 2: Poisson Regression alpha={best_alpha}",
        y_test,
        poisson.predict(Xte_s),
        f"validation R2={best_score:.4f}",
        "poisson.py",
    ))

    # XGBoost from xgb.py
    if package_available("xgboost"):
        import xgboost as xgb
        xgb_alphas = [0.1, 1, 5, 10, 100]
        best_alpha, best_score, _ = tune_by_validation_r2(
            lambda a: xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                reg_lambda=a,
            ),
            xgb_alphas,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            reg_lambda=best_alpha,
        )
        model.fit(X_train, y_train)
        rows.append(evaluate_regression(
            f"Repo Model 3: XGBoost reg_lambda={best_alpha}",
            y_test,
            model.predict(X_test),
            f"validation R2={best_score:.4f}",
            "xgb.py",
        ))
    else:
        rows.append({"model": "Repo Model 3: XGBoost", "source_file": "xgb.py", "error": "xgboost not installed"})

    # LightGBM from lgbm.py
    if package_available("lightgbm"):
        from lightgbm import LGBMRegressor
        lgbm_alphas = [0.1, 1, 5, 10, 100]
        best_alpha, best_score, _ = tune_by_validation_r2(
            lambda a: LGBMRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                reg_lambda=a,
                verbosity=-1,
            ),
            lgbm_alphas,
            Xtr_s,
            y_train,
            Xva_s,
            y_val,
        )
        model = LGBMRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            reg_lambda=best_alpha,
            verbosity=-1,
        )
        model.fit(Xtr_s, y_train)
        rows.append(evaluate_regression(
            f"Repo Model 4: LightGBM reg_lambda={best_alpha}",
            y_test,
            model.predict(Xte_s),
            f"validation R2={best_score:.4f}",
            "lgbm.py",
        ))
    else:
        rows.append({"model": "Repo Model 4: LightGBM", "source_file": "lgbm.py", "error": "lightgbm not installed"})

    # MLP from mlp.py. The original has a small bug: final_model uses alpha instead of best_alpha.
    # We report both bug-faithful and corrected so you can see the impact.
    mlp_alphas = [0.1, 1, 5, 10, 100]
    best_alpha = None
    best_score = -np.inf
    for alpha in mlp_alphas:
        model = MLPRegressor(hidden_layer_sizes=(200, 200), alpha=alpha, random_state=RANDOM_STATE, max_iter=1000)
        model.fit(Xtr_s, y_train)
        pred = model.predict(Xva_s)
        score = r2_score(y_val, pred) if len(np.unique(y_val)) > 1 else -np.inf
        if score > best_score:
            best_alpha = alpha
            best_score = score

    mlp_corrected = MLPRegressor(hidden_layer_sizes=(100, 100), alpha=best_alpha, random_state=RANDOM_STATE, max_iter=1000)
    mlp_corrected.fit(Xtr_s, y_train)
    rows.append(evaluate_regression(
        f"Repo Model 5: MLP corrected alpha={best_alpha}",
        y_test,
        mlp_corrected.predict(Xte_s),
        f"validation R2={best_score:.4f}",
        "mlp.py",
    ))

    bug_alpha = mlp_alphas[-1]
    mlp_bug = MLPRegressor(hidden_layer_sizes=(100, 100), alpha=bug_alpha, random_state=RANDOM_STATE, max_iter=1000)
    mlp_bug.fit(Xtr_s, y_train)
    rows.append(evaluate_regression(
        f"Repo Model 5b: MLP bug-faithful alpha={bug_alpha}",
        y_test,
        mlp_bug.predict(Xte_s),
        "original final alpha variable",
        "mlp.py",
    ))

    out = pd.DataFrame(rows)
    if "MAE_pct_points" in out.columns:
        out = out.sort_values("MAE_pct_points", ascending=True, na_position="last")
    return out


def drop_leaky_for_classification(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=[c for c in LEAKY_CLASSIFICATION_COLUMNS if c in X.columns])


def safe_roc_auc(y_true, prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, prob)


def evaluate_classifier(name: str, model, X_trainval, y_trainval_cls, X_test, y_test_cls, source_file: str) -> dict:
    model.fit(X_trainval, y_trainval_cls)
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    else:
        prob = pred
    return {
        "model": name,
        "source_file": source_file,
        "threshold_pct": ELEVATED_THRESHOLD_PCT,
        "Test_ROC_AUC": safe_roc_auc(y_test_cls, prob),
        "Test_F1": f1_score(y_test_cls, pred, zero_division=0),
        "Test_Precision": precision_score(y_test_cls, pred, zero_division=0),
        "Test_Recall": recall_score(y_test_cls, pred, zero_division=0),
        "Test_Accuracy": accuracy_score(y_test_cls, pred),
        "Test_positive_rate": float(np.mean(y_test_cls)),
    }


def run_repo_classification_models() -> pd.DataFrame:
    X_train, X_val, X_test, y_train, y_val, y_test = load_repo_split()

    Xtr = drop_leaky_for_classification(X_train)
    Xva = drop_leaky_for_classification(X_val)
    Xte = drop_leaky_for_classification(X_test)
    X_trainval = pd.concat([Xtr, Xva], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    y_trainval_cls = (y_trainval > ELEVATED_THRESHOLD_LOG).astype(int)
    y_test_cls = (y_test > ELEVATED_THRESHOLD_LOG).astype(int)

    rows: List[dict] = []

    # logreg.py style: LogisticRegressionCV threshold is train 75th percentile.
    threshold = float(y_train.quantile(0.75))
    y_train_cls_q = (y_train > threshold).astype(int)
    y_test_cls_q = (y_test > threshold).astype(int)
    try:
        logreg_cv = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegressionCV(cv=5, max_iter=1000, class_weight="balanced")),
        ])
        rows.append(evaluate_classifier(
            f"Repo logreg.py: LogisticRegressionCV threshold=train 75th percentile ({np.expm1(threshold):.2f}%)",
            logreg_cv,
            X_train,
            y_train_cls_q,
            X_test,
            y_test_cls_q,
            "logreg.py",
        ))
    except Exception as exc:
        rows.append({"model": "Repo logreg.py: LogisticRegressionCV", "source_file": "logreg.py", "error": str(exc)})

    # next_year.py model set.
    models: Dict[str, object] = {
        "Repo next_year.py: Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Repo next_year.py: Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=4, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Repo next_year.py: Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE
        ),
        "Repo next_year.py: SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
        ]),
    }
    if package_available("xgboost"):
        import xgboost as xgb
        pos = max(1, int((y_trainval_cls == 1).sum()))
        neg = max(1, int((y_trainval_cls == 0).sum()))
        models["Repo next_year.py: XGBoost Classifier"] = xgb.XGBClassifier(
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

    # CV metrics for next_year-style models.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        try:
            cv_auc = cross_val_score(model, X_trainval, y_trainval_cls, cv=cv, scoring="roc_auc").mean()
            cv_f1 = cross_val_score(model, X_trainval, y_trainval_cls, cv=cv, scoring="f1").mean()
            res = evaluate_classifier(name, model, X_trainval, y_trainval_cls, Xte, y_test_cls, "next_year.py")
            res["CV_ROC_AUC"] = cv_auc
            res["CV_F1"] = cv_f1
            rows.append(res)
        except Exception as exc:
            rows.append({"model": name, "source_file": "next_year.py", "error": str(exc)})

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["CV_ROC_AUC", "Test_ROC_AUC", "Test_F1"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False, na_position="last")
    return out


def run_optional_nf_script(timeout_seconds: int = 180) -> pd.DataFrame:
    if not Path("nf.py").exists():
        return pd.DataFrame([{"source_file": "nf.py", "status": "missing"}])
    if not package_available("neuralforecast"):
        return pd.DataFrame([{"source_file": "nf.py", "status": "skipped", "reason": "neuralforecast not installed"}])
    try:
        result = subprocess.run(
            [sys.executable, "nf.py"],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        Path("nf_stdout.txt").write_text(result.stdout, encoding="utf-8")
        Path("nf_stderr.txt").write_text(result.stderr, encoding="utf-8")
        return pd.DataFrame([{
            "source_file": "nf.py",
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout_saved_to": "nf_stdout.txt",
            "stderr_saved_to": "nf_stderr.txt",
        }])
    except subprocess.TimeoutExpired:
        return pd.DataFrame([{"source_file": "nf.py", "status": "timeout", "timeout_seconds": timeout_seconds}])
    except Exception as exc:
        return pd.DataFrame([{"source_file": "nf.py", "status": "failed", "reason": str(exc)}])


def scan_project_files() -> pd.DataFrame:
    rows = []
    for f in MODEL_FILES:
        p = Path(f)
        rows.append({
            "file": f,
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "covered_by_this_script": f in {
                "models_baseline.py", "poisson.py", "xgb.py", "lgbm.py", "mlp.py", "logreg.py", "next_year.py", "nf.py"
            },
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("Project model file scan:")
    scan = scan_project_files()
    print(scan.to_string(index=False))
    scan.to_csv("zip_model_file_scan.csv", index=False)

    X_train, X_val, X_test, y_train, y_val, y_test = load_repo_split()
    print("\nLoaded repo split:")
    print(f"  X_train={X_train.shape}, X_validate={X_val.shape}, X_test={X_test.shape}")
    print(f"  y_train={len(y_train)}, y_validate={len(y_val)}, y_test={len(y_test)}")
    print("  Repo target appears to be log1p(layoff percentage). This script also reports percentage-point errors.")

    reg_results = run_repo_regression_models()
    print_table("REGRESSION MODELS FROM ZIP — evaluated on repo test split", reg_results)
    reg_results.to_csv("zip_regression_model_comparison.csv", index=False)

    clf_results = run_repo_classification_models()
    print_table("CLASSIFICATION MODELS FROM ZIP — elevated layoff classification", clf_results)
    clf_results.to_csv("zip_classification_model_comparison.csv", index=False)

    nf_status = run_optional_nf_script(timeout_seconds=180)
    print_table("NEURALFORECAST nf.py STATUS", nf_status)
    nf_status.to_csv("zip_neuralforecast_status.csv", index=False)

    print("\nSaved:")
    print(" - zip_model_file_scan.csv")
    print(" - zip_regression_model_comparison.csv")
    print(" - zip_classification_model_comparison.csv")
    print(" - zip_neuralforecast_status.csv")
    if Path("nf_stdout.txt").exists():
        print(" - nf_stdout.txt")
    if Path("nf_stderr.txt").exists():
        print(" - nf_stderr.txt")

    print("\nInterpretation note:")
    print("The original repo scripts train on y_train.csv, which is log1p(layoff percentage).")
    print("Use MAE_pct_points for human-readable layoff-percentage error; use MAE_log_units only if comparing against old script output.")


if __name__ == "__main__":
    main()
